from ultralytics import YOLO
from collections import defaultdict, Counter, deque
import cv2
import time
import numpy as np
import threading
# ================== 1. MODEL ==================
detector = YOLO("/workspace/best_license_plate_detection.engine", task="detect")
ocr_model = YOLO("/workspace/best_license_plate_ocr_yolo.engine", task="detect")
dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
detector.predict(dummy_img, verbose=False)
ocr_model.predict(dummy_img, verbose=False)
# ================== 2. FUNCTION ==================
def fit_line_pca(points):
    points = np.array(points, dtype=float)
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    direction = vt[0]
    if direction[0] < 0:
        direction = -direction
    return centroid, direction
def perp_distance(point, centroid, direction):
    v = point - centroid
    proj = np.dot(v, direction) * direction
    return np.linalg.norm(v - proj)
def project_1d(point, centroid, direction):
    return np.dot(point - centroid, direction)
def assemble_plate_text_v4(chars, variance_ratio_threshold=0.25):
    if not chars:
        return None
    if len(chars) == 1:
        return chars[0]["char"]
    pts = np.array([[c["cx"], c["cy"]] for c in chars], dtype=float)
    centroid, direction = fit_line_pca(pts)
    normal = np.array([-direction[1], direction[0]])
    perp_vals = np.array([np.dot(p - centroid, normal) for p in pts])
    order_by_perp = np.argsort(perp_vals)
    sorted_vals = perp_vals[order_by_perp]
    n = len(sorted_vals)
    total_ss = np.sum((sorted_vals - sorted_vals.mean()) ** 2)
    best_split, best_ss = None, total_ss
    for k in range(1, n):
        left, right = sorted_vals[:k], sorted_vals[k:]
        ss = np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2)
        if ss < best_ss:
            best_ss, best_split = ss, k
    if best_split is None or total_ss == 0 or (best_ss / total_ss) > variance_ratio_threshold:
        order = np.argsort([project_1d(p, centroid, direction) for p in pts])
        return "".join(chars[i]["char"] for i in order)
    row0_idx = order_by_perp[:best_split]
    row1_idx = order_by_perp[best_split:]
    result = ""
    for idx in (row0_idx, row1_idx):
        row_pts = pts[idx]
        c, d = fit_line_pca(row_pts) if len(row_pts) >= 2 else (centroid, direction)
        order = np.array(idx)[np.argsort([project_1d(pts[i], c, d) for i in idx])]
        result += "".join(chars[i]["char"] for i in order)
    return result
def read_plate_fast(crop_img) -> str:
    results = ocr_model.predict(crop_img, imgsz=640, conf=0.4, iou=0.4, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    names = r.names
    chars = []
    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = box
        chars.append({"char": names[cls_id], "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "h": y2 - y1})
    return assemble_plate_text_v4(chars)
def is_sharp_enough(crop_img, threshold=100.0) -> bool:
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > threshold
# ================== 3. SHARED STATE ==================
frame_cond = threading.Condition()
latest_frame = None
frame_ready = False
frame_version = 0
results_lock = threading.Lock()
latest_results = []   # list[{"box":(x1,y1,x2,y2), "tid":int, "text":str}]
stop_event = threading.Event()
# ================== 4. THREAD CAPTURE ==================
def capture_thread_func(source=0):
    global latest_frame, frame_ready, frame_version
    cap = cv2.VideoCapture(source)
    is_file = isinstance(source, str)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps and fps > 0 else 0.033
    while not stop_event.is_set():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            with frame_cond:
                frame_cond.notify_all()
            break
        with frame_cond:
            latest_frame = frame
            frame_ready = True
            frame_version += 1
            frame_cond.notify_all()
        if is_file:
            sleep_time = frame_delay - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)
    cap.release()
# ================== 5. THREAD PROCESSING ==================
def processing_thread_func():
    global latest_results
    track_ocr_history = defaultdict(list)
    track_frame_count = defaultdict(int)
    track_final_result = {}
    track_last_seen = {}
    frame_idx = 0
    STALE_AFTER = 30
    OCR_EVERY_N_FRAMES = 5
    VOTES_TO_LOCK = 3
    proc_frame_times = deque(maxlen=30)
    proc_count = 0
    last_processed_version = -1
    while not stop_event.is_set():
        with frame_cond:
            got_new = frame_cond.wait_for(lambda: stop_event.is_set() or (frame_ready and frame_version != last_processed_version), timeout=1.0)
            if stop_event.is_set():
                break
            if not got_new:
                continue
            frame = latest_frame.copy()
            last_processed_version = frame_version
        frame_idx += 1
        frame_results = []
        t_proc_start = time.time()
        try:
            results = detector.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640, verbose=False)
            r = results[0]
            if r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                for box, tid in zip(boxes, track_ids):
                    track_last_seen[tid] = frame_idx
                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    track_frame_count[tid] += 1
                    if tid in track_final_result:
                        text = track_final_result[tid]
                    else:
                        text = None
                        if track_frame_count[tid] % OCR_EVERY_N_FRAMES == 0 and is_sharp_enough(crop):
                            pred = read_plate_fast(crop)
                            if pred:
                                track_ocr_history[tid].append(pred)
                                counts = Counter(track_ocr_history[tid])
                                best_text, best_count = counts.most_common(1)[0]
                                if best_count >= VOTES_TO_LOCK:
                                    track_final_result[tid] = best_text
                                    text = best_text
                        if text is None and track_ocr_history[tid]:
                            text = Counter(track_ocr_history[tid]).most_common(1)[0][0]
                    frame_results.append({"box": (x1, y1, x2, y2), "tid": tid, "text": text if text else f"ID:{tid} (dang doc...)"})
        except Exception as e:
            print(f"Loi xu ly frame: {e}")
        if frame_idx % STALE_AFTER == 0:
            stale_ids = [tid for tid, last in track_last_seen.items() if frame_idx - last > STALE_AFTER]
            for tid in stale_ids:
                track_ocr_history.pop(tid, None)
                track_frame_count.pop(tid, None)
                track_final_result.pop(tid, None)
                track_last_seen.pop(tid, None)
        with results_lock:
            latest_results = frame_results
        proc_frame_times.append(time.time() - t_proc_start)
        proc_count += 1
        if proc_count % 30 == 0:
            avg_latency = sum(proc_frame_times) / len(proc_frame_times)
            proc_fps = 1 / avg_latency if avg_latency > 0 else 0
            print(f"[Processing] ~{proc_fps:.1f} FPS | latency min={min(proc_frame_times)*1000:.1f}ms "
            f"max={max(proc_frame_times)*1000:.1f}ms")
            print(f"[Processing thread] ~{proc_fps:.1f} FPS (30 frame gần nhất)")
# ================== 6. MAIN THREAD ==================
def main():
    global detector, ocr_model
    t_capture = threading.Thread(target=capture_thread_func, args=('/workspace/a.mp4',), daemon=True)
    t_process = threading.Thread(target=processing_thread_func, daemon=True)
    t_capture.start()
    t_process.start()
    prev_time = time.time()
    fps_history = deque(maxlen=30)
    last_shown_version = -1
    while not stop_event.is_set():
        with frame_cond:
            got_new = frame_cond.wait_for(lambda: stop_event.is_set() or (frame_ready and frame_version != last_shown_version), timeout=1.0)
            if stop_event.is_set():
                break
            if not got_new:
                continue
            display_frame = latest_frame.copy()
            last_shown_version = frame_version
        with results_lock:
            results_to_draw = list(latest_results)
        for res in results_to_draw:
            x1, y1, x2, y2 = res["box"]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, res["text"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        curr_time = time.time()
        if curr_time > prev_time:
            fps_history.append(1 / (curr_time - prev_time))
        prev_time = curr_time
        # time.sleep(0.01) # log
        smoothed_fps = sum(fps_history) / len(fps_history) if fps_history else 0
        cv2.putText(display_frame, f"FPS: {int(smoothed_fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow("ALPR", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break
    cv2.destroyAllWindows()
    print("CLEAN MEM")
    stop_event.set()
    with frame_cond:
        frame_cond.notify_all()
    t_capture.join(timeout=2.0)
    t_process.join(timeout=2.0)
    try:
        del detector
        del ocr_model
    except NameError:
        pass
    print("Ket thuc")
if __name__ == "__main__":
    main()