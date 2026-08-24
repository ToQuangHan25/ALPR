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
