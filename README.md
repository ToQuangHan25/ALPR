# Hệ thống nhận diện và đọc chữ biển số xe (ALPR) trên thiết bị biên

Hệ thống tự động phát hiện (Detection) và đọc ký tự (OCR) trên biển số xe Việt Nam từ video theo thời gian thực, triển khai thử nghiệm trên hai môi trường: máy tính cá nhân (GPU NVIDIA RTX 3050 Ti) và thiết bị nhúng NVIDIA Jetson Xavier NX.

## Cấu trúc thư mục

```text
├── README.md
├── requirements.txt
├── report/                 # Báo cáo
├── training/               # Code huấn luyện 2 mô hình YOLO (Detection + OCR)
├── evaluation/             # Code đánh giá mô hình trên tập test (Precision/Recall/mAP)
├── inference/              # Mã nguồn thực thi chính
│   ├── main_laptop.py      # Chạy trên máy tính cá nhân (GPU NVIDIA RTX 3050 Ti)
│   └── main_jetson.py      # Chạy trên NVIDIA Jetson Xavier NX
```

## Môi trường và cài đặt

- Python 3.8+
- Thư viện chính: `ultralytics`, `opencv-python`, `numpy`, `torch`

```bash
pip install -r requirements.txt
```

> **Lưu ý:** Trên NVIDIA Jetson, cần đảm bảo môi trường JetPack đã tích hợp sẵn **TensorRT** và cấu hình biến môi trường CUDA tương thích.

## 📥 Tải trọng số mô hình (Pretrained Weights)

Do kích thước file lớn, các mô hình đã huấn luyện được lưu trữ trên Google Drive. Vui lòng tải về và đặt vào thư mục `models/`:

- 🔗 **[Models & Weights](https://drive.google.com/drive/folders/1xycIrytTuCKtMx1VwIGukGR_5Zlrm51G?usp=drive_link)**
  - `best_license_plate_detection.pt` (Mô hình phát hiện biển số YOLO26m)
  - `best_license_plate_ocr.pt` (Mô hình nhận diện ký tự YOLO26m)

## Hướng dẫn sử dụng

### 1. Huấn luyện mô hình (Tùy chọn nếu muốn huấn luyện lại từ đầu với tập dữ liệu mới):
```bash
python training/train_detection.py
python training/train_ocr.py
```

### 2. Đánh giá mô hình trên tập kiểm thử:
```bash
python evaluation/test_detection.py
python evaluation/test_ocr.py
```

### 3. Xuất mô hình sang TensorRT (.engine)
Hệ thống sử dụng định dạng `.engine` để tối ưu hóa tốc độ thực thi thời gian thực. Xuất mô hình bằng lệnh Ultralytics CLI:

```bash
# Export mô hình Detection
yolo export model=models/best_license_plate_detection.pt format=engine

# Export mô hình OCR
yolo export model=models/best_license_plate_ocr.pt format=engine
```

> **Lưu ý quan trọng:** File `.engine` được biên dịch trực tiếp dựa trên kiến trúc phần cứng GPU của từng máy, do đó **không thể sao chép file `.engine` từ Laptop sang Jetson** mà phải thực hiện lệnh `export` trực tiếp trên từng thiết bị.

### 4. Chạy hệ thống nhận diện (Inference)

```bash
# Chạy trên Laptop:
python inference/main_laptop.py

# Chạy trên NVIDIA Jetson Xavier NX:
python inference/main_jetson.py
```

*(Cập nhật đường dẫn file mô hình và đường dẫn video cần nhận diện trong file script tương ứng trước khi chạy).*

## Kết quả thực nghiệm

| Tiêu chí so sánh | Laptop (NVIDIA RTX 3050 Ti) | NVIDIA Jetson Xavier NX |
| :--- | :--- | :--- |
| **Tốc độ xử lý (Processing FPS)** | ~57 – 63 FPS | ~13 – 18 FPS |
| **Độ trễ xử lý (Latency)** | < 20 ms | ~40 – 260 ms |

- **[Demo Video](DÁN_LINK_VIDEO_DEMO_TẠI_ĐÂY)**

## Hạn chế và hướng phát triển

- Tốc độ xử lý AI trên Jetson Xavier NX thấp hơn máy tính cá nhân; có thể cải thiện bằng cách hạ độ phân giải khung hình hoặc áp dụng các mô hình nhẹ hơn.
- Thuật toán theo dõi đối tượng (ByteTrack) thỉnh thoảng bị mất dấu phương tiện.
- Thuật toán PCA ghép chữ cần bổ sung các bước căn chỉnh hình học đối với biển số bị nghiêng góc lớn.
- Thêm cơ chế stream trực tiếp từ Camera IP và tích hợp cơ sở dữ liệu để lưu kết quả.

## Tác giả

**Tô Quang Hân**  
Ngành Kỹ thuật điện tử và tin học — Khoa Vật lý, Trường Đại học Khoa học Tự nhiên, ĐHQGHN
