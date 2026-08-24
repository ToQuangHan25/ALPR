# Hệ thống nhận diện và đọc chữ biển số xe (ALPR) trên thiết bị biên

Hệ thống tự động phát hiện (detect) và đọc (OCR) biển số xe Việt Nam từ video theo thời gian thực, triển khai trên hai môi trường: máy tính có GPU rời (laptop) và thiết bị nhúng NVIDIA Jetson Xavier NX.

## Cấu trúc thư mục

```
├── README.md
├── requirements.txt
├── report/                  # Báo cáo hệ thống
├── training/                # Code huấn luyện 2 model YOLO (detect + OCR)
├── evaluation/              # Code đánh giá model trên tập test (Precision/Recall/mAP)
├── inference/               # Code chạy chính
│   ├── main_laptop.py       # Chạy trên máy tính có GPU (laptop)
│   └── main_jetson.py       # Chạy trên NVIDIA Jetson Xavier NX
├── models/                  # File model .pt (chưa export)
└── results/                 # Kết quả
```

## Môi trường và cài đặt

- Python 3.8+
- Thư viện chính: `ultralytics`, `opencv-python`, `numpy`, `torch`

```bash
pip install -r requirements.txt
```

Riêng khi chạy trên Jetson, cần cài thêm TensorRT (thường có sẵn trong JetPack) để export và chạy model dạng `.engine`.

## Cách sử dụng

**1. Huấn luyện model** (tùy chọn — có thể bỏ qua nếu chỉ cần chạy thử với model có sẵn trong `models/`):
```bash
python training/train_detection.py
python training/train_ocr.py
```

**2. Đánh giá model trên tập test:**
```bash
python evaluation/test_detection.py
python evaluation/test_ocr.py
```

**3. Export model sang TensorRT** (bắt buộc trước khi chạy inference, vì `app.py`/`app1.py` đọc file `.engine` chứ không đọc trực tiếp `.pt`):
```python
from ultralytics import YOLO
YOLO("models/best_license_plate_detection.pt").export(format="engine")
YOLO("models/best_license_plate_ocr_yolo.pt").export(format="engine")
```
Lưu ý: file `.engine` phải export riêng trên từng máy sẽ chạy nó (không dùng chéo được giữa laptop và Jetson do khác kiến trúc phần cứng).

**4. Chạy hệ thống:**
```bash
# Trên laptop
python inference/main_laptop.py

# Trên Jetson
python inference/main_jetson.py
```
Sửa đường dẫn model và đường dẫn video đầu vào trực tiếp trong file trước khi chạy.

## Kết quả

| Tiêu chí | Laptop (RTX 3050Ti) | Jetson Xavier NX |
|---|---|---|
| Tốc độ xử lý | ~57–63 FPS | ~13–18 FPS |
| Độ trễ xử lý | < 20 ms | ~40–260 ms |

Video demo: *(dán link Google Drive / YouTube tại đây)*

Ảnh minh họa kết quả nhận diện được lưu trong thư mục `results/`.

## Hạn chế và hướng phát triển

- FPS trên Jetson thấp hơn đáng kể so với laptop, có thể cải thiện bằng cách giảm độ phân giải đầu vào hoặc dùng model nhẹ hơn.
- Thuật toán tracking (ByteTrack) thỉnh thoảng bị mất dấu xe đang theo dõi.
- Hiện xử lý từ video quay sẵn, chưa hỗ trợ stream trực tiếp từ camera IP/điện thoại qua mạng.
- Chưa có cơ chế lưu kết quả nhận diện ra file/cơ sở dữ liệu.

## Tác giả

Tô Quang Hân — Kỹ thuật điện tử và tin học, Khoa Vật lý, Trường Đại học Khoa học Tự nhiên, ĐHQGHN
