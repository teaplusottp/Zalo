# 🐳 Docker Setup Guide - AeroEyes (Zalo AI Challenge 2025)

## Tổng Quan

Hướng dẫn này giúp bạn:
1. ✅ Build Docker image từ Dockerfile
2. ✅ Chạy Docker container với GPU support
3. ✅ Copy source code vào container
4. ✅ Cài đặt dependencies
5. ✅ Chạy prediction và JupyterLab
6. ✅ Commit image để submit

---

## 📋 Yêu Cầu Hệ Thống

- **Docker**: v24.0.5 trở lên
- **NVIDIA Docker**: Để support GPU
- **GPU**: CUDA 11.3 compatible (trong docker image)
- **RAM**: ≥ 8GB
- **Disk**: ≥ 20GB

---

## 🚀 Cách Chạy

### **Bước 1: Build Docker Image**

```bash
# Từ folder chứa Dockerfile
cd /path/to/zalo/project

# Build image
docker build -t zac2025:v1 .

# Xác nhận image được tạo
docker images | grep zac2025
```

**Output mong đợi:**
```
REPOSITORY   TAG    IMAGE ID      CREATED        SIZE
zac2025      v1     abc123def456  2 minutes ago   8.5GB
```

---

### **Bước 2: Khởi Động Container**

```bash
# Run container với GPU support
docker run --gpus all \
  --network host \
  -it \
  --name zac2025 \
  zac2025:v1 \
  /bin/bash
```

**Giải thích flags:**
- `--gpus all`: Sử dụng tất cả GPU devices
- `--network host`: Cho phép truy cập localhost (cho JupyterLab)
- `-it`: Interactive terminal
- `--name zac2025`: Đặt tên container
- `/bin/bash`: Shell entry point

**Bạn sẽ thấy prompt:**
```
root@container_id:/code#
```

---

### **Bước 3: Xác Nhận Cấu Hình**

```bash
# Kiểm tra Python
python3 --version
# Python 3.10.x

# Kiểm tra PyTorch & CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Kiểm tra folder structure
ls -la /code/
ls -la /result/
```

---

### **Bước 4: Chạy Prediction**

**Option A: Chạy predict.py trực tiếp (Nên chọn)**

```bash
# Trong container
cd /code
bash predict.sh
```

**predict.sh sẽ:**
1. Tạo folder `/result` nếu chưa tồn tại
2. Chạy `python3 predict.py`
3. Tạo output files:
   - `/result/submission.json`
   - `/result/time_submission.csv`

**Output mong đợi:**
```
✅ Starting prediction...
📊 Loaded models successfully
🎬 Processing video: LifeJacket_0
...
✅ Prediction complete!
📁 Output: /result/submission.json
⏱️  Time: /result/time_submission.csv
```

---

### **Bước 5: Chạy JupyterLab (Để Test Notebook)**

**Mở terminal khác (không đóng container), chạy:**

```bash
# Trong container terminal khác
cd /code
bash start_jupyter.sh
```

**start_jupyter.sh sẽ:**
- Khởi động JupyterLab trên port 9777
- Password: `zac2025`
- Token: `zac2025`

**Truy cập từ máy local:**
```
http://localhost:9777
```

**Credentials:**
- Password: `zac2025`
- Token: `zac2025`

**Hoặc manual run:**
```bash
jupyter lab --port 9777 --ip 0.0.0.0 \
  --NotebookApp.password='zac2025' \
  --NotebookApp.token='zac2025' \
  --allow-root \
  --no-browser
```

---

### **Bước 6: Copy Kết Quả Ra Ngoài Container** *(Optional)*

**Từ terminal máy local (không phải container):**

```bash
# Copy result folder ra ngoài
docker cp zac2025:/result /path/to/local/result

# Kiểm tra
ls -la /path/to/local/result/
```

---

### **Bước 7: Commit Image**

**Khi hoàn thành, commit container thành image:**

```bash
# Lấy container ID
docker ps -a | grep zac2025

# Commit
docker commit zac2025 zac2025:v1

# Verify
docker images | grep zac2025
```

**Output:**
```
REPOSITORY   TAG    IMAGE ID      CREATED        SIZE
zac2025      v1     xyz789abc123  Just now       8.5GB
```

---

## 🛑 Dừng & Xóa Container

```bash
# Dừng container (nếu đang chạy)
docker stop zac2025

# Xóa container
docker rm zac2025

# Xóa image (nếu cần)
docker rmi zac2025:v1
```

---

## 📝 Chỉnh Sửa Files Trong Container

### **Nếu cần sửa code trong container:**

```bash
# Dùng vim
vim /code/predict.py

# Hoặc copy file vào container rồi sửa
docker cp local_file.py zac2025:/code/
```

---

## 🔧 Troubleshooting

### **❌ Error: "docker: Error response from daemon: could not select device driver"**

**Giải pháp:** Cài NVIDIA Container Runtime
```bash
# Ubuntu/Debian
sudo apt-get install -y nvidia-container-runtime

# Restart Docker daemon
sudo systemctl restart docker
```

---

### **❌ Error: "RuntimeError: CUDA out of memory"**

**Giải pháp:** Giảm batch size hoặc dùng CPU
```bash
# Chỉ dùng CPU (chậm hơn)
docker run -it --name zac2025 zac2025:v1

# Hoặc set environment
export CUDA_VISIBLE_DEVICES=0
```

---

### **❌ Error: "ModuleNotFoundError: No module named 'mobilesamv2'"**

**Giải pháp:** Đảm bảo sys.path được setup đúng
```bash
# Check import path
python3 -c "import sys; print(sys.path)"

# Verify mobilesamv2 folder
ls -la /code/MobileSAMv2/mobilesamv2/
```

---

### **❌ JupyterLab không thể truy cập**

**Giải pháp:**
```bash
# Kiểm tra port
netstat -tulpn | grep 9777

# Hoặc dùng port khác
jupyter lab --port 9778 --ip 0.0.0.0 --allow-root --no-browser
```

---

## 📊 Workflow Hoàn Chỉnh

```
┌─────────────────────────────────────────┐
│ 1. Build Image                          │
│    docker build -t zac2025:v1 .        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. Run Container                        │
│    docker run --gpus all ... zac2025:v1│
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. Verify Setup                         │
│    python3 -c "import torch; ..."       │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 4. Run Prediction                       │
│    bash /code/predict.sh                │
│    ↓                                    │
│    /result/submission.json ✅           │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 5. Test Notebook (Optional)             │
│    bash /code/start_jupyter.sh          │
│    → http://localhost:9777              │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 6. Commit Image                         │
│    docker commit zac2025 zac2025:v1     │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 7. Submit to BTC ✅                     │
│    docker save zac2025:v1 | ...         │
└─────────────────────────────────────────┘
```

---

## 📂 Container Directory Structure

```
/code/                          (Working directory)
├── MobileSAMv2/               (Model folder)
├── weight/                    (Checkpoints)
├── segment_objects/           (Templates)
├── predict.py                 (Main script)
├── predict_notebook.ipynb     (Notebook)
├── final4_optimized.py        (Dev version)
├── predict.sh                 (Runner script)
├── start_jupyter.sh           (JupyterLab runner)
├── requirements.txt           (Dependencies)
├── Dockerfile                 (Build config)
└── README.md

/result/                        (Output directory - auto created)
├── submission.json
├── time_submission.csv
├── jupyter_submission.json    (from notebook)
└── jupyter_time_submission.csv
```

---

## ✅ Checklist Trước Khi Submit

- ✅ Dockerfile build thành công
- ✅ Container chạy với GPU support
- ✅ `python3 predict.py` chạy không lỗi
- ✅ `/result/submission.json` được tạo
- ✅ Notebook 4 cells chạy được (optional)
- ✅ Image commit thành công
- ✅ Models & weights giống với development

---

## 🔗 Liên Quan

- **Base Image**: pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
- **Python**: 3.10.x
- **PyTorch**: 1.12.1
- **CUDA**: 11.3
- **cuDNN**: 8

---

## 📞 Support

Nếu gặp vấn đề:

1. Kiểm tra Docker version: `docker --version`
2. Kiểm tra GPU: `nvidia-smi`
3. Xem container logs: `docker logs zac2025`
4. Debug imports: `python3 -c "import mobilesamv2; ..."`

---

**Team [HCMUS - FIT] DeepPL**  
**Zalo AI Challenge 2025 - AeroEyes**

Good luck! 🚀
