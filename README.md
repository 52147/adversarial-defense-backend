
# 🛡️ Adversarial Defense Backend

This backend, built with **FastAPI**, provides an **Adversarial Defense API** that supports:
- 🔹 **Adversarial Example Generation**
- 🔹 **Defensive Mechanisms**
- 🔹 **Image Classification (MNIST)**
- 🔹 **Multiple Defense Methods** (Gaussian Blur, Bilateral Filter, Median Filter)

---

## 📦 **Project Structure**
```bash
adversarial-defense-backend/
│── models/                  # Model-related files
│   ├── classify.py          # CNN classifier
│   ├── defense.py           # Defense mechanisms
│   ├── train.py             # Train the MNIST model
│── static/                  # Stores uploaded and processed images
│── main.py                  # FastAPI entry point
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
```
---

## 🚀 **Getting Started**
### **1️⃣ Install dependencies**
```bash
git clone https://github.com/52147/adversarial-defense-backend.git
cd adversarial-defense-backend
pip install -r requirements.txt
```
### **2️⃣ Start the Backend Server**

uvicorn main:app --host 0.0.0.0 --port 8000

The API will be available at http://127.0.0.1:8000.

Visit http://127.0.0.1:8000/docs for Swagger API documentation.

### **📌 API Endpoints**

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET`  | `/` | API health check |
| `POST` | `/upload/` | Upload and process an image |
| `GET`  | `/generate_adversarial_example?epsilon=0.3` | Generate adversarial example |
| `POST` | `/defend/` | Apply defense methods |
| `POST` | `/classify/` | Classify an image |

### **📡 Deployment**

This backend is deployed on Render:
🔗 Backend API: https://adversarial-defense-backend.onrender.com/

To deploy manually:

pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

### **📜 License**

MIT License - Free to use and modify.






# 🛡️ Adversarial Defense Backend

本後端使用 **FastAPI** 提供 **對抗樣本防禦（Adversarial Defense）** API，支援以下功能：
- 🔹 **對抗樣本生成**（Adversarial Example Generation）
- 🔹 **防禦對抗樣本**（Defense Mechanisms）
- 🔹 **圖像分類（MNIST）**（Image Classification）
- 🔹 **支援不同防禦方法**（Gaussian Blur, Bilateral Filter, Median Filter）

---

## 📦 **專案架構**
```
adversarial-defense-backend/
│── models/                  # 模型相關文件
│   ├── classify.py          # CNN 分類器
│   ├── defense.py           # 防禦方法
│   ├── train.py             # 訓練 MNIST 模型
│── static/                  # 儲存上傳與處理後的圖片
│── main.py                  # FastAPI 入口點
│── requirements.txt         # 依賴環境
│── README.md                # 專案說明文件
```
---

## 🚀 **快速開始**
### **1️⃣ 安裝依賴**
```bash
git clone https://github.com/52147/adversarial-defense-backend.git
cd adversarial-defense-backend
pip install -r requirements.txt
```
### **2️⃣ 啟動後端伺服器**

uvicorn main:app --host 0.0.0.0 --port 8000

API 伺服器會啟動於 http://127.0.0.1:8000。

你可以訪問 http://127.0.0.1:8000/docs 查看 Swagger API 文檔。

### **📌 API 端點**

| 方法 | 端點 | 描述 |
|------|------|------|
| `GET`  | `/` | API 健康檢查 |
| `POST` | `/upload/` | 上傳並處理圖片 |
| `GET`  | `/generate_adversarial_example?epsilon=0.3` | 生成對抗樣本 |
| `POST` | `/defend/` | 進行防禦 |
| `POST` | `/classify/` | 對圖片進行分類 |

### **📡 部屬**

此專案已部署於 Render：
🔗 後端 API: https://adversarial-defense-backend.onrender.com/

如果你需要自行部署：

pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

### **📜 License**

MIT License - 本專案可自由使用與修改。

---
