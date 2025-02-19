from fastapi import FastAPI, File, UploadFile, Query
from models.defense import defend_adversarial  
import shutil
import os
from fastapi.responses import FileResponse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from PIL import Image
from fastapi.responses import JSONResponse
from models.classify import classify_image
import io  # 這行是缺少的部分



app = FastAPI()

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Adversarial Defense API is running!"}

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 進行對抗樣本防禦
    processed_img = defend_adversarial(file_path)
    
    return {"filename": file.filename, "message": "Image processed!"}

# 生成對抗樣本的 API
@app.get("/generate_adversarial_example")
def generate_adversarial_example(epsilon: float = Query(0.2, ge=0.0, le=1.0)):  # 這裡的 Query 需要 import
    print("Generating adversarial example...")  # Debug 訊息
    # 1. 加載 MNIST 數據集
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    image, label = mnist[0]  # 取第一張圖片
    image = image.unsqueeze(0)  # 添加 batch 維度

    # 2. 生成對抗樣本（FGSM 近似）
    perturbation = torch.sign(torch.rand_like(image) - 0.5) * epsilon  # 生成隨機擾動
    perturbed_image = image + perturbation
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 限制值域 [0,1]

    # 3. 保存對抗樣本為 PNG
    adv_image = perturbed_image.squeeze().detach().numpy()
    adv_image = (adv_image * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_image)
    image_path = "adversarial_example.png"
    adv_pil.save(image_path)

    print(f"Saved adversarial image to {image_path}")  # Debug 訊息

    # 4. 返回圖片文件
    return FileResponse(image_path, media_type="image/png", filename="adversarial_example.png")

@app.post("/defend/")
async def defend_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 進行對抗樣本防禦
    processed_image_path = defend_adversarial(file_path)

    # **確保 API 返回圖片**
    if os.path.exists(processed_image_path):
        return FileResponse(processed_image_path, media_type="image/png", filename="defended_image.png")
    else:
        return {"error": "Failed to process image"}
    
# 確保你的模型正確加載
from models.classify import classify_image  # 確保這個函數已經正確導入

@app.post("/classify/")
async def classify_uploaded_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # 確保轉換為 PIL.Image
        predicted_label = classify_image(image)
        return {"predicted_label": predicted_label}
    except Exception as e:
        return {"error": str(e)}