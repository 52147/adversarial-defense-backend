import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

def defend_adversarial(image_path: str):
    """
    這個函數應用防禦技術來降低對抗樣本的影響：
    - Gaussian Blur（高斯模糊）: 平滑化圖像，減少對抗噪音
    - Bilateral Filter（雙邊濾波）: 保持圖片輪廓，同時消除噪音
    """
    # 1. 讀取圖片
    image = Image.open(image_path).convert("L")  # 轉換為灰階
    image_np = np.array(image)

    # 2. 應用防禦技術
    image_blurred = cv2.GaussianBlur(image_np, (3, 3), 1)  # 高斯模糊
    image_defended = cv2.bilateralFilter(image_blurred, 5, 75, 75)  # 雙邊濾波

    # 3. 儲存處理後的圖片
    output_path = "static/defended_image.png"
    defended_pil = Image.fromarray(image_defended)
    defended_pil.save(output_path)

    return output_path