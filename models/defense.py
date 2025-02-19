import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image



import cv2
import numpy as np
from PIL import Image

def defend_adversarial(image_path: str, method: str = "auto"):
    """
    這個函數應用防禦技術來降低對抗樣本的影響：
    - Gaussian Blur（高斯模糊）
    - Bilateral Filter（雙邊濾波）
    - Median Filter（中值濾波）
    """
    # 1. 讀取圖片
    image = Image.open(image_path).convert("L")  # 確保是灰階
    image_np = np.array(image)  # 轉成 NumPy 陣列

    # 2. 選擇防禦方法
    if method == "gaussian":
        image_np = cv2.GaussianBlur(image_np, (3, 3), 1)
    elif method == "bilateral":
        image_np = cv2.bilateralFilter(image_np, 5, 75, 75)
    elif method == "median":
        image_np = cv2.medianBlur(image_np, 3)
    elif method == "auto":  # ✅ 自動模式：同時套用多種方法
        image_np = cv2.GaussianBlur(image_np, (3, 3), 1)
        image_np = cv2.bilateralFilter(image_np, 5, 75, 75)
        image_np = cv2.medianBlur(image_np, 3)

    # 3. 確保數據範圍正確
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    # 4. 儲存處理後的圖片
    output_path = "static/defended_image.png"
    Image.fromarray(image_np).save(output_path)  # 確保正確儲存

    return output_path