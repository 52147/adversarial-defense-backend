import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path):
    """加载并处理图像"""
    img = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(img)
    return img_tensor