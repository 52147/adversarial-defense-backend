import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import urllib.request

# ✅ Google Drive 下載 URL
MODEL_URL = "https://drive.google.com/uc?id=13D1bcxVFpuMY62UrjXPBuULnfJglQIIm&export=download"

# ✅ 定義 SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ✅ 直接從 Google Drive 加載模型
def load_model_from_drive():
    print("🚀 直接從 Google Drive 加載模型...")
    try:
        response = urllib.request.urlopen(MODEL_URL)  # 讀取 `.pth` 文件
        model_data = io.BytesIO(response.read())  # 轉為記憶體流
        model = SimpleCNN()
        model.load_state_dict(torch.load(model_data, map_location=torch.device("cpu")))
        model.eval()
        print("✅ 成功直接從 Google Drive 加載模型！")
        return model
    except Exception as e:
        print(f"❌ 無法從 Google Drive 加載模型！錯誤: {e}")
        raise e  # 讓程序終止

# ✅ 加載模型
model = load_model_from_drive()

# ✅ 圖片轉換函數
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  # MNIST 圖像大小
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def classify_image(image):
    """
    ✅ `classify_image()` 現在只需要 **圖片作為參數**
    """
    if isinstance(image, str):  # 如果是路徑，先讀取圖片
        image = Image.open(image).convert("L")  

    image = transform(image).unsqueeze(0)  # 增加 batch 維度
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True).item()
    
    return pred  # 返回預測數字