import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import json

# ========== 路徑自己改 ==========
DATASET_DIR = r"C:\database\tzuchi\20250905\output\20250905\label_gray_he_244"
MODEL_PATH  = r"C:\Projects\ultralytics_yolov8\checkpoints\efficientnet_b0_best_244.pth"
CLASS_JSON  = r"C:\Projects\ultralytics_yolov8\checkpoints\class_names_244_with_space.json"
IMG_SIZE    = 244
BATCH_SIZE  = 32
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# =================================

# 載入類別名稱
with open(CLASS_JSON, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

# 資料集
tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # 灰階轉 3ch
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),        # 跟訓練一致
])
dataset = datasets.ImageFolder(DATASET_DIR, transform=tfms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型
model = models.efficientnet_b0(weights=None)
in_f = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_f, len(CLASS_NAMES))
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state, strict=True)
model.to(DEVICE).eval()

# 預測
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

# 評估
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
