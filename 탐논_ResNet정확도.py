import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 설정
DATA_DIR = r"C:\Users\지승훈\OneDrive\Desktop\processed__dataset_0"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전처리 (학습할 때랑 동일해야 함)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 데이터셋 로딩
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# 여기선 전체 데이터셋을 테스트용으로 사용 (원래는 분리된 test셋 써야 정석)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 구조 정의 (학습 때와 똑같이)
num_classes = len(dataset.classes)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet18_custom.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# 정확도 계산
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"정확도: {accuracy:.2f}%")