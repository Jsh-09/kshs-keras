import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 하이퍼파라미터
BATCH_SIZE = 40
EPOCHS = 5
LR = 0.001
DATA_DIR = r"C:\바탕 화면\processed__dataset_0"

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 전처리
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 흑백이면 3채널로 맞추기
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 데이터 로딩
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# 클래스 수 자동 감지
num_classes = len(dataset.classes)

# 사전 학습된 ResNet18 불러오기 (최신 문법)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 손실 함수, 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 학습 루프
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1} 시작")
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

# 모델 저장
torch.save(model.state_dict(), "resnet18_custom.pth")
print("모델 저장 완료")