import torch
from torchvision import models, transforms, datasets
from PIL import Image
import os

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"C:\Users\지승훈\OneDrive\Desktop\coding\resnet18_custom.pth"
data_dir = r"C:\Users\지승훈\OneDrive\Desktop\processed__dataset_0"  # 학습 데이터 폴더
image_path = str(input())     # 테스트 이미지 경로

# 클래스 이름 불러오기 (ImageFolder의 폴더 순서 그대로)
dataset = datasets.ImageFolder(data_dir)
class_names = dataset.classes  # ['경증', '양호', '중등도', '중증'] 같은 리스트일 거임

# 모델 불러오기
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 전처리
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 이미지 열기
image = Image.open(image_path).convert("L")
image = transform(image).unsqueeze(0).to(device)

# 예측
with torch.no_grad():
    output = model(image)
    predicted_idx = torch.argmax(output, 1).item()
    predicted_class = class_names[predicted_idx]

print(f"예측 클래스: {predicted_class}")