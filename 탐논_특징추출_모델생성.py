

import torch
import torchvision.models as models
import torch.nn as nn

def create_feature_extractor(model_path, num_classes=4):
    """
    학습된 ResNet-18 모델을 불러와서 마지막 분류 레이어를 제거하고,
    특징 추출기(feature extractor)로 사용될 수 있는 모델을 반환합니다.

    Args:
        model_path (str): 학습된 모델 가중치(.pth) 파일의 경로
        num_classes (int): 학습 시 사용했던 클래스의 개수

    Returns:
        torch.nn.Module: 마지막 레이어가 제거된 특징 추출기 모델
    """
    # 1. 사전 학습된 ResNet-18 모델 구조를 불러옵니다.
    model = models.resnet18(weights=None) 
    
    # 2. 모델의 마지막 출력층(fc layer)을 학습시켰던 클래스 개수에 맞게 수정합니다.
    #    이렇게 해야 저장된 가중치를 정확히 불러올 수 있습니다.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 3. 저장된 모델 가중치를 불러옵니다.
    model.load_state_dict(torch.load(model_path))
    
    # 4. 특징 추출을 위해 마지막 fc layer를 Identity layer로 교체합니다.
    #    Identity layer는 입력을 그대로 출력으로 내보내는 역할을 합니다.
    #    즉, fc layer 직전의 512차원 특징 벡터가 모델의 최종 출력이 됩니다.
    model.fc = nn.Identity()
    
    print("특징 추출기 모델이 성공적으로 생성되었습니다.")
    print("이제 이 모델은 이미지를 512차원의 특징 벡터로 변환합니다.")
    
    return model

if __name__ == '__main__':
    # 사용 예시
    # 기존에 학습시킨 모델 경로
    trained_model_path = 'C:\Users\지승훈\OneDrive\Desktop\coding\resnet18_custom.pth'
    
    # 특징 추출기 모델 생성
    feature_extractor = create_feature_extractor(model_path=trained_model_path, num_classes=4)
    
    # 모델을 평가 모드로 설정
    feature_extractor.eval()
    
    # 예시로, 랜덤한 이미지를 넣었을 때 출력 확인
    # 실제 사용 시에는 전처리된 실제 이미지 텐서를 넣어야 합니다.
    dummy_image = torch.randn(1, 3, 224, 224) # (배치크기, 채널, 높이, 너비)
    with torch.no_grad():
        features = feature_extractor(dummy_image)
    
    print(f"\n더미 이미지를 입력했을 때 추출된 특징 벡터의 크기: {features.shape}")

