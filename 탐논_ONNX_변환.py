import torch
import torchvision.models as models
import torch.nn as nn

def convert_to_onnx(pytorch_model_path, onnx_model_path, num_classes=4):
    """
    학습된 PyTorch 모델(.pth)을 ONNX 형식으로 변환하여 저장합니다.

    Args:
        pytorch_model_path (str): 변환할 PyTorch 모델 파일 경로
        onnx_model_path (str): 저장할 ONNX 모델 파일 경로
        num_classes (int): 모델이 학습한 클래스의 개수
    """
    # 1. PyTorch 모델 구조를 만들고, 학습된 가중치를 불러옵니다.
    device = torch.device("cpu") # ONNX 변환은 CPU에서 수행하는 것이 안전합니다.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    model.to(device)
    model.eval() # 모델을 반드시 평가 모드로 설정해야 합니다.

    # 2. 모델에 입력될 더미 데이터(dummy input)를 생성합니다.
    #    ONNX는 모델의 구조를 추적하기 위해 실제와 같은 크기의 입력이 필요합니다.
    #    (배치크기=1, 채널=3, 높이=224, 너비=224)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # 3. 모델을 ONNX 형식으로 변환합니다.
    try:
        torch.onnx.export(
            model,                          # 실행될 모델
            dummy_input,                    # 모델 입력값 (텐서)
            onnx_model_path,                # 저장될 ONNX 파일 경로
            export_params=True,             # 모델 파라미터도 함께 저장
            opset_version=11,               # ONNX 버전
            do_constant_folding=True,       # 최적화
            input_names=['input'],          # 입력값 이름
            output_names=['output'],        # 출력값 이름
            dynamic_axes={'input' : {0 : 'batch_size'},    # 동적 축 설정
                          'output' : {0 : 'batch_size'}}
        )
        print(f"모델이 성공적으로 ONNX 형식으로 변환되어 '{onnx_model_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"ONNX 변환 중 오류가 발생했습니다: {e}")

if __name__ == '__main__':
     # PyTorch 모델 파일 경로 (올바른 경로)
     pth_path = r"C:\coding\탐논\resnet18_custom.pth"
 
     # 저장될 ONNX 파일 경로 (올바른 경로)
     onnx_path = r"C:\coding\탐논\resnet18.onnx"
 
     # 클래스 개수는 4개 (경증, 양호, 중등도, 중증)
     convert_to_onnx(pth_path, onnx_path, num_classes=4)