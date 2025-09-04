import cv2
import os
import numpy as np

def imread_unicode(path):
    with open(path, 'rb') as f:
        img_array = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

input_root = r"C:\바탕 화면\탈모 데이터셋"
output_root = "processed_dataset_0"
os.makedirs(output_root, exist_ok=True)

for class_name in os.listdir(input_root):
    input_class_dir = os.path.join(input_root, class_name)
    if not os.path.isdir(input_class_dir):
        continue

    output_class_dir = os.path.join(output_root, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_name in os.listdir(input_class_dir):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(input_class_dir, img_name)
        img = imread_unicode(img_path)
        if img is None:
            print(f"못 불러온 이미지 있음: {img_path}")
            continue

        img_resized = cv2.resize(img, (224, 224))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        save_path = os.path.join(output_class_dir, img_name)
        cv2.imwrite(save_path, img_gray)

print("전처리 완료!")