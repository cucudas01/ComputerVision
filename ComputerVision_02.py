import cv2
from ultralytics import YOLO

# 1. 모델 설정
# YOLOv3 (기준 모델)와 YOLO11 (선택 모델)을 로드합니다. [cite: 7, 8]
model_v3 = YOLO('yolov3.pt')
model_v11 = YOLO('yolo11n.pt')  # n은 nano 모델로 속도가 빠릅니다.

# 2. 이미지 경로 설정
# 객체가 다양하고 개수가 많은 영상을 사용하세요. 
# 직접 촬영한 영상 권장, 인터넷 활용 시 출처 명시 필수 [cite: 12, 13]
video_path = r"C:\Users\4996y\OneDrive\Desktop\ComputerVision\ComputerVision\sample.mp4"
# 3. 객체 인식 수행 (Inference)
results_v3 = model_v3.predict(source=video_path, save=True)
results_v11 = model_v11.predict(source=video_path, save=True)

# 4. 결과 분석 및 성능 평가 (객체 개수 비교) 
def count_objects(result):
    counts = {}
    for box in result[0].boxes:
        cls_id = int(box.cls[0])
        label = result[0].names[cls_id]
        counts[label] = counts.get(label, 0) + 1
    return counts

v3_counts = count_objects(results_v3)
v11_counts = count_objects(results_v11)

# 5. 결과 출력 (보고서 작성용 수치) 
print("\n" + "="*30)
print("   [YOLO 성능 평가 결과]   ")
print("="*30)
print(f"{'객체 종류':<10} | {'YOLOv3':<7} | {'YOLO11':<7}")
print("-"*30)

# 두 모델이 찾은 모든 객체 종류를 합쳐서 출력
all_labels = set(v3_counts.keys()).union(set(v11_counts.keys()))
for label in sorted(all_labels):
    v3_num = v3_counts.get(label, 0)
    v11_num = v11_counts.get(label, 0)
    print(f"{label:<10} | {v3_num:<7} | {v11_num:<7}")

print("="*30)
print(f"결과 이미지가 'assignment/' 폴더에 저장되었습니다.")