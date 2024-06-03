# -*- coding: utf-8 -*-

import cv2
from ultralytics import YOLO
import numpy as np

# YOLOv8 모델 로드
model = YOLO('yolov8m.pt')  # yolov8n.pt 또는 yolov8s.pt, yolov8m.pt 등 모델 크기 선택 가능

# 클래스 이름과 인덱스 매핑
class_names = {v: k for k, v in model.names.items()}
knife_index = class_names.get('knife')

# 비디오 캡처 초기화
video_path = 'samplevideo.mp4'
cap = cv2.VideoCapture(video_path)

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 모델로 프레임 처리
    results = model(frame)
    detections = results[0].boxes.cpu().numpy()  # [x1, y1, x2, y2, confidence, class]

    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy[0]
        conf = detection.conf[0]
        cls = detection.cls[0]
        if int(cls) == knife_index:
            label = '{} {:.2f}'.format(model.names[int(cls)], conf)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 비디오 출력
    out.write(frame)

    # 화면에 결과 보여주기
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()