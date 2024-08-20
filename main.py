# -*- coding: utf-8 -*-

import cv2
from ultralytics import YOLO
import numpy as np
import os
import boto3
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import asyncio
import requests

load_dotenv()

# 환경 변수 설정
ID = os.getenv('ID')
SECRET = os.getenv('SECRET')
BUCKET_NAME = os.getenv('BUCKET_NAME')
MYREGION = os.getenv('REGION')
s3 = boto3.client('s3', aws_access_key_id=ID, aws_secret_access_key=SECRET, region_name=MYREGION)
baseurl = os.getenv('IP')

# YOLOv8 모델 로드
model = YOLO('yolov8m.pt')  # yolov8n.pt 또는 yolov8s.pt, yolov8m.pt 등 모델 크기 선택 가능

# 클래스 이름과 인덱스 매핑
class_names = {v: k for k, v in model.names.items()}
knife_index = class_names.get('knife')

# 모자이크 함수
def apply_mosaic(image, x1, y1, x2, y2, mosaic_scale=0.1):
    roi = image[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    
    # 축소 후 확대하여 모자이크 효과 적용
    roi = cv2.resize(roi, (int(roi_w * mosaic_scale), int(roi_h * mosaic_scale)), interpolation=cv2.INTER_LINEAR)
    roi = cv2.resize(roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    
    image[y1:y2, x1:x2] = roi
    return image

async def upload_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as file_object:
            s3.upload_fileobj(file_object, BUCKET_NAME, f"edit-video/{file_name}", ExtraArgs={"ContentType": "video/mp4"})
        return JSONResponse(content={"message": "File uploaded successfully", "filename": file_name})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

async def process_video(video_path, output_path):
    # 비디오 캡처 초기화
    cap = cv2.VideoCapture(video_path)

    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    mosaic_strength = 3  # 1: 약한 모자이크, 2: 중간 모자이크, 3: 강한 모자이크
    mosaic_scales = [0.05, 0.1, 0.2]

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
                # 감지된 객체에 모자이크 적용
                frame = apply_mosaic(frame, int(x1), int(y1), int(x2), int(y2))
                label = '{} {:.2f}'.format(model.names[int(cls)], conf)
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

    # 비디오 업로드
    await upload_file(output_path)

    # 알림 전송
    notify_complete(output_path)

def notify_complete(output_path):
    url = `http://${baseurl}/complete`
    try:
        # output_path를 JSON 형식으로 전송
        response = requests.post(url, json={"filename": output_path})
        response.raise_for_status()
        print("Notification sent successfully")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send notification: {e}")

app = FastAPI()

@app.post("/uploadvideo/")
async def upload_video(file: UploadFile = File(...)):
    temp_file_location = f"temp_{file.filename}"
    try:
        with open(temp_file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        output_file_location = f'./uploads/new_{file.filename}'
        # 비디오 처리 호출
        await process_video(temp_file_location, output_file_location)
        # 처리 후 임시 파일 삭제
        if os.path.exists(temp_file_location):
            os.remove(temp_file_location)
            print(f"Temporary file {temp_file_location} deleted")
        return {"info": "file processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
