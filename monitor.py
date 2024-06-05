# -*- coding: utf-8 -*-

import os
import time
import threading
import requests
from dotenv import load_dotenv
import boto3

load_dotenv()

# 환경 변수 설정
ID = os.getenv('ID')
SECRET = os.getenv('SECRET')
BUCKET_NAME = os.getenv('BUCKET_NAME')
MYREGION = os.getenv('REGION')
FROM_FOLDER = 'raw-video/'

# 다운로드 할 로컬 디렉토리
download_directory = './downloads'

# S3 클라이언트 생성
s3 = boto3.client('s3', aws_access_key_id=ID, aws_secret_access_key=SECRET, region_name=MYREGION)

last_check_time = None

def list_s3_objects():
    """S3 버킷에서 특정 폴더 내 객체 목록을 가져옴"""
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FROM_FOLDER)
    if 'Contents' in response:
        return response['Contents']
    else:
        return []

def download_new_objects():
    global last_check_time
    objects = list_s3_objects()
    
    for obj in objects:
        if last_check_time is None or obj['LastModified'].replace(tzinfo=None) > last_check_time:
            file_name = obj['Key']
            local_path = os.path.join(download_directory, file_name.replace(FROM_FOLDER, ''))
            local_dir = os.path.dirname(local_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            print(f'새 파일 다운로드: {file_name}')
            try:
                s3.download_file(BUCKET_NAME, file_name, local_path)
                notify_fastapi(local_path)  # 파일 다운로드 후 FastAPI 서버에 알림
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")
    
    # 마지막 확인 시간 갱신
    if objects:
        last_check_time = max(obj['LastModified'].replace(tzinfo=None) for obj in objects)

def notify_fastapi(file_path):
    """다운로드 완료 후 FastAPI 서버에 알림"""
    url = "http://192.168.1.222:3000/uploadvideo/"
    files = {'file': open(file_path, 'rb')}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        print("FastAPI 서버에 성공적으로 알림")
    except requests.exceptions.RequestException as e:
        print(f"FastAPI 서버 알림 실패: {e}")

def check_for_updates():
    while True:
        download_new_objects()
        time.sleep(10)

# 백그라운드에서 업데이트 확인 스레드 시작
update_thread = threading.Thread(target=check_for_updates, daemon=True)
update_thread.start()

# FastAPI 서버에 요청하는 부분은 기존의 main.py가 수행합니다
if __name__ == "__main__":
    while True:
        time.sleep(1)  # 메인 스레드는 유휴 상태로 둠
