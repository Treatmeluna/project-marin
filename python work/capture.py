import os
import time
from PIL import ImageGrab

# 초기 카운트 설정
count = 1

# 이미지를 저장할 디렉토리 경로 설정
save_directory = "C:\\Users\\it\\Desktop\\Python Work\\scraping\\train_dataset\\youtube Finless Porpoises"

# 지정된 디렉토리가 없다면 생성
os.makedirs(save_directory, exist_ok=True)

while True:
    # 화면을 캡쳐하고 이미지 객체로 저장
    screenshot = ImageGrab.grab()

    # 파일명 설정 및 카운트 증가
    file_name = f"screenshot_{count}.png"
    file_path = os.path.join(save_directory, file_name)
    count += 1

    # 캡쳐한 이미지를 파일로 저장 (PNG 형식)
    screenshot.save(file_path)

    # 캡쳐한 이미지를 화면에 표시
    screenshot.show()

    # 3초 대기
    time.sleep(1)
