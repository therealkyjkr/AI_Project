import cv2
from ultralytics import YOLO
import os

# 1. 파일 경로 확인 및 리소스 로드
video_path = 'drivetest.mp4'
if not os.path.exists(video_path):
    print(f"❌ '{video_path}' 파일이 없습니다.")
    exit()

# YOLOv8 AI 모델 및 비디오 파일 불러오기
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(video_path)

# 비디오의 해상도 및 총 프레임 수 추출
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 초기 관심 영역(ROI) 설정 및 일시정지 상태 변수
roi = (0, 0, width, height)
is_paused = False

# 슬라이더(Trackbar) 조작 시 해당 프레임으로 이동하는 함수
def on_trackbar(val):
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)

# 결과 표시를 위한 윈도우 생성 및 재생 바(Seek) 추가
win_name = "BEEP-BEEP Sensitive Tool"
cv2.namedWindow(win_name)
cv2.createTrackbar("Seek", win_name, 0, total_frames - 1, on_trackbar)

# 탐지 대상 클래스 번호: 사람(0), 자전거(1), 자동차(2), 오토바이(3), 버스(5), 트럭(7)
target_classes = [0, 1, 2, 3, 5, 7]

# 첫 번째 프레임 미리 읽기
success, frame = cap.read()

while cap.isOpened():
    # 윈도우 창이 닫히면 프로그램 종료
    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    # 일시정지 상태가 아닐 때만 다음 프레임을 읽음
    if not is_paused:
        success, frame = cap.read()
        # 영상이 끝나면 처음(0번 프레임)으로 되돌림
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
    
    # 원본 보호를 위해 복사본 프레임에서 시각화 작업 수행
    display_frame = frame.copy()

    # 2. AI 객체 탐지 수행 (설정된 클래스 및 확신도 기준)
    results = model(frame, classes=target_classes, conf=0.2, verbose=False)
    
    for result in results:
        for box in result.boxes:
            # 탐지된 객체의 좌표 추출
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            # 객체의 중앙점 계산
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            
            # 탐지된 객체의 중앙점이 설정된 ROI(관심 영역) 안에 있는지 확인
            rx, ry, rw, rh = roi
            if rx < cx < rx + rw and ry < cy < ry + rh:
                conf = float(box.conf[0])        # 확신도
                label = model.names[int(box.cls[0])] # 클래스 이름
                
                # 조건에 맞는 객체에 빨간색 박스와 라벨 표시
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(display_frame, f"{label} {conf:.2f}", (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 설정된 ROI 영역을 화면에 파란색 사각형으로 표시
    cv2.rectangle(display_frame, (int(roi[0]), int(roi[1])), 
                  (int(roi[0]+roi[2]), int(roi[1]+roi[3])), (255, 0, 0), 2)

    # 최종 결과 화면 출력
    cv2.imshow(win_name, display_frame)

    # 재생 중일 때만 하단 슬라이더 위치를 현재 프레임에 맞춰 업데이트
    if not is_paused:
        curr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos("Seek", win_name, curr)

    # 키 입력 처리
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'): # 'q' 누르면 종료
        break
    elif key == ord(' '): # 'Space' 누르면 일시정지/재생
        is_paused = not is_paused
    elif key == ord('r'): # 'r' 누르면 ROI 재설정 (드래그 후 Enter)
        new_roi = cv2.selectROI(win_name, frame, False)
        # 유효한 크기의 영역이 지정되었을 때만 ROI 업데이트
        if new_roi[2] > 20 and new_roi[3] > 20:
            roi = new_roi

# 자원 해제 및 모든 윈도우 닫기
cap.release()
cv2.destroyAllWindows()