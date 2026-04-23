import cv2
from ultralytics import YOLO
import os

# 1. 파일 및 모델 로드
video_path = 'drivetest.mp4'
if not os.path.exists(video_path):
    print(f"❌ '{video_path}' 파일이 없습니다.")
    exit()

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(video_path)

# 기본 정보 추출
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
roi = (0, 0, width, height)
is_paused = False

def on_trackbar(val):
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)

cv2.namedWindow("BEEP-BEEP Sensitive Tool")
cv2.createTrackbar("Seek", "BEEP-BEEP Sensitive Tool", 0, total_frames - 1, on_trackbar)

# 필터링 대상: 사람(0), 자전거(1), 자동차(2), 오토바이(3), 버스(5), 트럭(7)
target_classes = [0, 1, 2, 3, 5, 7]

while cap.isOpened():
    if not is_paused:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # 현재 위치 업데이트
        curr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos("Seek", "BEEP-BEEP Sensitive Tool", curr)
    
    display_frame = frame.copy()

    # 2. AI 탐지 수행 (conf=0.2로 대폭 낮춤)
    # 팀장님의 전략: "절대 놓치지 않기 위해" 감도를 올림
    results = model(frame, classes=target_classes, conf=0.2, verbose=False)
    
    for result in results:
        for box in result.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            
            # ROI 필터링 로직
            rx, ry, rw, rh = roi
            if rx < cx < rx + rw and ry < cy < ry + rh:
                conf = float(box.conf[0])
                label = model.names[int(box.cls[0])]
                
                # 시각화 (확신도에 따라 색상 강조 가능)
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(display_frame, f"{label} {conf:.2f}", (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 파란색 ROI 가이드라인
    cv2.rectangle(display_frame, (int(roi[0]), int(roi[1])), 
                  (int(roi[0]+roi[2]), int(roi[1]+roi[3])), (255, 0, 0), 2)

    cv2.imshow("BEEP-BEEP Sensitive Tool", display_frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '): is_paused = not is_paused
    elif key == ord('r'):
        new_roi = cv2.selectROI("BEEP-BEEP Sensitive Tool", frame, False)
        if new_roi[2] > 20 and new_roi[3] > 20:
            roi = new_roi

cap.release()
cv2.destroyAllWindows()