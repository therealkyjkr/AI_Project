import cv2
import time
import config

class Visualizer:
    """화면 렌더링 및 시각적 피드백을 담당하는 클래스"""

    def __init__(self):
        # 객체별 마지막 경고 시점을 저장할 딕셔너리
        self.last_alert_time_map = {}

    def draw_roi_zone(self, canvas):
        """설정된 ROI 영역과 가이드라인을 화면에 그립니다."""
        roi_x1, roi_y1, roi_x2, roi_y2 = config.ROI_BOUNDARY
        
        # ROI 사각형 그리기
        cv2.rectangle(canvas, (roi_x1, roi_y1), (roi_x2, roi_y2), config.COLOR_ROI_LINE, 1)
        
        # ROI 라벨 텍스트
        cv2.putText(canvas, "ROI ZONE", (roi_x1, roi_y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLOR_ROI_LINE, 1)

    def draw_object(self, canvas, box, track_id, final_label, risk_state, ttc_value):
        """객체의 박스, ID, 라벨 및 위험 상태별 색상을 화면에 표시합니다."""
        x1, y1, x2, y2 = box
        
        # 1. 상태별 색상 결정 및 터미널 로그 출력
        if risk_state == "DANGER":
            current_color = config.COLOR_DANGER
            self._print_terminal_log(track_id, final_label, ttc_value)
        elif risk_state == "CAUTION":
            current_color = config.COLOR_CAUTION
        else:
            current_color = config.COLOR_SAFE

        # 2. 객체 박스 렌더링
        caption = f"{track_id} {final_label}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), current_color, 2)
        
        # 3. 라벨 배경 (텍스트 가독성용)
        (text_w, text_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(canvas, (x1, y1 - 25), (x1 + text_w, y1), current_color, -1)
        
        # 4. 라벨 텍스트 (흰색)
        cv2.putText(canvas, caption, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _print_terminal_log(self, track_id, final_label, ttc_value):
        """위험 상태일 때 설정된 간격에 맞춰 터미널에 로그를 출력합니다."""
        now = time.time()
        if track_id not in self.last_alert_time_map or (now - self.last_alert_time_map[track_id] > config.LOG_INTERVAL_SEC):
            print(f"[!] 위험: {final_label}(ID:{track_id}) - TTC: {ttc_value:.2f}s")
            self.last_alert_time_map[track_id] = now