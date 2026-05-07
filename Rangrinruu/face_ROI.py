from __future__ import annotations

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("mediapipe가 필요합니다. 설치: pip install mediapipe")


# =============================
# 설정
# =============================
CAM_INDEX = 0
WINDOW_NAME = "Realtime Face ROI"
ROI_EXPAND_X = 0.12   # 얼굴 bbox 좌우 확장 비율
ROI_EXPAND_Y = 0.18   # 얼굴 bbox 상하 확장 비율
USE_ELLIPSE_MASK = False  # True면 타원형 ROI, False면 사각형 ROI


# =============================
# MediaPipe Face Detection
# =============================
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


# =============================
# ROI 계산
# =============================
def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def expand_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    bw = x2 - x1
    bh = y2 - y1

    ex = int(bw * ROI_EXPAND_X)
    ey = int(bh * ROI_EXPAND_Y)

    nx1 = clamp(x1 - ex, 0, w - 1)
    ny1 = clamp(y1 - ey, 0, h - 1)
    nx2 = clamp(x2 + ex, 0, w - 1)
    ny2 = clamp(y2 + ey, 0, h - 1)

    return nx1, ny1, nx2, ny2


def make_face_roi_mask(frame: np.ndarray, bbox):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if bbox is None:
        return mask

    x1, y1, x2, y2 = bbox

    if USE_ELLIPSE_MASK:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ax = max((x2 - x1) // 2, 1)
        ay = max((y2 - y1) // 2, 1)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    else:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


# =============================
# 얼굴 검출
# =============================
def detect_face_bbox(frame: np.ndarray):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detector.process(rgb)

    if not result.detections:
        return None

    # 가장 큰 얼굴 1개 선택
    best_bbox = None
    best_area = 0

    for detection in result.detections:
        rel_box = detection.location_data.relative_bounding_box

        x1 = int(rel_box.xmin * w)
        y1 = int(rel_box.ymin * h)
        bw = int(rel_box.width * w)
        bh = int(rel_box.height * h)

        x2 = x1 + bw
        y2 = y1 + bh

        x1 = clamp(x1, 0, w - 1)
        y1 = clamp(y1, 0, h - 1)
        x2 = clamp(x2, 0, w - 1)
        y2 = clamp(y2, 0, h - 1)

        area = max(1, (x2 - x1) * (y2 - y1))
        if area > best_area:
            best_area = area
            best_bbox = (x1, y1, x2, y2)

    if best_bbox is None:
        return None

    return expand_bbox(*best_bbox, w, h)


# =============================
# 메인 루프
# =============================
def main():
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패")
                break

            bbox = detect_face_bbox(frame)
            display = frame.copy()

            face_mask = make_face_roi_mask(frame, bbox)
            masked = cv2.bitwise_and(frame, frame, mask=face_mask)

            if bbox is not None:
                x1, y1, x2, y2 = bbox

                # 얼굴 위치 사각형 표시
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 텍스트 표시
                cv2.putText(
                    display,
                    "Face ROI",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    display,
                    "No Face Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

            # 좌우 비교 출력
            combined = np.hstack((display, masked))
            combined = cv2.resize(combined, (1400, 600))

            cv2.putText(
                combined,
                "Original + Face Box",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                combined,
                "Face ROI Masked",
                (720, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_detector.close()


if __name__ == "__main__":
    main()