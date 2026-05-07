from __future__ import annotations

import cv2
import numpy as np

CAM_INDEX = 0
WINDOW_NAME = "Face ROI (OpenCV 32-bit)"
ROI_EXPAND_X = 0.12
ROI_EXPAND_Y = 0.18
USE_ELLIPSE_MASK = False


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def expand_bbox(x: int, y: int, w_box: int, h_box: int, frame_w: int, frame_h: int):
    ex = int(w_box * ROI_EXPAND_X)
    ey = int(h_box * ROI_EXPAND_Y)

    x1 = clamp(x - ex, 0, frame_w - 1)
    y1 = clamp(y - ey, 0, frame_h - 1)
    x2 = clamp(x + w_box + ex, 0, frame_w - 1)
    y2 = clamp(y + h_box + ey, 0, frame_h - 1)

    return x1, y1, x2, y2


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


def detect_face_bbox(frame: np.ndarray, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    # 가장 큰 얼굴 선택
    x, y, w_box, h_box = max(faces, key=lambda f: f[2] * f[3])

    frame_h, frame_w = frame.shape[:2]
    return expand_bbox(x, y, w_box, h_box, frame_w, frame_h)


def main():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError(f"얼굴 cascade 로드 실패: {cascade_path}")

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

            bbox = detect_face_bbox(frame, face_cascade)
            display = frame.copy()

            face_mask = make_face_roi_mask(frame, bbox)
            masked = cv2.bitwise_and(frame, frame, mask=face_mask)

            if bbox is not None:
                x1, y1, x2, y2 = bbox

                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
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


if __name__ == "__main__":
    main()