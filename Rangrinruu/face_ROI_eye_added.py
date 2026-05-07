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
WINDOW_NAME = "Realtime Face ROI + Eye State"
ROI_EXPAND_X = 0.12
ROI_EXPAND_Y = 0.18
USE_ELLIPSE_MASK = False

# 눈 감김 판정(EAR) 임계값
EYE_AR_THRESHOLD = 0.20
# 너무 짧은 깜빡임 오검출 방지용
MIN_CLOSED_FRAMES_FOR_BLINK = 2

# 시선 방향(iris 상대 위치) 임계값
GAZE_LEFT_TH = 0.38
GAZE_RIGHT_TH = 0.62
GAZE_UP_TH = 0.38
GAZE_DOWN_TH = 0.62


# =============================
# MediaPipe
# 현재 코드는 solutions API 기준
# =============================
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# EAR 계산용 eye landmark
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# 시선 계산용 iris
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# 눈 경계(시선 계산용)
LEFT_EYE_REGION = [33, 133, 159, 145]
RIGHT_EYE_REGION = [362, 263, 386, 374]


# =============================
# 상태
# =============================
STATE = {
    "blink_count": 0,
    "closed_frame_count": 0,
    "prev_eye_closed": False,
    "eye_state": "Unknown",
    "gaze_direction": "Unknown",
}


# =============================
# 공통 유틸
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
# FaceMesh / 눈 상태
# =============================
def extract_face_landmarks(frame: np.ndarray):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    face_landmarks = result.multi_face_landmarks[0]
    pts = []
    for lm in face_landmarks.landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))
    return pts


def euclidean(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def eye_aspect_ratio(eye_pts):
    a = euclidean(eye_pts[1], eye_pts[5])
    b = euclidean(eye_pts[2], eye_pts[4])
    c = euclidean(eye_pts[0], eye_pts[3])

    if c == 0:
        return 1.0

    return (a + b) / (2.0 * c)


def iris_center(pts, iris_idx_list):
    xs = [pts[i][0] for i in iris_idx_list]
    ys = [pts[i][1] for i in iris_idx_list]
    return (int(np.mean(xs)), int(np.mean(ys)))


def gaze_ratio(pts, eye_region_idx, iris_idx):
    eye_points = [pts[i] for i in eye_region_idx]
    xs = [p[0] for p in eye_points]
    ys = [p[1] for p in eye_points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    cx, cy = iris_center(pts, iris_idx)

    if max_x == min_x:
        rx = 0.5
    else:
        rx = (cx - min_x) / (max_x - min_x)

    if max_y == min_y:
        ry = 0.5
    else:
        ry = (cy - min_y) / (max_y - min_y)

    return rx, ry, (cx, cy)


def estimate_eye_state_and_motion(pts):
    left_eye_pts = [pts[i] for i in LEFT_EYE_EAR]
    right_eye_pts = [pts[i] for i in RIGHT_EYE_EAR]

    ear_left = eye_aspect_ratio(left_eye_pts)
    ear_right = eye_aspect_ratio(right_eye_pts)
    ear_avg = (ear_left + ear_right) / 2.0

    eyes_closed = ear_avg < EYE_AR_THRESHOLD
    eye_state = "Closed" if eyes_closed else "Open"

    if eyes_closed:
        STATE["closed_frame_count"] += 1
    else:
        if STATE["prev_eye_closed"] and STATE["closed_frame_count"] >= MIN_CLOSED_FRAMES_FOR_BLINK:
            STATE["blink_count"] += 1
        STATE["closed_frame_count"] = 0

    STATE["prev_eye_closed"] = eyes_closed
    STATE["eye_state"] = eye_state

    left_rx, left_ry, left_iris_center = gaze_ratio(pts, LEFT_EYE_REGION, LEFT_IRIS)
    right_rx, right_ry, right_iris_center = gaze_ratio(pts, RIGHT_EYE_REGION, RIGHT_IRIS)

    rx = (left_rx + right_rx) / 2.0
    ry = (left_ry + right_ry) / 2.0

    if eyes_closed:
        gaze_direction = "Eyes Closed"
    else:
        horizontal = "Center"
        vertical = "Center"

        if rx < GAZE_LEFT_TH:
            horizontal = "Left"
        elif rx > GAZE_RIGHT_TH:
            horizontal = "Right"

        if ry < GAZE_UP_TH:
            vertical = "Up"
        elif ry > GAZE_DOWN_TH:
            vertical = "Down"

        if horizontal == "Center" and vertical == "Center":
            gaze_direction = "Center"
        elif vertical == "Center":
            gaze_direction = horizontal
        elif horizontal == "Center":
            gaze_direction = vertical
        else:
            gaze_direction = f"{vertical}-{horizontal}"

    STATE["gaze_direction"] = gaze_direction

    return {
        "ear_left": ear_left,
        "ear_right": ear_right,
        "ear_avg": ear_avg,
        "eyes_closed": eyes_closed,
        "eye_state": eye_state,
        "gaze_direction": gaze_direction,
        "left_iris_center": left_iris_center,
        "right_iris_center": right_iris_center,
        "left_eye_pts": left_eye_pts,
        "right_eye_pts": right_eye_pts,
    }


def draw_eye_info(display: np.ndarray, eye_result):
    for p in eye_result["left_eye_pts"]:
        cv2.circle(display, p, 2, (255, 255, 0), -1)
    for p in eye_result["right_eye_pts"]:
        cv2.circle(display, p, 2, (255, 255, 0), -1)

    cv2.circle(display, eye_result["left_iris_center"], 3, (0, 0, 255), -1)
    cv2.circle(display, eye_result["right_iris_center"], 3, (0, 0, 255), -1)

    cv2.putText(
        display,
        f"Eye State: {eye_result['eye_state']}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Gaze: {eye_result['gaze_direction']}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Blink Count: {STATE['blink_count']}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"L_EAR: {eye_result['ear_left']:.2f}  R_EAR: {eye_result['ear_right']:.2f}",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


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

            pts = extract_face_landmarks(frame)
            if pts is not None:
                eye_result = estimate_eye_state_and_motion(pts)
                draw_eye_info(display, eye_result)

            combined = np.hstack((display, masked))
            combined = cv2.resize(combined, (1400, 600))

            cv2.putText(
                combined,
                "Original + Face Box + Eye State",
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
        face_mesh.close()


if __name__ == "__main__":
    main()
