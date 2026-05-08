from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.pop("WAYLAND_DISPLAY", None)

import time
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("mediapipe가 필요합니다. 가상환경에서 설치 후 실행하세요: python -m pip install mediapipe")

try:
    from picamera2 import Picamera2
except ImportError:
    raise ImportError("picamera2가 필요합니다. sudo apt install -y python3-picamera2 후 실행하세요.")

BRIGHTNESS_ALPHA = 4.0
BRIGHTNESS_BETA = 20

def convert_picam_frame_for_display(frame_raw):
    # 2차원 MONO 프레임이면
    if frame_raw.ndim == 2:
        gray8 = cv2.normalize(frame_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gray8 = cv2.convertScaleAbs(gray8, alpha=BRIGHTNESS_ALPHA, beta=BRIGHTNESS_BETA)
        return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

    # 3채널이어도 실제론 매우 어두운 MONO일 수 있으니 gray로 바꿔서 밝게 만듦
    if frame_raw.ndim == 3 and frame_raw.shape[2] == 3:
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gray = cv2.convertScaleAbs(gray, alpha=BRIGHTNESS_ALPHA, beta=BRIGHTNESS_BETA)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return frame_raw
WINDOW_NAME = "Picamera2 Face ROI + Eye State"
CAMERA_FULL_SIZE = (5120, 800)
CAMERA_FORMAT = "RGB888"

ENABLE_QUAD_SPLIT = False
TARGET_VIEW_INDEX = 0  # 0=cam1, 1=cam2, 2=cam3, 3=cam4
PROCESS_VIEW_WIDTH = 960

ROI_EXPAND_X = 0.12
ROI_EXPAND_Y = 0.18
USE_ELLIPSE_MASK = False

EYE_AR_THRESHOLD = 0.20
MIN_CLOSED_FRAMES_FOR_BLINK = 2

GAZE_LEFT_TH = 0.38
GAZE_RIGHT_TH = 0.62
GAZE_UP_TH = 0.38
GAZE_DOWN_TH = 0.62


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

LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_REGION = [33, 133, 159, 145]
RIGHT_EYE_REGION = [362, 263, 386, 374]

STATE = {
    "blink_count": 0,
    "closed_frame_count": 0,
    "prev_eye_closed": False,
    "eye_state": "Unknown",
    "gaze_direction": "Unknown",
}


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


def split_quad_frame(frame: np.ndarray):
    h, w = frame.shape[:2]
    if not ENABLE_QUAD_SPLIT:
        return frame, "full"

    sub_w = w // 4
    views = [
        frame[:, 0:sub_w],
        frame[:, sub_w:sub_w * 2],
        frame[:, sub_w * 2:sub_w * 3],
        frame[:, sub_w * 3:sub_w * 4],
    ]
    index = max(0, min(3, TARGET_VIEW_INDEX))
    return views[index], f"cam{index + 1}"


def resize_for_processing(frame: np.ndarray):
    if PROCESS_VIEW_WIDTH is None:
        return frame
    h, w = frame.shape[:2]
    if w <= PROCESS_VIEW_WIDTH:
        return frame
    new_h = int(h * (PROCESS_VIEW_WIDTH / w))
    return cv2.resize(frame, (PROCESS_VIEW_WIDTH, new_h))


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


def extract_face_landmarks(frame: np.ndarray):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    pts = []
    for lm in result.multi_face_landmarks[0].landmark:
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
    return int(np.mean(xs)), int(np.mean(ys))


def gaze_ratio(pts, eye_region_idx, iris_idx):
    eye_points = [pts[i] for i in eye_region_idx]
    xs = [p[0] for p in eye_points]
    ys = [p[1] for p in eye_points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    cx, cy = iris_center(pts, iris_idx)
    rx = 0.5 if max_x == min_x else (cx - min_x) / (max_x - min_x)
    ry = 0.5 if max_y == min_y else (cy - min_y) / (max_y - min_y)
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

    left_rx, left_ry, _ = gaze_ratio(pts, LEFT_EYE_REGION, LEFT_IRIS)
    right_rx, right_ry, _ = gaze_ratio(pts, RIGHT_EYE_REGION, RIGHT_IRIS)

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
    }


def draw_eye_info(display: np.ndarray, eye_result):
    cv2.putText(display, f"Eye State: {eye_result['eye_state']}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display, f"Gaze: {eye_result['gaze_direction']}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(display, f"Blink Count: {STATE['blink_count']}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, f"L_EAR: {eye_result['ear_left']:.2f}  R_EAR: {eye_result['ear_right']:.2f}", (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def init_picamera2():
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"size": CAMERA_FULL_SIZE, "format": CAMERA_FORMAT}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.5)

    supported = picam2.camera_controls.keys()
    controls = {}

    if "AeEnable" in supported:
        controls["AeEnable"] = False
    if "AwbEnable" in supported:
        controls["AwbEnable"] = False
    if "ExposureTime" in supported:
        controls["ExposureTime"] = 3000
    if "AnalogueGain" in supported:
        controls["AnalogueGain"] = 1.0

    if controls:
        picam2.set_controls(controls)

    return picam2

    return picam2

    time.sleep(1.5)
    return picam2


def main():
    global TARGET_VIEW_INDEX

    picam2 = init_picamera2()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame_raw = picam2.capture_array()
            if frame_raw is None:
                print("프레임 읽기 실패")
                break

            frame_bgr = convert_picam_frame_for_display(frame_raw)
            selected_view, view_name = split_quad_frame(frame_bgr)
            frame = resize_for_processing(selected_view)

            display = frame.copy()
            bbox = detect_face_bbox(frame)
            face_mask = make_face_roi_mask(frame, bbox)
            masked = cv2.bitwise_and(frame, frame, mask=face_mask)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"Face ROI ({view_name})", (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display, f"No Face Detected ({view_name})", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            pts = extract_face_landmarks(frame)
            if pts is not None:
                eye_result = estimate_eye_state_and_motion(pts)
                draw_eye_info(display, eye_result)

            combined = np.hstack((display, masked))
            combined = cv2.resize(combined, (1400, 600))

            cv2.putText(combined, f"Original + Face ROI + Eye State ({view_name})", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, "Face ROI Masked", (720, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, "Press 1/2/3/4 to change camera view", (20, 570),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            elif key == ord("1"):
                TARGET_VIEW_INDEX = 0
            elif key == ord("2"):
                TARGET_VIEW_INDEX = 1
            elif key == ord("3"):
                TARGET_VIEW_INDEX = 2
            elif key == ord("4"):
                TARGET_VIEW_INDEX = 3

    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        face_detector.close()
        face_mesh.close()


if __name__ == "__main__":
    main()
