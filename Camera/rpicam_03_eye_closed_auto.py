from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep

import cv2
import numpy as np
from gpiozero import LED
from picamera2 import Picamera2

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "mediapipe is required. Install it in the Raspberry Pi environment first."
    ) from exc


# =========================
# Camera / save settings from rpicam_03.py
# =========================
CAPTURE_WIDTH = 5120
CAPTURE_HEIGHT = 800
SINGLE_WIDTH = CAPTURE_WIDTH // 4

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 400
SINGLE_VIEW_WIDTH = 1280
SINGLE_VIEW_HEIGHT = 800

MIN_EXPOSURE_MS = 1
MAX_EXPOSURE_MS = 100
INITIAL_EXPOSURE_MS = 8

WINDOW_NAME = "Eye Closed Auto UV Capture"

SAVE_ROOT = Path.home() / "Graduate_Project" / "captures"

CURRENT_GAIN = 1.0
SAVE_AS_PNG = True

ROTATE_DISPLAY = True
DISPLAY_ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE

ROTATE_SAVE = True
SAVE_ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE


# =========================
# Relay / UV LED settings from rpicam_03.py
# =========================
RELAY_PIN = 17
RELAY_ACTIVE_HIGH = False
RELAY_WARMUP_SEC = 0.3

relay = LED(RELAY_PIN, active_high=RELAY_ACTIVE_HIGH, initial_value=False)


CAMERA_INFO = {
    "cam2": {
        "label": "CAM 2 - NO FILTER",
        "folder": "cam2_no_filter",
        "filter": "no_filter",
        "x_start": SINGLE_WIDTH * 1,
        "x_end": SINGLE_WIDTH * 2,
        "sequence_order": 1,
    },
    "cam3": {
        "label": "CAM 3 - 405nm FILTER",
        "folder": "cam3_405nm",
        "filter": "405nm_filter",
        "x_start": SINGLE_WIDTH * 2,
        "x_end": SINGLE_WIDTH * 3,
        "sequence_order": 2,
    },
    "cam4": {
        "label": "CAM 4 - 660nm FILTER",
        "folder": "cam4_660nm",
        "filter": "660nm_filter",
        "x_start": SINGLE_WIDTH * 3,
        "x_end": SINGLE_WIDTH * 4,
        "sequence_order": 3,
    },
}


# =========================
# Eye detection settings from face_ROI_eye_added.py
# =========================
DETECTION_CAM_KEY = "cam2"
DETECTION_PROCESS_WIDTH = 960

ROI_EXPAND_X = 0.12
ROI_EXPAND_Y = 0.18

EYE_AR_THRESHOLD = 0.20
MIN_CLOSED_FRAMES_FOR_BLINK = 2

GAZE_LEFT_TH = 0.38
GAZE_RIGHT_TH = 0.62
GAZE_UP_TH = 0.38
GAZE_DOWN_TH = 0.62

EYES_CLOSED_DELAY_SEC = 2.0
CAPTURE_COOLDOWN_SEC = 3.0


mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5,
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
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
    "eyes_closed_started_at": None,
    "last_capture_at": 0.0,
    "waiting_for_reopen": False,
    "status": "Ready",
}


def nothing(_value):
    pass


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


def relay_on():
    relay.on()
    print("[Relay] ON")


def relay_off():
    relay.off()
    print("[Relay] OFF")


def set_manual_controls(camera, exposure_ms, gain):
    camera.set_controls(
        {
            "AeEnable": False,
            "ExposureTime": exposure_ms * 1000,
            "AnalogueGain": gain,
        }
    )


def get_current_exposure_ms():
    exposure_ms = cv2.getTrackbarPos("Exposure(ms)", WINDOW_NAME)
    if exposure_ms < MIN_EXPOSURE_MS:
        exposure_ms = MIN_EXPOSURE_MS
    return exposure_ms


def extract_cam_frame(full_frame_bgr, cam_key):
    info = CAMERA_INFO[cam_key]
    return full_frame_bgr[:, info["x_start"] : info["x_end"]]


def resize_for_eye_detection(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if w <= DETECTION_PROCESS_WIDTH:
        return frame_bgr, 1.0, 1.0

    scale = DETECTION_PROCESS_WIDTH / float(w)
    resized = cv2.resize(
        frame_bgr,
        (DETECTION_PROCESS_WIDTH, int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )
    scale_x = w / float(resized.shape[1])
    scale_y = h / float(resized.shape[0])
    return resized, scale_x, scale_y


def scale_bbox(bbox, scale_x, scale_y):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    )


def rotate_for_display(img):
    if ROTATE_DISPLAY:
        return cv2.rotate(img, DISPLAY_ROTATE_CODE)
    return img


def rotate_for_save(img):
    if ROTATE_SAVE:
        return cv2.rotate(img, SAVE_ROTATE_CODE)
    return img


def draw_text(img, title, exposure_ms, extra_text=None):
    out = img.copy()

    cv2.putText(
        out,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        out,
        f"Exposure: {exposure_ms} ms",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    if extra_text is not None:
        cv2.putText(
            out,
            extra_text,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )

    return out


def save_image(path, img_bgr):
    if path.suffix.lower() == ".png":
        ok = cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        ok = cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])

    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


def save_one_camera_image(
    cam_key,
    frame_bgr,
    timestamp,
    exposure_ms,
    gain,
    ext,
    trigger_metadata=None,
):
    info = CAMERA_INFO[cam_key]

    date_folder = SAVE_ROOT / info["folder"] / timestamp.strftime("%Y-%m-%d")
    date_folder.mkdir(parents=True, exist_ok=True)

    base_name = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]

    image_path = date_folder / f"{base_name}_{cam_key}.{ext}"
    meta_path = date_folder / f"{base_name}_{cam_key}_metadata.json"

    save_image(image_path, frame_bgr)

    metadata = {
        "captured_at": timestamp.isoformat(),
        "camera_name": cam_key,
        "camera_label": info["label"],
        "filter_type": info["filter"],
        "sequence_order": info["sequence_order"],
        "capture_type": "eye_closed_auto_high_quality_still_fullframe_then_crop",
        "file_format": ext,
        "camera_mode": {
            "capture_width": CAPTURE_WIDTH,
            "capture_height": CAPTURE_HEIGHT,
            "single_width": SINGLE_WIDTH,
        },
        "camera_control": {
            "AeEnable": False,
            "ExposureTime_ms": exposure_ms,
            "ExposureTime_us": exposure_ms * 1000,
            "AnalogueGain": gain,
        },
        "relay_control": {
            "relay_pin": RELAY_PIN,
            "active_high": RELAY_ACTIVE_HIGH,
            "relay_used": True,
            "relay_warmup_sec": RELAY_WARMUP_SEC,
        },
        "eye_closed_trigger": trigger_metadata,
        "crop_region": {
            "x_start": info["x_start"],
            "x_end": info["x_end"],
            "y_start": 0,
            "y_end": CAPTURE_HEIGHT,
        },
        "saved_file": str(image_path),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"[Saved] {image_path}")
    print(f"[Saved] {meta_path}")


def show_single_camera(window_name, frame_bgr, cam_key, exposure_ms, hold_ms=250):
    info = CAMERA_INFO[cam_key]

    single_view = cv2.resize(
        frame_bgr,
        (SINGLE_VIEW_WIDTH, SINGLE_VIEW_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )
    single_view = rotate_for_display(single_view)
    single_view = draw_text(
        single_view,
        info["label"],
        exposure_ms,
        extra_text="Capturing...",
    )

    cv2.imshow(window_name, single_view)
    cv2.waitKey(hold_ms)


def capture_high_quality_full_frame(cam, preview_config, still_config, exposure_ms, gain):
    try:
        cam.stop()
        cam.configure(still_config)
        cam.start()
        set_manual_controls(cam, exposure_ms, gain)
        sleep(0.15)

        still_frame = cam.capture_array()
        print("FRAME INFO:", still_frame.shape, still_frame.dtype)
        return cv2.cvtColor(still_frame, cv2.COLOR_RGB2BGR)

    finally:
        cam.stop()
        cam.configure(preview_config)
        cam.start()
        set_manual_controls(cam, exposure_ms, gain)


def capture_sequence(
    cam,
    preview_config,
    still_config,
    exposure_ms,
    gain,
    trigger_metadata=None,
    save_as_png=True,
):
    ext = "png" if save_as_png else "jpg"

    relay_on()
    sleep(RELAY_WARMUP_SEC)

    try:
        full_frame_bgr = capture_high_quality_full_frame(
            cam=cam,
            preview_config=preview_config,
            still_config=still_config,
            exposure_ms=exposure_ms,
            gain=gain,
        )

        capture_timestamp = datetime.now()

        for cam_key in ["cam2", "cam3", "cam4"]:
            target_frame = extract_cam_frame(full_frame_bgr, cam_key).copy()
            show_single_camera(
                WINDOW_NAME,
                target_frame,
                cam_key,
                exposure_ms,
                hold_ms=250,
            )

            save_frame = rotate_for_save(target_frame)
            save_one_camera_image(
                cam_key=cam_key,
                frame_bgr=save_frame,
                timestamp=capture_timestamp,
                exposure_ms=exposure_ms,
                gain=gain,
                ext=ext,
                trigger_metadata=trigger_metadata,
            )

        STATE["last_capture_at"] = monotonic()

    finally:
        relay_off()


def detect_face_bbox(frame_bgr):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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


def extract_face_landmarks(frame_bgr):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
        if (
            STATE["prev_eye_closed"]
            and STATE["closed_frame_count"] >= MIN_CLOSED_FRAMES_FOR_BLINK
        ):
            STATE["blink_count"] += 1
        STATE["closed_frame_count"] = 0

    STATE["prev_eye_closed"] = eyes_closed
    STATE["eye_state"] = eye_state

    left_rx, left_ry, left_iris_center = gaze_ratio(pts, LEFT_EYE_REGION, LEFT_IRIS)
    right_rx, right_ry, right_iris_center = gaze_ratio(
        pts,
        RIGHT_EYE_REGION,
        RIGHT_IRIS,
    )

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
    }


def analyze_eye_from_camera_frame(cam_frame_bgr):
    process_frame, scale_x, scale_y = resize_for_eye_detection(cam_frame_bgr)

    face_bbox_process = detect_face_bbox(process_frame)
    face_bbox_original = scale_bbox(face_bbox_process, scale_x, scale_y)

    pts = extract_face_landmarks(process_frame)
    if pts is None:
        return None, face_bbox_original

    return estimate_eye_state_and_motion(pts), face_bbox_original


def update_eye_closed_trigger(eye_result):
    now = monotonic()

    if eye_result is None:
        STATE["eyes_closed_started_at"] = None
        STATE["status"] = "No face landmarks"
        return False

    if not eye_result["eyes_closed"]:
        STATE["eyes_closed_started_at"] = None
        if STATE["waiting_for_reopen"]:
            STATE["waiting_for_reopen"] = False
            STATE["status"] = "Ready"
        else:
            STATE["status"] = "Ready - close eyes"
        return False

    if STATE["waiting_for_reopen"]:
        STATE["status"] = "Open eyes to re-arm"
        return False

    if now - STATE["last_capture_at"] < CAPTURE_COOLDOWN_SEC:
        STATE["status"] = "Capture cooldown"
        return False

    if STATE["eyes_closed_started_at"] is None:
        STATE["eyes_closed_started_at"] = now

    elapsed = now - STATE["eyes_closed_started_at"]
    remaining = max(0.0, EYES_CLOSED_DELAY_SEC - elapsed)
    STATE["status"] = f"Eyes closed - capture in {remaining:.1f}s"

    return elapsed >= EYES_CLOSED_DELAY_SEC


def build_trigger_metadata(eye_result, face_bbox):
    if eye_result is None:
        return None

    started_at = STATE["eyes_closed_started_at"]
    held_sec = 0.0 if started_at is None else monotonic() - started_at

    return {
        "type": "eyes_closed_for_2_seconds",
        "detection_camera": DETECTION_CAM_KEY,
        "threshold_ear": EYE_AR_THRESHOLD,
        "held_sec": held_sec,
        "eye_state": eye_result["eye_state"],
        "gaze_direction": eye_result["gaze_direction"],
        "ear_left": eye_result["ear_left"],
        "ear_right": eye_result["ear_right"],
        "ear_avg": eye_result["ear_avg"],
        "face_bbox": list(face_bbox) if face_bbox is not None else None,
    }


def draw_face_box(frame_bgr, face_bbox):
    out = frame_bgr.copy()
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def draw_eye_status(img, eye_result):
    y = 155
    lines = [
        f"Auto: {STATE['status']}",
        f"Eye: {STATE['eye_state']}  Blink: {STATE['blink_count']}",
    ]
    if eye_result is not None:
        lines.append(
            f"EAR L:{eye_result['ear_left']:.2f} R:{eye_result['ear_right']:.2f}"
        )
    lines.append("q: quit  c/p: capture now  o/f: relay")

    for line in lines:
        cv2.putText(
            img,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        y += 32

    return img


def initialize_camera():
    camera_list = Picamera2.global_camera_info()
    print("Detected cameras:", camera_list)

    if len(camera_list) == 0:
        relay_off()
        raise RuntimeError("No Picamera2 camera detected.")

    cam = Picamera2(0)

    preview_config = cam.create_preview_configuration(
        main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "XRGB8888"}
    )
    still_config = cam.create_still_configuration(
        main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
    )

    cam.configure(preview_config)
    cam.options["quality"] = 100
    cam.options["compress_level"] = 0

    cam.start()
    sleep(0.5)
    set_manual_controls(cam, INITIAL_EXPOSURE_MS, CURRENT_GAIN)
    sleep(0.5)
    set_manual_controls(cam, INITIAL_EXPOSURE_MS, CURRENT_GAIN)

    relay_off()
    return cam, preview_config, still_config


def initialize_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if ROTATE_DISPLAY:
        cv2.resizeWindow(WINDOW_NAME, PREVIEW_HEIGHT * 3, PREVIEW_WIDTH)
    else:
        cv2.resizeWindow(WINDOW_NAME, PREVIEW_WIDTH * 3, PREVIEW_HEIGHT)

    cv2.createTrackbar(
        "Exposure(ms)",
        WINDOW_NAME,
        INITIAL_EXPOSURE_MS,
        MAX_EXPOSURE_MS,
        nothing,
    )
    cv2.setTrackbarMin("Exposure(ms)", WINDOW_NAME, MIN_EXPOSURE_MS)


def main():
    cam, preview_config, still_config = initialize_camera()
    initialize_window()
    prev_exposure_ms = INITIAL_EXPOSURE_MS

    try:
        while True:
            exposure_ms = get_current_exposure_ms()

            if exposure_ms != prev_exposure_ms:
                set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
                prev_exposure_ms = exposure_ms

            frame = cam.capture_array()
            full_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            cam2 = extract_cam_frame(full_frame_bgr, "cam2")
            cam3 = extract_cam_frame(full_frame_bgr, "cam3")
            cam4 = extract_cam_frame(full_frame_bgr, "cam4")

            detection_frame = extract_cam_frame(full_frame_bgr, DETECTION_CAM_KEY)
            eye_result, face_bbox = analyze_eye_from_camera_frame(detection_frame)
            should_capture = update_eye_closed_trigger(eye_result)

            cam2_annotated = draw_face_box(cam2, face_bbox)

            cam2_view = cv2.resize(
                cam2_annotated,
                (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                interpolation=cv2.INTER_CUBIC,
            )
            cam3_view = cv2.resize(
                cam3,
                (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                interpolation=cv2.INTER_CUBIC,
            )
            cam4_view = cv2.resize(
                cam4,
                (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                interpolation=cv2.INTER_CUBIC,
            )

            cam2_view = rotate_for_display(cam2_view)
            cam3_view = rotate_for_display(cam3_view)
            cam4_view = rotate_for_display(cam4_view)

            cam2_view = draw_text(
                cam2_view,
                "CAM 2 - NO FILTER",
                exposure_ms,
                extra_text="Eye detection source",
            )
            cam2_view = draw_eye_status(cam2_view, eye_result)
            cam3_view = draw_text(
                cam3_view,
                "CAM 3 - 405nm FILTER",
                exposure_ms,
                extra_text="UV capture path",
            )
            cam4_view = draw_text(
                cam4_view,
                "CAM 4 - 660nm FILTER",
                exposure_ms,
                extra_text="UV capture path",
            )

            preview = cv2.hconcat([cam2_view, cam3_view, cam4_view])
            cv2.imshow(WINDOW_NAME, preview)

            if should_capture:
                STATE["status"] = "Capturing"
                trigger_metadata = build_trigger_metadata(eye_result, face_bbox)
                exposure_ms = get_current_exposure_ms()
                set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
                capture_sequence(
                    cam=cam,
                    preview_config=preview_config,
                    still_config=still_config,
                    exposure_ms=exposure_ms,
                    gain=CURRENT_GAIN,
                    trigger_metadata=trigger_metadata,
                    save_as_png=SAVE_AS_PNG,
                )
                set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
                prev_exposure_ms = exposure_ms
                STATE["eyes_closed_started_at"] = None
                STATE["waiting_for_reopen"] = True
                STATE["status"] = "Capture complete"

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("c"), ord("p")):
                STATE["status"] = "Manual capture"
                exposure_ms = get_current_exposure_ms()
                set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
                capture_sequence(
                    cam=cam,
                    preview_config=preview_config,
                    still_config=still_config,
                    exposure_ms=exposure_ms,
                    gain=CURRENT_GAIN,
                    trigger_metadata={"type": "manual"},
                    save_as_png=SAVE_AS_PNG,
                )
                set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
                prev_exposure_ms = exposure_ms

            if key == ord("o"):
                relay_on()

            if key == ord("f"):
                relay_off()

            if key == ord("q") or key == 27:
                break

    finally:
        relay_off()
        cam.stop()
        cv2.destroyAllWindows()
        face_detector.close()
        face_mesh.close()


if __name__ == "__main__":
    main()
