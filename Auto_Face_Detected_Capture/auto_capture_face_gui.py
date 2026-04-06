import cv2
import json
import math
import time
from pathlib import Path
from datetime import datetime
from time import sleep

import mediapipe as mp
import numpy as np
from gpiozero import LED
from picamera2 import Picamera2

# =========================
# 기본 설정
# =========================
CAPTURE_WIDTH = 5120
CAPTURE_HEIGHT = 800
SINGLE_WIDTH = CAPTURE_WIDTH // 4   # 1280

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 400
SINGLE_VIEW_WIDTH = 1280
SINGLE_VIEW_HEIGHT = 800

MIN_EXPOSURE_MS = 1
MAX_EXPOSURE_MS = 100
INITIAL_EXPOSURE_MS = 8
CURRENT_GAIN = 1.0

WINDOW_NAME = "Auto Capture GUI"
SAVE_ROOT = Path.home() / "Graduate_Project" / "captures"
SAVE_AS_PNG = True

# =========================
# 릴레이 / LED 설정
# =========================
RELAY_PIN = 17
RELAY_ACTIVE_HIGH = False
RELAY_WARMUP_SEC = 0.3
relay = LED(RELAY_PIN, active_high=RELAY_ACTIVE_HIGH, initial_value=False)

# =========================
# 자동 촬영 조건
# =========================
FACE_CENTER_TOL_X = 0.18     # 얼굴 중심 허용 오차 (가로)
FACE_CENTER_TOL_Y = 0.22     # 얼굴 중심 허용 오차 (세로)
MIN_FACE_WIDTH_RATIO = 0.22  # 얼굴이 너무 작으면 촬영 안 함
MAX_ABS_YAW = 12.0
MAX_ABS_PITCH = 12.0
MAX_ABS_ROLL = 10.0
EYE_AR_THRESHOLD = 0.22
EYES_CLOSED_HOLD_FRAMES = 5
CAPTURE_COOLDOWN_SEC = 2.0

CAMERA_INFO = {
    "cam2": {
        "label": "CAM 2 - NO FILTER",
        "folder": "cam2_no_filter",
        "filter": "no_filter",
        "x_start": SINGLE_WIDTH * 1,
        "x_end": SINGLE_WIDTH * 2,
        "sequence_order": 1
    },
    "cam3": {
        "label": "CAM 3 - 405nm FILTER",
        "folder": "cam3_405nm",
        "filter": "405nm_filter",
        "x_start": SINGLE_WIDTH * 2,
        "x_end": SINGLE_WIDTH * 3,
        "sequence_order": 2
    },
    "cam4": {
        "label": "CAM 4 - 660nm FILTER",
        "folder": "cam4_660nm",
        "filter": "660nm_filter",
        "x_start": SINGLE_WIDTH * 3,
        "x_end": SINGLE_WIDTH * 4,
        "sequence_order": 3
    }
}

# =========================
# UI 상태
# =========================
STATE = {
    "armed": False,
    "status_text": "START 버튼을 누르면 자동 촬영 대기",
    "closed_counter": 0,
    "cooldown_until": 0.0,
}

BUTTONS = {
    "start": (20, 20, 220, 70),
    "cancel": (240, 20, 440, 70),
}

# =========================
# MediaPipe FaceMesh
# =========================
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
POSE_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}


# =========================
# 트랙바 콜백
# =========================
def nothing(x):
    pass


# =========================
# 릴레이 제어
# =========================
def relay_on():
    relay.on()
    print("[릴레이] ON")


def relay_off():
    relay.off()
    print("[릴레이] OFF")


# =========================
# 수동 카메라 제어
# =========================
def set_manual_controls(camera, exposure_ms, gain):
    camera.set_controls({
        "AeEnable": False,
        "ExposureTime": exposure_ms * 1000,
        "AnalogueGain": gain,
    })


# =========================
# 현재 트랙바 값 읽기
# =========================
def get_current_exposure_ms():
    exposure_ms = cv2.getTrackbarPos("Exposure(ms)", WINDOW_NAME)
    if exposure_ms < MIN_EXPOSURE_MS:
        exposure_ms = MIN_EXPOSURE_MS
    return exposure_ms


# =========================
# 이미지 추출 / 저장
# =========================
def extract_cam_frame(full_frame_bgr, cam_key):
    info = CAMERA_INFO[cam_key]
    return full_frame_bgr[:, info["x_start"]:info["x_end"]]


def save_image(path, img_bgr):
    if path.suffix.lower() == ".png":
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])


def save_one_camera_image(cam_key, frame_bgr, timestamp, exposure_ms, gain, ext, trigger_meta):
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
        "capture_type": "auto_face_eye_trigger_then_high_quality_still_fullframe_then_crop",
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
        },
        "trigger": trigger_meta,
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

    print(f"[저장 완료] {image_path}")
    print(f"[저장 완료] {meta_path}")


# =========================
# UI 그리기
# =========================
def draw_text(img, title, exposure_ms, extra_text=None):
    out = img.copy()
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(out, f"Exposure: {exposure_ms} ms", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if extra_text is not None:
        cv2.putText(out, extra_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def draw_buttons_and_status(preview_img, armed, status_text):
    img = preview_img.copy()
    # 반투명 헤더
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 95), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    sx1, sy1, sx2, sy2 = BUTTONS["start"]
    cx1, cy1, cx2, cy2 = BUTTONS["cancel"]

    start_color = (0, 160, 0) if not armed else (80, 80, 80)
    cancel_color = (0, 80, 180) if armed else (80, 80, 80)

    cv2.rectangle(img, (sx1, sy1), (sx2, sy2), start_color, -1)
    cv2.rectangle(img, (cx1, cy1), (cx2, cy2), cancel_color, -1)

    cv2.putText(img, "START", (sx1 + 48, sy1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, "CANCEL", (cx1 + 38, cy1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    state_text = "ARMED" if armed else "IDLE"
    state_color = (0, 255, 0) if armed else (180, 180, 180)
    cv2.putText(img, f"STATE: {state_text}", (470, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, state_color, 2)
    cv2.putText(img, status_text, (470, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    return img


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    sx1, sy1, sx2, sy2 = BUTTONS["start"]
    cx1, cy1, cx2, cy2 = BUTTONS["cancel"]

    if sx1 <= x <= sx2 and sy1 <= y <= sy2:
        STATE["armed"] = True
        STATE["closed_counter"] = 0
        STATE["status_text"] = "자동 촬영 대기 중: 얼굴/각도/눈감김 확인"
    elif cx1 <= x <= cx2 and cy1 <= y <= cy2:
        STATE["armed"] = False
        STATE["closed_counter"] = 0
        STATE["status_text"] = "자동 촬영 취소"


# =========================
# 얼굴 분석 유틸
# =========================
def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def calc_ear(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    # p1, p2, p3, p4, p5, p6
    p1, p2, p3, p4, p5, p6 = pts
    ear = (distance(p2, p6) + distance(p3, p5)) / (2.0 * max(distance(p1, p4), 1e-6))
    return ear


def get_face_bbox(landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = int(max(0, min(xs))), int(min(w - 1, max(xs)))
    y1, y2 = int(max(0, min(ys))), int(min(h - 1, max(ys)))
    return x1, y1, x2, y2


def estimate_head_pose(landmarks, w, h):
    image_points = np.array([
        (landmarks[POSE_IDX["nose_tip"]].x * w, landmarks[POSE_IDX["nose_tip"]].y * h),
        (landmarks[POSE_IDX["chin"]].x * w, landmarks[POSE_IDX["chin"]].y * h),
        (landmarks[POSE_IDX["left_eye_outer"]].x * w, landmarks[POSE_IDX["left_eye_outer"]].y * h),
        (landmarks[POSE_IDX["right_eye_outer"]].x * w, landmarks[POSE_IDX["right_eye_outer"]].y * h),
        (landmarks[POSE_IDX["mouth_left"]].x * w, landmarks[POSE_IDX["mouth_left"]].y * h),
        (landmarks[POSE_IDX["mouth_right"]].x * w, landmarks[POSE_IDX["mouth_right"]].y * h),
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rot_mat, _ = cv2.Rodrigues(rotation_vec)
    angles, *_ = cv2.RQDecomp3x3(rot_mat)
    pitch, yaw, roll = [float(a) for a in angles]
    return pitch, yaw, roll


def analyze_face_and_eyes(frame_bgr, face_mesh):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    info = {
        "face_ok": False,
        "eyes_closed": False,
        "bbox": None,
        "center_ok": False,
        "size_ok": False,
        "angles_ok": False,
        "pitch": None,
        "yaw": None,
        "roll": None,
        "ear_left": None,
        "ear_right": None,
    }

    if not results.multi_face_landmarks:
        return info

    landmarks = results.multi_face_landmarks[0].landmark
    x1, y1, x2, y2 = get_face_bbox(landmarks, w, h)
    info["bbox"] = (x1, y1, x2, y2)

    face_cx = (x1 + x2) / 2.0
    face_cy = (y1 + y2) / 2.0
    face_w = x2 - x1

    info["center_ok"] = abs(face_cx - (w / 2.0)) <= (w * FACE_CENTER_TOL_X) and \
                        abs(face_cy - (h / 2.0)) <= (h * FACE_CENTER_TOL_Y)
    info["size_ok"] = face_w >= (w * MIN_FACE_WIDTH_RATIO)

    pose = estimate_head_pose(landmarks, w, h)
    if pose is not None:
        pitch, yaw, roll = pose
        info["pitch"] = pitch
        info["yaw"] = yaw
        info["roll"] = roll
        info["angles_ok"] = abs(yaw) <= MAX_ABS_YAW and abs(pitch) <= MAX_ABS_PITCH and abs(roll) <= MAX_ABS_ROLL

    ear_left = calc_ear(landmarks, LEFT_EYE, w, h)
    ear_right = calc_ear(landmarks, RIGHT_EYE, w, h)
    info["ear_left"] = ear_left
    info["ear_right"] = ear_right
    info["eyes_closed"] = ear_left < EYE_AR_THRESHOLD and ear_right < EYE_AR_THRESHOLD

    info["face_ok"] = info["center_ok"] and info["size_ok"] and info["angles_ok"]
    return info


# =========================
# 조건 시각화
# =========================
def draw_detection_status(cam2_view, detection_info):
    img = cam2_view.copy()
    h, w = img.shape[:2]

    if detection_info["bbox"] is not None:
        x1, y1, x2, y2 = detection_info["bbox"]
        sx = PREVIEW_WIDTH / 1280.0
        sy = PREVIEW_HEIGHT / 800.0
        x1, x2 = int(x1 * sx), int(x2 * sx)
        y1, y2 = int(y1 * sy), int(y2 * sy)
        color = (0, 255, 0) if detection_info["face_ok"] else (0, 180, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    line_y = PREVIEW_HEIGHT - 85
    checks = [
        ("CENTER", detection_info["center_ok"]),
        ("SIZE", detection_info["size_ok"]),
        ("ANGLE", detection_info["angles_ok"]),
        ("EYES CLOSED", detection_info["eyes_closed"]),
    ]
    x = 10
    for label, ok in checks:
        c = (0, 255, 0) if ok else (0, 0, 255)
        cv2.putText(img, label, (x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        x += 150

    if detection_info["yaw"] is not None:
        cv2.putText(
            img,
            f"Pitch:{detection_info['pitch']:.1f} Yaw:{detection_info['yaw']:.1f} Roll:{detection_info['roll']:.1f}",
            (10, PREVIEW_HEIGHT - 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    if detection_info["ear_left"] is not None:
        cv2.putText(
            img,
            f"EAR L:{detection_info['ear_left']:.3f} R:{detection_info['ear_right']:.3f}",
            (10, PREVIEW_HEIGHT - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    return img


# =========================
# 고화질 전체 프레임 캡처
# =========================
def capture_high_quality_full_frame(cam, preview_config, still_config, exposure_ms, gain):
    relay_on()
    sleep(RELAY_WARMUP_SEC)

    try:
        cam.stop()
        cam.configure(still_config)
        cam.start()
        set_manual_controls(cam, exposure_ms, gain)

        still_frame = cam.capture_array()   # RGB888 예상
        full_frame_bgr = cv2.cvtColor(still_frame, cv2.COLOR_RGB2BGR)
    finally:
        relay_off()
        cam.stop()
        cam.configure(preview_config)
        cam.start()
        set_manual_controls(cam, exposure_ms, gain)

    return full_frame_bgr


# =========================
# 자동 촬영 시퀀스
# =========================
def capture_sequence(cam, preview_config, still_config, exposure_ms, gain, trigger_meta, save_as_png=True):
    ext = "png" if save_as_png else "jpg"

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

        show_single_camera(WINDOW_NAME, target_frame, cam_key, exposure_ms, hold_ms=500)

        save_one_camera_image(
            cam_key=cam_key,
            frame_bgr=target_frame,
            timestamp=capture_timestamp,
            exposure_ms=exposure_ms,
            gain=gain,
            ext=ext,
            trigger_meta=trigger_meta,
        )


# =========================
# 단독 출력
# =========================
def show_single_camera(window_name, frame_bgr, cam_key, exposure_ms, hold_ms=700):
    info = CAMERA_INFO[cam_key]

    single_view = cv2.resize(frame_bgr, (SINGLE_VIEW_WIDTH, SINGLE_VIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
    single_view = draw_text(single_view, info["label"], exposure_ms, extra_text="Capturing...")

    cv2.imshow(window_name, single_view)
    cv2.waitKey(hold_ms)


# =========================
# 카메라 초기화
# =========================
def init_camera():
    camera_list = Picamera2.global_camera_info()
    print("Detected cameras:", camera_list)

    if len(camera_list) == 0:
        relay_off()
        raise RuntimeError("카메라가 감지되지 않았습니다.")

    camera = Picamera2(0)

    preview_cfg = camera.create_preview_configuration(
        main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "XRGB8888"}
    )
    still_cfg = camera.create_still_configuration(
        main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
    )

    camera.configure(preview_cfg)
    camera.options["quality"] = 100
    camera.options["compress_level"] = 0

    set_manual_controls(camera, INITIAL_EXPOSURE_MS, CURRENT_GAIN)
    camera.start()
    relay_off()
    return camera, preview_cfg, still_cfg


# =========================
# 메인
# =========================
def main():
    cam, preview_config, still_config = init_camera()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, PREVIEW_WIDTH * 3, PREVIEW_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    cv2.createTrackbar("Exposure(ms)", WINDOW_NAME, INITIAL_EXPOSURE_MS, MAX_EXPOSURE_MS, nothing)
    cv2.setTrackbarMin("Exposure(ms)", WINDOW_NAME, MIN_EXPOSURE_MS)

    prev_exposure_ms = INITIAL_EXPOSURE_MS

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
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

                detection_info = analyze_face_and_eyes(cam2, face_mesh)

                now = time.monotonic()
                if now < STATE["cooldown_until"]:
                    STATE["status_text"] = "촬영 직후 대기 중..."
                elif not STATE["armed"]:
                    STATE["closed_counter"] = 0
                else:
                    if detection_info["bbox"] is None:
                        STATE["closed_counter"] = 0
                        STATE["status_text"] = "얼굴을 화면 안에 넣어주세요"
                    elif not detection_info["center_ok"]:
                        STATE["closed_counter"] = 0
                        STATE["status_text"] = "얼굴을 가운데로 맞춰주세요"
                    elif not detection_info["size_ok"]:
                        STATE["closed_counter"] = 0
                        STATE["status_text"] = "얼굴을 조금 더 가까이 해주세요"
                    elif not detection_info["angles_ok"]:
                        STATE["closed_counter"] = 0
                        STATE["status_text"] = "정면 각도로 맞춰주세요"
                    elif not detection_info["eyes_closed"]:
                        STATE["closed_counter"] = 0
                        STATE["status_text"] = "눈을 감아주세요"
                    else:
                        STATE["closed_counter"] += 1
                        STATE["status_text"] = f"눈 감김 확인 중... {STATE['closed_counter']}/{EYES_CLOSED_HOLD_FRAMES}"

                        if STATE["closed_counter"] >= EYES_CLOSED_HOLD_FRAMES:
                            trigger_meta = {
                                "type": "auto_face_and_eyes_closed",
                                "face_ok": detection_info["face_ok"],
                                "center_ok": detection_info["center_ok"],
                                "size_ok": detection_info["size_ok"],
                                "angles_ok": detection_info["angles_ok"],
                                "eyes_closed": detection_info["eyes_closed"],
                                "pitch": detection_info["pitch"],
                                "yaw": detection_info["yaw"],
                                "roll": detection_info["roll"],
                                "ear_left": detection_info["ear_left"],
                                "ear_right": detection_info["ear_right"],
                            }

                            capture_sequence(
                                cam=cam,
                                preview_config=preview_config,
                                still_config=still_config,
                                exposure_ms=exposure_ms,
                                gain=CURRENT_GAIN,
                                trigger_meta=trigger_meta,
                                save_as_png=SAVE_AS_PNG,
                            )

                            STATE["armed"] = False
                            STATE["closed_counter"] = 0
                            STATE["cooldown_until"] = time.monotonic() + CAPTURE_COOLDOWN_SEC
                            STATE["status_text"] = "촬영 완료"

                            exposure_ms = get_current_exposure_ms()
                            set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
                            prev_exposure_ms = exposure_ms

                cam2_view = cv2.resize(cam2, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
                cam3_view = cv2.resize(cam3, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
                cam4_view = cv2.resize(cam4, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)

                cam2_view = draw_text(cam2_view, "CAM 2 - NO FILTER", exposure_ms)
                cam3_view = draw_text(cam3_view, "CAM 3 - 405nm FILTER", exposure_ms)
                cam4_view = draw_text(cam4_view, "CAM 4 - 660nm FILTER", exposure_ms)

                cam2_view = draw_detection_status(cam2_view, detection_info)
                cv2.putText(cam3_view, "자동 촬영용 판단은 CAM2 기준", (20, PREVIEW_HEIGHT - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(cam4_view, "조건 만족 시 릴레이 ON -> 촬영 -> OFF", (20, PREVIEW_HEIGHT - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

                preview = cv2.hconcat([cam2_view, cam3_view, cam4_view])
                preview = draw_buttons_and_status(preview, STATE["armed"], STATE["status_text"])
                cv2.imshow(WINDOW_NAME, preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('o'):
                    relay_on()
                elif key == ord('f'):
                    relay_off()
                elif key == ord('s'):
                    STATE["armed"] = True
                    STATE["closed_counter"] = 0
                    STATE["status_text"] = "자동 촬영 대기 중: 얼굴/각도/눈감김 확인"
                elif key == ord('x'):
                    STATE["armed"] = False
                    STATE["closed_counter"] = 0
                    STATE["status_text"] = "자동 촬영 취소"
        finally:
            relay_off()
            cam.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
