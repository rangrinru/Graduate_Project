import cv2
import json
from pathlib import Path
from datetime import datetime
from time import sleep
from gpiozero import LED
from picamera2 import Picamera2

try:
    import mediapipe as mp
except ImportError:
    mp = None

# =========================
# 기본 설정
# =========================
CAPTURE_WIDTH = 5120
CAPTURE_HEIGHT = 800
SINGLE_WIDTH = CAPTURE_WIDTH // 4   # 1280

# 미리보기용 표시 크기 (회전 전 기준)
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 400

# 개별 촬영 시 단독 표시 크기 (회전 전 기준)
SINGLE_VIEW_WIDTH = 1280
SINGLE_VIEW_HEIGHT = 800

MIN_EXPOSURE_MS = 1
MAX_EXPOSURE_MS = 100
INITIAL_EXPOSURE_MS = 8

WINDOW_NAME = "Cam2 | Cam3 | Cam4 Preview"

SAVE_ROOT = Path.home() / "Graduate_Project" / "captures"

# 공통 Gain (현재 구조상 카메라별 독립 설정 불가)
CURRENT_GAIN = 1.0

# 저장 형식
# True  -> PNG (무손실, 분석용 추천)
# False -> JPG (용량 절약)
SAVE_AS_PNG = True

# =========================
# 표시 회전 설정
# True면 화면 출력만 90도 회전
# 저장 원본은 회전하지 않음
# =========================
ROTATE_DISPLAY = True
DISPLAY_ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE

# =========================
# 릴레이 / 촬영 LED 설정 (GPIO17)
# =========================
RELAY_PIN = 17
RELAY_ACTIVE_HIGH = False
RELAY_WARMUP_SEC = 0.3
relay = LED(RELAY_PIN, active_high=RELAY_ACTIVE_HIGH, initial_value=False)

# =========================
# 백색 LED 설정 (GPIO22)
# 얼굴 위치 맞춤 / 눈감음 확인용
# =========================
WHITE_LED_PIN = 22
WHITE_LED_ACTIVE_HIGH = True
WHITE_LED_OFF_BEFORE_CAPTURE_SEC = 0.15
white_led = LED(WHITE_LED_PIN, active_high=WHITE_LED_ACTIVE_HIGH, initial_value=False)

# =========================
# 자동 얼굴 촬영 설정
# =========================
AUTO_GUIDE_ENABLED = True
FACE_CENTER_TOL_X = 0.18
FACE_CENTER_TOL_Y = 0.20
FACE_MIN_AREA_RATIO = 0.08
FACE_MAX_AREA_RATIO = 0.70
EYE_AR_THRESHOLD = 0.18
EYES_CLOSED_HOLD_FRAMES = 4

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

AUTO_STATE = {
    "armed": False,
    "eyes_closed_count": 0,
    "status": "대기 중",
}

# =========================
# MediaPipe 설정
# =========================
if mp is not None:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
else:
    face_mesh = None

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def nothing(x):
    pass


def relay_on():
    relay.on()
    print("[릴레이] ON")


def relay_off():
    relay.off()
    print("[릴레이] OFF")


def white_led_on():
    white_led.on()
    print("[백색 LED] ON")


def white_led_off():
    white_led.off()
    print("[백색 LED] OFF")


def set_manual_controls(camera, exposure_ms, gain):
    camera.set_controls({
        "AeEnable": False,
        "ExposureTime": exposure_ms * 1000,
        "AnalogueGain": gain
    })


def get_current_exposure_ms():
    exposure_ms = cv2.getTrackbarPos("Exposure(ms)", WINDOW_NAME)
    if exposure_ms < MIN_EXPOSURE_MS:
        exposure_ms = MIN_EXPOSURE_MS
    return exposure_ms


def extract_cam_frame(full_frame_bgr, cam_key):
    info = CAMERA_INFO[cam_key]
    return full_frame_bgr[:, info["x_start"]:info["x_end"]]


def rotate_for_display(img):
    if ROTATE_DISPLAY:
        return cv2.rotate(img, DISPLAY_ROTATE_CODE)
    return img


def draw_text(img, title, exposure_ms, extra_text=None):
    out = img.copy()
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(out, f"Exposure: {exposure_ms} ms", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if extra_text is not None:
        cv2.putText(out, extra_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def save_image(path, img_bgr):
    if path.suffix.lower() == ".png":
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])


def save_one_camera_image(cam_key, frame_bgr, timestamp, exposure_ms, gain, ext):
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
        "capture_type": "high_quality_still_fullframe_then_crop",
        "file_format": ext,
        "camera_mode": {
            "capture_width": CAPTURE_WIDTH,
            "capture_height": CAPTURE_HEIGHT,
            "single_width": SINGLE_WIDTH
        },
        "camera_control": {
            "AeEnable": False,
            "ExposureTime_ms": exposure_ms,
            "ExposureTime_us": exposure_ms * 1000,
            "AnalogueGain": gain
        },
        "relay_control": {
            "relay_pin": RELAY_PIN,
            "active_high": RELAY_ACTIVE_HIGH,
            "relay_used": True
        },
        "white_led_control": {
            "white_led_pin": WHITE_LED_PIN,
            "white_led_active_high": WHITE_LED_ACTIVE_HIGH,
            "used_for_face_alignment": True
        },
        "crop_region": {
            "x_start": info["x_start"],
            "x_end": info["x_end"],
            "y_start": 0,
            "y_end": CAPTURE_HEIGHT
        },
        "saved_file": str(image_path)
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"[저장 완료] {image_path}")
    print(f"[저장 완료] {meta_path}")


def show_single_camera(window_name, frame_bgr, cam_key, exposure_ms, hold_ms=700):
    info = CAMERA_INFO[cam_key]
    single_view = cv2.resize(frame_bgr, (SINGLE_VIEW_WIDTH, SINGLE_VIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
    single_view = rotate_for_display(single_view)
    single_view = draw_text(single_view, info["label"], exposure_ms, extra_text="Capturing...")
    cv2.imshow(window_name, single_view)
    cv2.waitKey(hold_ms)


def capture_high_quality_full_frame(cam, preview_config, still_config, exposure_ms, gain):
    try:
        cam.stop()
        cam.configure(still_config)
        cam.start()
        set_manual_controls(cam, exposure_ms, gain)
        still_frame = cam.capture_array()
        full_frame_bgr = cv2.cvtColor(still_frame, cv2.COLOR_RGB2BGR)
    finally:
        cam.stop()
        cam.configure(preview_config)
        cam.start()
        set_manual_controls(cam, exposure_ms, gain)
    return full_frame_bgr


def capture_sequence(cam, preview_config, still_config, exposure_ms, gain, save_as_png=True):
    ext = "png" if save_as_png else "jpg"
    relay_on()
    sleep(RELAY_WARMUP_SEC)

    try:
        full_frame_bgr = capture_high_quality_full_frame(
            cam=cam,
            preview_config=preview_config,
            still_config=still_config,
            exposure_ms=exposure_ms,
            gain=gain
        )

        capture_timestamp = datetime.now()
        for cam_key in ["cam2", "cam3", "cam4"]:
            target_frame = extract_cam_frame(full_frame_bgr, cam_key).copy()
            show_single_camera(WINDOW_NAME, target_frame, cam_key, exposure_ms, hold_ms=700)
            save_one_camera_image(
                cam_key=cam_key,
                frame_bgr=target_frame,
                timestamp=capture_timestamp,
                exposure_ms=exposure_ms,
                gain=gain,
                ext=ext
            )
    finally:
        relay_off()


def _euclidean(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _eye_aspect_ratio(eye_pts):
    a = _euclidean(eye_pts[1], eye_pts[5])
    b = _euclidean(eye_pts[2], eye_pts[4])
    c = _euclidean(eye_pts[0], eye_pts[3])
    if c == 0:
        return 1.0
    return (a + b) / (2.0 * c)


def analyze_face(cam2_bgr):
    h, w = cam2_bgr.shape[:2]
    result = {
        "face_found": False,
        "center_ok": False,
        "size_ok": False,
        "eyes_closed": False,
        "ear_left": 1.0,
        "ear_right": 1.0,
        "bbox": None,
        "guide_text": "얼굴을 화면에 맞춰주세요",
    }

    if face_mesh is None:
        result["guide_text"] = "mediapipe 미설치: pip install mediapipe"
        return result

    rgb = cv2.cvtColor(cam2_bgr, cv2.COLOR_BGR2RGB)
    mesh_result = face_mesh.process(rgb)
    if not mesh_result.multi_face_landmarks:
        result["guide_text"] = "얼굴이 감지되지 않습니다"
        return result

    face_landmarks = mesh_result.multi_face_landmarks[0]
    pts = []
    for lm in face_landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
    y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area_ratio = (bw * bh) / float(w * h)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    norm_dx = abs(cx - (w / 2.0)) / w
    norm_dy = abs(cy - (h / 2.0)) / h

    left_eye_pts = [pts[i] for i in LEFT_EYE]
    right_eye_pts = [pts[i] for i in RIGHT_EYE]
    ear_left = _eye_aspect_ratio(left_eye_pts)
    ear_right = _eye_aspect_ratio(right_eye_pts)
    eyes_closed = (ear_left < EYE_AR_THRESHOLD) and (ear_right < EYE_AR_THRESHOLD)

    center_ok = (norm_dx <= FACE_CENTER_TOL_X) and (norm_dy <= FACE_CENTER_TOL_Y)
    size_ok = (FACE_MIN_AREA_RATIO <= area_ratio <= FACE_MAX_AREA_RATIO)

    result.update({
        "face_found": True,
        "center_ok": center_ok,
        "size_ok": size_ok,
        "eyes_closed": eyes_closed,
        "ear_left": ear_left,
        "ear_right": ear_right,
        "bbox": (x1, y1, x2, y2),
    })

    if not center_ok:
        result["guide_text"] = "얼굴을 화면 중앙에 맞춰주세요"
    elif not size_ok:
        result["guide_text"] = "얼굴 거리를 조정해주세요"
    elif not eyes_closed:
        result["guide_text"] = "눈을 감아주세요"
    else:
        result["guide_text"] = "조건 충족"

    return result


def draw_face_guide(cam2_bgr, detection):
    out = cam2_bgr.copy()
    h, w = out.shape[:2]

    cx1 = int(w * (0.5 - FACE_CENTER_TOL_X))
    cx2 = int(w * (0.5 + FACE_CENTER_TOL_X))
    cy1 = int(h * (0.5 - FACE_CENTER_TOL_Y))
    cy2 = int(h * (0.5 + FACE_CENTER_TOL_Y))
    cv2.rectangle(out, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)

    if detection["bbox"] is not None:
        x1, y1, x2, y2 = detection["bbox"]
        color = (0, 255, 0) if (detection["center_ok"] and detection["size_ok"] and detection["eyes_closed"]) else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    status = detection["guide_text"]
    cv2.putText(out, status, (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(out, f"L_EAR:{detection['ear_left']:.2f}  R_EAR:{detection['ear_right']:.2f}", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


camera_list = Picamera2.global_camera_info()
print("Detected cameras:", camera_list)
if len(camera_list) == 0:
    relay_off()
    white_led_off()
    raise RuntimeError("카메라가 감지되지 않았습니다.")

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
set_manual_controls(cam, INITIAL_EXPOSURE_MS, CURRENT_GAIN)
cam.start()
relay_off()
white_led_off()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
if ROTATE_DISPLAY:
    cv2.resizeWindow(WINDOW_NAME, PREVIEW_HEIGHT * 3, PREVIEW_WIDTH)
else:
    cv2.resizeWindow(WINDOW_NAME, PREVIEW_WIDTH * 3, PREVIEW_HEIGHT)

cv2.createTrackbar("Exposure(ms)", WINDOW_NAME, INITIAL_EXPOSURE_MS, MAX_EXPOSURE_MS, nothing)
cv2.setTrackbarMin("Exposure(ms)", WINDOW_NAME, MIN_EXPOSURE_MS)

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

        detection = analyze_face(cam2)
        cam2 = draw_face_guide(cam2, detection)

        if AUTO_STATE["armed"]:
            white_led_on()
            if detection["face_found"] and detection["center_ok"] and detection["size_ok"] and detection["eyes_closed"]:
                AUTO_STATE["eyes_closed_count"] += 1
                AUTO_STATE["status"] = f"눈감음 확인 중 {AUTO_STATE['eyes_closed_count']}/{EYES_CLOSED_HOLD_FRAMES}"
            else:
                AUTO_STATE["eyes_closed_count"] = 0
                AUTO_STATE["status"] = detection["guide_text"]

            if AUTO_STATE["eyes_closed_count"] >= EYES_CLOSED_HOLD_FRAMES:
                AUTO_STATE["status"] = "조건 충족: 백색 LED OFF 후 촬영"
                white_led_off()
                sleep(WHITE_LED_OFF_BEFORE_CAPTURE_SEC)

                exposure_ms = get_current_exposure_ms()
                set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
                capture_sequence(
                    cam=cam,
                    preview_config=preview_config,
                    still_config=still_config,
                    exposure_ms=exposure_ms,
                    gain=CURRENT_GAIN,
                    save_as_png=SAVE_AS_PNG,
                )
                exposure_ms = get_current_exposure_ms()
                set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
                prev_exposure_ms = exposure_ms
                AUTO_STATE["armed"] = False
                AUTO_STATE["eyes_closed_count"] = 0
                AUTO_STATE["status"] = "촬영 완료"
        else:
            white_led_off()

        cam2_view = cv2.resize(cam2, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
        cam3_view = cv2.resize(cam3, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
        cam4_view = cv2.resize(cam4, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)

        cam2_view = rotate_for_display(cam2_view)
        cam3_view = rotate_for_display(cam3_view)
        cam4_view = rotate_for_display(cam4_view)

        cam2_view = draw_text(cam2_view, "CAM 2 - NO FILTER", exposure_ms,
                              extra_text="c: 자동정렬촬영 / p: 즉시촬영")
        cam3_view = draw_text(cam3_view, "CAM 3 - 405nm FILTER", exposure_ms,
                              extra_text=f"Auto: {AUTO_STATE['status']}")
        cam4_view = draw_text(cam4_view, "CAM 4 - 660nm FILTER", exposure_ms,
                              extra_text="x: 취소 / q: 종료")

        preview = cv2.hconcat([cam2_view, cam3_view, cam4_view])
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            AUTO_STATE["armed"] = True
            AUTO_STATE["eyes_closed_count"] = 0
            AUTO_STATE["status"] = "얼굴 위치를 맞추고 눈을 감아주세요"
            white_led_on()

        if key == ord('x'):
            AUTO_STATE["armed"] = False
            AUTO_STATE["eyes_closed_count"] = 0
            AUTO_STATE["status"] = "자동촬영 취소"
            white_led_off()

        if key == ord('p'):
            AUTO_STATE["armed"] = False
            AUTO_STATE["eyes_closed_count"] = 0
            AUTO_STATE["status"] = "즉시 촬영"
            white_led_off()
            exposure_ms = get_current_exposure_ms()
            set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
            capture_sequence(
                cam=cam,
                preview_config=preview_config,
                still_config=still_config,
                exposure_ms=exposure_ms,
                gain=CURRENT_GAIN,
                save_as_png=SAVE_AS_PNG,
            )
            exposure_ms = get_current_exposure_ms()
            set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
            prev_exposure_ms = exposure_ms
            AUTO_STATE["status"] = "즉시 촬영 완료"

        if key == ord('o'):
            relay_on()
        if key == ord('f'):
            relay_off()
        if key == ord('q'):
            break

finally:
    white_led_off()
    relay_off()
    cam.stop()
    cv2.destroyAllWindows()
    if face_mesh is not None:
        face_mesh.close()
