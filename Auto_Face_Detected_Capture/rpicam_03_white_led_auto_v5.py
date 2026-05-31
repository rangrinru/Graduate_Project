
import cv2
import json
import math
import logging
import threading
from pathlib import Path
from datetime import datetime
from time import sleep, monotonic
from collections import deque

from gpiozero import LED
from picamera2 import Picamera2

try:
    import mediapipe as mp
except ImportError:
    mp = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rpicam_auto_v5")

# =========================
# 기본 설정
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

WINDOW_NAME = "Cam2 | Cam3 | Cam4 Preview"

SAVE_ROOT = Path.home() / "Graduate_Project" / "captures"
SESSION_ROOT = SAVE_ROOT / "sessions"
SESSION_ROOT.mkdir(parents=True, exist_ok=True)

CURRENT_GAIN = 1.0
SAVE_AS_PNG = True

ROTATE_DISPLAY = True
DISPLAY_ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE

# =========================
# 릴레이 / LED 설정
# =========================
RELAY_PIN = 17
RELAY_ACTIVE_HIGH = False
RELAY_WARMUP_SEC = 0.3
relay = LED(RELAY_PIN, active_high=RELAY_ACTIVE_HIGH, initial_value=False)

WHITE_LED_PIN = 22
WHITE_LED_ACTIVE_HIGH = True
WHITE_LED_OFF_BEFORE_CAPTURE_SEC = 0.15
white_led = LED(WHITE_LED_PIN, active_high=WHITE_LED_ACTIVE_HIGH, initial_value=False)

# =========================
# 자동 얼굴 촬영 설정
# =========================
FACE_CENTER_TOL_X = 0.18
FACE_CENTER_TOL_Y = 0.20
FACE_MIN_AREA_RATIO = 0.08
FACE_MAX_AREA_RATIO = 0.70

MAX_ABS_ROLL_DEG = 8.0
MAX_ABS_YAW_SCORE = 0.12
MAX_ABS_PITCH_SCORE = 0.12

DEFAULT_EYE_AR_THRESHOLD = 0.18
MIN_DYNAMIC_EYE_AR_THRESHOLD = 0.14
MAX_DYNAMIC_EYE_AR_THRESHOLD = 0.26
EYE_THRESHOLD_RATIO = 0.72

STABLE_FACE_HOLD_FRAMES = 6
OPEN_EYE_BASELINE_SAMPLES = 15
EYES_CLOSED_HOLD_FRAMES = 4
CAPTURE_COOLDOWN_SEC = 1.5

# =========================
# 포르피린 분석 설정
# =========================
ANALYSIS_ENABLED = True
ANALYSIS_BACKGROUND = True
PORPHYRIN_MIN_AREA = 20
ROI_MARGIN_RATIO = 0.15

CAMERA_INFO = {
    "cam2": {"label": "CAM 2 - NO FILTER", "filter": "no_filter", "x_start": SINGLE_WIDTH * 1, "x_end": SINGLE_WIDTH * 2, "sequence_order": 1},
    "cam3": {"label": "CAM 3 - 405nm FILTER", "filter": "405nm_filter", "x_start": SINGLE_WIDTH * 2, "x_end": SINGLE_WIDTH * 3, "sequence_order": 2},
    "cam4": {"label": "CAM 4 - 660nm FILTER", "filter": "660nm_filter", "x_start": SINGLE_WIDTH * 3, "x_end": SINGLE_WIDTH * 4, "sequence_order": 3},
}

STATE = {
    "armed": False,
    "eyes_closed_count": 0,
    "stable_face_count": 0,
    "status": "대기 중",
    "white_led_is_on": False,
    "relay_is_on": False,
    "last_session_dir": None,
    "last_capture_monotonic": 0.0,
    "dynamic_eye_threshold": DEFAULT_EYE_AR_THRESHOLD,
    "last_analysis_report": None,
    "analysis_in_progress": False,
    "last_detection_snapshot": None,
}

analysis_lock = threading.Lock()

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

NOSE_TIP_IDX = 1
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263
LEFT_FACE_IDX = 234
RIGHT_FACE_IDX = 454
UPPER_FACE_IDX = 10
LOWER_FACE_IDX = 152
MOUTH_LEFT_IDX = 61
MOUTH_RIGHT_IDX = 291

OPEN_EYE_EAR_HISTORY = deque(maxlen=OPEN_EYE_BASELINE_SAMPLES)


def nothing(x):
    pass


def relay_on():
    if not STATE["relay_is_on"]:
        relay.on()
        STATE["relay_is_on"] = True
        logger.info("[릴레이] ON")


def relay_off():
    if STATE["relay_is_on"]:
        relay.off()
        STATE["relay_is_on"] = False
        logger.info("[릴레이] OFF")


def white_led_on():
    if not STATE["white_led_is_on"]:
        white_led.on()
        STATE["white_led_is_on"] = True
        logger.info("[백색 LED] ON")


def white_led_off():
    if STATE["white_led_is_on"]:
        white_led.off()
        STATE["white_led_is_on"] = False
        logger.info("[백색 LED] OFF")


def set_manual_controls(camera, exposure_ms, gain):
    camera.set_controls({
        "AeEnable": False,
        "ExposureTime": exposure_ms * 1000,
        "AnalogueGain": gain
    })


def get_current_exposure_ms():
    exposure_ms = cv2.getTrackbarPos("Exposure(ms)", WINDOW_NAME)
    return max(exposure_ms, MIN_EXPOSURE_MS)


def safe_capture_array(cam):
    try:
        return cam.capture_array()
    except Exception as e:
        logger.exception("프레임 획득 실패: %s", e)
        return None


def extract_cam_frame(full_frame_bgr, cam_key):
    info = CAMERA_INFO[cam_key]
    return full_frame_bgr[:, info["x_start"]:info["x_end"]]


def rotate_for_display(img):
    return cv2.rotate(img, DISPLAY_ROTATE_CODE) if ROTATE_DISPLAY else img


def draw_text(img, title, exposure_ms, extra_text=None):
    out = img.copy()
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(out, f"Exposure: {exposure_ms} ms", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if extra_text is not None:
        cv2.putText(out, extra_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def save_image(path, img_bgr):
    ok = False
    if path.suffix.lower() == ".png":
        ok = cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        ok = cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")


def make_session_dir():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    session_dir = SESSION_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_id, session_dir


def save_one_camera_image(session_dir, cam_key, frame_bgr, timestamp, exposure_ms, gain, ext):
    info = CAMERA_INFO[cam_key]
    image_path = session_dir / f"{cam_key}.{ext}"
    meta_path = session_dir / f"{cam_key}_metadata.json"

    save_image(image_path, frame_bgr)

    metadata = {
        "captured_at": timestamp.isoformat(),
        "camera_name": cam_key,
        "camera_label": info["label"],
        "filter_type": info["filter"],
        "sequence_order": info["sequence_order"],
        "file_format": ext,
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
        "dynamic_eye_threshold": STATE["dynamic_eye_threshold"],
        "saved_file": str(image_path),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    logger.info("[저장 완료] %s", image_path)
    logger.info("[저장 완료] %s", meta_path)


def save_session_metadata(session_dir, timestamp, exposure_ms, gain, detection_snapshot=None):
    session_meta_path = session_dir / "session_metadata.json"
    data = {
        "session_dir": str(session_dir),
        "captured_at": timestamp.isoformat(),
        "exposure_ms": exposure_ms,
        "gain": gain,
        "dynamic_eye_threshold": STATE["dynamic_eye_threshold"],
        "open_eye_baseline_samples": list(OPEN_EYE_EAR_HISTORY),
        "trigger_detection": detection_snapshot,
        "files": {
            "cam2": str(session_dir / f"cam2.{'png' if SAVE_AS_PNG else 'jpg'}"),
            "cam3": str(session_dir / f"cam3.{'png' if SAVE_AS_PNG else 'jpg'}"),
            "cam4": str(session_dir / f"cam4.{'png' if SAVE_AS_PNG else 'jpg'}"),
        },
    }
    with open(session_meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def show_single_camera(window_name, frame_bgr, cam_key, exposure_ms, hold_ms=180):
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
        still_frame = safe_capture_array(cam)
        if still_frame is None:
            raise RuntimeError("고화질 프레임 획득 실패")
        return cv2.cvtColor(still_frame, cv2.COLOR_RGB2BGR)
    finally:
        cam.stop()
        cam.configure(preview_config)
        cam.start()
        set_manual_controls(cam, exposure_ms, gain)


def load_trigger_bbox(session_dir):
    session_meta = Path(session_dir) / "session_metadata.json"
    if not session_meta.exists():
        return None
    try:
        data = json.loads(session_meta.read_text(encoding="utf-8"))
        trig = data.get("trigger_detection")
        if not trig:
            return None
        bbox = trig.get("bbox")
        if bbox and len(bbox) == 4:
            return tuple(int(v) for v in bbox)
    except Exception as e:
        logger.warning("trigger bbox 로드 실패: %s", e)
    return None


def build_face_roi_mask(shape, bbox):
    h, w = shape[:2]
    mask = cv2.GaussianBlur(cv2.cvtColor(cv2.UMat(h, w, cv2.CV_8UC1).get(), cv2.COLOR_GRAY2BGR), (1, 1), 0)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[:] = 0

    if bbox is None:
        mask[:] = 255
        return mask

    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * ROI_MARGIN_RATIO)
    my = int(bh * ROI_MARGIN_RATIO)

    rx1 = max(0, x1 - mx)
    ry1 = max(0, y1 - my)
    rx2 = min(w, x2 + mx)
    ry2 = min(h, y2 + my)

    cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, thickness=-1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask


def analyze_porphyrin_session(session_dir):
    session_dir = Path(session_dir)
    cam2_path = session_dir / f"cam2.{'png' if SAVE_AS_PNG else 'jpg'}"
    cam3_path = session_dir / f"cam3.{'png' if SAVE_AS_PNG else 'jpg'}"
    cam4_path = session_dir / f"cam4.{'png' if SAVE_AS_PNG else 'jpg'}"

    cam2 = cv2.imread(str(cam2_path), cv2.IMREAD_GRAYSCALE)
    cam3 = cv2.imread(str(cam3_path), cv2.IMREAD_GRAYSCALE)
    cam4 = cv2.imread(str(cam4_path), cv2.IMREAD_GRAYSCALE)

    if cam4 is None:
        raise RuntimeError("cam4 이미지를 불러오지 못했습니다.")
    if cam2 is None:
        cam2 = cv2.GaussianBlur(cam4, (3, 3), 0)
    if cam3 is None:
        cam3 = cv2.GaussianBlur(cam4, (3, 3), 0)

    bbox = load_trigger_bbox(session_dir)
    roi_mask = build_face_roi_mask(cam4.shape, bbox)

    score = (
        0.65 * cam4.astype("float32")
        + 0.20 * cv2.max(cam4.astype("float32") - cam3.astype("float32"), 0)
        + 0.15 * cv2.max(cam4.astype("float32") - cam2.astype("float32"), 0)
    )

    # 얼굴 ROI 내부 비중 강화
    score = score * (0.35 + 0.65 * (roi_mask.astype("float32") / 255.0))

    score_norm = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    _, mask = cv2.threshold(score_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(mask, roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = mask.copy()
    clean_mask[:] = 0

    porphyrin_count = 0
    porphyrin_area = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= PORPHYRIN_MIN_AREA:
            clean_mask[labels == label_idx] = 255
            porphyrin_count += 1
            porphyrin_area += area

    cam4_bgr = cv2.imread(str(cam4_path), cv2.IMREAD_COLOR)
    if cam4_bgr is None:
        raise RuntimeError("cam4 컬러 이미지를 불러오지 못했습니다.")

    overlay = cam4_bgr.copy()
    overlay[clean_mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(cam4_bgr, 0.7, overlay, 0.3, 0)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)

    analysis_dir = session_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    mask_path = analysis_dir / "porphyrin_mask.png"
    overlay_path = analysis_dir / "porphyrin_overlay.png"
    roi_mask_path = analysis_dir / "face_roi_mask.png"
    report_path = analysis_dir / "report.json"

    save_image(mask_path, clean_mask)
    save_image(overlay_path, overlay)
    save_image(roi_mask_path, roi_mask)

    mean_intensity = float(score_norm[clean_mask > 0].mean()) if (clean_mask > 0).any() else 0.0

    report = {
        "session_dir": str(session_dir),
        "porphyrin_count": porphyrin_count,
        "porphyrin_area": porphyrin_area,
        "mean_intensity": mean_intensity,
        "face_bbox": bbox,
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "roi_mask_path": str(roi_mask_path),
        "report_path": str(report_path),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    STATE["last_analysis_report"] = report
    logger.info("분석 완료: count=%s area=%s intensity=%.2f", porphyrin_count, porphyrin_area, mean_intensity)
    return report


def start_background_analysis(session_dir):
    if not ANALYSIS_ENABLED:
        return

    def _job():
        with analysis_lock:
            STATE["analysis_in_progress"] = True
            try:
                analyze_porphyrin_session(session_dir)
                report = STATE["last_analysis_report"]
                STATE["status"] = f"촬영+분석 완료 (P:{report['porphyrin_count']})"
            except Exception as e:
                logger.exception("자동 분석 실패: %s", e)
                STATE["status"] = "촬영 완료 (분석 실패)"
            finally:
                STATE["analysis_in_progress"] = False

    if ANALYSIS_BACKGROUND:
        threading.Thread(target=_job, daemon=True).start()
    else:
        _job()


def capture_sequence(cam, preview_config, still_config, exposure_ms, gain, detection_snapshot=None, save_as_png=True):
    ext = "png" if save_as_png else "jpg"
    relay_on()
    sleep(RELAY_WARMUP_SEC)
    _, session_dir = make_session_dir()

    try:
        full_frame_bgr = capture_high_quality_full_frame(cam, preview_config, still_config, exposure_ms, gain)
        capture_timestamp = datetime.now()

        for cam_key in ["cam2", "cam3", "cam4"]:
            target_frame = extract_cam_frame(full_frame_bgr, cam_key).copy()
            show_single_camera(WINDOW_NAME, target_frame, cam_key, exposure_ms, hold_ms=180)
            save_one_camera_image(session_dir, cam_key, target_frame, capture_timestamp, exposure_ms, gain, ext)

        save_session_metadata(session_dir, capture_timestamp, exposure_ms, gain, detection_snapshot=detection_snapshot)
        STATE["last_session_dir"] = str(session_dir)
        STATE["last_capture_monotonic"] = monotonic()
        logger.info("세션 저장 완료: %s", session_dir)
        start_background_analysis(session_dir)
        return str(session_dir)
    finally:
        relay_off()


def _euclidean(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _eye_aspect_ratio(eye_pts):
    a = _euclidean(eye_pts[1], eye_pts[5])
    b = _euclidean(eye_pts[2], eye_pts[4])
    c = _euclidean(eye_pts[0], eye_pts[3])
    return 1.0 if c == 0 else (a + b) / (2.0 * c)


def update_dynamic_eye_threshold(avg_ear):
    OPEN_EYE_EAR_HISTORY.append(avg_ear)
    if len(OPEN_EYE_EAR_HISTORY) >= max(8, OPEN_EYE_BASELINE_SAMPLES // 2):
        baseline = sum(OPEN_EYE_EAR_HISTORY) / len(OPEN_EYE_EAR_HISTORY)
        dynamic = baseline * EYE_THRESHOLD_RATIO
        dynamic = max(MIN_DYNAMIC_EYE_AR_THRESHOLD, min(MAX_DYNAMIC_EYE_AR_THRESHOLD, dynamic))
        STATE["dynamic_eye_threshold"] = dynamic


def analyze_face(cam2_bgr):
    h, w = cam2_bgr.shape[:2]
    result = {
        "face_found": False,
        "center_ok": False,
        "size_ok": False,
        "eyes_closed": False,
        "roll_ok": False,
        "yaw_ok": False,
        "pitch_ok": False,
        "angles_ok": False,
        "ear_left": 1.0,
        "ear_right": 1.0,
        "avg_ear": 1.0,
        "roll_deg": 0.0,
        "yaw_score": 0.0,
        "pitch_score": 0.0,
        "bbox": None,
        "guide_text": "얼굴을 화면에 맞춰주세요",
    }

    if face_mesh is None:
        result["guide_text"] = "mediapipe 미설치: 자동촬영 사용 불가"
        return result

    rgb = cv2.cvtColor(cam2_bgr, cv2.COLOR_BGR2RGB)
    mesh_result = face_mesh.process(rgb)
    if not mesh_result.multi_face_landmarks:
        result["guide_text"] = "얼굴이 감지되지 않습니다"
        return result

    pts = []
    for lm in mesh_result.multi_face_landmarks[0].landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))

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
    avg_ear = (ear_left + ear_right) / 2.0

    center_ok = (norm_dx <= FACE_CENTER_TOL_X) and (norm_dy <= FACE_CENTER_TOL_Y)
    size_ok = (FACE_MIN_AREA_RATIO <= area_ratio <= FACE_MAX_AREA_RATIO)

    nose = pts[NOSE_TIP_IDX]
    le = pts[LEFT_EYE_OUTER_IDX]
    re = pts[RIGHT_EYE_OUTER_IDX]
    lf = pts[LEFT_FACE_IDX]
    rf = pts[RIGHT_FACE_IDX]
    upper = pts[UPPER_FACE_IDX]
    lower = pts[LOWER_FACE_IDX]
    ml = pts[MOUTH_LEFT_IDX]
    mr = pts[MOUTH_RIGHT_IDX]

    eye_mid = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
    mouth_mid = ((ml[0] + mr[0]) / 2.0, (ml[1] + mr[1]) / 2.0)
    face_mid_x = (lf[0] + rf[0]) / 2.0
    face_width = max(1.0, float(rf[0] - lf[0]))
    face_height = max(1.0, float(lower[1] - upper[1]))

    roll_deg = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    yaw_score = abs(nose[0] - face_mid_x) / face_width
    pitch_score = abs(nose[1] - ((eye_mid[1] + mouth_mid[1]) / 2.0)) / face_height

    roll_ok = abs(roll_deg) <= MAX_ABS_ROLL_DEG
    yaw_ok = yaw_score <= MAX_ABS_YAW_SCORE
    pitch_ok = pitch_score <= MAX_ABS_PITCH_SCORE
    angles_ok = roll_ok and yaw_ok and pitch_ok

    threshold = STATE["dynamic_eye_threshold"]
    eyes_closed = (ear_left < threshold) and (ear_right < threshold)

    result.update({
        "face_found": True,
        "center_ok": center_ok,
        "size_ok": size_ok,
        "eyes_closed": eyes_closed,
        "roll_ok": roll_ok,
        "yaw_ok": yaw_ok,
        "pitch_ok": pitch_ok,
        "angles_ok": angles_ok,
        "ear_left": ear_left,
        "ear_right": ear_right,
        "avg_ear": avg_ear,
        "roll_deg": roll_deg,
        "yaw_score": yaw_score,
        "pitch_score": pitch_score,
        "bbox": (x1, y1, x2, y2),
    })

    if not center_ok:
        result["guide_text"] = "얼굴을 화면 중앙에 맞춰주세요"
    elif not size_ok:
        result["guide_text"] = "얼굴 거리를 조정해주세요"
    elif not angles_ok:
        result["guide_text"] = "얼굴 각도를 정면으로 맞춰주세요"
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
        ok_all = (
            detection["center_ok"]
            and detection["size_ok"]
            and detection["angles_ok"]
            and detection["eyes_closed"]
        )
        color = (0, 255, 0) if ok_all else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    cv2.putText(out, detection["guide_text"], (20, h - 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        out,
        f"L_EAR:{detection['ear_left']:.2f} R_EAR:{detection['ear_right']:.2f} TH:{STATE['dynamic_eye_threshold']:.2f}",
        (20, h - 110),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )
    cv2.putText(
        out,
        f"Roll:{detection['roll_deg']:.1f} Yaw:{detection['yaw_score']:.2f} Pitch:{detection['pitch_score']:.2f}",
        (20, h - 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )
    cv2.putText(
        out,
        f"Stable:{STATE['stable_face_count']}/{STABLE_FACE_HOLD_FRAMES}",
        (20, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )

    if STATE["analysis_in_progress"]:
        cv2.putText(out, "Analysis: running...", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    if STATE["last_analysis_report"] is not None:
        report = STATE["last_analysis_report"]
        cv2.putText(
            out,
            f"P-count:{report['porphyrin_count']} Area:{report['porphyrin_area']} Int:{report['mean_intensity']:.1f}",
            (20, h - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )

    return out


def initialize_camera():
    camera_list = Picamera2.global_camera_info()
    logger.info("Detected cameras: %s", camera_list)
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
    return cam, preview_config, still_config


def initialize_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if ROTATE_DISPLAY:
        cv2.resizeWindow(WINDOW_NAME, PREVIEW_HEIGHT * 3, PREVIEW_WIDTH)
    else:
        cv2.resizeWindow(WINDOW_NAME, PREVIEW_WIDTH * 3, PREVIEW_HEIGHT)

    cv2.createTrackbar("Exposure(ms)", WINDOW_NAME, INITIAL_EXPOSURE_MS, MAX_EXPOSURE_MS, nothing)
    cv2.setTrackbarMin("Exposure(ms)", WINDOW_NAME, MIN_EXPOSURE_MS)


def handle_auto_state(detection):
    now = monotonic()

    if now - STATE["last_capture_monotonic"] < CAPTURE_COOLDOWN_SEC:
        STATE["status"] = "촬영 후 안정화 대기 중"
        STATE["stable_face_count"] = 0
        STATE["eyes_closed_count"] = 0
        return False

    stable_face_ok = (
        detection["face_found"]
        and detection["center_ok"]
        and detection["size_ok"]
        and detection["angles_ok"]
    )

    if stable_face_ok:
        STATE["stable_face_count"] += 1
    else:
        STATE["stable_face_count"] = 0
        STATE["eyes_closed_count"] = 0
        STATE["status"] = detection["guide_text"]
        return False

    if STATE["stable_face_count"] >= STABLE_FACE_HOLD_FRAMES and not detection["eyes_closed"]:
        update_dynamic_eye_threshold(detection["avg_ear"])
        STATE["status"] = f"눈을 감아주세요 (기준 EAR 수집 {len(OPEN_EYE_EAR_HISTORY)}/{OPEN_EYE_BASELINE_SAMPLES})"
        STATE["eyes_closed_count"] = 0
        return False

    if STATE["stable_face_count"] < STABLE_FACE_HOLD_FRAMES:
        STATE["status"] = f"얼굴 고정 중 {STATE['stable_face_count']}/{STABLE_FACE_HOLD_FRAMES}"
        STATE["eyes_closed_count"] = 0
        return False

    if detection["eyes_closed"]:
        STATE["eyes_closed_count"] += 1
        STATE["status"] = f"눈감음 확인 중 {STATE['eyes_closed_count']}/{EYES_CLOSED_HOLD_FRAMES}"
    else:
        STATE["eyes_closed_count"] = 0
        STATE["status"] = "얼굴을 유지한 채 눈을 감아주세요"

    return STATE["eyes_closed_count"] >= EYES_CLOSED_HOLD_FRAMES


def run():
    cam, preview_config, still_config = initialize_camera()
    initialize_window()
    prev_exposure_ms = INITIAL_EXPOSURE_MS

    try:
        while True:
            exposure_ms = get_current_exposure_ms()
            if exposure_ms != prev_exposure_ms:
                set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
                prev_exposure_ms = exposure_ms

            frame = safe_capture_array(cam)
            if frame is None:
                STATE["status"] = "프레임 획득 실패"
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            full_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            cam2 = extract_cam_frame(full_frame_bgr, "cam2")
            cam3 = extract_cam_frame(full_frame_bgr, "cam3")
            cam4 = extract_cam_frame(full_frame_bgr, "cam4")

            detection = analyze_face(cam2)
            cam2 = draw_face_guide(cam2, detection)

            if STATE["armed"]:
                white_led_on()
                should_capture = handle_auto_state(detection)

                if should_capture:
                    STATE["status"] = "조건 충족: 백색 LED OFF 후 촬영"
                    white_led_off()
                    sleep(WHITE_LED_OFF_BEFORE_CAPTURE_SEC)

                    exposure_ms = get_current_exposure_ms()
                    set_manual_controls(cam, exposure_ms, CURRENT_GAIN)

                    detection_snapshot = {
                        "bbox": list(detection["bbox"]) if detection["bbox"] is not None else None,
                        "roll_deg": detection["roll_deg"],
                        "yaw_score": detection["yaw_score"],
                        "pitch_score": detection["pitch_score"],
                        "dynamic_eye_threshold": STATE["dynamic_eye_threshold"],
                    }

                    capture_sequence(
                        cam=cam,
                        preview_config=preview_config,
                        still_config=still_config,
                        exposure_ms=exposure_ms,
                        gain=CURRENT_GAIN,
                        detection_snapshot=detection_snapshot,
                        save_as_png=SAVE_AS_PNG,
                    )
                    exposure_ms = get_current_exposure_ms()
                    set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
                    prev_exposure_ms = exposure_ms

                    STATE["armed"] = False
                    STATE["eyes_closed_count"] = 0
                    STATE["stable_face_count"] = 0
                    STATE["status"] = "촬영 완료 (분석 시작)"
            else:
                white_led_off()
                STATE["stable_face_count"] = 0
                STATE["eyes_closed_count"] = 0

            cam2_view = cv2.resize(cam2, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
            cam3_view = cv2.resize(cam3, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
            cam4_view = cv2.resize(cam4, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)

            cam2_view = rotate_for_display(cam2_view)
            cam3_view = rotate_for_display(cam3_view)
            cam4_view = rotate_for_display(cam4_view)

            cam2_view = draw_text(cam2_view, "CAM 2 - NO FILTER", exposure_ms, extra_text="c: 자동정렬촬영 / p: 즉시촬영 / z: EAR 재보정")
            cam3_view = draw_text(cam3_view, "CAM 3 - 405nm FILTER", exposure_ms, extra_text=f"Auto: {STATE['status']}")
            cam4_view = draw_text(cam4_view, "CAM 4 - 660nm FILTER", exposure_ms, extra_text="a: 마지막 세션 재분석 / x: 취소 / q: 종료")

            preview = cv2.hconcat([cam2_view, cam3_view, cam4_view])
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                if face_mesh is None:
                    STATE["status"] = "mediapipe 미설치: 자동촬영 불가"
                else:
                    STATE["armed"] = True
                    STATE["eyes_closed_count"] = 0
                    STATE["stable_face_count"] = 0
                    STATE["status"] = "얼굴 위치/각도를 맞추고 눈을 감아주세요"
                    white_led_on()

            if key == ord('x'):
                STATE["armed"] = False
                STATE["eyes_closed_count"] = 0
                STATE["stable_face_count"] = 0
                STATE["status"] = "자동촬영 취소"
                white_led_off()

            if key == ord('z'):
                OPEN_EYE_EAR_HISTORY.clear()
                STATE["dynamic_eye_threshold"] = DEFAULT_EYE_AR_THRESHOLD
                STATE["status"] = "EAR 기준값 초기화 완료"

            if key == ord('a'):
                if STATE["last_session_dir"] is not None and not STATE["analysis_in_progress"]:
                    start_background_analysis(STATE["last_session_dir"])
                    STATE["status"] = "재분석 시작"
                elif STATE["analysis_in_progress"]:
                    STATE["status"] = "분석이 이미 진행 중입니다"
                else:
                    STATE["status"] = "재분석할 세션이 없습니다"

            if key == ord('p'):
                STATE["armed"] = False
                STATE["eyes_closed_count"] = 0
                STATE["stable_face_count"] = 0
                STATE["status"] = "즉시 촬영"
                white_led_off()
                exposure_ms = get_current_exposure_ms()
                set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
                capture_sequence(
                    cam=cam,
                    preview_config=preview_config,
                    still_config=still_config,
                    exposure_ms=exposure_ms,
                    gain=CURRENT_GAIN,
                    detection_snapshot=None,
                    save_as_png=SAVE_AS_PNG,
                )
                exposure_ms = get_current_exposure_ms()
                set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
                prev_exposure_ms = exposure_ms
                STATE["status"] = "즉시 촬영 완료 (분석 시작)"

            if key == ord('o'):
                relay_on()
            if key == ord('f'):
                relay_off()
            if key == ord('q'):
                break

    except Exception as e:
        logger.exception("실행 중 오류 발생: %s", e)
        raise
    finally:
        white_led_off()
        relay_off()
        cam.stop()
        cv2.destroyAllWindows()
        if face_mesh is not None:
            face_mesh.close()


if __name__ == "__main__":
    run()
