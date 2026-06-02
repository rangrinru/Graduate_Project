# Flask 서버와 JSON 응답을 위한 모듈 import
from flask import Flask, jsonify, Response, request, send_file

# CORS 허용을 위한 모듈 import
from flask_cors import CORS

# 이미지 저장/변환을 위한 OpenCV import
import cv2

from porphyrin_analysis import analyze_porphyrin_heatmap_v04
from trouble_risk_analysis import analyze_trouble_risk_map
from config import *
from state import *
from led_controller import (
    get_white_led_status,
    relay_off,
    relay_on,
    white_led_off,
    white_led_on,
)
from profile_service import (
    create_profile,
    delete_profile,
    find_profile_by_id,
    load_profiles,
)
from capture_service import extract_cam_frame, save_one_camera_image, save_white_cam4_reference_image
from history_service import (
    delete_capture_history,
    get_capture_detail,
    get_capture_history,
    get_profile_root,
    get_porphyrin_analysis_report,
    resolve_analysis_image_path,
    resolve_image_path,
    resolve_white_cam4_reference_path,
)
from auto_capture_service import (
    calculate_eye_aspect_ratio_from_points,
    clamp,
)
from picamera2 import Picamera2

# 카메라 동기화용 스레드 락 import
import threading

# 날짜/시간 처리용 import
from datetime import datetime

# 대기 시간과 경과 시간 측정용 import
from time import sleep, monotonic

# 자동 얼굴 촬영 조건 확인은 rpicam_03_eye_closed_auto.py와 같은 MediaPipe FaceMesh EAR 방식으로 수행합니다.
# 눈 감음이 2초 유지되면 기존 촬영 저장 흐름을 호출합니다.
try:
    # MediaPipe 기본 모듈 import
    import mediapipe as mp

    # MediaPipe import 성공 표시
    MEDIAPIPE_IMPORT_ERROR = None

except Exception as e:
    # MediaPipe import 실패 시 서버는 켜지되 자동 촬영만 사용할 수 없게 처리
    mp = None
    MEDIAPIPE_IMPORT_ERROR = e


# Flask 앱 생성
app = Flask(__name__)

# 다른 포트의 프론트엔드에서도 접근 가능하도록 CORS 허용
CORS(app)


# =========================
# MediaPipe FaceMesh 설정
# =========================

# rpicam_03_eye_closed_auto.py와 같은 solutions API 객체
face_detector = None
face_mesh = None

# MediaPipe 얼굴/눈 인식 준비 여부
mediapipe_face_ready = False

# MediaPipe 얼굴/눈 인식 오류 메시지
mediapipe_face_error = None


def init_mediapipe_face_mesh():
    # 전역 MediaPipe 상태값 사용 선언
    global face_detector, face_mesh, mediapipe_face_ready, mediapipe_face_error

    # MediaPipe import 자체가 실패한 경우
    if MEDIAPIPE_IMPORT_ERROR is not None:
        mediapipe_face_ready = False
        mediapipe_face_error = f"MediaPipe import 실패: {MEDIAPIPE_IMPORT_ERROR}"
        return

    try:
        # face_ROI_eye_added.py / rpicam_03_eye_closed_auto.py와 같은 방식으로 초기화
        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 준비 완료 표시
        mediapipe_face_ready = True
        mediapipe_face_error = None

    except Exception as e:
        # 생성 실패 시 오류 저장
        face_detector = None
        face_mesh = None
        mediapipe_face_ready = False
        mediapipe_face_error = f"MediaPipe FaceMesh 초기화 실패: {e}"


# 서버 시작 시 MediaPipe FaceMesh 초기화
init_mediapipe_face_mesh()


# =========================
# 카메라 수동 제어 함수
# =========================

def set_manual_controls(camera, exposure_ms, gain):
    if exposure_ms is None:
        exposure_ms = PREVIEW_EXPOSURE_MS
    if gain is None:
        gain = CURRENT_GAIN

    camera.set_controls({
        "AeEnable": False,
        "ExposureTime": int(round(float(exposure_ms) * 1000)),
        "AnalogueGain": float(gain),
    })


def picamera_frame_to_bgr(frame, source_format):
    if frame is None:
        raise RuntimeError("Picamera2 프레임을 받지 못했습니다.")

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    channels = frame.shape[2] if frame.ndim == 3 else 0
    if channels == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if channels == 3 and source_format == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if channels == 3:
        return frame.copy()

    raise RuntimeError(f"지원하지 않는 Picamera2 프레임 형식입니다: shape={frame.shape}")


def read_picamera_full_frame_bgr():
    if picam2 is None:
        raise RuntimeError("Picamera2가 초기화되지 않았습니다.")

    frame = picam2.capture_array()
    return picamera_frame_to_bgr(frame, source_format="xrgb")


def read_picamera_cam_frame_bgr(cam_key):
    full_frame_bgr = read_picamera_full_frame_bgr()
    return extract_cam_frame(full_frame_bgr, cam_key).copy()


# =========================
# 카메라 초기화
# =========================

def init_camera():
    # 전역 변수 사용 선언
    global picam2, preview_config, still_config, camera_ready

    camera_list = Picamera2.global_camera_info()
    print("Detected cameras:", camera_list)

    if len(camera_list) == 0:
        raise RuntimeError("Picamera2 카메라가 감지되지 않았습니다.")

    picam2 = Picamera2(0)
    preview_config = picam2.create_preview_configuration(
        main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "XRGB8888"}
    )
    still_config = picam2.create_still_configuration(
        main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
    )

    picam2.configure(preview_config)
    picam2.options["quality"] = 100
    picam2.options["compress_level"] = 0
    picam2.start()
    sleep(PICAMERA_PREVIEW_WARMUP_SEC)
    set_manual_controls(picam2, PREVIEW_EXPOSURE_MS, CURRENT_GAIN)
    sleep(PICAMERA_CONTROL_SETTLE_SEC)

    # 테스트 프레임 1장을 읽어서 카메라 상태 확인
    test_frame = read_preview_frame()

    # 프레임 크기 검증
    if test_frame.shape[:2] != (CAPTURE_HEIGHT, CAPTURE_WIDTH):
        raise RuntimeError(f"카메라 프레임 크기가 올바르지 않습니다: {test_frame.shape}")

    # 카메라 준비 완료 표시
    camera_ready = True

    # 상태 출력
    print(f"[Picamera2] camera ready: frame={test_frame.shape}")


# =========================
# 프리뷰 프레임 읽기
# =========================

def read_preview_frame():
    return read_picamera_full_frame_bgr()


# =========================
# 고화질 전체 프레임 촬영
# =========================

def capture_high_quality_full_frame(exposure_ms, gain):
    frames_by_exposure = capture_high_quality_full_frames_by_exposure([exposure_ms], gain)
    return frames_by_exposure[exposure_ms]


def capture_picamera_still_frames_by_exposure(exposure_values_ms, gain):
    if picam2 is None or preview_config is None or still_config is None:
        raise RuntimeError("Picamera2가 초기화되지 않았습니다.")

    frames_by_exposure = {}

    picam2.stop()
    picam2.configure(still_config)
    picam2.start()

    try:
        for exposure_ms in exposure_values_ms:
            set_manual_controls(picam2, exposure_ms, gain)

            # Picamera2는 제어값 변경 직후 이전 노출 프레임이 큐에 남을 수 있어
            # 저장할 프레임을 받기 전에 몇 장을 버려 실제 노출값이 반영되게 합니다.
            settle_sec = max(
                PICAMERA_CONTROL_SETTLE_SEC,
                (float(exposure_ms) / 1000.0) * PICAMERA_STILL_DISCARD_FRAMES,
            )
            sleep(settle_sec)

            for _ in range(PICAMERA_STILL_DISCARD_FRAMES):
                picam2.capture_array()

            still_frame = picam2.capture_array()
            frames_by_exposure[exposure_ms] = picamera_frame_to_bgr(
                still_frame,
                source_format="rgb",
            )

    finally:
        picam2.stop()
        picam2.configure(preview_config)
        picam2.start()
        set_manual_controls(picam2, PREVIEW_EXPOSURE_MS, CURRENT_GAIN)
        sleep(PICAMERA_CONTROL_SETTLE_SEC)

    return frames_by_exposure


def capture_high_quality_full_frames_by_exposure(exposure_values_ms, gain):
    relay_on()
    sleep(RELAY_WARMUP_SEC)

    frames_by_exposure = {}

    try:
        with camera_lock:
            frames_by_exposure = capture_picamera_still_frames_by_exposure(
                exposure_values_ms=exposure_values_ms,
                gain=gain,
            )

    finally:
        relay_off()

    return frames_by_exposure


# =========================
# 자동 얼굴 촬영 유틸 함수 - rpicam_03_eye_closed_auto.py 방식
# =========================

# MediaPipe FaceMesh 눈 EAR 인덱스
LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]

# 얼굴 ROI 표시용 확장 비율
AUTO_ROI_EXPAND_X = 0.12
AUTO_ROI_EXPAND_Y = 0.18

# 너무 짧은 깜빡임을 blink로 세기 위한 최소 프레임
MIN_CLOSED_FRAMES_FOR_BLINK = 2



def expand_auto_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    bw = x2 - x1
    bh = y2 - y1
    ex = int(bw * AUTO_ROI_EXPAND_X)
    ey = int(bh * AUTO_ROI_EXPAND_Y)
    return (
        clamp(x1 - ex, 0, w - 1),
        clamp(y1 - ey, 0, h - 1),
        clamp(x2 + ex, 0, w - 1),
        clamp(y2 + ey, 0, h - 1),
    )


def resize_for_auto_detection(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if w <= AUTO_DETECTION_PROCESS_WIDTH:
        return frame_bgr, 1.0, 1.0

    scale = AUTO_DETECTION_PROCESS_WIDTH / float(w)
    resized = cv2.resize(
        frame_bgr,
        (AUTO_DETECTION_PROCESS_WIDTH, int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )
    scale_x = w / float(resized.shape[1])
    scale_y = h / float(resized.shape[0])
    return resized, scale_x, scale_y


def scale_bbox_to_original(bbox, scale_x, scale_y):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    )


def detect_face_bbox_for_auto(frame_bgr):
    if face_detector is None:
        return None

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

    return expand_auto_bbox(*best_bbox, w, h)


def extract_face_landmarks_for_auto(frame_bgr):
    if face_mesh is None:
        return None

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    pts = []
    for lm in result.multi_face_landmarks[0].landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))
    return pts


def extract_face_landmarks_for_analysis(frame_bgr):
    if face_mesh is None:
        return None

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    pts = []
    for lm in result.multi_face_landmarks[0].landmark:
        pts.append((
            clamp(int(lm.x * w), 0, w - 1),
            clamp(int(lm.y * h), 0, h - 1),
        ))
    return pts


def bbox_from_landmark_points(pts, width, height):
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return expand_auto_bbox(
        clamp(min(xs), 0, width - 1),
        clamp(min(ys), 0, height - 1),
        clamp(max(xs), 0, width - 1),
        clamp(max(ys), 0, height - 1),
        width,
        height,
    )



def estimate_eye_state_and_motion(pts):
    left_eye_pts = [pts[i] for i in LEFT_EYE_EAR_INDICES]
    right_eye_pts = [pts[i] for i in RIGHT_EYE_EAR_INDICES]

    ear_left = calculate_eye_aspect_ratio_from_points(left_eye_pts)
    ear_right = calculate_eye_aspect_ratio_from_points(right_eye_pts)
    ear_avg = (ear_left + ear_right) / 2.0

    eyes_closed = ear_avg < DEFAULT_EYE_AR_THRESHOLD
    eye_state = "Closed" if eyes_closed else "Open"
    gaze_direction = "Eyes Closed" if eyes_closed else "Unknown"

    AUTO_EYE_STATE["eye_state"] = eye_state
    AUTO_EYE_STATE["gaze_direction"] = gaze_direction

    return {
        "ear_left": float(ear_left),
        "ear_right": float(ear_right),
        "avg_ear": float(ear_avg),
        "eyes_closed": eyes_closed,
        "eye_state": eye_state,
        "gaze_direction": gaze_direction,
    }


def update_dynamic_eye_threshold(_avg_ear):
    # rpicam_03_eye_closed_auto.py는 고정 EAR 기준을 사용합니다.
    with auto_state_lock:
        AUTO_STATE["dynamic_eye_threshold"] = DEFAULT_EYE_AR_THRESHOLD


def analyze_face_for_auto(cam2_bgr):
    process_frame, scale_x, scale_y = resize_for_auto_detection(cam2_bgr)
    h, w = process_frame.shape[:2]

    result = {
        "face_found": False,
        "center_ok": False,
        "size_ok": False,
        "eyes_closed": False,
        "roll_ok": True,
        "yaw_ok": True,
        "pitch_ok": True,
        "angles_ok": False,
        "eyes_visible": False,
        "eye_count": 0,
        "ear_left": 1.0,
        "ear_right": 1.0,
        "avg_ear": 1.0,
        "roll_deg": 0.0,
        "yaw_score": 0.0,
        "pitch_score": 0.0,
        "bbox": None,
        "guide_text": "얼굴을 화면에 맞춰주세요",
    }

    if not mediapipe_face_ready or face_mesh is None:
        result["guide_text"] = mediapipe_face_error or "MediaPipe FaceMesh를 사용할 수 없습니다"
        return result

    try:
        face_bbox = detect_face_bbox_for_auto(process_frame)
        pts = extract_face_landmarks_for_auto(process_frame)
    except Exception as e:
        result["guide_text"] = f"MediaPipe 얼굴/눈 분석 오류: {e}"
        return result

    if pts is None:
        if face_bbox is not None:
            result["face_found"] = True
            result["bbox"] = scale_bbox_to_original(face_bbox, scale_x, scale_y)
            result["guide_text"] = "얼굴 랜드마크를 찾는 중입니다"
        else:
            result["guide_text"] = "얼굴이 감지되지 않습니다"
        return result

    if face_bbox is None:
        face_bbox = bbox_from_landmark_points(pts, w, h)

    eye_result = estimate_eye_state_and_motion(pts)
    face_bbox_original = scale_bbox_to_original(face_bbox, scale_x, scale_y)

    result.update({
        "face_found": True,
        "center_ok": True,
        "size_ok": True,
        "eyes_closed": bool(eye_result["eyes_closed"]),
        "roll_ok": True,
        "yaw_ok": True,
        "pitch_ok": True,
        "angles_ok": True,
        "eyes_visible": True,
        "eye_count": 2,
        "ear_left": eye_result["ear_left"],
        "ear_right": eye_result["ear_right"],
        "avg_ear": eye_result["avg_ear"],
        "bbox": face_bbox_original,
    })

    if eye_result["eyes_closed"]:
        result["guide_text"] = "눈 감음 감지됨"
    else:
        result["guide_text"] = "눈을 감으면 2초 후 자동 촬영합니다"

    return result


def build_auto_checks(detection):
    # 자동 촬영 체크 결과 구성
    face_found = bool(detection.get("face_found"))
    return {
        "face_found": face_found,
        "center_ok": face_found,
        "size_ok": face_found,
        "angle_ok": face_found,
        "eyes_closed": bool(detection.get("eyes_closed")),
        "stable_ok": False,
    }


def update_auto_state_from_detection(detection):
    # 현재 검사 결과 생성
    checks = build_auto_checks(detection)

    now = monotonic()

    # 자동 촬영 상태 갱신을 위해 락 획득
    with auto_state_lock:
        # 얼굴이 없으면 눈 감음 타이머 초기화
        if not checks["face_found"]:
            AUTO_STATE["status"] = "얼굴이 감지되지 않습니다"
            AUTO_STATE["stable_face_count"] = 0
            AUTO_STATE["eyes_closed_count"] = 0
            AUTO_EYE_STATE["eyes_closed_started_at"] = None
            AUTO_EYE_STATE["closed_frame_count"] = 0
            AUTO_EYE_STATE["prev_eye_closed"] = False

        # 눈을 뜨면 2초 타이머를 다시 시작하도록 초기화
        elif not checks["eyes_closed"]:
            AUTO_STATE["eyes_closed_count"] = 0
            AUTO_STATE["stable_face_count"] = 1
            AUTO_STATE["status"] = "눈을 감으면 2초 후 자동 촬영합니다"
            AUTO_EYE_STATE["eyes_closed_started_at"] = None

            if (
                AUTO_EYE_STATE["prev_eye_closed"]
                and AUTO_EYE_STATE["closed_frame_count"] >= MIN_CLOSED_FRAMES_FOR_BLINK
            ):
                AUTO_EYE_STATE["blink_count"] += 1

            AUTO_EYE_STATE["closed_frame_count"] = 0
            AUTO_EYE_STATE["prev_eye_closed"] = False

        # 눈 감음이 유지되면 초 단위로 2초를 기다림
        else:
            AUTO_EYE_STATE["closed_frame_count"] += 1
            AUTO_EYE_STATE["prev_eye_closed"] = True

            if AUTO_EYE_STATE["eyes_closed_started_at"] is None:
                AUTO_EYE_STATE["eyes_closed_started_at"] = now

            elapsed = now - AUTO_EYE_STATE["eyes_closed_started_at"]
            remaining = max(0.0, EYES_CLOSED_DELAY_SEC - elapsed)
            checks["stable_ok"] = elapsed >= EYES_CLOSED_DELAY_SEC
            AUTO_STATE["stable_face_count"] = int(min(elapsed, EYES_CLOSED_DELAY_SEC) * 10)
            AUTO_STATE["eyes_closed_count"] = int(elapsed * 10)

            if checks["stable_ok"]:
                AUTO_STATE["status"] = "조건 충족: 자동 촬영 준비"
            else:
                AUTO_STATE["status"] = f"눈감음 유지 중 {elapsed:.1f}/{EYES_CLOSED_DELAY_SEC:.1f}초, {remaining:.1f}초 남음"

        # 검사 결과 저장
        AUTO_STATE["checks"] = checks

        # 최근 갱신 시각 저장
        AUTO_STATE["last_update"] = datetime.now().isoformat()

        # 2초 눈 감음 유지 여부 반환
        return checks["stable_ok"]


def snapshot_detection_for_metadata(detection):
    # 메타데이터 저장용 감지 결과 생성
    return {
        "bbox": list(detection["bbox"]) if detection.get("bbox") is not None else None,
        "roll_deg": float(detection.get("roll_deg", 0.0)),
        "yaw_score": float(detection.get("yaw_score", 0.0)),
        "pitch_score": float(detection.get("pitch_score", 0.0)),
        "mediapipe_eye_count": int(detection.get("eye_count", 0)),
        "mediapipe_eyes_visible": bool(detection.get("eyes_visible", False)),
        "ear_left": float(detection.get("ear_left", 1.0)),
        "ear_right": float(detection.get("ear_right", 1.0)),
        "avg_ear": float(detection.get("avg_ear", 1.0)),
        "dynamic_eye_threshold": float(AUTO_STATE.get("dynamic_eye_threshold", DEFAULT_EYE_AR_THRESHOLD)),
    }


def build_auto_trigger_metadata(detection):
    started_at = AUTO_EYE_STATE.get("eyes_closed_started_at")
    held_sec = 0.0 if started_at is None else monotonic() - started_at
    metadata = snapshot_detection_for_metadata(detection)
    metadata.update({
        "type": "eyes_closed_for_2_seconds",
        "detection_camera": AUTO_DETECTION_CAM_KEY,
        "threshold_ear": DEFAULT_EYE_AR_THRESHOLD,
        "held_sec": float(held_sec),
        "eye_state": AUTO_EYE_STATE["eye_state"],
        "gaze_direction": AUTO_EYE_STATE["gaze_direction"],
        "blink_count": AUTO_EYE_STATE["blink_count"],
    })
    return metadata


def perform_capture_for_profile(profile_id: str, trigger_metadata=None):
    # profileId가 비어 있으면 예외 발생
    if not str(profile_id).strip():
        raise ValueError("profileId가 필요합니다.")

    # 프로필 정보 조회
    profile = find_profile_by_id(profile_id)

    # 프로필이 없으면 예외 발생
    if profile is None:
        raise ValueError("존재하지 않는 프로필입니다.")

    # 표시용 프로필 이름 읽기
    profile_name = profile["name"]

    # 폴더용 프로필 ID 읽기
    folder_id = profile["folderId"]

    # 프로필 루트 경로 생성
    profile_root = SAVE_ROOT / folder_id

    # 프로필 폴더가 없으면 예외 발생
    if not profile_root.exists():
        raise ValueError("프로필 폴더가 존재하지 않습니다.")

    # 촬영 gain 설정
    gain = CURRENT_GAIN

    # 저장 확장자 결정
    ext = "png" if SAVE_AS_PNG else "jpg"

    capture_exposure_plan = {
        cam_key: CAPTURE_EXPOSURE_MS_BY_CAMERA.get(cam_key, INITIAL_EXPOSURE_MS)
        for cam_key in ["cam2", "cam3", "cam4"]
    }
    exposure_values_ms = list(dict.fromkeys(capture_exposure_plan.values()))

    # Manual capture can be triggered while the white LED is already on.
    # Keep the UV/fluorescence capture clean, then turn white LED back on only for cam4_white.
    white_led_off()
    sleep(WHITE_LED_OFF_BEFORE_CAPTURE_SEC)

    # 네 카메라 영역이 하나의 전체 프레임에 들어오므로 노출별 전체 프레임을 찍은 뒤 필요한 영역만 저장합니다.
    frames_by_exposure = capture_high_quality_full_frames_by_exposure(
        exposure_values_ms=exposure_values_ms,
        gain=gain
    )

    # 촬영 시각 기록
    capture_timestamp = datetime.now()

    # 캡처 ID 생성
    capture_id = capture_timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]

    # 저장 파일 목록 초기화
    saved_files = []

    # cam2, cam3, cam4 순서대로 저장
    for cam_key in ["cam2", "cam3", "cam4"]:
        exposure_ms = capture_exposure_plan[cam_key]
        full_frame_bgr = frames_by_exposure[exposure_ms]

        # 개별 카메라 영역 추출
        target_frame = extract_cam_frame(full_frame_bgr, cam_key).copy()

        # 개별 이미지 저장
        result = save_one_camera_image(
            cam_key=cam_key,
            frame_bgr=target_frame,
            profile_root=profile_root,
            capture_id=capture_id,
            timestamp=capture_timestamp,
            exposure_ms=exposure_ms,
            gain=gain,
            ext=ext,
            profile_name=profile_name,
            folder_id=folder_id,
            trigger_metadata=trigger_metadata,
        )

        # 저장 파일 목록에 추가
        saved_files.append(result)

    try:
        white_led_on()
        sleep(WHITE_LED_WARMUP_SEC)

        with camera_lock:
            white_frames_by_exposure = capture_picamera_still_frames_by_exposure(
                exposure_values_ms=[WHITE_REFERENCE_EXPOSURE_MS],
                gain=gain,
            )

        white_full_frame_bgr = white_frames_by_exposure[WHITE_REFERENCE_EXPOSURE_MS]
        white_cam4_frame = extract_cam_frame(white_full_frame_bgr, "cam4").copy()
        white_capture_timestamp = datetime.now()

        white_result = save_white_cam4_reference_image(
            frame_bgr=white_cam4_frame,
            profile_root=profile_root,
            capture_id=capture_id,
            timestamp=white_capture_timestamp,
            exposure_ms=WHITE_REFERENCE_EXPOSURE_MS,
            gain=gain,
            ext=ext,
            profile_name=profile_name,
            folder_id=folder_id,
            trigger_metadata=trigger_metadata,
        )
        saved_files.append(white_result)

    finally:
        white_led_off()

    # 촬영 결과 반환
    return {
        "ok": True,
        "captured_at": capture_timestamp.isoformat(),
        "profile_name": profile_name,
        "profile_id": folder_id,
        "capture_id": capture_id,
        "exposure_plan_ms": capture_exposure_plan,
        "files": saved_files
    }


def get_auto_state_copy():
    # 자동 촬영 상태를 안전하게 복사하기 위해 락 획득
    with auto_state_lock:
        # 응답용 상태 복사본 반환
        return {
            "running": AUTO_STATE["running"],
            "captured": AUTO_STATE["captured"],
            "profile_id": AUTO_STATE["profile_id"],
            "capture_id": AUTO_STATE["capture_id"],
            "status": AUTO_STATE["status"],
            "error": AUTO_STATE["error"],
            "checks": dict(AUTO_STATE["checks"]),
            "stable_face_count": AUTO_STATE["stable_face_count"],
            "eyes_closed_count": AUTO_STATE["eyes_closed_count"],
            "dynamic_eye_threshold": AUTO_STATE["dynamic_eye_threshold"],
            "white_led_is_on": get_white_led_status(),
            "last_update": AUTO_STATE["last_update"],
        }


def reset_auto_state(profile_id=None, running=False):
    # 자동 촬영 상태 초기화를 위해 락 획득
    with auto_state_lock:
        # 실행 여부 설정
        AUTO_STATE["running"] = running

        # 촬영 완료 여부 초기화
        AUTO_STATE["captured"] = False

        # 프로필 ID 저장
        AUTO_STATE["profile_id"] = profile_id

        # 촬영 ID 초기화
        AUTO_STATE["capture_id"] = None

        # 오류 메시지 초기화
        AUTO_STATE["error"] = None

        # 검사 상태 초기화
        AUTO_STATE["checks"] = make_default_auto_checks()

        # 얼굴 안정 카운트 초기화
        AUTO_STATE["stable_face_count"] = 0

        # 눈 감음 카운트 초기화
        AUTO_STATE["eyes_closed_count"] = 0

        # MediaPipe EAR 기준값 초기화
        AUTO_STATE["dynamic_eye_threshold"] = DEFAULT_EYE_AR_THRESHOLD
        AUTO_EYE_STATE["blink_count"] = 0
        AUTO_EYE_STATE["closed_frame_count"] = 0
        AUTO_EYE_STATE["prev_eye_closed"] = False
        AUTO_EYE_STATE["eyes_closed_started_at"] = None
        AUTO_EYE_STATE["eye_state"] = "Unknown"
        AUTO_EYE_STATE["gaze_direction"] = "Unknown"

        # 상태 문구 초기화
        AUTO_STATE["status"] = "눈을 감으면 2초 후 자동 촬영합니다" if running else "자동 촬영 대기 중"

        # 최근 갱신 시각 저장
        AUTO_STATE["last_update"] = datetime.now().isoformat()


def auto_capture_worker(profile_id: str):
    # 자동 촬영 백그라운드 작업 실행
    try:
        # 자동 촬영 루프 시작
        while True:
            # 현재 실행 여부 확인
            with auto_state_lock:
                running = AUTO_STATE["running"]

            # 실행 중이 아니면 종료
            if not running:
                break

            # 카메라가 준비되지 않았으면 상태 표시 후 대기
            if not camera_ready:
                with auto_state_lock:
                    AUTO_STATE["status"] = "카메라가 아직 준비되지 않았습니다"
                    AUTO_STATE["checks"] = make_default_auto_checks()
                sleep(0.3)
                continue

            # 카메라 락을 잡고 자동감지에 필요한 cam2 영역만 읽기
            with camera_lock:
                cam2_bgr = read_picamera_cam_frame_bgr(AUTO_DETECTION_CAM_KEY)

            # 얼굴 상태 분석
            detection = analyze_face_for_auto(cam2_bgr)

            # rpicam_03_eye_closed_auto.py의 고정 EAR 기준값을 응답 상태에도 반영
            update_dynamic_eye_threshold(detection.get("avg_ear", 1.0))

            # 자동 촬영 조건 갱신 후 촬영 여부 판단
            should_capture = update_auto_state_from_detection(detection)

            # 조건이 충족되면 촬영 실행
            if should_capture:
                # 촬영 직전 백색 LED 자동 OFF
                white_led_off()

                # 백색 LED가 꺼질 시간을 잠시 확보
                sleep(WHITE_LED_OFF_BEFORE_CAPTURE_SEC)

                # 촬영 중 상태 표시
                with auto_state_lock:
                    AUTO_STATE["status"] = "조건 충족: 자동 촬영 중"

                # 기존 프로필 저장 구조에 맞춰 촬영 실행
                capture_result = perform_capture_for_profile(
                    profile_id,
                    trigger_metadata=build_auto_trigger_metadata(detection)
                )

                # 촬영 완료 상태 저장
                with auto_state_lock:
                    AUTO_STATE["running"] = False
                    AUTO_STATE["captured"] = True
                    AUTO_STATE["capture_id"] = capture_result["capture_id"]
                    AUTO_STATE["status"] = "자동 촬영 완료"
                    AUTO_STATE["checks"] = {
                        "face_found": True,
                        "center_ok": True,
                        "size_ok": True,
                        "angle_ok": True,
                        "eyes_closed": True,
                        "stable_ok": True,
                    }
                    AUTO_STATE["last_update"] = datetime.now().isoformat()
                break

            # 다음 검사 전 짧게 대기
            sleep(AUTO_CAPTURE_INTERVAL_SEC)

    except Exception as e:
        # 오류 발생 시 자동 촬영 상태에 오류 저장
        with auto_state_lock:
            AUTO_STATE["running"] = False
            AUTO_STATE["captured"] = False
            AUTO_STATE["error"] = str(e)
            AUTO_STATE["status"] = f"자동 촬영 오류: {e}"
            AUTO_STATE["last_update"] = datetime.now().isoformat()


# =========================
# 기본 상태 확인 API
# =========================

@app.route("/health")
def health():
    # 서버 상태 반환
    return jsonify({"ok": True, "camera_ready": camera_ready})


# =========================
# 프로필 목록 조회 API
# =========================

@app.route("/profiles", methods=["GET"])
def get_profiles():
    # 전체 프로필 목록 반환
    return jsonify({
        "ok": True,
        "profiles": load_profiles()
    })


# =========================
# 프로필 생성 API
# =========================

@app.route("/profiles", methods=["POST"])
def create_profile_api():
    try:
        # JSON 바디 읽기
        body = request.get_json(silent=True) or {}

        # 이름 추출
        profile_name = body.get("name", "")

        # 프로필 생성
        new_profile = create_profile(profile_name)

        # 성공 응답
        return jsonify({
            "ok": True,
            "profile": new_profile
        })

    except Exception as e:
        # 실패 응답
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400


# =========================
# 프로필 삭제 API
# =========================

@app.route("/profiles/<profile_id>", methods=["DELETE"])
def delete_profile_api(profile_id):
    try:
        # 프로필 삭제
        delete_profile(profile_id)

        # 성공 응답
        return jsonify({"ok": True})

    except Exception as e:
        # 실패 응답
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400


# =========================
# CAM4 스트리밍 생성기
# =========================

def build_cam4_preview_jpeg():
    # 위치 확인 화면은 PREVIEW_CAM_KEY로 지정한 카메라를 사용합니다.
    with camera_lock:
        if not camera_ready:
            return None

        preview_bgr = read_picamera_cam_frame_bgr(PREVIEW_CAM_KEY)

    # 세로 키오스크 화면에 맞게 영상을 서버에서 세로 방향으로 회전
    preview_bgr = cv2.rotate(preview_bgr, cv2.ROTATE_90_CLOCKWISE)

    if PREVIEW_MIRROR_HORIZONTAL:
        preview_bgr = cv2.flip(preview_bgr, 1)

    # 화면 표시용 크기로 리사이즈
    preview_bgr = cv2.resize(
        preview_bgr,
        (STREAM_WIDTH, STREAM_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # 브라우저에 바로 보낼 JPEG로 인코딩
    ok, buffer = cv2.imencode(
        ".jpg",
        preview_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
    )

    if not ok:
        return None

    return buffer.tobytes()


def cam4_jpeg_worker():
    # 백그라운드에서 최신 CAM4 JPEG만 계속 갱신합니다.
    global latest_cam4_jpeg, latest_cam4_jpeg_time, latest_cam4_jpeg_seq

    frame_delay = 1.0 / STREAM_FPS

    while not cam4_jpeg_stop_event.is_set():
        start_time = monotonic()

        try:
            jpg_bytes = build_cam4_preview_jpeg()

            if jpg_bytes is not None:
                with cam4_jpeg_condition:
                    latest_cam4_jpeg = jpg_bytes
                    latest_cam4_jpeg_time = monotonic()
                    latest_cam4_jpeg_seq += 1
                    cam4_jpeg_condition.notify_all()

            elapsed = monotonic() - start_time
            remain = frame_delay - elapsed

            if remain > 0:
                cam4_jpeg_stop_event.wait(remain)

        except Exception as e:
            print("[CAM4 JPEG worker error]", e)
            cam4_jpeg_stop_event.wait(0.1)


def start_cam4_jpeg_worker():
    # CAM4 JPEG 백그라운드 스레드 시작
    global cam4_jpeg_thread

    if cam4_jpeg_thread is not None and cam4_jpeg_thread.is_alive():
        return

    cam4_jpeg_stop_event.clear()
    cam4_jpeg_thread = threading.Thread(target=cam4_jpeg_worker, daemon=True)
    cam4_jpeg_thread.start()


def generate_cam4_stream():
    # 스트리밍 요청은 직접 인코딩하지 않고 백그라운드 스레드의 최신 JPEG를 내보냅니다.
    start_cam4_jpeg_worker()

    last_seq = -1

    while True:
        try:
            with cam4_jpeg_condition:
                has_new_frame = cam4_jpeg_condition.wait_for(
                    lambda: latest_cam4_jpeg is not None and latest_cam4_jpeg_seq != last_seq,
                    timeout=1.0
                )

                if not has_new_frame and latest_cam4_jpeg is None:
                    continue

                jpg_bytes = latest_cam4_jpeg
                last_seq = latest_cam4_jpeg_seq

            if jpg_bytes is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n" +
                jpg_bytes +
                b"\r\n"
            )

        except GeneratorExit:
            break

        except Exception as e:
            print("[stream error]", e)
            sleep(0.1)


def generate_cam4_stream_inline():
    # 목표 프레임 간격 계산
    frame_delay = 1.0 / STREAM_FPS

    # 무한 스트리밍 루프
    while True:
        # 이번 프레임 처리 시작 시간 기록
        start_time = monotonic()

        try:
            # 스트리밍에서는 전체 프레임을 받은 뒤 프리뷰 카메라 영역만 잘라냅니다.
            with camera_lock:
                if not camera_ready:
                    sleep(0.05)
                    continue

                preview_bgr = read_picamera_cam_frame_bgr(PREVIEW_CAM_KEY)

            # 세로 키오스크 화면에 맞게 프리뷰 영상을 서버에서 세로 방향으로 회전
            preview_bgr = cv2.rotate(preview_bgr, cv2.ROTATE_90_CLOCKWISE)

            if PREVIEW_MIRROR_HORIZONTAL:
                preview_bgr = cv2.flip(preview_bgr, 1)

            # 800x1280 원본 세로 프레임에서 720x1152로 약간만 줄여 품질과 부드러움을 균형 있게 맞춤
            preview_bgr = cv2.resize(
                preview_bgr,
                (STREAM_WIDTH, STREAM_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            ok, buffer = cv2.imencode(
                ".jpg",
                preview_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
            )

            if not ok:
                sleep(0.01)
                continue

            jpg_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n" +
                jpg_bytes +
                b"\r\n"
            )

            elapsed = monotonic() - start_time
            remain = frame_delay - elapsed

            if remain > 0:
                sleep(remain)

        except GeneratorExit:
            break

        except Exception as e:
            print("[스트림 오류]", e)
            sleep(0.1)


# =========================
# CAM4 스트리밍 API
# =========================

@app.route("/stream-cam4")
def stream_cam4():
    # MJPEG 스트림 반환
    return Response(
        generate_cam4_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =========================
# 전체 촬영 API
# =========================

@app.route("/capture-all", methods=["POST"])
def capture_all():
    try:
        # JSON 바디 읽기
        body = request.get_json(silent=True) or {}

        # 선택된 profileId 읽기
        profile_id = body.get("profileId", "")

        # 기존 저장 구조에 맞춰 촬영 실행
        result = perform_capture_for_profile(profile_id)

        # 촬영 성공 응답 반환
        return jsonify(result)

    except Exception as e:
        # 서버 콘솔에 오류 출력
        print("[촬영 오류]", e)

        # 실패 응답
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


# =========================
# 백색 LED 제어 API
# =========================

@app.route("/white-led/on", methods=["POST"])
def white_led_on_api():
    try:
        # 백색 LED 켜기
        white_led_on()

        # 성공 응답 반환
        return jsonify({
            "ok": True,
            "white_led_is_on": True
        })

    except Exception as e:
        # 실패 응답 반환
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route("/white-led/off", methods=["POST"])
def white_led_off_api():
    try:
        # 백색 LED 끄기
        white_led_off()

        # 성공 응답 반환
        return jsonify({
            "ok": True,
            "white_led_is_on": False
        })

    except Exception as e:
        # 실패 응답 반환
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route("/white-led/status", methods=["GET"])
def white_led_status_api():
    # 현재 백색 LED 상태 반환
    return jsonify({
        "ok": True,
        "white_led_is_on": get_white_led_status()
    })


# =========================
# 자동 얼굴 촬영 API
# =========================

@app.route("/auto-capture/start", methods=["POST"])
def auto_capture_start_api():
    global auto_capture_thread

    try:
        # JSON 바디 읽기
        body = request.get_json(silent=True) or {}

        # 선택된 profileId 읽기
        profile_id = body.get("profileId", "")

        # profileId가 비어 있으면 에러
        if not str(profile_id).strip():
            return jsonify({
                "ok": False,
                "error": "profileId가 필요합니다."
            }), 400

        # 프로필 존재 여부 확인
        if find_profile_by_id(profile_id) is None:
            return jsonify({
                "ok": False,
                "error": "존재하지 않는 프로필입니다."
            }), 400

        # MediaPipe FaceMesh 준비 여부 확인
        if not mediapipe_face_ready:
            return jsonify({
                "ok": False,
                "error": mediapipe_face_error or "MediaPipe FaceMesh를 사용할 수 없어 자동 촬영을 사용할 수 없습니다."
            }), 400

        # 카메라 준비 여부 확인
        if not camera_ready:
            return jsonify({
                "ok": False,
                "error": "카메라가 아직 준비되지 않았습니다."
            }), 400

        # 이미 자동 촬영 중인지 확인
        with auto_state_lock:
            already_running = AUTO_STATE["running"]

        # 이미 실행 중이면 현재 상태 반환
        if already_running:
            return jsonify({
                "ok": True,
                **get_auto_state_copy()
            })

        # 자동 촬영 안내 단계에서는 얼굴을 맞추기 쉽도록 백색 LED를 켭니다.
        white_led_on()

        # 자동 촬영 상태 초기화
        reset_auto_state(profile_id=profile_id, running=True)

        # 자동 촬영 스레드 생성
        auto_capture_thread = threading.Thread(
            target=auto_capture_worker,
            args=(profile_id,),
            daemon=True
        )

        # 자동 촬영 스레드 시작
        auto_capture_thread.start()

        # 현재 상태 반환
        return jsonify({
            "ok": True,
            **get_auto_state_copy()
        })

    except Exception as e:
        # 실패 응답 반환
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route("/auto-capture/status", methods=["GET"])
def auto_capture_status_api():
    # 현재 자동 촬영 상태 반환
    return jsonify({
        "ok": True,
        **get_auto_state_copy()
    })


@app.route("/auto-capture/cancel", methods=["POST"])
def auto_capture_cancel_api():
    # 자동 촬영 상태를 중지로 변경
    with auto_state_lock:
        AUTO_STATE["running"] = False
        AUTO_STATE["status"] = "자동 촬영 취소"
        AUTO_STATE["last_update"] = datetime.now().isoformat()

    # 취소 후 현재 상태 반환
    return jsonify({
        "ok": True,
        **get_auto_state_copy()
    })


# =========================
# 이전 기록 목록 조회 API
# =========================

@app.route("/profiles/<profile_id>/history", methods=["GET"])
def get_history_api(profile_id):
    try:
        # 기록 목록 조회
        history = get_capture_history(profile_id)

        # 성공 응답
        return jsonify({
            "ok": True,
            "profileId": profile_id,
            "history": history
        })

    except Exception as e:
        # 실패 응답
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400


# =========================
# 특정 기록 상세 조회 API
# =========================

@app.route("/profiles/<profile_id>/history/<capture_id>", methods=["GET"])
def get_history_detail_api(profile_id, capture_id):
    try:
        # 상세 정보 조회
        detail = get_capture_detail(profile_id, capture_id)

        # 성공 응답
        return jsonify({
            "ok": True,
            **detail
        })

    except Exception as e:
        # 실패 응답
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400


# =========================
# 특정 촬영 기록 삭제 API
# =========================

@app.route("/profiles/<profile_id>/history/<capture_id>", methods=["DELETE"])
def delete_history_api(profile_id, capture_id):
    try:
        # 특정 촬영 기록 삭제
        delete_capture_history(profile_id, capture_id)

        # 성공 응답
        return jsonify({
            "ok": True
        })

    except Exception as e:
        # 실패 응답
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400




# =========================
# 특정 촬영 기록 포르피린 분석 API
# =========================

@app.route("/profiles/<profile_id>/history/<capture_id>/analyze-porphyrin", methods=["POST"])
def analyze_porphyrin_api(profile_id, capture_id):
    try:
        # cam4_660nm 이미지 경로 찾기
        image_path = resolve_image_path(
            profile_id=profile_id,
            capture_id=capture_id,
            filter_type="660nm_filter"
        )
        try:
            face_reference_path = resolve_white_cam4_reference_path(profile_id, capture_id)
        except Exception:
            try:
                face_reference_path = resolve_image_path(
                    profile_id=profile_id,
                    capture_id=capture_id,
                    filter_type="no_filter"
                )
            except Exception:
                face_reference_path = None

        try:
            white_reference_path = resolve_white_cam4_reference_path(profile_id, capture_id)
        except Exception:
            white_reference_path = None

        # 프로필 루트 경로 가져오기
        profile_root = get_profile_root(profile_id)

        # 분석 결과 저장 폴더 경로
        analysis_dir = (
            profile_root
            / CAMERA_INFO["cam4"]["folder"]
            / capture_id
            / "analysis"
        )

        # 포르피린 분석 실행
        report = analyze_porphyrin_heatmap_v04(
            image_path,
            analysis_dir,
            face_reference_path,
            extract_face_landmarks_for_analysis,
            white_reference_path
        )

        # 성공 응답 반환
        return jsonify({
            "ok": True,
            "captureId": capture_id,
            "porphyrin_count": report["porphyrin_count"],
            "porphyrin_area": report["porphyrin_area"],
            "detection_rate_percent": report["detection_rate_percent"],
            "porphyrin_mean_brightness": report["porphyrin_mean_brightness"],
            "porphyrin_top5_max_brightness": report["porphyrin_top5_max_brightness"],
            "face_area_pixels": report["face_area_pixels"],
            "grade": report["grade"],
            "skin_score": report["skin_score"],
            "region_analysis": report["region_analysis"],
            "threshold_percentile": report["threshold_percentile"],
            "threshold_value": report["threshold_value"],
            "min_area": report["min_area"],
            "max_area": report["max_area"],
            "face_landmarks_used": report["face_landmarks_used"],
            "heatmap_url": (
                f"/profiles/{profile_id}/history/{capture_id}/analysis/porphyrin-white-overlay"
                if report.get("white_overlay_path")
                else f"/profiles/{profile_id}/history/{capture_id}/analysis/porphyrin-heatmap"
            ),
            "uv_heatmap_url": f"/profiles/{profile_id}/history/{capture_id}/analysis/porphyrin-heatmap",
            "white_overlay_url": (
                f"/profiles/{profile_id}/history/{capture_id}/analysis/porphyrin-white-overlay"
                if report.get("white_overlay_path")
                else None
            ),
        })

    except Exception as e:
        # 실패 응답 반환
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400


# =========================
# 특정 촬영 기록 트러블 위험/집중 케어 분석 API
# =========================

@app.route("/profiles/<profile_id>/history/<capture_id>/analyze-trouble-risk", methods=["POST"])
def analyze_trouble_risk_api(profile_id, capture_id):
    try:
        image_path = resolve_image_path(
            profile_id=profile_id,
            capture_id=capture_id,
            filter_type="660nm_filter"
        )

        try:
            face_reference_path = resolve_image_path(
                profile_id=profile_id,
                capture_id=capture_id,
                filter_type="no_filter"
            )
        except Exception:
            face_reference_path = None

        profile_root = get_profile_root(profile_id)
        analysis_dir = (
            profile_root
            / CAMERA_INFO["cam4"]["folder"]
            / capture_id
            / "analysis"
        )

        report = analyze_trouble_risk_map(
            image_path,
            analysis_dir,
            face_reference_path,
            extract_face_landmarks_for_analysis
        )

        return jsonify({
            "ok": True,
            "captureId": capture_id,
            "risk_area": report["risk_area"],
            "risk_rate_percent": report["risk_rate_percent"],
            "risk_grade": report["risk_grade"],
            "region_analysis": report["region_analysis"],
            "focus_areas": report["focus_areas"],
            "top_region": report["top_region"],
            "threshold_value": report["threshold_value"],
            "face_area_pixels": report["face_area_pixels"],
            "face_landmarks_used": report["face_landmarks_used"],
            "risk_heatmap_url": f"/profiles/{profile_id}/history/{capture_id}/analysis/trouble-risk-heatmap",
            "focus_overlay_url": f"/profiles/{profile_id}/history/{capture_id}/analysis/focus-care-overlay",
            "risk_mask_url": f"/profiles/{profile_id}/history/{capture_id}/analysis/trouble-risk-mask"
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400


# =========================
# 특정 촬영 기록 포르피린 분석 리포트 조회 API
# =========================

@app.route("/profiles/<profile_id>/history/<capture_id>/analysis/porphyrin-report", methods=["GET"])
def get_porphyrin_analysis_report_api(profile_id, capture_id):
    try:
        report = get_porphyrin_analysis_report(profile_id, capture_id)

        return jsonify({
            "ok": True,
            "captureId": capture_id,
            "porphyrin_count": report.get("porphyrin_count", 0),
            "porphyrin_area": report.get("porphyrin_area", 0),
            "detection_rate_percent": report.get("detection_rate_percent", 0),
            "grade": report.get("grade", "-"),
            "skin_score": report.get("skin_score"),
            "region_analysis": report.get("region_analysis", {}),
            "threshold_percentile": report.get("threshold_percentile", 0),
            "threshold_value": report.get("threshold_value", 0),
            "min_area": report.get("min_area", 0),
            "max_area": report.get("max_area", 0),
            "heatmap_url": f"/profiles/{profile_id}/history/{capture_id}/analysis/porphyrin-heatmap",
            "uv_heatmap_url": f"/profiles/{profile_id}/history/{capture_id}/analysis/porphyrin-heatmap",
            "white_overlay_url": (
                f"/profiles/{profile_id}/history/{capture_id}/analysis/porphyrin-white-overlay"
                if report.get("white_overlay_path")
                else None
            ),
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 404


# =========================
# 포르피린 분석 이미지 반환 API
# =========================

@app.route("/profiles/<profile_id>/history/<capture_id>/analysis/<result_type>", methods=["GET"])
def get_porphyrin_analysis_image_api(profile_id, capture_id, result_type):
    try:
        # 분석 이미지 경로 가져오기
        image_path = resolve_analysis_image_path(profile_id, capture_id, result_type)

        # 이미지 파일 반환
        return send_file(image_path)

    except Exception as e:
        # 실패 시 JSON 반환
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 404


# =========================
# 특정 기록 이미지 직접 반환 API
# =========================

@app.route("/profiles/<profile_id>/history/<capture_id>/image/<filter_type>", methods=["GET"])
def get_history_image_api(profile_id, capture_id, filter_type):
    try:
        # 실제 파일 경로 찾기
        image_path = resolve_image_path(profile_id, capture_id, filter_type)

        # 이미지 파일 반환
        return send_file(image_path)

    except Exception as e:
        # 실패 시 JSON 반환
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 404


# =========================
# 메인 실행부
# =========================

if __name__ == "__main__":
    # 서버 시작 전에 릴레이 OFF 보장
    relay_off()

    # 서버 시작 전에 백색 LED OFF 보장
    white_led_off()

    # 카메라 초기화
    init_camera()

    # CAM4 미리보기 JPEG를 백그라운드에서 미리 만들어 화면 지연을 줄입니다.
    start_cam4_jpeg_worker()

    # Flask 서버 실행
    app.run(host="0.0.0.0", port=8000, threaded=True)





