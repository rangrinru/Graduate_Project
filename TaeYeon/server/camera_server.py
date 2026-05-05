# Flask 서버와 JSON 응답을 위한 모듈 import
from flask import Flask, jsonify, Response, request, send_file

# CORS 허용을 위한 모듈 import
from flask_cors import CORS

# 이미지 저장/변환을 위한 OpenCV import
import cv2

# 포르피린 분석을 위한 NumPy import
import numpy as np

# JSON 파일 읽기/쓰기용 import
import json

# 얼굴 각도 계산을 위한 math import
import math

# 카메라 동기화용 스레드 락 import
import threading

# 문자열 정리용 정규식 모듈 import
import re

# 폴더 삭제용 모듈 import
import shutil

# 경로 처리를 위한 Path import
from pathlib import Path

# 날짜/시간 처리용 import
from datetime import datetime

# 대기 시간과 경과 시간 측정용 import
from time import sleep, monotonic

# 눈 감음 기준값 누적용 deque import
from collections import deque

# 릴레이 제어용 GPIO import
from gpiozero import LED

# Picamera2 import
from picamera2 import Picamera2

# MediaPipe 얼굴 인식 모듈은 설치되어 있을 때만 사용
try:
    import mediapipe as mp
except ImportError:
    mp = None


# Flask 앱 생성
app = Flask(__name__)

# 다른 포트의 프론트엔드에서도 접근 가능하도록 CORS 허용
CORS(app)


# =========================
# 기본 설정값
# =========================

# 전체 캡처 이미지 가로 해상도
CAPTURE_WIDTH = 5120

# 전체 캡처 이미지 세로 해상도
CAPTURE_HEIGHT = 800

# 전체 이미지가 4분할 구조라고 가정하고 단일 폭 계산
SINGLE_WIDTH = CAPTURE_WIDTH // 4

# 저장 루트 경로 설정
SAVE_ROOT = Path.home() / "Graduate_Project" / "Final_Project" / "captures"

# 저장 루트 폴더가 없으면 생성
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# 프로필 목록을 저장하는 JSON 파일 경로
PROFILES_FILE = SAVE_ROOT / "profiles.json"

# 현재 고정 gain 값
CURRENT_GAIN = 1.0

# PNG 저장 여부
SAVE_AS_PNG = True

# 초기 노출 시간(ms)
INITIAL_EXPOSURE_MS = 100

# 스트리밍 FPS
STREAM_FPS = 12

# 릴레이 연결 GPIO 번호
RELAY_PIN = 17

# active_high 여부
RELAY_ACTIVE_HIGH = False

# 릴레이 켠 후 안정화 대기 시간
RELAY_WARMUP_SEC = 0.3

# 백색 LED 연결 GPIO 번호
WHITE_LED_PIN = 22

# 백색 LED active_high 여부
WHITE_LED_ACTIVE_HIGH = True

# 자동 촬영 직전 백색 LED를 끈 뒤 대기하는 시간
WHITE_LED_OFF_BEFORE_CAPTURE_SEC = 0.15

# 얼굴 중앙 허용 오차 X축 비율
FACE_CENTER_TOL_X = 0.18

# 얼굴 중앙 허용 오차 Y축 비율
FACE_CENTER_TOL_Y = 0.20

# 얼굴이 너무 작은지 판단하는 최소 면적 비율
FACE_MIN_AREA_RATIO = 0.08

# 얼굴이 너무 큰지 판단하는 최대 면적 비율
FACE_MAX_AREA_RATIO = 0.70

# 얼굴 좌우 기울기 허용 각도
MAX_ABS_ROLL_DEG = 8.0

# 얼굴 좌우 회전 점수 허용값
MAX_ABS_YAW_SCORE = 0.12

# 얼굴 위아래 회전 점수 허용값
MAX_ABS_PITCH_SCORE = 0.12

# 기본 눈 감음 EAR 기준값
DEFAULT_EYE_AR_THRESHOLD = 0.18

# 자동 보정 EAR 최소값
MIN_DYNAMIC_EYE_AR_THRESHOLD = 0.14

# 자동 보정 EAR 최대값
MAX_DYNAMIC_EYE_AR_THRESHOLD = 0.26

# 열린 눈 평균값에 곱할 비율
EYE_THRESHOLD_RATIO = 0.72

# 얼굴 정렬이 유지되어야 하는 프레임 수
STABLE_FACE_HOLD_FRAMES = 6

# 열린 눈 기준값을 계산할 샘플 수
OPEN_EYE_BASELINE_SAMPLES = 15

# 눈 감음이 유지되어야 하는 프레임 수
EYES_CLOSED_HOLD_FRAMES = 4

# 자동 촬영 상태 갱신 주기
AUTO_CAPTURE_INTERVAL_SEC = 0.12


# =========================
# 릴레이 객체 생성
# =========================

# 릴레이 LED 객체 생성
relay = LED(RELAY_PIN, active_high=RELAY_ACTIVE_HIGH, initial_value=False)

# 백색 LED 객체 생성
white_led = LED(WHITE_LED_PIN, active_high=WHITE_LED_ACTIVE_HIGH, initial_value=False)

# 백색 LED 상태값
white_led_is_on = False


# =========================
# 카메라별 설정 정보
# =========================

CAMERA_INFO = {
    "cam2": {
        "label": "CAM 2 - NO FILTER",
        "folder": "cam2_no_filter",
        "filter": "no_filter",
        "display_name": "No_Filter",
        "x_start": SINGLE_WIDTH * 1,
        "x_end": SINGLE_WIDTH * 2,
        "sequence_order": 1
    },
    "cam3": {
        "label": "CAM 3 - 405nm FILTER",
        "folder": "cam3_405nm",
        "filter": "405nm_filter",
        "display_name": "405nm_Filter",
        "x_start": SINGLE_WIDTH * 2,
        "x_end": SINGLE_WIDTH * 3,
        "sequence_order": 2
    },
    "cam4": {
        "label": "CAM 4 - 660nm FILTER",
        "folder": "cam4_660nm",
        "filter": "660nm_filter",
        "display_name": "660nm_Filter",
        "x_start": SINGLE_WIDTH * 3,
        "x_end": SINGLE_WIDTH * 4,
        "sequence_order": 3
    }
}


# =========================
# 전역 카메라 상태값
# =========================

# 카메라 동시 접근 방지를 위한 락
camera_lock = threading.Lock()

# 카메라 준비 여부
camera_ready = False

# Picamera2 객체
picam2 = None

# 프리뷰 설정
preview_config = None

# 스틸 캡처 설정
still_config = None

# 자동 촬영 스레드 객체
auto_capture_thread = None

# 자동 촬영 상태 동기화용 락
auto_state_lock = threading.Lock()

# 열린 눈 EAR 기준값 누적 배열
OPEN_EYE_EAR_HISTORY = deque(maxlen=OPEN_EYE_BASELINE_SAMPLES)

# MediaPipe 얼굴 메시 객체 생성
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

# 왼쪽 눈 랜드마크 인덱스
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# 오른쪽 눈 랜드마크 인덱스
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# 코 끝 랜드마크 인덱스
NOSE_TIP_IDX = 1

# 왼쪽 눈 바깥쪽 랜드마크 인덱스
LEFT_EYE_OUTER_IDX = 33

# 오른쪽 눈 바깥쪽 랜드마크 인덱스
RIGHT_EYE_OUTER_IDX = 263

# 얼굴 왼쪽 끝 랜드마크 인덱스
LEFT_FACE_IDX = 234

# 얼굴 오른쪽 끝 랜드마크 인덱스
RIGHT_FACE_IDX = 454

# 얼굴 위쪽 랜드마크 인덱스
UPPER_FACE_IDX = 10

# 얼굴 아래쪽 랜드마크 인덱스
LOWER_FACE_IDX = 152

# 입 왼쪽 랜드마크 인덱스
MOUTH_LEFT_IDX = 61

# 입 오른쪽 랜드마크 인덱스
MOUTH_RIGHT_IDX = 291


def make_default_auto_checks():
    # 자동 촬영 조건 기본값 생성
    return {
        "face_found": False,
        "center_ok": False,
        "size_ok": False,
        "angle_ok": False,
        "eyes_closed": False,
        "stable_ok": False,
    }


# 자동 촬영 상태값
AUTO_STATE = {
    "running": False,
    "captured": False,
    "profile_id": None,
    "capture_id": None,
    "status": "자동 촬영 대기 중",
    "error": None,
    "checks": make_default_auto_checks(),
    "stable_face_count": 0,
    "eyes_closed_count": 0,
    "dynamic_eye_threshold": DEFAULT_EYE_AR_THRESHOLD,
    "last_update": None,
}


# =========================
# 릴레이 제어 함수
# =========================

def relay_on():
    # 릴레이 켜기
    relay.on()


def relay_off():
    # 릴레이 끄기
    relay.off()


def white_led_on():
    # 전역 백색 LED 상태값 사용 선언
    global white_led_is_on

    # 백색 LED 켜기
    white_led.on()

    # 백색 LED 상태값 갱신
    white_led_is_on = True


def white_led_off():
    # 전역 백색 LED 상태값 사용 선언
    global white_led_is_on

    # 백색 LED 끄기
    white_led.off()

    # 백색 LED 상태값 갱신
    white_led_is_on = False


# =========================
# 카메라 수동 제어 함수
# =========================

def set_manual_controls(camera, exposure_ms, gain):
    # 자동 노출을 끄고 수동 노출/게인 적용
    camera.set_controls({
        "AeEnable": False,
        "ExposureTime": exposure_ms * 1000,
        "AnalogueGain": gain
    })


# =========================
# 전체 프레임에서 개별 카메라 영역 자르기
# =========================

def extract_cam_frame(full_frame_bgr, cam_key):
    # 해당 카메라 정보 읽기
    info = CAMERA_INFO[cam_key]

    # 전체 프레임에서 해당 영역만 잘라서 반환
    return full_frame_bgr[:, info["x_start"]:info["x_end"]]


# =========================
# 이미지 저장 함수
# =========================

def save_image(path, img_bgr):
    # PNG 저장일 경우 무압축 저장
    if path.suffix.lower() == ".png":
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # JPG 저장일 경우 고품질 저장
    else:
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])


# =========================
# 프로필 이름 정리
# =========================

def sanitize_profile_name(profile_name: str) -> str:
    # 앞뒤 공백 제거
    cleaned = profile_name.strip()

    # 연속 공백을 하나로 정리
    cleaned = re.sub(r"\s+", " ", cleaned)

    # 비어 있으면 예외 발생
    if not cleaned:
        raise ValueError("유효한 프로필 이름이 아닙니다.")

    # 정리된 이름 반환
    return cleaned


# =========================
# 실제 폴더용 profile 폴더 ID 생성
# =========================

def make_folder_id() -> str:
    # 현재 timestamp 기반으로 고유 폴더 ID 생성
    return f"profile_{int(datetime.now().timestamp() * 1000)}"


# =========================
# 프로필 목록 로드
# =========================

def load_profiles():
    # 프로필 파일이 없으면 빈 리스트 반환
    if not PROFILES_FILE.exists():
        return []

    try:
        # UTF-8로 JSON 읽기
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        # 파싱 실패 시 빈 리스트 반환
        return []


# =========================
# 프로필 목록 저장
# =========================

def save_profiles(profiles):
    # UTF-8로 JSON 저장
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


# =========================
# 프로필 기본 폴더 구조 생성
# =========================

def ensure_profile_dirs(folder_id: str):
    # 프로필 루트 폴더 경로 생성
    profile_root = SAVE_ROOT / folder_id

    # 프로필 루트 생성
    profile_root.mkdir(parents=True, exist_ok=True)

    # 카메라별 폴더 생성
    for cam in CAMERA_INFO.values():
        (profile_root / cam["folder"]).mkdir(parents=True, exist_ok=True)

    # 생성된 프로필 루트 반환
    return profile_root


# =========================
# ID로 프로필 찾기
# =========================

def find_profile_by_id(profile_id: str):
    # 현재 프로필 목록 로드
    profiles = load_profiles()

    # 일치하는 folderId 탐색
    for profile in profiles:
        if profile["folderId"] == profile_id:
            return profile

    # 없으면 None 반환
    return None


# =========================
# 프로필 생성
# =========================

def create_profile(profile_name: str):
    # 사용자 표시용 이름 정리
    display_name = sanitize_profile_name(profile_name)

    # 기존 프로필 목록 로드
    profiles = load_profiles()

    # 이름 중복 검사
    exists = any(p["name"] == display_name for p in profiles)

    # 이미 있으면 예외
    if exists:
        raise ValueError("이미 존재하는 프로필입니다.")

    # 생성 날짜 문자열
    created_at = datetime.now().strftime("%Y.%m.%d")

    # 화면용 숫자 ID
    profile_id = int(datetime.now().timestamp() * 1000)

    # 실제 폴더용 안전한 ID
    folder_id = make_folder_id()

    # 새 프로필 정보 구성
    new_profile = {
        "id": profile_id,
        "name": display_name,
        "folderId": folder_id,
        "createdAt": created_at
    }

    # 목록에 추가
    profiles.append(new_profile)

    # JSON 저장
    save_profiles(profiles)

    # 폴더 생성
    ensure_profile_dirs(folder_id)

    # 생성된 프로필 반환
    return new_profile


# =========================
# 프로필 삭제
# =========================

def delete_profile(profile_id: str):
    # 프로필 목록 로드
    profiles = load_profiles()

    # 삭제 대상 초기화
    target = None

    # 삭제할 프로필 찾기
    for p in profiles:
        if p["folderId"] == profile_id:
            target = p
            break

    # 없으면 예외
    if target is None:
        raise ValueError("삭제할 프로필이 없습니다.")

    # 삭제 대상 제외한 목록 생성
    new_profiles = [p for p in profiles if p["folderId"] != profile_id]

    # 프로필 루트 경로
    profile_root = SAVE_ROOT / target["folderId"]

    # 실제 폴더가 있으면 전체 삭제
    if profile_root.exists() and profile_root.is_dir():
        shutil.rmtree(profile_root)

    # 갱신된 목록 저장
    save_profiles(new_profiles)


# =========================
# 개별 카메라 이미지 저장
# =========================

def save_one_camera_image(
    cam_key,
    frame_bgr,
    profile_root,
    capture_id,
    timestamp,
    exposure_ms,
    gain,
    ext,
    profile_name,
    folder_id
):
    # 카메라 정보 조회
    info = CAMERA_INFO[cam_key]

    # 세로 미러 형태에 맞게 시계 방향 90도 회전
    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

    # 해당 캡처 시각 폴더 생성
    target_dir = profile_root / info["folder"] / capture_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 경로 생성
    image_path = target_dir / f"{cam_key}.{ext}"

    # 메타데이터 경로 생성
    meta_path = target_dir / "metadata.json"

    # 이미지 저장
    save_image(image_path, frame_bgr)

    # 메타데이터 구성
    metadata = {
        "captured_at": timestamp.isoformat(),
        "profile_name": profile_name,
        "profile_folder_id": folder_id,
        "camera_name": cam_key,
        "camera_label": info["label"],
        "filter_type": info["filter"],
        "display_name": info["display_name"],
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
        "saved_file": str(image_path),
        "rotation_applied": "ROTATE_90_CLOCKWISE"
    }

    # 메타데이터 저장
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    # 저장 결과 반환
    return {
        "camera": cam_key,
        "filter_type": info["filter"],
        "display_name": info["display_name"],
        "image_path": str(image_path),
        "metadata_path": str(meta_path)
    }


# =========================
# 카메라 초기화
# =========================

def init_camera():
    # 전역 변수 사용 선언
    global picam2, preview_config, still_config, camera_ready

    # 연결된 카메라 목록 조회
    camera_list = Picamera2.global_camera_info()

    # 카메라가 없으면 예외
    if len(camera_list) == 0:
        raise RuntimeError("카메라가 감지되지 않았습니다.")

    # 첫 번째 카메라 사용
    picam2 = Picamera2(0)

    # 프리뷰 설정 생성
    preview_config = picam2.create_preview_configuration(
        main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "XRGB8888"}
    )

    # 스틸 설정 생성
    still_config = picam2.create_still_configuration(
        main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
    )

    # 프리뷰 설정 적용
    picam2.configure(preview_config)

    # 품질 옵션 설정
    picam2.options["quality"] = 100
    picam2.options["compress_level"] = 0

    # 카메라 시작
    picam2.start()

    # 초기 노출/게인 설정
    set_manual_controls(picam2, INITIAL_EXPOSURE_MS, CURRENT_GAIN)

    # 카메라 준비 완료 표시
    camera_ready = True


# =========================
# 프리뷰 프레임 읽기
# =========================

def read_preview_frame():
    # 현재 프레임 캡처
    frame = picam2.capture_array()

    # BGRA -> BGR 변환 후 반환
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


# =========================
# 고화질 전체 프레임 촬영
# =========================

def capture_high_quality_full_frame(exposure_ms, gain):
    # 전역 변수 사용
    global picam2, preview_config, still_config

    # 릴레이 ON
    relay_on()

    # 릴레이 안정화 대기
    sleep(RELAY_WARMUP_SEC)

    try:
        # 프리뷰 중지
        picam2.stop()

        # 스틸 설정 적용
        picam2.configure(still_config)

        # 카메라 재시작
        picam2.start()

        # 수동 노출/게인 적용
        set_manual_controls(picam2, exposure_ms, gain)

        # 노출 안정화 대기
        sleep(1)

        # 스틸 프레임 캡처
        still_frame = picam2.capture_array()

        # RGB -> BGR 변환
        full_frame_bgr = cv2.cvtColor(still_frame, cv2.COLOR_RGB2BGR)

    finally:
        # 릴레이 OFF
        relay_off()

        # 카메라 정지
        picam2.stop()

        # 프리뷰 설정 복귀
        picam2.configure(preview_config)

        # 카메라 재시작
        picam2.start()

        # 프리뷰에도 동일 수동 제어 재적용
        set_manual_controls(picam2, exposure_ms, gain)

    # 촬영 결과 반환
    return full_frame_bgr


# =========================
# 촬영 기록 유틸 함수
# =========================

def format_capture_id_to_text(capture_id: str) -> str:
    # 예: 20260408_132215_123 -> 2026-04-08 13:22:15.123 형태로 변환 시도
    try:
        dt = datetime.strptime(capture_id, "%Y%m%d_%H%M%S_%f")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return capture_id


def get_profile_root(profile_id: str) -> Path:
    # 프로필 존재 여부 확인
    profile = find_profile_by_id(profile_id)

    # 없으면 예외
    if profile is None:
        raise ValueError("존재하지 않는 프로필입니다.")

    # 루트 경로 반환
    return SAVE_ROOT / profile["folderId"]


def get_capture_history(profile_id: str):
    # 프로필 루트 경로 가져오기
    profile_root = get_profile_root(profile_id)

    # 프로필 조회
    profile = find_profile_by_id(profile_id)

    # 프로필 루트가 없으면 빈 배열 반환
    if not profile_root.exists():
        return []

    # 기준이 되는 cam2 폴더 경로
    cam2_root = profile_root / CAMERA_INFO["cam2"]["folder"]

    # cam2 폴더가 없으면 빈 배열 반환
    if not cam2_root.exists():
        return []

    # 기록 목록 저장 배열
    history = []

    # capture_id 폴더 순회
    for capture_dir in cam2_root.iterdir():
        # 폴더만 처리
        if not capture_dir.is_dir():
            continue

        # capture ID는 폴더 이름
        capture_id = capture_dir.name

        # 기본 captured_at 값
        captured_at = None

        # 메타데이터 파일 경로
        meta_path = capture_dir / "metadata.json"

        # 메타데이터가 있으면 실제 촬영 시각 사용
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    captured_at = meta.get("captured_at")
            except Exception:
                captured_at = None

        # 메타데이터에서 못 읽었으면 capture_id를 기반으로 표시 문자열 구성
        if not captured_at:
            captured_at = capture_id

        # 목록에 추가
        history.append({
            "captureId": capture_id,
            "capturedAt": captured_at,
            "displayTime": format_capture_id_to_text(capture_id),
            "profileId": profile["folderId"],
            "profileName": profile["name"]
        })

    # 최신 촬영이 위로 오도록 정렬
    history.sort(key=lambda x: x["captureId"], reverse=True)

    # 정렬된 기록 반환
    return history


def get_capture_detail(profile_id: str, capture_id: str):
    # 프로필 루트 확인
    profile_root = get_profile_root(profile_id)

    # 프로필 정보 조회
    profile = find_profile_by_id(profile_id)

    # 결과 이미지 딕셔너리
    images = {}

    # 대표 촬영 시각
    captured_at = None

    # 각 카메라 폴더 순회
    for cam_key, info in CAMERA_INFO.items():
        # 해당 캡처 폴더 경로
        target_dir = profile_root / info["folder"] / capture_id

        # 파일 확장자 후보 목록
        candidates = [
            target_dir / f"{cam_key}.png",
            target_dir / f"{cam_key}.jpg",
            target_dir / f"{cam_key}.jpeg",
        ]

        # 실제 존재하는 이미지 파일 찾기
        image_path = None
        for candidate in candidates:
            if candidate.exists():
                image_path = candidate
                break

        # 메타데이터 경로
        meta_path = target_dir / "metadata.json"

        # 메타데이터 읽기
        meta_data = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
            except Exception:
                meta_data = {}

        # captured_at은 하나만 대표로 사용
        if captured_at is None:
            captured_at = meta_data.get("captured_at")

        # 이미지가 있으면 응답용 정보 생성
        images[info["filter"]] = {
            "camera": cam_key,
            "display_name": info["display_name"],
            "filter_type": info["filter"],
            "exists": image_path is not None,
            "image_url": f"/profiles/{profile_id}/history/{capture_id}/image/{info['filter']}" if image_path else None,
            "metadata": meta_data
        }

    # 대표 촬영 시각이 없으면 capture_id 기반 문자열 사용
    if captured_at is None:
        captured_at = capture_id

    # 최소 하나라도 이미지가 있어야 유효한 기록으로 판단
    has_any = any(item["exists"] for item in images.values())

    # 하나도 없으면 예외
    if not has_any:
        raise ValueError("해당 촬영 기록을 찾을 수 없습니다.")

    # 최종 상세 정보 반환
    return {
        "captureId": capture_id,
        "capturedAt": captured_at,
        "displayTime": format_capture_id_to_text(capture_id),
        "profileId": profile["folderId"],
        "profileName": profile["name"],
        "images": images
    }


def delete_capture_history(profile_id: str, capture_id: str):
    # 프로필 루트 경로 가져오기
    profile_root = get_profile_root(profile_id)

    # 하나라도 삭제되었는지 확인하기 위한 플래그
    deleted_any = False

    # cam2, cam3, cam4의 동일 capture_id 폴더를 모두 삭제
    for cam_key, info in CAMERA_INFO.items():
        # 삭제 대상 폴더 경로 생성
        target_dir = profile_root / info["folder"] / capture_id

        # 실제 폴더가 있으면 내부 이미지와 metadata.json까지 전체 삭제
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir)
            deleted_any = True

    # 하나도 삭제되지 않았으면 잘못된 기록으로 판단
    if not deleted_any:
        raise ValueError("삭제할 촬영 기록이 없습니다.")


def resolve_image_path(profile_id: str, capture_id: str, filter_type: str) -> Path:
    # 프로필 루트 확인
    profile_root = get_profile_root(profile_id)

    # filter_type에 맞는 카메라 찾기
    matched_cam_key = None
    matched_info = None

    for cam_key, info in CAMERA_INFO.items():
        if info["filter"] == filter_type:
            matched_cam_key = cam_key
            matched_info = info
            break

    # 없으면 예외
    if matched_cam_key is None:
        raise ValueError("유효하지 않은 필터 타입입니다.")

    # 캡처 폴더 경로
    target_dir = profile_root / matched_info["folder"] / capture_id

    # 후보 파일 목록
    candidates = [
        target_dir / f"{matched_cam_key}.png",
        target_dir / f"{matched_cam_key}.jpg",
        target_dir / f"{matched_cam_key}.jpeg",
    ]

    # 존재하는 파일 찾기
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    # 없으면 예외
    raise ValueError("이미지 파일이 존재하지 않습니다.")




# =========================
# 자동 얼굴 촬영 유틸 함수
# =========================

def _euclidean(p1, p2):
    # 두 점 사이의 유클리드 거리 계산
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _eye_aspect_ratio(eye_pts):
    # 눈 위아래 거리 1 계산
    a = _euclidean(eye_pts[1], eye_pts[5])

    # 눈 위아래 거리 2 계산
    b = _euclidean(eye_pts[2], eye_pts[4])

    # 눈 좌우 거리 계산
    c = _euclidean(eye_pts[0], eye_pts[3])

    # 0으로 나누는 상황 방지
    if c == 0:
        return 1.0

    # EAR 값 반환
    return (a + b) / (2.0 * c)


def update_dynamic_eye_threshold(avg_ear):
    # 열린 눈 EAR 값을 누적
    OPEN_EYE_EAR_HISTORY.append(avg_ear)

    # 충분한 샘플이 쌓이면 동적 기준값 계산
    if len(OPEN_EYE_EAR_HISTORY) >= max(8, OPEN_EYE_BASELINE_SAMPLES // 2):
        # 열린 눈 평균값 계산
        baseline = sum(OPEN_EYE_EAR_HISTORY) / len(OPEN_EYE_EAR_HISTORY)

        # 열린 눈 평균값에 비율을 곱해 눈 감음 기준값 계산
        dynamic = baseline * EYE_THRESHOLD_RATIO

        # 너무 낮거나 높은 값으로 튀지 않게 제한
        dynamic = max(MIN_DYNAMIC_EYE_AR_THRESHOLD, min(MAX_DYNAMIC_EYE_AR_THRESHOLD, dynamic))

        # 자동 촬영 상태에 동적 기준값 저장
        with auto_state_lock:
            AUTO_STATE["dynamic_eye_threshold"] = dynamic


def analyze_face_for_auto(cam2_bgr):
    # 이미지 높이와 폭 읽기
    h, w = cam2_bgr.shape[:2]

    # 기본 분석 결과 구성
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

    # MediaPipe가 설치되어 있지 않으면 분석 불가 처리
    if face_mesh is None:
        result["guide_text"] = "mediapipe가 설치되어 있지 않습니다"
        return result

    # BGR 이미지를 RGB로 변환
    rgb = cv2.cvtColor(cam2_bgr, cv2.COLOR_BGR2RGB)

    # MediaPipe 얼굴 메시 분석 실행
    mesh_result = face_mesh.process(rgb)

    # 얼굴이 감지되지 않으면 반환
    if not mesh_result.multi_face_landmarks:
        result["guide_text"] = "얼굴이 감지되지 않습니다"
        return result

    # 랜드마크 좌표 배열 생성
    pts = []

    # 첫 번째 얼굴의 모든 랜드마크 순회
    for lm in mesh_result.multi_face_landmarks[0].landmark:
        # 정규화 좌표를 픽셀 좌표로 변환하여 저장
        pts.append((int(lm.x * w), int(lm.y * h)))

    # 얼굴 전체 x 좌표 배열 생성
    xs = [p[0] for p in pts]

    # 얼굴 전체 y 좌표 배열 생성
    ys = [p[1] for p in pts]

    # 얼굴 박스 왼쪽 좌표 계산
    x1 = max(0, min(xs))

    # 얼굴 박스 오른쪽 좌표 계산
    x2 = min(w - 1, max(xs))

    # 얼굴 박스 위쪽 좌표 계산
    y1 = max(0, min(ys))

    # 얼굴 박스 아래쪽 좌표 계산
    y2 = min(h - 1, max(ys))

    # 얼굴 박스 폭 계산
    bw = max(1, x2 - x1)

    # 얼굴 박스 높이 계산
    bh = max(1, y2 - y1)

    # 얼굴 면적 비율 계산
    area_ratio = (bw * bh) / float(w * h)

    # 얼굴 중심 x 좌표 계산
    cx = (x1 + x2) / 2.0

    # 얼굴 중심 y 좌표 계산
    cy = (y1 + y2) / 2.0

    # 중앙에서 벗어난 x 비율 계산
    norm_dx = abs(cx - (w / 2.0)) / w

    # 중앙에서 벗어난 y 비율 계산
    norm_dy = abs(cy - (h / 2.0)) / h

    # 왼쪽 눈 좌표 배열 생성
    left_eye_pts = [pts[i] for i in LEFT_EYE]

    # 오른쪽 눈 좌표 배열 생성
    right_eye_pts = [pts[i] for i in RIGHT_EYE]

    # 왼쪽 눈 EAR 계산
    ear_left = _eye_aspect_ratio(left_eye_pts)

    # 오른쪽 눈 EAR 계산
    ear_right = _eye_aspect_ratio(right_eye_pts)

    # 양쪽 눈 평균 EAR 계산
    avg_ear = (ear_left + ear_right) / 2.0

    # 얼굴 중앙 정렬 여부 계산
    center_ok = (norm_dx <= FACE_CENTER_TOL_X) and (norm_dy <= FACE_CENTER_TOL_Y)

    # 얼굴 크기 적정 여부 계산
    size_ok = (FACE_MIN_AREA_RATIO <= area_ratio <= FACE_MAX_AREA_RATIO)

    # 코 끝 좌표 읽기
    nose = pts[NOSE_TIP_IDX]

    # 왼쪽 눈 바깥쪽 좌표 읽기
    le = pts[LEFT_EYE_OUTER_IDX]

    # 오른쪽 눈 바깥쪽 좌표 읽기
    re = pts[RIGHT_EYE_OUTER_IDX]

    # 얼굴 왼쪽 끝 좌표 읽기
    lf = pts[LEFT_FACE_IDX]

    # 얼굴 오른쪽 끝 좌표 읽기
    rf = pts[RIGHT_FACE_IDX]

    # 얼굴 위쪽 좌표 읽기
    upper = pts[UPPER_FACE_IDX]

    # 얼굴 아래쪽 좌표 읽기
    lower = pts[LOWER_FACE_IDX]

    # 입 왼쪽 좌표 읽기
    ml = pts[MOUTH_LEFT_IDX]

    # 입 오른쪽 좌표 읽기
    mr = pts[MOUTH_RIGHT_IDX]

    # 양쪽 눈 중심 좌표 계산
    eye_mid = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)

    # 입 중심 좌표 계산
    mouth_mid = ((ml[0] + mr[0]) / 2.0, (ml[1] + mr[1]) / 2.0)

    # 얼굴 중심 x 좌표 계산
    face_mid_x = (lf[0] + rf[0]) / 2.0

    # 얼굴 폭 계산
    face_width = max(1.0, float(rf[0] - lf[0]))

    # 얼굴 높이 계산
    face_height = max(1.0, float(lower[1] - upper[1]))

    # 얼굴 좌우 기울기 각도 계산
    roll_deg = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))

    # 좌우 회전 점수 계산
    yaw_score = abs(nose[0] - face_mid_x) / face_width

    # 위아래 회전 점수 계산
    pitch_score = abs(nose[1] - ((eye_mid[1] + mouth_mid[1]) / 2.0)) / face_height

    # 좌우 기울기 허용 여부 계산
    roll_ok = abs(roll_deg) <= MAX_ABS_ROLL_DEG

    # 좌우 회전 허용 여부 계산
    yaw_ok = yaw_score <= MAX_ABS_YAW_SCORE

    # 위아래 회전 허용 여부 계산
    pitch_ok = pitch_score <= MAX_ABS_PITCH_SCORE

    # 전체 각도 허용 여부 계산
    angles_ok = roll_ok and yaw_ok and pitch_ok

    # 현재 눈 감음 기준값 읽기
    with auto_state_lock:
        threshold = AUTO_STATE["dynamic_eye_threshold"]

    # 양쪽 눈이 모두 기준값보다 낮으면 눈 감음 처리
    eyes_closed = (ear_left < threshold) and (ear_right < threshold)

    # 분석 결과 갱신
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

    # 얼굴 위치가 틀어진 경우 안내 문구 설정
    if not center_ok:
        result["guide_text"] = "얼굴을 화면 중앙에 맞춰주세요"

    # 얼굴 크기가 부적절한 경우 안내 문구 설정
    elif not size_ok:
        result["guide_text"] = "얼굴 거리를 조정해주세요"

    # 얼굴 각도가 부적절한 경우 안내 문구 설정
    elif not angles_ok:
        result["guide_text"] = "얼굴 각도를 정면으로 맞춰주세요"

    # 눈을 아직 감지 않은 경우 안내 문구 설정
    elif not eyes_closed:
        result["guide_text"] = "눈을 감아주세요"

    # 모든 기본 조건이 충족된 경우 안내 문구 설정
    else:
        result["guide_text"] = "조건 충족"

    # 분석 결과 반환
    return result


def build_auto_checks(detection):
    # 자동 촬영 체크 결과 구성
    return {
        "face_found": bool(detection.get("face_found")),
        "center_ok": bool(detection.get("center_ok")),
        "size_ok": bool(detection.get("size_ok")),
        "angle_ok": bool(detection.get("angles_ok")),
        "eyes_closed": bool(detection.get("eyes_closed")),
        "stable_ok": False,
    }


def update_auto_state_from_detection(detection):
    # 현재 검사 결과 생성
    checks = build_auto_checks(detection)

    # 기본 안정 조건 계산
    stable_face_ok = (
        checks["face_found"]
        and checks["center_ok"]
        and checks["size_ok"]
        and checks["angle_ok"]
    )

    # 자동 촬영 상태 갱신을 위해 락 획득
    with auto_state_lock:
        # 얼굴 기본 조건이 안정적이면 카운트 증가
        if stable_face_ok:
            AUTO_STATE["stable_face_count"] += 1

        # 얼굴 기본 조건이 불안정하면 카운트 초기화
        else:
            AUTO_STATE["stable_face_count"] = 0
            AUTO_STATE["eyes_closed_count"] = 0

        # 안정 유지 조건 갱신
        checks["stable_ok"] = AUTO_STATE["stable_face_count"] >= STABLE_FACE_HOLD_FRAMES

        # 얼굴을 못 찾은 경우 상태 문구 설정
        if not checks["face_found"]:
            AUTO_STATE["status"] = "얼굴이 감지되지 않습니다"

        # 중앙 정렬이 안 된 경우 상태 문구 설정
        elif not checks["center_ok"]:
            AUTO_STATE["status"] = "얼굴을 화면 중앙에 맞춰주세요"

        # 얼굴 크기가 안 맞는 경우 상태 문구 설정
        elif not checks["size_ok"]:
            AUTO_STATE["status"] = "얼굴 거리를 조정해주세요"

        # 얼굴 각도가 안 맞는 경우 상태 문구 설정
        elif not checks["angle_ok"]:
            AUTO_STATE["status"] = "얼굴 각도를 정면으로 맞춰주세요"

        # 얼굴 안정 유지 프레임이 부족한 경우 상태 문구 설정
        elif not checks["stable_ok"]:
            AUTO_STATE["status"] = f"얼굴 고정 중 {AUTO_STATE['stable_face_count']}/{STABLE_FACE_HOLD_FRAMES}"

        # 눈을 아직 감지 않은 경우 상태 문구 설정
        elif not checks["eyes_closed"]:
            AUTO_STATE["eyes_closed_count"] = 0
            AUTO_STATE["status"] = "얼굴을 유지한 채 눈을 감아주세요"

        # 눈 감음이 확인된 경우 상태 문구 설정
        else:
            AUTO_STATE["eyes_closed_count"] += 1
            AUTO_STATE["status"] = f"눈감음 확인 중 {AUTO_STATE['eyes_closed_count']}/{EYES_CLOSED_HOLD_FRAMES}"

        # 검사 결과 저장
        AUTO_STATE["checks"] = checks

        # 최근 갱신 시각 저장
        AUTO_STATE["last_update"] = datetime.now().isoformat()

        # 눈 감음 유지 여부 반환
        return AUTO_STATE["eyes_closed_count"] >= EYES_CLOSED_HOLD_FRAMES


def snapshot_detection_for_metadata(detection):
    # 메타데이터 저장용 감지 결과 생성
    return {
        "bbox": list(detection["bbox"]) if detection.get("bbox") is not None else None,
        "roll_deg": float(detection.get("roll_deg", 0.0)),
        "yaw_score": float(detection.get("yaw_score", 0.0)),
        "pitch_score": float(detection.get("pitch_score", 0.0)),
        "dynamic_eye_threshold": float(AUTO_STATE.get("dynamic_eye_threshold", DEFAULT_EYE_AR_THRESHOLD)),
    }


def perform_capture_for_profile(profile_id: str):
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

    # 촬영 노출 시간 설정
    exposure_ms = INITIAL_EXPOSURE_MS

    # 촬영 gain 설정
    gain = CURRENT_GAIN

    # 저장 확장자 결정
    ext = "png" if SAVE_AS_PNG else "jpg"

    # 카메라 락을 잡고 고화질 전체 프레임 촬영
    with camera_lock:
        full_frame_bgr = capture_high_quality_full_frame(
            exposure_ms=exposure_ms,
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
            folder_id=folder_id
        )

        # 저장 파일 목록에 추가
        saved_files.append(result)

    # 촬영 결과 반환
    return {
        "ok": True,
        "captured_at": capture_timestamp.isoformat(),
        "profile_name": profile_name,
        "profile_id": folder_id,
        "capture_id": capture_id,
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
            "white_led_is_on": white_led_is_on,
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

        # 상태 문구 초기화
        AUTO_STATE["status"] = "자동 촬영 조건 확인 중" if running else "자동 촬영 대기 중"

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

            # 카메라 락을 잡고 프리뷰 프레임 읽기
            with camera_lock:
                full_frame_bgr = read_preview_frame()

            # cam2 영역을 얼굴 인식용으로 사용
            cam2_bgr = extract_cam_frame(full_frame_bgr, "cam2").copy()

            # 얼굴 상태 분석
            detection = analyze_face_for_auto(cam2_bgr)

            # 얼굴이 안정적이고 눈을 뜬 상태라면 열린 눈 기준값 갱신
            if detection["face_found"] and detection["center_ok"] and detection["size_ok"] and detection["angles_ok"] and not detection["eyes_closed"]:
                update_dynamic_eye_threshold(detection["avg_ear"])

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
                capture_result = perform_capture_for_profile(profile_id)

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
# 포르피린 분석 함수
# =========================

def analyze_porphyrin_image(image_path: Path, output_dir: Path):
    # 분석할 이미지 읽기
    img = cv2.imread(str(image_path))

    # 이미지가 없으면 예외 발생
    if img is None:
        raise RuntimeError("이미지 로드 실패")

    # 결과 표시용 원본 복사
    output = img.copy()

    # BGR 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE로 국소 대비 강화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 대비 강화 이미지 생성
    enhanced = clahe.apply(gray)

    # 작은 노이즈 완화를 위한 Gaussian Blur
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 상위 밝기 영역만 추출
    threshold_value = np.percentile(blur, 97)

    # 이진화
    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)

    # 노이즈 제거용 커널 생성
    kernel = np.ones((3, 3), np.uint8)

    # 작은 점 노이즈 제거
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 외곽선 검출
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 검출 개수 초기화
    count = 0

    # 검출 면적 초기화
    total_area = 0.0

    # 각 컨투어 순회
    for cnt in contours:
        # 컨투어 면적 계산
        area = cv2.contourArea(cnt)

        # 너무 작은 노이즈 제거
        if area < 10:
            continue

        # 컨투어를 감싸는 최소 원 계산
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        # 포르피린 후보 점 크기 범위 제한
        if 3 < radius < 15:
            # 중심 좌표 정수화
            center = (int(x), int(y))

            # 결과 이미지에 빨간 원 표시
            cv2.circle(output, center, int(radius), (0, 0, 255), 2)

            # 검출 개수 증가
            count += 1

            # 검출 면적 누적
            total_area += float(area)

    # 분석 결과 폴더 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 결과 이미지 경로
    overlay_path = output_dir / "porphyrin_overlay.jpg"

    # 마스크 이미지 경로
    mask_path = output_dir / "porphyrin_mask.jpg"

    # 비교 이미지 경로
    compare_path = output_dir / "porphyrin_compare.jpg"

    # 리포트 JSON 경로
    report_path = output_dir / "porphyrin_report.json"

    # 결과 이미지 저장
    cv2.imwrite(str(overlay_path), output)

    # 마스크 이미지 저장
    cv2.imwrite(str(mask_path), thresh)

    # 원본과 결과 비교 이미지 생성
    combined = np.hstack((img, output))

    # 원본 이미지 크기 읽기
    h, w = img.shape[:2]

    # 원본 텍스트 표시
    cv2.putText(
        combined,
        "Original",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # 검출 결과 텍스트 표시
    cv2.putText(
        combined,
        "Detection",
        (w + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # 비교 이미지 저장
    cv2.imwrite(str(compare_path), combined)

    # 분석 결과 구성
    report = {
        "porphyrin_count": count,
        "porphyrin_area": total_area,
        "threshold_value": float(threshold_value),
        "overlay_path": str(overlay_path),
        "mask_path": str(mask_path),
        "compare_path": str(compare_path),
        "report_path": str(report_path)
    }

    # 리포트 JSON 저장
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 분석 결과 반환
    return report


def resolve_analysis_image_path(profile_id: str, capture_id: str, result_type: str) -> Path:
    # 프로필 루트 경로 가져오기
    profile_root = get_profile_root(profile_id)

    # 분석 결과 폴더 경로
    analysis_dir = (
        profile_root
        / CAMERA_INFO["cam4"]["folder"]
        / capture_id
        / "analysis"
    )

    # 요청 타입별 파일명 매핑
    file_map = {
        "porphyrin-overlay": "porphyrin_overlay.jpg",
        "porphyrin-mask": "porphyrin_mask.jpg",
        "porphyrin-compare": "porphyrin_compare.jpg",
    }

    # 유효하지 않은 요청이면 예외 발생
    if result_type not in file_map:
        raise ValueError("유효하지 않은 분석 이미지 타입입니다.")

    # 실제 이미지 경로 생성
    image_path = analysis_dir / file_map[result_type]

    # 이미지 존재 확인
    if not image_path.exists() or not image_path.is_file():
        raise ValueError("분석 결과 이미지가 없습니다. 먼저 포르피린 분석을 실행하세요.")

    # 이미지 경로 반환
    return image_path


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

def generate_cam4_stream():
    # 프레임 간 지연 시간 계산
    frame_delay = 1.0 / STREAM_FPS

    # 무한 스트리밍 루프
    while True:
        try:
            with camera_lock:
                # 카메라 준비 안 되었으면 잠시 대기
                if not camera_ready:
                    sleep(0.2)
                    continue

                # 전체 프레임 읽기
                full_frame_bgr = read_preview_frame()

                # cam4 부분만 잘라 복사
                cam4_frame = extract_cam_frame(full_frame_bgr, "cam4").copy()

                # JPEG 인코딩
                ok, buffer = cv2.imencode(".jpg", cam4_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # 인코딩 실패 시 다음 루프로
            if not ok:
                sleep(frame_delay)
                continue

            # 바이트 변환
            jpg_bytes = buffer.tobytes()

            # multipart 응답 데이터 yield
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpg_bytes +
                b"\r\n"
            )

            # FPS 맞추기 위한 대기
            sleep(frame_delay)

        except Exception as e:
            # 스트리밍 오류 출력
            print("[스트림 오류]", e)

            # 잠시 쉬고 재시도
            sleep(0.3)


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
        "white_led_is_on": white_led_is_on
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

        # MediaPipe 설치 여부 확인
        if face_mesh is None:
            return jsonify({
                "ok": False,
                "error": "mediapipe가 설치되어 있지 않아 자동 촬영을 사용할 수 없습니다."
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
        report = analyze_porphyrin_image(image_path, analysis_dir)

        # 성공 응답 반환
        return jsonify({
            "ok": True,
            "captureId": capture_id,
            "porphyrin_count": report["porphyrin_count"],
            "porphyrin_area": report["porphyrin_area"],
            "threshold_value": report["threshold_value"],
            "overlay_url": f"/profiles/{profile_id}/history/{capture_id}/analysis/porphyrin-overlay",
            "mask_url": f"/profiles/{profile_id}/history/{capture_id}/analysis/porphyrin-mask",
            "compare_url": f"/profiles/{profile_id}/history/{capture_id}/analysis/porphyrin-compare"
        })

    except Exception as e:
        # 실패 응답 반환
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400


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

    # Flask 서버 실행
    app.run(host="0.0.0.0", port=8000, threaded=True)
