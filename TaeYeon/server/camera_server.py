# Flask 서버와 JSON 응답을 위한 모듈 import
from flask import Flask, jsonify, Response, request, send_file

# CORS 허용을 위한 모듈 import
from flask_cors import CORS

# 이미지 저장/변환을 위한 OpenCV import
import cv2

# 포르피린 분석을 위한 NumPy import
import numpy as np

from porphyrin_analysis import analyze_porphyrin_heatmap_v04

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

# 릴레이와 RGB LED 제어용 GPIO import
from gpiozero import LED, RGBLED

# 외부 명령 실행용 import
import subprocess

# UC-788 FIFO 스트리밍용 OS/select import
import os
import select
import errno

# UC-788 Rev.B는 Picamera2 RGB 프리뷰가 검게 나오는 문제가 있어
# /dev/video0의 Y10P raw를 직접 읽어서 화면용 이미지로 변환합니다.

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
# 기본 설정값
# =========================

# 전체 캡처 이미지 가로 해상도
CAPTURE_WIDTH = 5120

# 전체 캡처 이미지 세로 해상도
CAPTURE_HEIGHT = 800

# 전체 이미지가 4분할 구조라고 가정하고 단일 폭 계산
SINGLE_WIDTH = CAPTURE_WIDTH // 4

# 저장 루트 경로 설정
SAVE_ROOT = Path.home() / "Graduate_Project" / "TaeYeon" / "captures"

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
STREAM_FPS = 15

# 스트리밍용 화면 가로 크기
STREAM_WIDTH = 720

# 스트리밍용 화면 세로 크기
STREAM_HEIGHT = 1152

# 스트리밍용 JPEG 품질
STREAM_JPEG_QUALITY = 70

# 릴레이 연결 GPIO 번호
RELAY_PIN = 17

# active_high 여부
RELAY_ACTIVE_HIGH = False

# 릴레이 켠 후 안정화 대기 시간
RELAY_WARMUP_SEC = 2.4

# RGB LED 1 GPIO 번호
RGB1_RED_PIN = 27
RGB1_GREEN_PIN = 22
RGB1_BLUE_PIN = 23

# RGB LED 2 GPIO 번호
RGB2_RED_PIN = 5
RGB2_GREEN_PIN = 6
RGB2_BLUE_PIN = 13

# RGB LED active_high 여부
# 사용자가 테스트한 RGBLED 예제와 동일하게 기본 active_high=True로 사용합니다.
RGB_LED_ACTIVE_HIGH = True

# 백색 조명으로 사용할 RGB 색상값
WHITE_LED_COLOR = (1, 1, 1)

# 자동 촬영 직전 백색 LED를 끈 뒤 대기하는 시간
WHITE_LED_OFF_BEFORE_CAPTURE_SEC = 0.15

# 기본 눈 감음 EAR 기준값
DEFAULT_EYE_AR_THRESHOLD = 0.20

# rpicam_03_eye_closed_auto.py와 동일한 눈 감음 자동 촬영 대기 시간
EYES_CLOSED_DELAY_SEC = 2.0

# 자동 촬영 얼굴/눈 인식에 사용할 카메라 영역
AUTO_DETECTION_CAM_KEY = "cam2"

# MediaPipe 처리 속도를 위해 cam2 영역을 줄일 목표 폭
AUTO_DETECTION_PROCESS_WIDTH = 960

# 자동 촬영 상태 갱신 주기
AUTO_CAPTURE_INTERVAL_SEC = 0.12


# =========================
# UC-788 Rev.B / Arducam Pivariety RAW 설정
# =========================

# UC-788 Rev.B 전체 raw 가로 해상도
RAW_WIDTH = CAPTURE_WIDTH

# UC-788 Rev.B 전체 raw 세로 해상도
RAW_HEIGHT = CAPTURE_HEIGHT

# V4L2 카메라 장치 경로
RAW_VIDEO_DEVICE = "/dev/video0"

# Y10P raw 임시 저장 경로
RAW_FRAME_PATH = Path("/dev/shm/uc788_raw_frame.bin")

# Y10P 한 프레임의 정상 바이트 크기: 5120 * 800 * 10bit / 8 = 5,120,000 bytes
RAW_EXPECTED_BYTES = RAW_WIDTH * RAW_HEIGHT * 5 // 4

# 부드러운 스트리밍을 위해 v4l2-ctl이 raw를 계속 써주는 FIFO 경로
RAW_FIFO_PATH = Path("/dev/shm/uc788_y10p_fifo")

# FIFO 스트림 상태값
raw_stream_process = None
raw_stream_thread = None
raw_stream_stop_event = threading.Event()
raw_stream_lock = threading.Lock()
latest_preview_gray8 = None
latest_preview_frame_time = 0.0
raw_fifo_fd = None

# CAM4 미리보기는 백그라운드에서 최신 JPEG만 만들어두고 스트림 응답은 이를 재사용합니다.
cam4_jpeg_thread = None
cam4_jpeg_stop_event = threading.Event()
cam4_jpeg_condition = threading.Condition()
latest_cam4_jpeg = None
latest_cam4_jpeg_time = 0.0
latest_cam4_jpeg_seq = 0

# raw 프레임을 화면용 8bit로 만들 때 사용할 하위/상위 퍼센타일
RAW_NORMALIZE_LOW_PERCENTILE = 1
RAW_NORMALIZE_HIGH_PERCENTILE = 99

# UC-788 직접 V4L2 캡처용 기본 노출값
# 너무 밝으면 400~600, 너무 어두우면 1000~2000 정도로 조정하세요.
UC788_TRIGGER_MODE = 0
UC788_EXPOSURE = 800
UC788_ANALOGUE_GAIN = 100

# 현재 검색된 Arducam media device 경로
arducam_media_device = None

# media-ctl 포맷 설정 완료 여부
uc788_media_configured = False


# =========================
# 릴레이 객체 생성
# =========================

# 릴레이 LED 객체 생성
relay = LED(RELAY_PIN, active_high=RELAY_ACTIVE_HIGH, initial_value=False)

# RGB LED 1 객체 생성
# initial_value=(0, 0, 0)이므로 서버가 켜져도 처음에는 LED가 켜지지 않습니다.
rgb1 = RGBLED(
    red=RGB1_RED_PIN,
    green=RGB1_GREEN_PIN,
    blue=RGB1_BLUE_PIN,
    active_high=RGB_LED_ACTIVE_HIGH,
    initial_value=(0, 0, 0)
)

# RGB LED 2 객체 생성
# initial_value=(0, 0, 0)이므로 서버가 켜져도 처음에는 LED가 켜지지 않습니다.
rgb2 = RGBLED(
    red=RGB2_RED_PIN,
    green=RGB2_GREEN_PIN,
    blue=RGB2_BLUE_PIN,
    active_high=RGB_LED_ACTIVE_HIGH,
    initial_value=(0, 0, 0)
)

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

# 기존 코드 호환을 위해 남겨두는 자리값입니다.
# UC-788 Rev.B는 Picamera2 대신 V4L2 raw 캡처를 사용합니다.
picam2 = None
preview_config = None
still_config = None

# 자동 촬영 스레드 객체
auto_capture_thread = None

# 자동 촬영 상태 동기화용 락
auto_state_lock = threading.Lock()

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

# rpicam_03_eye_closed_auto.py의 눈 감음 타이머 상태
AUTO_EYE_STATE = {
    "blink_count": 0,
    "closed_frame_count": 0,
    "prev_eye_closed": False,
    "eyes_closed_started_at": None,
    "eye_state": "Unknown",
    "gaze_direction": "Unknown",
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

    # RGB LED 2개를 모두 흰색으로 켜기
    rgb1.color = WHITE_LED_COLOR
    rgb2.color = WHITE_LED_COLOR

    # 백색 LED 상태값 갱신
    white_led_is_on = True


def white_led_off():
    # 전역 백색 LED 상태값 사용 선언
    global white_led_is_on

    # RGB LED 2개를 모두 끄기
    rgb1.off()
    rgb2.off()

    # 백색 LED 상태값 갱신
    white_led_is_on = False


# =========================
# 카메라 수동 제어 함수
# =========================

def set_manual_controls(camera, exposure_ms, gain):
    # UC-788 Rev.B raw 캡처는 Picamera2 제어를 사용하지 않습니다.
    # 노출/게인 제어가 필요하면 v4l2-ctl --list-ctrls로 지원 컨트롤명을 확인한 뒤 별도 적용해야 합니다.
    return


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
    folder_id,
    trigger_metadata=None
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
        "capture_type": "uc788_y10p_raw_v4l2_fullframe_then_crop",
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
        "rotation_applied": "ROTATE_90_CLOCKWISE",
        "auto_capture_trigger": trigger_metadata,
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
# UC-788 Rev.B RAW 카메라 유틸 함수
# =========================

def run_command(command, check=True, capture_output=True):
    # 외부 명령을 실행하고 결과를 반환
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_output else subprocess.DEVNULL,
    )


def find_arducam_media_device():
    # /dev/media* 목록을 순회하면서 arducam-pivariety가 붙은 unicam media 장치를 찾음
    for media_path in sorted(Path("/dev").glob("media*")):
        try:
            # media topology 조회
            result = run_command(["media-ctl", "-d", str(media_path), "-p"])

            # 출력 문자열 결합
            output = (result.stdout or "") + (result.stderr or "")

            # Arducam Pivariety 센서가 있는 media 장치인지 확인
            if "arducam-pivariety" in output and "unicam" in output:
                return str(media_path)

        except Exception:
            # 접근 실패한 media 장치는 무시
            continue

    # 찾지 못하면 예외 발생
    raise RuntimeError("arducam-pivariety가 연결된 /dev/media* 장치를 찾지 못했습니다.")


def configure_uc788_media(force=False):
    # 전역 media 상태값 사용 선언
    global arducam_media_device, uc788_media_configured

    # 이미 설정되어 있고 강제 재설정이 아니면 종료
    if uc788_media_configured and not force:
        return

    # Arducam media 장치가 없으면 검색
    if arducam_media_device is None or force:
        arducam_media_device = find_arducam_media_device()

    # UC-788 Rev.B 센서 출력 포맷을 5120x800 Y10으로 설정
    run_command(
        [
            "media-ctl",
            "-d", arducam_media_device,
            "--set-v4l2",
            "'arducam-pivariety 10-000c':0 [fmt:Y10_1X10/5120x800 field:none]",
        ],
        capture_output=True,
    )

    # 설정 완료 표시
    uc788_media_configured = True




def apply_uc788_controls():
    # UC-788 Rev.B 직접 V4L2 캡처용 노출/게인/트리거 기본값 적용
    # /dev/v4l-subdev0 컨트롤이 순간적으로 준비되지 않을 수 있으므로 실패해도 서버는 계속 실행합니다.
    commands = [
        ["v4l2-ctl", "-d", "/dev/v4l-subdev0", "-c", f"trigger_mode={UC788_TRIGGER_MODE}"],
        ["v4l2-ctl", "-d", "/dev/v4l-subdev0", "-c", f"exposure={UC788_EXPOSURE}"],
        ["v4l2-ctl", "-d", "/dev/v4l-subdev0", "-c", f"analogue_gain={UC788_ANALOGUE_GAIN}"],
    ]

    for command in commands:
        try:
            run_command(command, check=False, capture_output=True)
        except Exception as e:
            print(f"[UC-788 컨트롤 적용 경고] {command}: {e}")

def y10p_high8_to_gray8(raw_bytes):
    # Y10P는 4픽셀을 5바이트에 저장합니다.
    # 화면 프리뷰는 속도를 위해 하위 2비트를 버리고 상위 8비트만 사용합니다.
    data = np.frombuffer(raw_bytes, dtype=np.uint8)

    # 크기 확인
    if data.size != RAW_EXPECTED_BYTES:
        raise RuntimeError(f"raw 크기 오류: expected={RAW_EXPECTED_BYTES}, actual={data.size}")

    # 한 줄 바이트 수 계산
    row_stride = RAW_WIDTH * 5 // 4

    # 행 단위로 재구성
    packed = data.reshape(RAW_HEIGHT, row_stride)

    # 5바이트 그룹 중 앞 4바이트가 각 픽셀의 상위 8비트입니다.
    groups = packed.reshape(RAW_HEIGHT, RAW_WIDTH // 4, 5)
    gray8 = groups[:, :, :4].reshape(RAW_HEIGHT, RAW_WIDTH).copy()

    # 화면 확인용 밝기 보정
    gray8 = cv2.convertScaleAbs(gray8, alpha=1.35, beta=0)

    return gray8


def unpack_y10p_to_gray8(raw_bytes):
    # 저장/분석용으로 10비트를 복원한 뒤 보기 좋게 정규화합니다.
    data = np.frombuffer(raw_bytes, dtype=np.uint8)

    # Y10P는 4픽셀을 5바이트에 저장하므로 5바이트 단위로 재구성
    data = data.reshape(-1, 5)

    # 5바이트에서 10비트 픽셀 4개 복원
    p0 = (data[:, 0].astype(np.uint16) << 2) | ((data[:, 4] >> 0) & 0x03)
    p1 = (data[:, 1].astype(np.uint16) << 2) | ((data[:, 4] >> 2) & 0x03)
    p2 = (data[:, 2].astype(np.uint16) << 2) | ((data[:, 4] >> 4) & 0x03)
    p3 = (data[:, 3].astype(np.uint16) << 2) | ((data[:, 4] >> 6) & 0x03)

    # 전체 10비트 이미지 배열 생성
    img10 = np.empty(RAW_WIDTH * RAW_HEIGHT, dtype=np.uint16)

    # 4픽셀씩 순서대로 배치
    img10[0::4] = p0
    img10[1::4] = p1
    img10[2::4] = p2
    img10[3::4] = p3

    # 2차원 이미지로 변환
    img10 = img10.reshape(RAW_HEIGHT, RAW_WIDTH)

    # 화면에서 잘 보이도록 퍼센타일 기반 자동 대비 계산
    low = np.percentile(img10, RAW_NORMALIZE_LOW_PERCENTILE)
    high = np.percentile(img10, RAW_NORMALIZE_HIGH_PERCENTILE)

    # 대비 계산이 불가능하면 단순 8비트 축소 사용
    if high <= low:
        return np.clip(img10 >> 2, 0, 255).astype(np.uint8)

    # 10비트 raw를 0~255 화면용 gray 이미지로 변환
    gray8 = np.clip((img10.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)

    return gray8


def stop_uc788_stream():
    # 전역 스트림 상태값 사용 선언
    global raw_stream_process, raw_stream_thread, raw_fifo_fd

    # 종료 요청
    raw_stream_stop_event.set()

    # FIFO fd 닫기
    if raw_fifo_fd is not None:
        try:
            os.close(raw_fifo_fd)
        except Exception:
            pass
        raw_fifo_fd = None

    # v4l2-ctl 프로세스 종료
    if raw_stream_process is not None:
        try:
            raw_stream_process.terminate()
            raw_stream_process.wait(timeout=1.0)
        except Exception:
            try:
                raw_stream_process.kill()
            except Exception:
                pass
        raw_stream_process = None

    # 스레드 참조 초기화
    raw_stream_thread = None


def start_uc788_stream():
    # 전역 스트림 상태값 사용 선언
    global raw_stream_process, raw_stream_thread, raw_fifo_fd, latest_preview_gray8

    # 이미 살아 있으면 종료
    if raw_stream_process is not None and raw_stream_process.poll() is None and raw_stream_thread is not None:
        return

    # 기존 스트림 정리
    stop_uc788_stream()

    # 최신 프레임 초기화
    latest_preview_gray8 = None

    # 종료 이벤트 초기화
    raw_stream_stop_event.clear()

    # media 포맷 설정
    configure_uc788_media(force=True)

    # UC-788 직접 V4L2 캡처용 노출/게인/트리거 기본값 적용
    apply_uc788_controls()

    # 기존 FIFO 삭제 후 새로 생성
    try:
        RAW_FIFO_PATH.unlink(missing_ok=True)
    except Exception:
        pass

    os.mkfifo(RAW_FIFO_PATH, 0o666)

    # 읽기 fd를 먼저 열어 v4l2-ctl writer가 막히지 않게 함
    raw_fifo_fd = os.open(str(RAW_FIFO_PATH), os.O_RDONLY | os.O_NONBLOCK)

    # v4l2-ctl이 FIFO에 raw 프레임을 연속으로 쓰도록 실행
    command = [
        "v4l2-ctl",
        "--silent",
        "-d", RAW_VIDEO_DEVICE,
        "--set-fmt-video=width=5120,height=800,pixelformat=Y10P",
        "--stream-mmap=3",
        "--stream-skip=5",
        "--stream-count=100000000",
        f"--stream-to={RAW_FIFO_PATH}",
    ]

    raw_stream_process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # FIFO reader 스레드 시작
    raw_stream_thread = threading.Thread(target=uc788_fifo_reader_loop, daemon=True)
    raw_stream_thread.start()


def read_exact_from_fifo(size, timeout_sec=2.0):
    # 전역 FIFO fd 사용
    global raw_fifo_fd

    # 수신 버퍼 생성
    chunks = []
    received = 0
    deadline = monotonic() + timeout_sec

    # 필요한 크기만큼 반복해서 읽기
    while received < size:
        remaining_time = deadline - monotonic()
        if remaining_time <= 0:
            raise TimeoutError(f"FIFO raw 프레임 수신 시간 초과: {received}/{size}")

        readable, _, _ = select.select([raw_fifo_fd], [], [], remaining_time)
        if not readable:
            continue

        try:
            chunk = os.read(raw_fifo_fd, size - received)
        except BlockingIOError:
            sleep(0.001)
            continue
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EINTR):
                sleep(0.001)
                continue
            raise

        if not chunk:
            # writer가 아직 준비 전이면 잠깐 대기
            sleep(0.002)
            continue

        chunks.append(chunk)
        received += len(chunk)

    return b"".join(chunks)


def uc788_fifo_reader_loop():
    # 전역 최신 프레임 상태값 사용 선언
    global latest_preview_gray8, latest_preview_frame_time

    # 스트림 반복
    while not raw_stream_stop_event.is_set():
        try:
            # v4l2-ctl이 죽었으면 재시작
            if raw_stream_process is None or raw_stream_process.poll() is not None:
                sleep(0.05)
                start_uc788_stream()
                continue

            # FIFO에서 raw 프레임 1장 읽기
            raw_bytes = read_exact_from_fifo(RAW_EXPECTED_BYTES, timeout_sec=2.0)

            # 전체 5120x800을 1채널 gray로만 보관합니다.
            # 기존처럼 BGR 3채널 전체 프레임으로 만들면 메모리/CPU가 3배 가까이 늘어
            # 세로 키오스크 모드에서 프레임이 툭툭 끊길 수 있습니다.
            gray8 = y10p_high8_to_gray8(raw_bytes)

            # 최신 프레임 갱신
            with raw_stream_lock:
                latest_preview_gray8 = gray8
                latest_preview_frame_time = monotonic()

        except Exception as e:
            print("[UC-788 FIFO 스트림 오류]", e)
            sleep(0.05)


def capture_y10p_raw_bytes():
    # 부드러운 스트림에서 프레임을 직접 하나 읽습니다.
    start_uc788_stream()
    return read_exact_from_fifo(RAW_EXPECTED_BYTES, timeout_sec=2.0)


def read_uc788_full_frame_gray8():
    # 스트림 시작
    start_uc788_stream()

    # 최신 gray 프레임 대기
    deadline = monotonic() + 2.0

    while monotonic() < deadline:
        with raw_stream_lock:
            if latest_preview_gray8 is not None:
                return latest_preview_gray8.copy()

        sleep(0.005)

    raise RuntimeError("UC-788 FIFO 프리뷰 프레임을 받지 못했습니다.")


def read_uc788_cam_frame_gray8(cam_key):
    # 스트리밍/자동감지에서는 전체 프레임 복사 대신 필요한 카메라 영역만 복사합니다.
    start_uc788_stream()

    info = CAMERA_INFO[cam_key]
    deadline = monotonic() + 2.0

    while monotonic() < deadline:
        with raw_stream_lock:
            if latest_preview_gray8 is not None:
                return latest_preview_gray8[:, info["x_start"]:info["x_end"]].copy()

        sleep(0.005)

    raise RuntimeError("UC-788 FIFO 프리뷰 프레임을 받지 못했습니다.")


def read_uc788_full_frame_bgr():
    # 기존 자동 촬영/저장 로직 호환용으로 필요할 때만 BGR 변환합니다.
    gray8 = read_uc788_full_frame_gray8()
    return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)


# =========================
# 카메라 초기화
# =========================

def init_camera():
    # 전역 변수 사용 선언
    global camera_ready

    # Arducam media 장치 검색 및 포맷 설정
    configure_uc788_media(force=True)

    # 테스트 raw 프레임 1장을 읽어서 카메라 상태 확인
    test_frame = read_uc788_full_frame_bgr()

    # 프레임 크기 검증
    if test_frame.shape[:2] != (CAPTURE_HEIGHT, CAPTURE_WIDTH):
        raise RuntimeError(f"카메라 프레임 크기가 올바르지 않습니다: {test_frame.shape}")

    # 카메라 준비 완료 표시
    camera_ready = True

    # 상태 출력
    print(f"[UC-788] camera ready: {RAW_VIDEO_DEVICE}, media={arducam_media_device}, frame={test_frame.shape}")


# =========================
# 프리뷰 프레임 읽기
# =========================

def read_preview_frame():
    # UC-788 Rev.B Y10P raw를 직접 읽어서 BGR 이미지로 반환
    return read_uc788_full_frame_bgr()


# =========================
# 고화질 전체 프레임 촬영
# =========================

def capture_high_quality_full_frame(exposure_ms, gain):
    # 릴레이 ON
    relay_on()

    # 릴레이 안정화 대기
    sleep(RELAY_WARMUP_SEC)

    try:
        # UC-788 Rev.B raw 기반 전체 프레임 촬영
        with camera_lock:
            full_frame_bgr = read_uc788_full_frame_bgr()

    finally:
        # 릴레이 OFF
        relay_off()

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


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


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


def point_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_eye_aspect_ratio_from_points(eye_pts):
    vertical_1 = point_distance(eye_pts[1], eye_pts[5])
    vertical_2 = point_distance(eye_pts[2], eye_pts[4])
    horizontal = point_distance(eye_pts[0], eye_pts[3])

    if horizontal <= 0:
        return 1.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


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

    # 촬영 노출 시간 설정
    exposure_ms = INITIAL_EXPOSURE_MS

    # 촬영 gain 설정
    gain = CURRENT_GAIN

    # 저장 확장자 결정
    ext = "png" if SAVE_AS_PNG else "jpg"

    # 릴레이 예열 중에는 스트리밍이 멈추지 않도록 실제 프레임 읽기 순간에만 카메라 락을 잡습니다.
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
            folder_id=folder_id,
            trigger_metadata=trigger_metadata
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
                cam2_gray = read_uc788_cam_frame_gray8(AUTO_DETECTION_CAM_KEY)

            # rpicam_03_eye_closed_auto.py와 동일하게 cam2 영역을 얼굴/눈 인식용으로 사용
            cam2_bgr = cv2.cvtColor(cam2_gray, cv2.COLOR_GRAY2BGR)

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
        "porphyrin-face-mask": "porphyrin_face_mask.jpg",
        "porphyrin-compare": "porphyrin_compare.jpg",
        "porphyrin-heatmap": "porphyrin_heatmap.jpg",
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

def build_cam4_preview_jpeg():
    # 스트리밍용 CAM4 프레임을 만들고 JPEG로 인코딩합니다.
    with camera_lock:
        if not camera_ready:
            return None

        cam4_gray = read_uc788_cam_frame_gray8("cam4")

    # 세로 키오스크 화면에 맞게 CAM4 영상을 서버에서 세로 방향으로 회전
    cam4_gray = cv2.rotate(cam4_gray, cv2.ROTATE_90_CLOCKWISE)

    # 화면 표시용 크기로 리사이즈
    cam4_gray = cv2.resize(
        cam4_gray,
        (STREAM_WIDTH, STREAM_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # 브라우저에 바로 보낼 grayscale JPEG로 인코딩
    ok, buffer = cv2.imencode(
        ".jpg",
        cam4_gray,
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
            # 스트리밍에서는 전체 BGR 변환을 하지 않고 1채널 gray에서 CAM4만 잘라냅니다.
            # 이렇게 해야 15.6인치 세로 키오스크에서도 프레임이 덜 끊깁니다.
            with camera_lock:
                if not camera_ready:
                    sleep(0.05)
                    continue

                cam4_gray = read_uc788_cam_frame_gray8("cam4")

            # 세로 키오스크 화면에 맞게 CAM4 영상을 서버에서 세로 방향으로 회전
            cam4_gray = cv2.rotate(cam4_gray, cv2.ROTATE_90_CLOCKWISE)

            # 800x1280 원본 세로 프레임에서 720x1152로 약간만 줄여 품질과 부드러움을 균형 있게 맞춤
            cam4_gray = cv2.resize(
                cam4_gray,
                (STREAM_WIDTH, STREAM_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            # 흑백 이미지는 BGR로 바꾸지 않고 그대로 JPEG 인코딩합니다.
            # 브라우저는 grayscale JPEG도 정상 표시합니다.
            ok, buffer = cv2.imencode(
                ".jpg",
                cam4_gray,
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
            face_reference_path = resolve_image_path(
                profile_id=profile_id,
                capture_id=capture_id,
                filter_type="no_filter"
            )
        except Exception:
            face_reference_path = None

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
            extract_face_landmarks_for_analysis
        )

        # 성공 응답 반환
        return jsonify({
            "ok": True,
            "captureId": capture_id,
            "porphyrin_count": report["porphyrin_count"],
            "porphyrin_area": report["porphyrin_area"],
            "detection_rate_percent": report["detection_rate_percent"],
            "face_area_pixels": report["face_area_pixels"],
            "grade": report["grade"],
            "region_analysis": report["region_analysis"],
            "threshold_percentile": report["threshold_percentile"],
            "threshold_value": report["threshold_value"],
            "min_area": report["min_area"],
            "max_area": report["max_area"],
            "face_landmarks_used": report["face_landmarks_used"],
            "heatmap_url": f"/profiles/{profile_id}/history/{capture_id}/analysis/porphyrin-heatmap"
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

    # CAM4 미리보기 JPEG를 백그라운드에서 미리 만들어 화면 지연을 줄입니다.
    start_cam4_jpeg_worker()

    # Flask 서버 실행
    app.run(host="0.0.0.0", port=8000, threaded=True)


