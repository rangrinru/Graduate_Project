# Flask 서버와 JSON 응답을 위한 모듈 import
from flask import Flask, jsonify, Response, request, send_file

# CORS 허용을 위한 모듈 import
from flask_cors import CORS

# 이미지 저장/변환을 위한 OpenCV import
import cv2

# 포르피린 분석을 위한 NumPy import
import numpy as np

from porphyrin_analysis import analyze_porphyrin_heatmap_v04
from trouble_risk_analysis import analyze_trouble_risk_map
from skin_aging_analysis import analyze_skin_aging_405nm
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
    ensure_profile_dirs,
    find_profile_by_id,
    load_profiles,
    make_folder_id,
    sanitize_profile_name,
    save_profiles,
)
from capture_service import extract_cam_frame, save_image, save_one_camera_image
from camera_uc788 import run_command, unpack_y10p_to_gray8, y10p_high8_to_gray8
from history_service import (
    delete_capture_history,
    format_capture_id_to_text,
    get_capture_detail,
    get_capture_history,
    get_profile_root,
    resolve_analysis_image_path,
    resolve_image_path,
)
from auto_capture_service import (
    calculate_eye_aspect_ratio_from_points,
    clamp,
    point_distance,
)

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
    # UC-788 Rev.B raw 캡처는 Picamera2 제어를 사용하지 않습니다.
    # 노출/게인 제어가 필요하면 v4l2-ctl --list-ctrls로 지원 컨트롤명을 확인한 뒤 별도 적용해야 합니다.
    return


def exposure_ms_to_uc788_value(exposure_ms):
    if exposure_ms is None:
        return UC788_EXPOSURE

    if exposure_ms == REFERENCE_EXPOSURE_MS:
        return UC788_REFERENCE_EXPOSURE

    if exposure_ms == FLUORESCENCE_EXPOSURE_MS:
        return UC788_FLUORESCENCE_EXPOSURE

    if exposure_ms == PREVIEW_EXPOSURE_MS:
        return UC788_PREVIEW_EXPOSURE

    if INITIAL_EXPOSURE_MS <= 0:
        return max(0, int(round(exposure_ms)))

    scale = UC788_EXPOSURE / float(INITIAL_EXPOSURE_MS)
    return max(0, int(round(float(exposure_ms) * scale)))


def gain_to_uc788_value(gain):
    if gain is None:
        return UC788_ANALOGUE_GAIN

    if CURRENT_GAIN <= 0:
        return UC788_ANALOGUE_GAIN

    scale = UC788_ANALOGUE_GAIN / float(CURRENT_GAIN)
    return max(0, int(round(float(gain) * scale)))


# =========================
# UC-788 Rev.B RAW 카메라 유틸 함수
# =========================


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




def apply_uc788_controls(exposure_ms=None, gain=None, strict=False):
    # UC-788 Rev.B 직접 V4L2 캡처용 노출/게인/트리거 기본값 적용
    # /dev/v4l-subdev0 컨트롤이 순간적으로 준비되지 않을 수 있으므로 실패해도 서버는 계속 실행합니다.
    exposure_value = exposure_ms_to_uc788_value(exposure_ms)
    gain_value = gain_to_uc788_value(gain)

    commands = [
        ["v4l2-ctl", "-d", "/dev/v4l-subdev0", "-c", f"trigger_mode={UC788_TRIGGER_MODE}"],
        ["v4l2-ctl", "-d", "/dev/v4l-subdev0", "-c", f"exposure={exposure_value}"],
        ["v4l2-ctl", "-d", "/dev/v4l-subdev0", "-c", f"analogue_gain={gain_value}"],
    ]

    for command in commands:
        try:
            run_command(command, check=strict, capture_output=True)
        except Exception as e:
            if strict:
                raise RuntimeError(f"UC-788 컨트롤 적용 실패: {command}: {e}") from e
            print(f"[UC-788 컨트롤 적용 경고] {command}: {e}")


def stop_uc788_stream():
    # 전역 스트림 상태값 사용 선언
    global raw_stream_process, raw_stream_thread, raw_fifo_fd

    thread_to_join = raw_stream_thread

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

    if (
        thread_to_join is not None
        and thread_to_join.is_alive()
        and thread_to_join is not threading.current_thread()
    ):
        try:
            thread_to_join.join(timeout=1.0)
        except Exception:
            pass

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

    # 위치 확인 프리뷰는 저장용 0ms가 아니라 얼굴을 볼 수 있는 별도 노출로 유지합니다.
    apply_uc788_controls(exposure_ms=PREVIEW_EXPOSURE_MS, gain=CURRENT_GAIN)

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
        stderr=subprocess.PIPE,
        text=True,
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
        if raw_stream_process is not None and raw_stream_process.poll() is not None:
            stderr = ""
            try:
                stderr = raw_stream_process.stderr.read() if raw_stream_process.stderr else ""
            except Exception:
                stderr = ""
            raise RuntimeError(
                f"v4l2-ctl 스트림이 종료되었습니다: code={raw_stream_process.returncode}, stderr={stderr.strip()}"
            )

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
                break

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
            break


def capture_y10p_raw_bytes():
    # 부드러운 스트림에서 프레임을 직접 하나 읽습니다.
    start_uc788_stream()
    return read_exact_from_fifo(RAW_EXPECTED_BYTES, timeout_sec=2.0)


def capture_y10p_raw_bytes_direct(exposure_ms=None, gain=None, strict_controls=False):
    stop_uc788_stream()
    configure_uc788_media(force=True)
    apply_uc788_controls(exposure_ms=exposure_ms, gain=gain, strict=strict_controls)
    sleep(DIRECT_CAPTURE_EXPOSURE_SETTLE_SEC)

    try:
        RAW_FRAME_PATH.unlink(missing_ok=True)
    except Exception:
        pass

    command = [
        "v4l2-ctl",
        "-d", RAW_VIDEO_DEVICE,
        "--set-fmt-video=width=5120,height=800,pixelformat=Y10P",
        "--stream-mmap=3",
        f"--stream-skip={DIRECT_CAPTURE_STREAM_SKIP}",
        "--stream-count=1",
        f"--stream-to={RAW_FRAME_PATH}",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"v4l2-ctl 단발 캡처 실패: {result.stderr.strip()}")

    raw_bytes = RAW_FRAME_PATH.read_bytes()
    if len(raw_bytes) != RAW_EXPECTED_BYTES:
        raise RuntimeError(f"raw 크기 오류: expected={RAW_EXPECTED_BYTES}, actual={len(raw_bytes)}")

    return raw_bytes


def capture_uc788_full_frame_bgr_direct(exposure_ms, gain):
    raw_bytes = capture_y10p_raw_bytes_direct(
        exposure_ms=exposure_ms,
        gain=gain,
        strict_controls=True
    )
    gray8 = y10p_high8_to_gray8(raw_bytes)
    return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)


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

    print("[UC-788] FIFO 프리뷰 프레임을 받지 못해 단발 캡처로 확인합니다.")
    raw_bytes = capture_y10p_raw_bytes_direct()
    return y10p_high8_to_gray8(raw_bytes)


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
            full_frame_bgr = capture_uc788_full_frame_bgr_direct(exposure_ms, gain)

    finally:
        # 릴레이 OFF
        relay_off()

    # 촬영 결과 반환
    return full_frame_bgr


def capture_high_quality_full_frames_by_exposure(exposure_values_ms, gain):
    relay_on()
    sleep(RELAY_WARMUP_SEC)

    frames_by_exposure = {}

    try:
        with camera_lock:
            for exposure_ms in exposure_values_ms:
                frames_by_exposure[exposure_ms] = capture_uc788_full_frame_bgr_direct(
                    exposure_ms=exposure_ms,
                    gain=gain
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

    # UC-788은 네 카메라가 하나의 프레임으로 동작하므로 노출별 전체 프레임을 찍은 뒤 필요한 카메라 영역만 저장합니다.
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
            uc788_exposure_value=exposure_ms_to_uc788_value(exposure_ms),
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

        preview_gray = read_uc788_cam_frame_gray8(PREVIEW_CAM_KEY)

    # 세로 키오스크 화면에 맞게 영상을 서버에서 세로 방향으로 회전
    preview_gray = cv2.rotate(preview_gray, cv2.ROTATE_90_CLOCKWISE)

    if PREVIEW_MIRROR_HORIZONTAL:
        preview_gray = cv2.flip(preview_gray, 1)

    # 화면 표시용 크기로 리사이즈
    preview_gray = cv2.resize(
        preview_gray,
        (STREAM_WIDTH, STREAM_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # 브라우저에 바로 보낼 grayscale JPEG로 인코딩
    ok, buffer = cv2.imencode(
        ".jpg",
        preview_gray,
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
            # 스트리밍에서는 전체 BGR 변환을 하지 않고 1채널 gray에서 프리뷰 카메라만 잘라냅니다.
            # 이렇게 해야 15.6인치 세로 키오스크에서도 프레임이 덜 끊깁니다.
            with camera_lock:
                if not camera_ready:
                    sleep(0.05)
                    continue

                preview_gray = read_uc788_cam_frame_gray8(PREVIEW_CAM_KEY)

            # 세로 키오스크 화면에 맞게 프리뷰 영상을 서버에서 세로 방향으로 회전
            preview_gray = cv2.rotate(preview_gray, cv2.ROTATE_90_CLOCKWISE)

            if PREVIEW_MIRROR_HORIZONTAL:
                preview_gray = cv2.flip(preview_gray, 1)

            # 800x1280 원본 세로 프레임에서 720x1152로 약간만 줄여 품질과 부드러움을 균형 있게 맞춤
            preview_gray = cv2.resize(
                preview_gray,
                (STREAM_WIDTH, STREAM_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            # 흑백 이미지는 BGR로 바꾸지 않고 그대로 JPEG 인코딩합니다.
            # 브라우저는 grayscale JPEG도 정상 표시합니다.
            ok, buffer = cv2.imencode(
                ".jpg",
                preview_gray,
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
            "skin_score": report["skin_score"],
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
# 포르피린 분석 이미지 반환 API
# =========================

@app.route("/profiles/<profile_id>/history/<capture_id>/analyze-aging", methods=["POST"])
def analyze_skin_aging_api(profile_id, capture_id):
    try:
        image_path = resolve_image_path(
            profile_id=profile_id,
            capture_id=capture_id,
            filter_type="405nm_filter"
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
            / CAMERA_INFO["cam3"]["folder"]
            / capture_id
            / "analysis"
        )

        report = analyze_skin_aging_405nm(
            image_path,
            analysis_dir,
            face_reference_path,
            extract_face_landmarks_for_analysis
        )

        return jsonify({
            "ok": True,
            "captureId": capture_id,
            "freckle_count": report["freckle_count"],
            "freckle_area": report["freckle_area"],
            "freckle_area_rate_percent": report["freckle_area_rate_percent"],
            "predicted_skin_age": report["predicted_skin_age"],
            "skin_age_offset": report["skin_age_offset"],
            "skin_age_level": report["skin_age_level"],
            "grade": report["grade"],
            "label": report["label"],
            "basis": report["basis"],
            "base_age": report["base_age"],
            "threshold_value": report["threshold_value"],
            "min_area": report["min_area"],
            "max_area": report["max_area"],
            "face_detection_method": report["face_detection_method"],
            "face_landmarks_used": report["face_landmarks_used"],
            "result_url": f"/profiles/{profile_id}/history/{capture_id}/analysis/skin-aging-result",
            "mask_url": f"/profiles/{profile_id}/history/{capture_id}/analysis/skin-aging-mask"
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400

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





