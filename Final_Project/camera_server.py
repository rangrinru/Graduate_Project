# Flask 서버와 JSON 응답을 위한 모듈 import
from flask import Flask, jsonify, Response, request, send_file

# CORS 허용을 위한 모듈 import
from flask_cors import CORS

# 이미지 저장/변환을 위한 OpenCV import
import cv2

# JSON 파일 읽기/쓰기용 import
import json

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

# 대기 시간용 import
from time import sleep

# 릴레이 제어용 GPIO import
from gpiozero import LED

# Picamera2 import
from picamera2 import Picamera2


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


# =========================
# 릴레이 객체 생성
# =========================

# 릴레이 LED 객체 생성
relay = LED(RELAY_PIN, active_high=RELAY_ACTIVE_HIGH, initial_value=False)


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


# =========================
# 릴레이 제어 함수
# =========================

def relay_on():
    # 릴레이 켜기
    relay.on()


def relay_off():
    # 릴레이 끄기
    relay.off()


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

        # 비어 있으면 에러
        if not str(profile_id).strip():
            return jsonify({
                "ok": False,
                "error": "profileId가 필요합니다."
            }), 400

        # 프로필 찾기
        profile = find_profile_by_id(profile_id)

        # 존재하지 않으면 에러
        if profile is None:
            return jsonify({
                "ok": False,
                "error": "존재하지 않는 프로필입니다."
            }), 400

        # 표시용 이름
        profile_name = profile["name"]

        # 폴더용 ID
        folder_id = profile["folderId"]

        # 프로필 루트 경로
        profile_root = SAVE_ROOT / folder_id

        # 프로필 폴더가 없으면 에러
        if not profile_root.exists():
            return jsonify({
                "ok": False,
                "error": "프로필 폴더가 존재하지 않습니다."
            }), 400

        # 촬영 노출 시간 설정
        exposure_ms = INITIAL_EXPOSURE_MS

        # 촬영 게인 설정
        gain = CURRENT_GAIN

        # 저장 확장자 결정
        ext = "png" if SAVE_AS_PNG else "jpg"

        # 카메라 락 걸고 전체 프레임 촬영
        with camera_lock:
            full_frame_bgr = capture_high_quality_full_frame(
                exposure_ms=exposure_ms,
                gain=gain
            )

        # 촬영 시각 기록
        capture_timestamp = datetime.now()

        # 캡처 ID 생성
        capture_id = capture_timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # 저장 파일 목록
        saved_files = []

        # cam2, cam3, cam4 순서대로 저장
        for cam_key in ["cam2", "cam3", "cam4"]:
            # 개별 영역 잘라내기
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

            # 저장 결과 목록에 추가
            saved_files.append(result)

        # 촬영 성공 응답
        return jsonify({
            "ok": True,
            "captured_at": capture_timestamp.isoformat(),
            "profile_name": profile_name,
            "profile_id": folder_id,
            "capture_id": capture_id,
            "files": saved_files
        })

    except Exception as e:
        # 서버 콘솔에 오류 출력
        print("[촬영 오류]", e)

        # 실패 응답
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


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

    # 카메라 초기화
    init_camera()

    # Flask 서버 실행
    app.run(host="0.0.0.0", port=8000, threaded=True)
