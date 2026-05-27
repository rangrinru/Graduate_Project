# Flask 웹 서버를 사용하기 위한 모듈을 가져옵니다.
from flask import Flask, Response, jsonify, send_file, request

# 브라우저에서 다른 주소로 접속해도 차단되지 않도록 CORS를 가져옵니다.
from flask_cors import CORS

# 이미지 변환과 JPEG 인코딩을 위해 OpenCV를 가져옵니다.
import cv2

# 배열 형태의 이미지 처리를 위해 NumPy를 가져옵니다.
import numpy as np

# 여러 요청이 동시에 카메라를 건드리지 않도록 Lock을 가져옵니다.
import threading

# 프레임 간 대기 시간을 주기 위해 sleep을 가져옵니다.
from time import sleep

# 저장 경로 처리를 위해 Path를 가져옵니다.
from pathlib import Path

# 저장 파일명에 시간을 넣기 위해 datetime을 가져옵니다.
from datetime import datetime

# 라즈베리파이 카메라 제어를 위해 Picamera2를 가져옵니다.
from picamera2 import Picamera2


# =========================
# 기본 설정
# =========================

# Flask 앱을 생성합니다.
app = Flask(__name__)

# React나 다른 PC 브라우저에서 접근해도 허용합니다.
CORS(app)

# 기존 프로젝트에서 쓰는 전체 가로 해상도입니다.
CAPTURE_WIDTH = 5120

# 기존 프로젝트에서 쓰는 전체 세로 해상도입니다.
CAPTURE_HEIGHT = 800

# 전체 화면이 4개 카메라 화면으로 가로 분할되어 있다고 보고 단일 폭을 계산합니다.
SINGLE_WIDTH = CAPTURE_WIDTH // 4

# 웹에서 너무 큰 이미지를 바로 보내지 않도록 전체 보기용 축소 가로 크기입니다.
FULL_DISPLAY_WIDTH = 1280

# 개별 카메라 보기용 축소 가로 크기입니다.
SINGLE_DISPLAY_WIDTH = 640

# 스트리밍 JPEG 품질입니다.
JPEG_QUALITY = 75

# 카메라 테스트 이미지 저장 폴더입니다.
SAVE_DIR = Path.home() / "camera_check_output"

# 저장 폴더가 없으면 생성합니다.
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 카메라 동시 접근을 막기 위한 Lock입니다.
camera_lock = threading.Lock()

# Picamera2 객체를 저장할 전역 변수입니다.
picam2 = None

# 카메라 준비 여부를 저장합니다.
camera_ready = False

# 카메라 초기화 오류 메시지를 저장합니다.
camera_error = None

# 마지막 프레임 모양을 저장합니다.
last_frame_shape = None


# =========================
# 카메라 영역 정보
# =========================

# 전체 프레임에서 확인할 수 있는 영역 목록입니다.
CAMERA_REGION = {
    "full": {
        "label": "FULL - 5120x800 전체 화면",
        "x_start": 0,
        "x_end": CAPTURE_WIDTH,
    },
    "cam1": {
        "label": "CAM 1 - 첫 번째 영역",
        "x_start": SINGLE_WIDTH * 0,
        "x_end": SINGLE_WIDTH * 1,
    },
    "cam2": {
        "label": "CAM 2 - 기존 코드 얼굴 인식/NO FILTER 영역",
        "x_start": SINGLE_WIDTH * 1,
        "x_end": SINGLE_WIDTH * 2,
    },
    "cam3": {
        "label": "CAM 3 - 기존 코드 405nm 영역",
        "x_start": SINGLE_WIDTH * 2,
        "x_end": SINGLE_WIDTH * 3,
    },
    "cam4": {
        "label": "CAM 4 - 기존 코드 660nm 영역",
        "x_start": SINGLE_WIDTH * 3,
        "x_end": SINGLE_WIDTH * 4,
    },
}


# =========================
# 카메라 초기화 함수
# =========================

def init_camera():
    # 전역 카메라 상태값을 수정하기 위해 global을 선언합니다.
    global picam2, camera_ready, camera_error

    try:
        # 연결된 카메라 목록을 가져옵니다.
        camera_list = Picamera2.global_camera_info()

        # 카메라 목록이 비어 있으면 예외를 발생시킵니다.
        if len(camera_list) == 0:
            raise RuntimeError("Picamera2에서 감지된 카메라가 없습니다.")

        # 첫 번째 카메라 장치를 사용합니다.
        picam2 = Picamera2(0)

        # 기존 프로젝트와 같은 5120x800 프리뷰 설정을 만듭니다.
        preview_config = picam2.create_preview_configuration(
            main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "XRGB8888"}
        )

        # 프리뷰 설정을 카메라에 적용합니다.
        picam2.configure(preview_config)

        # 카메라를 시작합니다.
        picam2.start()

        # 카메라가 켜질 시간을 조금 줍니다.
        sleep(0.8)

        # 카메라 준비 완료로 표시합니다.
        camera_ready = True

        # 오류 메시지를 비웁니다.
        camera_error = None

        # 초기화 성공 로그를 출력합니다.
        print("[OK] camera started")

    except Exception as e:
        # 카메라 준비 실패로 표시합니다.
        camera_ready = False

        # 오류 메시지를 저장합니다.
        camera_error = str(e)

        # 초기화 실패 로그를 출력합니다.
        print(f"[ERROR] camera init failed: {camera_error}")


# =========================
# 프레임 변환 함수
# =========================

def normalize_to_bgr(frame):
    # Picamera2 프레임을 OpenCV에서 쓰는 BGR 형식으로 변환합니다.
    if frame is None:
        # 프레임이 비어 있으면 예외를 발생시킵니다.
        raise RuntimeError("카메라 프레임이 None입니다.")

    # 프레임이 2차원이라면 그레이스케일로 보고 BGR로 바꿉니다.
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # 프레임 채널 수를 확인합니다.
    channels = frame.shape[2]

    # XRGB8888 또는 BGRA처럼 4채널이면 BGR로 변환합니다.
    if channels == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # RGB888처럼 3채널이면 RGB에서 BGR로 변환합니다.
    if channels == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 알 수 없는 채널 수면 예외를 발생시킵니다.
    raise RuntimeError(f"지원하지 않는 프레임 채널 수입니다: {channels}")


# =========================
# 프레임 읽기 함수
# =========================

def read_full_frame_bgr():
    # 전역 마지막 프레임 모양을 수정하기 위해 global을 선언합니다.
    global last_frame_shape

    # 카메라가 준비되지 않았으면 예외를 발생시킵니다.
    if not camera_ready or picam2 is None:
        raise RuntimeError(f"카메라가 준비되지 않았습니다: {camera_error}")

    # 카메라 동시 접근을 막기 위해 Lock을 잡습니다.
    with camera_lock:
        # Picamera2에서 현재 프레임을 가져옵니다.
        frame = picam2.capture_array()

    # 마지막 프레임 모양을 저장합니다.
    last_frame_shape = tuple(frame.shape)

    # 프레임을 BGR로 변환해서 반환합니다.
    return normalize_to_bgr(frame)


# =========================
# 영역 자르기 함수
# =========================

def crop_region(full_bgr, cam_key):
    # 요청한 cam_key가 없으면 full로 처리합니다.
    if cam_key not in CAMERA_REGION:
        cam_key = "full"

    # 선택한 영역 정보를 가져옵니다.
    region = CAMERA_REGION[cam_key]

    # full이면 전체 이미지를 반환합니다.
    if cam_key == "full":
        return full_bgr

    # 이미지의 실제 높이와 폭을 가져옵니다.
    h, w = full_bgr.shape[:2]

    # 요청 영역의 시작 x좌표를 실제 폭 안으로 제한합니다.
    x1 = max(0, min(w, region["x_start"]))

    # 요청 영역의 끝 x좌표를 실제 폭 안으로 제한합니다.
    x2 = max(0, min(w, region["x_end"]))

    # 영역 폭이 잘못되었으면 전체 이미지를 반환합니다.
    if x2 <= x1:
        return full_bgr

    # 해당 영역만 잘라서 반환합니다.
    return full_bgr[:, x1:x2]


# =========================
# 웹 표시용 리사이즈 함수
# =========================

def resize_for_web(img_bgr, cam_key):
    # 이미지 높이와 폭을 가져옵니다.
    h, w = img_bgr.shape[:2]

    # 전체 화면이면 전체 보기용 폭을 사용합니다.
    target_width = FULL_DISPLAY_WIDTH if cam_key == "full" else SINGLE_DISPLAY_WIDTH

    # 원본 폭이 목표 폭보다 작거나 같으면 그대로 반환합니다.
    if w <= target_width:
        return img_bgr

    # 비율을 유지하기 위한 목표 높이를 계산합니다.
    target_height = int(h * (target_width / w))

    # 이미지를 리사이즈해서 반환합니다.
    return cv2.resize(img_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


# =========================
# JPEG 인코딩 함수
# =========================

def encode_jpeg(img_bgr):
    # 이미지를 JPEG 바이트로 인코딩합니다.
    ok, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    # 인코딩에 실패하면 예외를 발생시킵니다.
    if not ok:
        raise RuntimeError("JPEG 인코딩 실패")

    # JPEG 바이트를 반환합니다.
    return buffer.tobytes()


# =========================
# 상태 확인 API
# =========================

@app.route("/status")
def status():
    # Picamera2가 보는 카메라 목록을 안전하게 가져옵니다.
    try:
        # 카메라 목록을 조회합니다.
        camera_list = Picamera2.global_camera_info()
    except Exception as e:
        # 조회 실패 시 오류 문자열로 저장합니다.
        camera_list = f"camera info error: {e}"

    # 현재 상태를 JSON으로 반환합니다.
    return jsonify({
        "ok": True,
        "camera_ready": camera_ready,
        "camera_error": camera_error,
        "camera_list": camera_list,
        "last_frame_shape": last_frame_shape,
        "capture_width": CAPTURE_WIDTH,
        "capture_height": CAPTURE_HEIGHT,
        "single_width": SINGLE_WIDTH,
        "available_views": list(CAMERA_REGION.keys()),
    })


# =========================
# 단일 이미지 확인 API
# =========================

@app.route("/snapshot")
def snapshot():
    try:
        # URL에서 cam 값을 읽습니다.
        cam_key = request.args.get("cam", "full")

        # 전체 프레임을 읽습니다.
        full_bgr = read_full_frame_bgr()

        # 요청한 영역만 자릅니다.
        view_bgr = crop_region(full_bgr, cam_key)

        # 웹 표시용으로 리사이즈합니다.
        view_bgr = resize_for_web(view_bgr, cam_key)

        # JPEG로 인코딩합니다.
        jpg_bytes = encode_jpeg(view_bgr)

        # JPEG 응답을 반환합니다.
        return Response(jpg_bytes, mimetype="image/jpeg")

    except Exception as e:
        # 오류 내용을 JSON으로 반환합니다.
        return jsonify({"ok": False, "error": str(e)}), 500


# =========================
# 테스트 이미지 저장 API
# =========================

@app.route("/save-test")
def save_test():
    try:
        # 전체 프레임을 읽습니다.
        full_bgr = read_full_frame_bgr()

        # 현재 시간을 파일명으로 사용할 문자열로 만듭니다.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 저장 결과 목록을 만듭니다.
        saved = []

        # full, cam1, cam2, cam3, cam4를 각각 저장합니다.
        for cam_key in CAMERA_REGION.keys():
            # 해당 영역을 자릅니다.
            img = crop_region(full_bgr, cam_key)

            # 저장 파일 경로를 만듭니다.
            path = SAVE_DIR / f"{stamp}_{cam_key}.jpg"

            # 이미지를 저장합니다.
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 저장 경로를 결과에 추가합니다.
            saved.append(str(path))

        # 저장 결과를 JSON으로 반환합니다.
        return jsonify({"ok": True, "saved": saved})

    except Exception as e:
        # 오류 내용을 JSON으로 반환합니다.
        return jsonify({"ok": False, "error": str(e)}), 500


# =========================
# 실시간 스트리밍 생성 함수
# =========================

def generate_stream(cam_key):
    # 무한 반복으로 MJPEG 프레임을 생성합니다.
    while True:
        try:
            # 전체 프레임을 읽습니다.
            full_bgr = read_full_frame_bgr()

            # 요청한 카메라 영역을 자릅니다.
            view_bgr = crop_region(full_bgr, cam_key)

            # 웹 표시용으로 리사이즈합니다.
            view_bgr = resize_for_web(view_bgr, cam_key)

            # JPEG로 인코딩합니다.
            jpg_bytes = encode_jpeg(view_bgr)

            # MJPEG 형식으로 한 프레임을 전송합니다.
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )

            # 너무 빠르게 돌지 않도록 잠시 쉽니다.
            sleep(0.05)

        except GeneratorExit:
            # 브라우저가 연결을 끊으면 반복을 종료합니다.
            break

        except Exception as e:
            # 오류가 나면 로그를 출력합니다.
            print(f"[STREAM ERROR] {e}")

            # 오류가 반복되지 않도록 잠시 쉽니다.
            sleep(0.5)


# =========================
# 실시간 스트리밍 API
# =========================

@app.route("/stream")
def stream():
    # URL에서 cam 값을 읽습니다.
    cam_key = request.args.get("cam", "full")

    # 없는 cam 값이면 full로 바꿉니다.
    if cam_key not in CAMERA_REGION:
        cam_key = "full"

    # MJPEG 스트리밍 응답을 반환합니다.
    return Response(
        generate_stream(cam_key),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =========================
# 브라우저 확인용 메인 페이지
# =========================

@app.route("/")
def index():
    # 브라우저에서 바로 확인할 수 있는 HTML을 반환합니다.
    return """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Camera Check</title>
  <style>
    body { margin: 0; background: #0b1220; color: white; font-family: Arial, sans-serif; }
    header { padding: 18px; background: #111827; position: sticky; top: 0; z-index: 10; }
    h1 { margin: 0 0 8px 0; font-size: 22px; }
    p { margin: 4px 0; color: #cbd5e1; }
    .links { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    a, button { background: #22d3ee; color: #0f172a; border: 0; border-radius: 10px; padding: 10px 12px; font-weight: 800; text-decoration: none; cursor: pointer; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 16px; padding: 16px; }
    .card { background: #111827; border: 1px solid rgba(255,255,255,.12); border-radius: 16px; padding: 12px; }
    .card h2 { margin: 0 0 10px 0; font-size: 18px; }
    img { width: 100%; background: black; border-radius: 12px; display: block; }
    pre { white-space: pre-wrap; color: #e5e7eb; background: #020617; padding: 12px; border-radius: 12px; overflow: auto; }
  </style>
</head>
<body>
  <header>
    <h1>Picamera2 Camera Check</h1>
    <p>기존 서버/GPIO/MediaPipe 없이 카메라 프레임만 확인하는 페이지입니다.</p>
    <div class="links">
      <a href="/status" target="_blank">상태 JSON</a>
      <a href="/save-test" target="_blank">테스트 이미지 저장</a>
      <a href="/snapshot?cam=full" target="_blank">전체 스냅샷</a>
      <a href="/snapshot?cam=cam1" target="_blank">CAM1 스냅샷</a>
      <a href="/snapshot?cam=cam2" target="_blank">CAM2 스냅샷</a>
      <a href="/snapshot?cam=cam3" target="_blank">CAM3 스냅샷</a>
      <a href="/snapshot?cam=cam4" target="_blank">CAM4 스냅샷</a>
    </div>
  </header>
  <main class="grid">
    <section class="card">
      <h2>FULL</h2>
      <img src="/stream?cam=full" />
    </section>
    <section class="card">
      <h2>CAM1</h2>
      <img src="/stream?cam=cam1" />
    </section>
    <section class="card">
      <h2>CAM2</h2>
      <img src="/stream?cam=cam2" />
    </section>
    <section class="card">
      <h2>CAM3</h2>
      <img src="/stream?cam=cam3" />
    </section>
    <section class="card">
      <h2>CAM4</h2>
      <img src="/stream?cam=cam4" />
    </section>
  </main>
</body>
</html>
"""


# =========================
# 서버 실행부
# =========================

if __name__ == "__main__":
    # 서버 시작 전에 카메라를 초기화합니다.
    init_camera()

    # Flask 서버를 8010번 포트로 실행합니다.
    app.run(host="0.0.0.0", port=8010, threaded=True)
