import cv2
import json
from pathlib import Path
from datetime import datetime
from picamera2 import Picamera2

# =========================
# 기본 설정
# =========================
CAPTURE_WIDTH = 5120
CAPTURE_HEIGHT = 800
SINGLE_WIDTH = CAPTURE_WIDTH // 4   # 1280

# 미리보기용 표시 크기
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 400

# 개별 촬영 시 단독 표시 크기
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
# 트랙바 콜백
# =========================
def nothing(x):
    pass


# =========================
# 공통 수동 제어 적용
# =========================
def set_manual_controls(camera, exposure_ms, gain):
    camera.set_controls({
        "AeEnable": False,
        "ExposureTime": exposure_ms * 1000,   # ms -> us
        "AnalogueGain": gain
    })


# =========================
# 프레임 추출
# =========================
def extract_cam_frame(full_frame_bgr, cam_key):
    info = CAMERA_INFO[cam_key]
    return full_frame_bgr[:, info["x_start"]:info["x_end"]]


# =========================
# 텍스트 표시
# =========================
def draw_text(img, title, exposure_ms, extra_text=None):
    out = img.copy()

    cv2.putText(
        out, title, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (255, 255, 255), 2
    )
    cv2.putText(
        out, f"Exposure: {exposure_ms} ms", (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2
    )

    if extra_text is not None:
        cv2.putText(
            out, extra_text, (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2
        )

    return out


# =========================
# 이미지 저장
# =========================
def save_image(path, img_bgr):
    if path.suffix.lower() == ".png":
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])


# =========================
# 개별 카메라 이미지 + metadata 저장
# =========================
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


# =========================
# 단독 출력
# =========================
def show_single_camera(window_name, frame_bgr, cam_key, exposure_ms, hold_ms=700):
    info = CAMERA_INFO[cam_key]

    single_view = cv2.resize(
        frame_bgr,
        (SINGLE_VIEW_WIDTH, SINGLE_VIEW_HEIGHT),
        interpolation=cv2.INTER_CUBIC
    )

    single_view = draw_text(
        single_view,
        info["label"],
        exposure_ms,
        extra_text="Capturing..."
    )

    cv2.imshow(window_name, single_view)
    cv2.waitKey(hold_ms)


# =========================
# 고화질 전체 프레임 캡처
# =========================
def capture_high_quality_full_frame(cam, preview_config, still_config, exposure_ms, gain):
    """
    미리보기 모드 -> 고화질 still 모드 -> 전체 프레임 캡처 -> 다시 미리보기 모드 복귀
    """
    # still 모드로 전환
    cam.stop()
    cam.configure(still_config)
    cam.start()
    set_manual_controls(cam, exposure_ms, gain)

    still_frame = cam.capture_array()   # RGB888 예상
    full_frame_bgr = cv2.cvtColor(still_frame, cv2.COLOR_RGB2BGR)

    # 다시 preview 모드로 복귀
    cam.stop()
    cam.configure(preview_config)
    cam.start()
    set_manual_controls(cam, exposure_ms, gain)

    return full_frame_bgr


# =========================
# 순차 촬영
# Cam2 -> Cam3 -> Cam4
# =========================
def capture_sequence(cam, preview_config, still_config, exposure_ms, gain, save_as_png=True):
    """
    고화질 전체 프레임 1장을 still 모드로 캡처한 뒤,
    Cam2 -> Cam3 -> Cam4 순서로 단독 출력하고 각각 다른 폴더에 저장
    """
    ext = "png" if save_as_png else "jpg"

    # 집계 카메라 구조상 전체 프레임 1장을 고화질로 받아서 crop
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

        # 단독 표시
        show_single_camera(WINDOW_NAME, target_frame, cam_key, exposure_ms, hold_ms=700)

        # 개별 저장
        save_one_camera_image(
            cam_key=cam_key,
            frame_bgr=target_frame,
            timestamp=capture_timestamp,
            exposure_ms=exposure_ms,
            gain=gain,
            ext=ext
        )


# =========================
# 카메라 확인
# =========================
camera_list = Picamera2.global_camera_info()
print("Detected cameras:", camera_list)

if len(camera_list) == 0:
    raise RuntimeError("카메라가 감지되지 않았습니다.")

cam = Picamera2(0)

# =========================
# 설정 분리
# preview용 / still용
# =========================
preview_config = cam.create_preview_configuration(
    main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "XRGB8888"}
)

still_config = cam.create_still_configuration(
    main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
)

cam.configure(preview_config)

# Picamera2 저장 옵션
cam.options["quality"] = 100
cam.options["compress_level"] = 0

set_manual_controls(cam, INITIAL_EXPOSURE_MS, CURRENT_GAIN)
cam.start()

# =========================
# 창 / 트랙바 생성
# =========================
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, PREVIEW_WIDTH * 3, PREVIEW_HEIGHT)

cv2.createTrackbar(
    "Exposure(ms)",
    WINDOW_NAME,
    INITIAL_EXPOSURE_MS,
    MAX_EXPOSURE_MS,
    nothing
)

prev_exposure_ms = INITIAL_EXPOSURE_MS

try:
    while True:
        # -------------------------
        # ExposureTime 읽기
        # -------------------------
        exposure_ms = cv2.getTrackbarPos("Exposure(ms)", WINDOW_NAME)
        if exposure_ms < MIN_EXPOSURE_MS:
            exposure_ms = MIN_EXPOSURE_MS

        if exposure_ms != prev_exposure_ms:
            set_manual_controls(cam, exposure_ms, CURRENT_GAIN)
            prev_exposure_ms = exposure_ms

        # -------------------------
        # preview 프레임 읽기
        # -------------------------
        frame = cam.capture_array()
        full_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # -------------------------
        # Cam2 / Cam3 / Cam4 추출
        # -------------------------
        cam2 = extract_cam_frame(full_frame_bgr, "cam2")
        cam3 = extract_cam_frame(full_frame_bgr, "cam3")
        cam4 = extract_cam_frame(full_frame_bgr, "cam4")

        # -------------------------
        # 표시용 크기 조절
        # -------------------------
        cam2_view = cv2.resize(cam2, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
        cam3_view = cv2.resize(cam3, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
        cam4_view = cv2.resize(cam4, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)

        # -------------------------
        # 라벨 / 안내문 표시
        # -------------------------
        cam2_view = draw_text(cam2_view, "CAM 2 - NO FILTER", exposure_ms)
        cam3_view = draw_text(cam3_view, "CAM 3 - 405nm FILTER", exposure_ms)
        cam4_view = draw_text(cam4_view, "CAM 4 - 660nm FILTER", exposure_ms)

        cv2.putText(cam2_view, "Press 'c' to capture sequence", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(cam3_view, "Cam2 -> Cam3 -> Cam4", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(cam4_view, "Press 'q' to quit", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # -------------------------
        # 3개 가로 출력
        # -------------------------
        preview = cv2.hconcat([cam2_view, cam3_view, cam4_view])
        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(1) & 0xFF

        # -------------------------
        # 고화질 촬영 시퀀스 실행
        # -------------------------
        if key == ord('c'):
            capture_sequence(
                cam=cam,
                preview_config=preview_config,
                still_config=still_config,
                exposure_ms=exposure_ms,
                gain=CURRENT_GAIN,
                save_as_png=SAVE_AS_PNG
            )

        # 종료
        if key == ord('q'):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()