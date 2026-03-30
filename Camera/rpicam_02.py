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

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 600

MIN_EXPOSURE_MS = 1
MAX_EXPOSURE_MS = 100
INITIAL_EXPOSURE_MS = 8

WINDOW_NAME = "CAM 3 + CAM 4"

SAVE_ROOT = Path.home() / "Graduate_Project" / "captures"


# =========================
# 트랙바 콜백
# =========================
def nothing(x):
    pass


# =========================
# 저장 함수
# =========================
def save_capture(full_frame_bgr, cam3_bgr, cam4_bgr, exposure_ms, gain):
    now = datetime.now()
    folder = SAVE_ROOT / now.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)

    base_name = now.strftime("%Y%m%d_%H%M%S")

    full_path = folder / f"{base_name}_full.jpg"
    cam3_path = folder / f"{base_name}_cam3.jpg"
    cam4_path = folder / f"{base_name}_cam4.jpg"
    meta_path = folder / f"{base_name}_metadata.json"

    # 이미지 저장
    cv2.imwrite(str(full_path), full_frame_bgr)
    cv2.imwrite(str(cam3_path), cam3_bgr)
    cv2.imwrite(str(cam4_path), cam4_bgr)

    metadata = {
        "captured_at": now.isoformat(),
        "camera_mode": {
            "capture_width": CAPTURE_WIDTH,
            "capture_height": CAPTURE_HEIGHT,
            "single_width": SINGLE_WIDTH
        },
        "display_mode": {
            "display_width": DISPLAY_WIDTH,
            "display_height": DISPLAY_HEIGHT
        },
        "camera_control": {
            "AeEnable": False,
            "ExposureTime_ms": exposure_ms,
            "ExposureTime_us": exposure_ms * 1000,
            "AnalogueGain": gain
        },
        "channels": {
            "cam3_region": {
                "x_start": SINGLE_WIDTH * 2,
                "x_end": SINGLE_WIDTH * 3,
                "y_start": 0,
                "y_end": CAPTURE_HEIGHT
            },
            "cam4_region": {
                "x_start": SINGLE_WIDTH * 3,
                "x_end": SINGLE_WIDTH * 4,
                "y_start": 0,
                "y_end": CAPTURE_HEIGHT
            }
        },
        "saved_files": {
            "full_image": str(full_path),
            "cam3_image": str(cam3_path),
            "cam4_image": str(cam4_path)
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"[저장 완료] {full_path}")
    print(f"[저장 완료] {cam3_path}")
    print(f"[저장 완료] {cam4_path}")
    print(f"[저장 완료] {meta_path}")


# =========================
# 카메라 확인
# =========================
camera_list = Picamera2.global_camera_info()
print("Detected cameras:", camera_list)

if len(camera_list) == 0:
    raise RuntimeError("카메라가 감지되지 않았습니다.")

cam = Picamera2(0)

config = cam.create_preview_configuration(
    main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "XRGB8888"}
)
cam.configure(config)

current_gain = 1.0

cam.set_controls({
    "AeEnable": False,
    "ExposureTime": INITIAL_EXPOSURE_MS * 1000,
    "AnalogueGain": current_gain
})

cam.start()

# =========================
# 창 / 트랙바 생성
# =========================
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH * 2, DISPLAY_HEIGHT)

cv2.createTrackbar(
    "Exposure(ms)",
    WINDOW_NAME,
    INITIAL_EXPOSURE_MS,
    MAX_EXPOSURE_MS,
    nothing
)

cv2.setTrackbarMin("Exposure(ms)", WINDOW_NAME, MIN_EXPOSURE_MS)

prev_exposure_ms = INITIAL_EXPOSURE_MS

try:
    while True:
        # -------------------------
        # ExposureTime 트랙바 값 읽기
        # -------------------------
        exposure_ms = cv2.getTrackbarPos("Exposure(ms)", WINDOW_NAME)
        if exposure_ms < MIN_EXPOSURE_MS:
            exposure_ms = MIN_EXPOSURE_MS

        if exposure_ms != prev_exposure_ms:
            cam.set_controls({
                "AeEnable": False,
                "ExposureTime": exposure_ms * 1000
            })
            prev_exposure_ms = exposure_ms

        # -------------------------
        # 전체 프레임 읽기
        # -------------------------
        frame = cam.capture_array()

        # XRGB8888 -> BGR
        full_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # -------------------------
        # cam3 / cam4 추출
        # -------------------------
        cam3 = full_frame_bgr[:, SINGLE_WIDTH * 2:SINGLE_WIDTH * 3]
        cam4 = full_frame_bgr[:, SINGLE_WIDTH * 3:SINGLE_WIDTH * 4]

        # 저장용 원본 복사
        cam3_save = cam3.copy()
        cam4_save = cam4.copy()

        # -------------------------
        # 표시용 크기 조절
        # -------------------------
        cam3_view = cv2.resize(cam3, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_CUBIC)
        cam4_view = cv2.resize(cam4, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_CUBIC)

        # -------------------------
        # 라벨 표시
        # -------------------------
        cv2.putText(cam3_view, "CAM 3", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(cam4_view, "CAM 4", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(cam3_view, f"Exp: {exposure_ms} ms", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(cam4_view, f"Exp: {exposure_ms} ms", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(cam3_view, "Press 'c' to capture", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(cam4_view, "Press 'q' to quit", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # -------------------------
        # cam3 | cam4 가로 배치
        # -------------------------
        final_view = cv2.hconcat([cam3_view, cam4_view])

        cv2.imshow(WINDOW_NAME, final_view)

        key = cv2.waitKey(1) & 0xFF

        # 촬영 및 저장
        if key == ord('c'):
            save_capture(
                full_frame_bgr=full_frame_bgr,
                cam3_bgr=cam3_save,
                cam4_bgr=cam4_save,
                exposure_ms=exposure_ms,
                gain=current_gain
            )

        # 종료
        if key == ord('q'):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()