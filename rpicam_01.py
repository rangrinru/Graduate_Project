import cv2
import numpy as np
from picamera2 import Picamera2

# =========================
# 사용자 설정
# =========================

# 집계 프레임 전체 크기 (4카메라 합쳐진 입력)
CAPTURE_WIDTH = 5120
CAPTURE_HEIGHT = 800

# 각 카메라 원본 조각 크기
SINGLE_WIDTH = CAPTURE_WIDTH // 4   # 1280
SINGLE_HEIGHT = CAPTURE_HEIGHT      # 800

# 화면에 표시할 각 카메라 크기
# 너무 크게 하면 무거워질 수 있으니 적당히 조절
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 600

# 공통 카메라 설정
EXPOSURE_TIME = 50000    # us, 50000 = 50ms
ANALOG_GAIN = 2.0

# 화면 보정 파라미터
ENABLE_CLAHE = True
ENABLE_SHARPEN = True
BRIGHTNESS_BETA = 10     # 밝기 증가
CONTRAST_ALPHA = 1.1     # 대비 증가

# =========================
# 보정 함수들
# =========================

def improve_image(frame):
    """
    보기 좋게 만드는 후처리:
    1) 밝기/대비 보정
    2) CLAHE로 지역 대비 향상
    3) 샤프닝
    """
    # 1. 밝기/대비 보정
    frame = cv2.convertScaleAbs(frame, alpha=CONTRAST_ALPHA, beta=BRIGHTNESS_BETA)

    # 2. CLAHE (흑백 카메라라 대비 향상에 도움)
    if ENABLE_CLAHE:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 3. 샤프닝
    if ENABLE_SHARPEN:
        kernel = np.array([
            [0, -1,  0],
            [-1, 5, -1],
            [0, -1,  0]
        ], dtype=np.float32)
        frame = cv2.filter2D(frame, -1, kernel)

    return frame


def resize_for_display(frame):
    """
    표시용 업스케일
    """
    return cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_CUBIC)


def put_label(img, text):
    """
    각 화면에 라벨 표시
    """
    cv2.putText(
        img, text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    return img


# =========================
# 카메라 초기화
# =========================

cam = Picamera2(0)

config = cam.create_preview_configuration(
    main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "XRGB8888"}
)
cam.configure(config)

cam.set_controls({
    "AeEnable": False,
    "ExposureTime": EXPOSURE_TIME,
    "AnalogueGain": ANALOG_GAIN
})

cam.start()

# =========================
# 메인 루프
# =========================

try:
    while True:
        frame = cam.capture_array()

        # XRGB8888 -> BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # -------------------------
        # 4등분 (좌 -> 우)
        # -------------------------
        cam1 = frame[:, 0:SINGLE_WIDTH]
        cam2 = frame[:, SINGLE_WIDTH:SINGLE_WIDTH * 2]
        cam3 = frame[:, SINGLE_WIDTH * 2:SINGLE_WIDTH * 3]
        cam4 = frame[:, SINGLE_WIDTH * 3:SINGLE_WIDTH * 4]

        # -------------------------
        # 각 화면 보정
        # -------------------------
        cam1 = improve_image(cam1)
        cam2 = improve_image(cam2)
        cam3 = improve_image(cam3)
        cam4 = improve_image(cam4)

        # -------------------------
        # 각 화면 크기 확대
        # -------------------------
        cam1 = resize_for_display(cam1)
        cam2 = resize_for_display(cam2)
        cam3 = resize_for_display(cam3)
        cam4 = resize_for_display(cam4)

        # -------------------------
        # 라벨 표시
        # -------------------------
        cam1 = put_label(cam1, "CAM 1")
        cam2 = put_label(cam2, "CAM 2")
        cam3 = put_label(cam3, "CAM 3")
        cam4 = put_label(cam4, "CAM 4")

        # -------------------------
        # 2x2 배치
        # 위: cam1 cam2
        # 아래: cam3 cam4
        # -------------------------
        top_row = cv2.hconcat([cam1, cam2])
        bottom_row = cv2.hconcat([cam3, cam4])
        final_view = cv2.vconcat([top_row, bottom_row])

        cv2.imshow("4 Camera 2x2 View", final_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()