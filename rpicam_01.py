import cv2
from picamera2 import Picamera2

# =========================
# 기본 설정
# =========================
CAPTURE_WIDTH = 5120
CAPTURE_HEIGHT = 800

SINGLE_WIDTH = CAPTURE_WIDTH // 4   # 1280

# 각 화면 표시 크기
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 600

MIN_EXPOSURE_MS = 1
MAX_EXPOSURE_MS = 100
INITIAL_EXPOSURE_MS = 8

WINDOW_NAME = "CAM 3 + CAM 4"

# =========================
# 트랙바 콜백 함수
# =========================
def nothing(x):
    pass

# =========================
# 카메라 확인
# =========================
camera_list = Picamera2.global_camera_info()
print("Detected cameras:", camera_list)

if len(camera_list) == 0:
    raise RuntimeError("카메라가 감지되지 않았습니다.")

# 집계된 카메라 1개만 사용
cam = Picamera2(0)

config = cam.create_preview_configuration(
    main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "XRGB8888"}
)
cam.configure(config)

# 자동노출 끄고 시작값 적용
cam.set_controls({
    "AeEnable": False,
    "ExposureTime": INITIAL_EXPOSURE_MS * 1000,   # us 단위
    "AnalogueGain": 1.0
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
        # 트랙바 값 읽기
        # -------------------------
        exposure_ms = cv2.getTrackbarPos("Exposure(ms)", WINDOW_NAME)
        if exposure_ms < MIN_EXPOSURE_MS:
            exposure_ms = MIN_EXPOSURE_MS

        # 값이 바뀐 경우만 적용
        if exposure_ms != prev_exposure_ms:
            cam.set_controls({
                "AeEnable": False,
                "ExposureTime": exposure_ms * 1000   # ms -> us
            })
            prev_exposure_ms = exposure_ms

        # -------------------------
        # 전체 프레임 읽기
        # -------------------------
        frame = cam.capture_array()

        # XRGB8888 -> BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # -------------------------
        # cam3 / cam4 영역만 추출
        # cam1:   0 ~ 1279
        # cam2: 1280 ~ 2559
        # cam3: 2560 ~ 3839
        # cam4: 3840 ~ 5119
        # -------------------------
        cam3 = frame[:, SINGLE_WIDTH * 2:SINGLE_WIDTH * 3]
        cam4 = frame[:, SINGLE_WIDTH * 3:SINGLE_WIDTH * 4]

        # -------------------------
        # 표시용 크기 조절
        # -------------------------
        cam3 = cv2.resize(cam3, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_CUBIC)
        cam4 = cv2.resize(cam4, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_CUBIC)

        # -------------------------
        # 라벨 표시
        # -------------------------
        cv2.putText(cam3, "CAM 3", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(cam4, "CAM 4", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(cam3, f"Exp: {exposure_ms} ms", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(cam4, f"Exp: {exposure_ms} ms", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # -------------------------
        # cam3 | cam4 가로 배치
        # -------------------------
        final_view = cv2.hconcat([cam3, cam4])

        cv2.imshow(WINDOW_NAME, final_view)

        # q 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()