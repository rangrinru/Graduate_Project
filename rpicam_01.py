import cv2
from picamera2 import Picamera2

CAPTURE_WIDTH = 5120
CAPTURE_HEIGHT = 800

SINGLE_WIDTH = CAPTURE_WIDTH // 4
SINGLE_HEIGHT = CAPTURE_HEIGHT

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 400

cam = Picamera2(0)

config = cam.create_preview_configuration(
    main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "XRGB8888"}
)
cam.configure(config)

# 먼저 아주 보수적으로 시작
cam.set_controls({
    "AeEnable": False,
    "ExposureTime": 8000,   # 8ms
    "AnalogueGain": 1.0
})

cam.start()

try:
    while True:
        frame = cam.capture_array()

        # XRGB8888 -> BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 4개 분할
        cam1 = frame[:, 0:SINGLE_WIDTH]
        cam2 = frame[:, SINGLE_WIDTH:SINGLE_WIDTH * 2]
        cam3 = frame[:, SINGLE_WIDTH * 2:SINGLE_WIDTH * 3]
        cam4 = frame[:, SINGLE_WIDTH * 3:SINGLE_WIDTH * 4]

        # 표시용 축소/확대
        cam1 = cv2.resize(cam1, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
        cam2 = cv2.resize(cam2, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
        cam3 = cv2.resize(cam3, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
        cam4 = cv2.resize(cam4, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)

        # 라벨
        cv2.putText(cam1, "CAM 1", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(cam2, "CAM 2", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(cam3, "CAM 3", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(cam4, "CAM 4", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 2x2 배치
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