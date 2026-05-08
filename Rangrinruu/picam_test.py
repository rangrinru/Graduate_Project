from picamera2 import Picamera2
import cv2
import numpy as np

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (5120, 800), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()

    h, w = frame.shape[:2]
    sub_w = w // 4

    cam1 = frame[:, 0:sub_w]
    cam2 = frame[:, sub_w:sub_w*2]
    cam3 = frame[:, sub_w*2:sub_w*3]
    cam4 = frame[:, sub_w*3:sub_w*4]

    # 예시: cam2만 보기
    display = cam2.copy()

    cv2.imshow("Cam2", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()