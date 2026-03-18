import cv2
from picamera2 import Picamera2

cam = Picamera2(0)

config = cam.create_preview_configuration(
    main={"size": (1280, 200), "format": "XRGB8888"}
)
cam.configure(config)

cam.set_controls({
    "AeEnable": False,
    "ExposureTime": 100000,   # 100ms
    "AnalogueGain": 4.0
})

cam.start()

try:
    while True:
        frame = cam.capture_array()

        # 밝기 추가 보정
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)

        cv2.imshow("Single Camera Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()