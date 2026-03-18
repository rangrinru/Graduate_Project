from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1280, 200), "format": "RGB888"}
)
picam2.configure(config)

picam2.set_controls({
    "AeEnable": False,
    "ExposureTime": 50000,   # 50ms
    "AnalogueGain": 1.0
})

# VNC/원격 화면에서는 QT가 더 적합
picam2.start_preview(Preview.QT)
picam2.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    picam2.stop_preview()
    picam2.stop()