from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()

# 네 카메라(5120x800)의 비율을 유지한 축소 미리보기
config = picam2.create_preview_configuration(
    main={"size": (1280, 200), "format": "RGB888"}
)
picam2.configure(config)

# 노출시간/게인 고정
picam2.set_controls({
    "AeEnable": False,
    "ExposureTime": 50000,   # 50ms
    "AnalogueGain": 1.0
})

# 미리보기 창 시작 -> 그 다음 카메라 시작
picam2.start_preview(Preview.QTGL, width=1280, height=200)
picam2.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    picam2.stop_preview()
    picam2.stop()