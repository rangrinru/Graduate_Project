from picamera2 import Picamera2
import time

picam2 = Picamera2()

# 네 카메라 최대 모드 기준
capture_config = picam2.create_still_configuration(
    main={"size": (5120, 800)}
)

picam2.configure(capture_config)

# 자동 노출 끄고 노출시간/게인 고정
picam2.set_controls({
    "AeEnable": False,
    "ExposureTime": 50000,   # 50ms = 50000us
    "AnalogueGain": 1.0
})

picam2.start()
time.sleep(2)  # 카메라 안정화
picam2.capture_file("capture_5120x800_exp50ms.jpg")
picam2.stop()