import cv2
from picamera2 import Picamera2

# -------------------------------
# 카메라 번호 설정
# 0 = 405 필터 카메라
# 1 = 660 필터 카메라
# 2 = 필터 없는 카메라
# -------------------------------
cam405 = Picamera2(0)
cam660 = Picamera2(1)
camNo  = Picamera2(2)

# 화면 비율에 맞춘 미리보기 해상도
preview_size = (1280, 200)

# 공통 설정
config405 = cam405.create_preview_configuration(
    main={"size": preview_size, "format": "RGB888"}
)
config660 = cam660.create_preview_configuration(
    main={"size": preview_size, "format": "RGB888"}
)
configNo = camNo.create_preview_configuration(
    main={"size": preview_size, "format": "RGB888"}
)

cam405.configure(config405)
cam660.configure(config660)
camNo.configure(configNo)

# -------------------------------
# 밝기 설정
# ExposureTime 단위 = us(마이크로초)
# 100000 = 100ms
# -------------------------------

# 405 필터 카메라: 가장 어두우므로 더 강하게 밝게
cam405.set_controls({
    "AeEnable": False,
    "ExposureTime": 120000,   # 120ms
    "AnalogueGain": 6.0
})

# 660 필터 카메라: 중간 정도로 밝게
cam660.set_controls({
    "AeEnable": False,
    "ExposureTime": 80000,    # 80ms
    "AnalogueGain": 4.0
})

# 필터 없는 카메라: 기준 화면
camNo.set_controls({
    "AeEnable": False,
    "ExposureTime": 20000,    # 20ms
    "AnalogueGain": 1.0
})

cam405.start()
cam660.start()
camNo.start()

def brighten_frame(frame, alpha=1.0, beta=0):
    """
    alpha: 대비(contrast)
    beta : 밝기(brightness)
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

try:
    while True:
        # 프레임 읽기
        frame405 = cam405.capture_array()
        frame660 = cam660.capture_array()
        frameNo  = camNo.capture_array()

        # RGB -> BGR (OpenCV 표시용)
        frame405 = cv2.cvtColor(frame405, cv2.COLOR_RGB2BGR)
        frame660 = cv2.cvtColor(frame660, cv2.COLOR_RGB2BGR)
        frameNo  = cv2.cvtColor(frameNo,  cv2.COLOR_RGB2BGR)

        # -------------------------------
        # 추가 밝기 보정
        # 405 / 660만 더 밝게
        # -------------------------------
        frame405 = brighten_frame(frame405, alpha=1.25, beta=35)
        frame660 = brighten_frame(frame660, alpha=1.15, beta=20)

        # 화면 합치기
        combined = cv2.hconcat([frame405, frame660, frameNo])

        cv2.imshow("405 / 660 / No Filter", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cam405.stop()
    cam660.stop()
    camNo.stop()
    cv2.destroyAllWindows()