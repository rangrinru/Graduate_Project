from gpiozero import RGBLED
from time import sleep

# RGB LED 1
rgb1 = RGBLED(red=27, green=22, blue=23)

# RGB LED 2
rgb2 = RGBLED(red=5, green=6, blue=13)

while True:

    # 둘 다 흰색
    rgb1.color = (1, 1, 1)
    rgb2.color = (1, 1, 1)
    sleep(1)

    # 첫 번째 빨강 / 두 번째 파랑
    rgb1.color = (1, 0, 0)
    rgb2.color = (0, 0, 1)
    sleep(1)

    # 첫 번째 초록 / 두 번째 보라
    rgb1.color = (0, 1, 0)
    rgb2.color = (1, 0, 1)
    sleep(1)

    # 둘 다 OFF
    rgb1.off()
    rgb2.off()
    sleep(1)