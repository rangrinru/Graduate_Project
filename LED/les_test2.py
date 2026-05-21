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

    # 둘 다 OFF
    rgb1.off()
    rgb2.off()
    sleep(1)