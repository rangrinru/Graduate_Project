from gpiozero import LED
from time import sleep

relay_22 = LED(22, active_high=False)
relay_17 = LED(17, active_high=False)

relay_22.on()
relay_17.off()
sleep(5)
relay_17.on()
relay_22.off()
sleep(5)
