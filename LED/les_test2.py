from gpiozero import LED
from time import sleep

relay = LED(22, active_high=False)


relay.on()
sleep(5)
relay.off()
sleep(5)
