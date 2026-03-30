from gpiozero import LED
from time import sleep

relay = LED(17, active_high=False)

while True:
	relay.on()
	sleep(1)
	relay.off()
	sleep(1)

