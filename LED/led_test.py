from gpiozero import LED
from time import sleep

relay = LED(22, active_high=False)

while True:
	relay.on()
	print("relay on")
	sleep(1)
	relay.off()
	print("relay off")
	sleep(1)

