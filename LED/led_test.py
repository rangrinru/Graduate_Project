from gpiozero import LED
from time import sleep

relay2 = LED(17, active_high=False)

while True:
	relay2.off()
	print("relay on")
	sleep(1)
	relay2.on()
	print("relay off")
	sleep(1)

