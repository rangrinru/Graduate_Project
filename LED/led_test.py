from gpiozero import LED
from time import sleep

relay = LED(22, active_high=False)
<<<<<<< HEAD
relay2 = LED(17, active_high=False)

while True:
	relay2.off()
	print("relay on")
	sleep(1)
	relay2.on()
=======

while True:
	relay.on()
	print("relay on")
	sleep(1)
	relay.off()
>>>>>>> 00b1a35732c14ae8aa9c5bf4e62e81badb28c0c4
	print("relay off")
	sleep(1)

