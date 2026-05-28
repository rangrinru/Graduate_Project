from gpiozero import LED, RGBLED
from config import (
    RELAY_ACTIVE_HIGH,
    RELAY_PIN,
    RGB1_BLUE_PIN,
    RGB1_GREEN_PIN,
    RGB1_RED_PIN,
    RGB2_BLUE_PIN,
    RGB2_GREEN_PIN,
    RGB2_RED_PIN,
    RGB_LED_ACTIVE_HIGH,
    WHITE_LED_COLOR,
)

relay = LED(RELAY_PIN, active_high=RELAY_ACTIVE_HIGH, initial_value=False)
rgb1 = RGBLED(
    red=RGB1_RED_PIN,
    green=RGB1_GREEN_PIN,
    blue=RGB1_BLUE_PIN,
    active_high=RGB_LED_ACTIVE_HIGH,
    initial_value=(0, 0, 0),
)
rgb2 = RGBLED(
    red=RGB2_RED_PIN,
    green=RGB2_GREEN_PIN,
    blue=RGB2_BLUE_PIN,
    active_high=RGB_LED_ACTIVE_HIGH,
    initial_value=(0, 0, 0),
)

white_led_is_on = False


def relay_on():
    relay.on()


def relay_off():
    relay.off()


def white_led_on():
    global white_led_is_on
    rgb1.color = WHITE_LED_COLOR
    rgb2.color = WHITE_LED_COLOR
    white_led_is_on = True


def white_led_off():
    global white_led_is_on
    rgb1.off()
    rgb2.off()
    white_led_is_on = False


def get_white_led_status():
    return white_led_is_on
