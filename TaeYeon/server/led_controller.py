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

try:
    from gpiozero import LED, RGBLED

    GPIO_IMPORT_ERROR = None
except Exception as e:
    LED = None
    RGBLED = None
    GPIO_IMPORT_ERROR = e


class NoopLED:
    def __init__(self, *args, **kwargs):
        self.is_lit = False

    def on(self):
        self.is_lit = True

    def off(self):
        self.is_lit = False


class NoopRGBLED(NoopLED):
    def __init__(self, *args, initial_value=(0, 0, 0), **kwargs):
        super().__init__()
        self._color = initial_value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value
        self.is_lit = any(bool(channel) for channel in value)

    def off(self):
        self.color = (0, 0, 0)


def create_gpio_devices():
    if LED is None or RGBLED is None:
        print(f"[GPIO] gpiozero를 사용할 수 없어 LED 제어를 비활성화합니다: {GPIO_IMPORT_ERROR}")
        return NoopLED(), NoopRGBLED(), NoopRGBLED()

    try:
        return (
            LED(RELAY_PIN, active_high=RELAY_ACTIVE_HIGH, initial_value=False),
            RGBLED(
                red=RGB1_RED_PIN,
                green=RGB1_GREEN_PIN,
                blue=RGB1_BLUE_PIN,
                active_high=RGB_LED_ACTIVE_HIGH,
                initial_value=(0, 0, 0),
            ),
            RGBLED(
                red=RGB2_RED_PIN,
                green=RGB2_GREEN_PIN,
                blue=RGB2_BLUE_PIN,
                active_high=RGB_LED_ACTIVE_HIGH,
                initial_value=(0, 0, 0),
            ),
        )
    except Exception as e:
        print(f"[GPIO] GPIO 장치 초기화 실패로 LED 제어를 비활성화합니다: {e}")
        return NoopLED(), NoopRGBLED(), NoopRGBLED()


relay, rgb1, rgb2 = create_gpio_devices()
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
