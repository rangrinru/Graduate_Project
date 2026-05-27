# camera_preview_only_debug.py
# 원래 미리보기 방식(XRGB8888)만 사용해서 검은 화면 원인을 확인하는 코드
# - raw/still 설정을 쓰지 않음
# - 릴레이는 이 코드 안에서도 켤 수 있음
# - 프레임 통계(min/max/mean)를 출력하고 jpg 저장

import argparse
from pathlib import Path
from datetime import datetime
from time import sleep

import cv2
import numpy as np
from picamera2 import Picamera2

try:
    from gpiozero import LED
except Exception:
    LED = None


CAPTURE_WIDTH = 5120
CAPTURE_HEIGHT = 800
SAVE_ROOT = Path.home() / "camera_preview_only_debug"


def str_to_bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def normalize_to_uint8(img):
    arr = img.astype(np.float32)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))

    if max_v <= min_v:
        return np.zeros(arr.shape[:2], dtype=np.uint8)

    arr = (arr - min_v) * 255.0 / (max_v - min_v)
    return np.clip(arr, 0, 255).astype(np.uint8)


def print_stats(label, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(
        f"[{label}] "
        f"min={int(np.min(gray))}, "
        f"max={int(np.max(gray))}, "
        f"mean={float(np.mean(gray)):.2f}, "
        f"std={float(np.std(gray)):.2f}"
    )

    return gray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-on", action="store_true")
    parser.add_argument("--relay-pin", type=int, default=17)
    parser.add_argument("--relay-active-high", type=str, default="false")
    parser.add_argument("--exposure-ms", type=int, default=200)
    parser.add_argument("--gain", type=float, default=15.9)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--delay-sec", type=float, default=0.2)
    args = parser.parse_args()

    save_dir = SAVE_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    relay = None
    picam2 = None

    try:
        if args.relay_on:
            if LED is None:
                raise RuntimeError("gpiozero import 실패")
            active_high = str_to_bool(args.relay_active_high)
            relay = LED(args.relay_pin, active_high=active_high, initial_value=False)
            print(f"[릴레이] GPIO{args.relay_pin} ON / active_high={active_high}")
            relay.on()
            sleep(1.0)

        print("[카메라] 목록")
        print(Picamera2.global_camera_info())

        picam2 = Picamera2(0)

        preview_config = picam2.create_preview_configuration(
            main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "XRGB8888"}
        )

        picam2.configure(preview_config)
        picam2.start()

        picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": int(args.exposure_ms * 1000),
            "AnalogueGain": float(args.gain),
        })

        print(f"[카메라] exposure={args.exposure_ms}ms, gain={args.gain}")
        sleep(1.0)

        last_bgr = None

        for i in range(args.frames):
            frame = picam2.capture_array()

            if frame.ndim == 3 and frame.shape[2] == 4:
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                raise RuntimeError(f"알 수 없는 프레임 형태: {frame.shape}, {frame.dtype}")

            last_bgr = bgr
            print_stats(f"frame {i+1}", bgr)
            sleep(args.delay_sec)

        if last_bgr is None:
            raise RuntimeError("프레임을 받지 못했습니다.")

        original_path = save_dir / "preview_original.jpg"
        cv2.imwrite(str(original_path), last_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        gray = cv2.cvtColor(last_bgr, cv2.COLOR_BGR2GRAY)
        auto = normalize_to_uint8(gray)
        auto_path = save_dir / "preview_auto_brightness.jpg"
        cv2.imwrite(str(auto_path), auto, [cv2.IMWRITE_JPEG_QUALITY, 95])

        single_w = CAPTURE_WIDTH // 4
        for idx in range(4):
            crop = last_bgr[:, single_w * idx:single_w * (idx + 1)]
            crop_path = save_dir / f"cam{idx+1}.jpg"
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print("\n================ 결과 ================")
        print(f"저장 폴더: {save_dir}")
        print(f"원본: {original_path}")
        print(f"자동밝기: {auto_path}")
        print("cam1~cam4 crop도 같이 저장했습니다.")

    finally:
        if picam2 is not None:
            try:
                picam2.stop()
            except Exception:
                pass

        if relay is not None:
            print("[릴레이] OFF")
            relay.off()


if __name__ == "__main__":
    main()
