# camera_deep_debug.py
# 검은 화면 원인 분리용 코드
# 1) GPIO17 릴레이가 실제로 켜지는지 확인
# 2) Picamera2 main 화면이 검은지 확인
# 3) raw 센서 값 자체가 들어오는지 확인
# 4) metadata로 실제 노출/게인이 적용됐는지 확인

import argparse
import json
from pathlib import Path
from datetime import datetime
from time import sleep

import cv2
import numpy as np
from config import PROJECT_ROOT

try:
    from picamera2 import Picamera2

    PICAMERA_IMPORT_ERROR = None
except Exception as e:
    Picamera2 = None
    PICAMERA_IMPORT_ERROR = e

try:
    from gpiozero import LED
except Exception:
    LED = None


CAPTURE_WIDTH = 5120
CAPTURE_HEIGHT = 800

SAVE_ROOT = PROJECT_ROOT / "camera_deep_debug"


def str_to_bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)

    min_v = float(np.min(arr))
    max_v = float(np.max(arr))

    if max_v <= min_v:
        return np.zeros(arr.shape[:2], dtype=np.uint8)

    arr = (arr - min_v) * 255.0 / (max_v - min_v)
    return np.clip(arr, 0, 255).astype(np.uint8)


def stats_of_array(name: str, arr: np.ndarray) -> dict:
    arr_float = arr.astype(np.float32)

    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr_float)),
        "max": float(np.max(arr_float)),
        "mean": float(np.mean(arr_float)),
        "std": float(np.std(arr_float)),
    }


def save_main_image(save_dir: Path, main: np.ndarray) -> dict:
    result = {}

    # 원본 main 배열 저장
    npy_path = save_dir / "main_array.npy"
    np.save(str(npy_path), main)
    result["main_array_npy"] = str(npy_path)

    # XRGB8888이면 4채널이 들어옵니다.
    if main.ndim == 3 and main.shape[2] == 4:
        # 화면 확인용 변환
        bgr = cv2.cvtColor(main, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # RGB888이면 3채널이 들어옵니다.
    elif main.ndim == 3 and main.shape[2] == 3:
        bgr = cv2.cvtColor(main, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 혹시 모노로 들어오면 그대로 처리합니다.
    elif main.ndim == 2:
        gray = normalize_to_uint8(main)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    else:
        raise RuntimeError(f"알 수 없는 main shape입니다: {main.shape}, dtype={main.dtype}")

    # 원본에 가까운 jpg
    main_jpg = save_dir / "main_original.jpg"
    cv2.imwrite(str(main_jpg), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    result["main_original_jpg"] = str(main_jpg)

    # 자동 밝기 보정 jpg
    main_norm = normalize_to_uint8(gray)
    main_norm_bgr = cv2.cvtColor(main_norm, cv2.COLOR_GRAY2BGR)
    main_norm_jpg = save_dir / "main_auto_brightness.jpg"
    cv2.imwrite(str(main_norm_jpg), main_norm_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    result["main_auto_brightness_jpg"] = str(main_norm_jpg)

    result["main_gray_stats"] = stats_of_array("main_gray", gray)

    return result


def save_raw_image(save_dir: Path, raw: np.ndarray | None) -> dict:
    result = {}

    if raw is None:
        result["raw_available"] = False
        return result

    result["raw_available"] = True

    raw_npy = save_dir / "raw_array.npy"
    np.save(str(raw_npy), raw)
    result["raw_array_npy"] = str(raw_npy)

    result["raw_stats"] = stats_of_array("raw", raw)

    # raw가 2차원이라고 가정하고 자동 밝기 보정 이미지 저장
    # packed raw면 가로 비율이 이상하게 보일 수 있지만, 밝기값이 들어오는지 확인하는 용도입니다.
    if raw.ndim == 2:
        raw_norm = normalize_to_uint8(raw)
    elif raw.ndim == 3:
        # 채널이 있으면 첫 채널 기준으로 확인
        raw_norm = normalize_to_uint8(raw[:, :, 0])
    else:
        raw_norm = normalize_to_uint8(raw.reshape(raw.shape[0], -1))

    raw_jpg = save_dir / "raw_auto_brightness.jpg"
    cv2.imwrite(str(raw_jpg), raw_norm, [cv2.IMWRITE_JPEG_QUALITY, 95])
    result["raw_auto_brightness_jpg"] = str(raw_jpg)

    return result


def relay_test(relay_pin: int, relay_active_high: bool, relay_on: bool, warmup_sec: float):
    relay = None

    if not relay_on:
        return None

    if LED is None:
        raise RuntimeError("gpiozero를 import할 수 없습니다. 릴레이 제어를 할 수 없습니다.")

    relay = LED(relay_pin, active_high=relay_active_high, initial_value=False)

    print(f"[릴레이] GPIO{relay_pin} ON 시도 / active_high={relay_active_high}")
    relay.on()
    print("[릴레이] 지금 실제 LED/조명이 켜졌는지 눈으로 확인하세요.")
    sleep(warmup_sec)

    return relay


def main():
    if Picamera2 is None:
        raise RuntimeError(f"Picamera2를 사용할 수 없습니다: {PICAMERA_IMPORT_ERROR}")

    parser = argparse.ArgumentParser()

    parser.add_argument("--exposure-ms", type=int, default=1000)
    parser.add_argument("--gain", type=float, default=16.0)

    parser.add_argument("--relay-on", action="store_true")
    parser.add_argument("--relay-pin", type=int, default=17)
    parser.add_argument("--relay-active-high", type=str, default="false")
    parser.add_argument("--warmup-sec", type=float, default=1.0)

    parser.add_argument("--width", type=int, default=CAPTURE_WIDTH)
    parser.add_argument("--height", type=int, default=CAPTURE_HEIGHT)

    args = parser.parse_args()

    relay_active_high = str_to_bool(args.relay_active_high)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = SAVE_ROOT / ts
    save_dir.mkdir(parents=True, exist_ok=True)

    relay = None
    picam2 = None

    report = {
        "save_dir": str(save_dir),
        "args": vars(args),
        "relay_active_high_bool": relay_active_high,
    }

    try:
        print("[1] 카메라 목록 확인")
        camera_list = Picamera2.global_camera_info()
        print(json.dumps(camera_list, ensure_ascii=False, indent=2, default=str))
        report["camera_list"] = camera_list

        if len(camera_list) == 0:
            raise RuntimeError("카메라가 감지되지 않았습니다.")

        print("[2] 릴레이 테스트")
        relay = relay_test(
            relay_pin=args.relay_pin,
            relay_active_high=relay_active_high,
            relay_on=args.relay_on,
            warmup_sec=args.warmup_sec,
        )

        print("[3] Picamera2 설정")
        picam2 = Picamera2(0)

        # main + raw 동시 캡처 설정
        config = picam2.create_still_configuration(
            main={"size": (args.width, args.height), "format": "XRGB8888"},
            raw={"size": (args.width, args.height)}
        )

        picam2.configure(config)

        print("[4] 지원 컨트롤 확인")
        controls = picam2.camera_controls
        report["camera_controls"] = {k: str(v) for k, v in controls.items()}
        print(json.dumps(report["camera_controls"], ensure_ascii=False, indent=2))

        print("[5] 카메라 시작")
        picam2.start()

        print(f"[6] 수동 노출 적용: exposure={args.exposure_ms}ms, gain={args.gain}")
        picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": int(args.exposure_ms * 1000),
            "AnalogueGain": float(args.gain),
        })

        # 긴 노출이 실제 반영되도록 대기
        sleep(max(2.0, args.exposure_ms / 1000.0 + 0.5))

        print("[7] 캡처")
        request = picam2.capture_request()

        metadata = request.get_metadata()
        report["metadata"] = metadata
        print("[metadata]")
        print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))

        main_array = request.make_array("main")

        raw_array = None
        try:
            raw_array = request.make_array("raw")
        except Exception as e:
            report["raw_error"] = str(e)
            print("[WARN] raw array 읽기 실패:", e)

        request.release()

        print("[8] 이미지/배열 저장")
        report["main"] = save_main_image(save_dir, main_array)
        report["raw"] = save_raw_image(save_dir, raw_array)

        report_path = save_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        print("\n================ 결과 ================")
        print(f"저장 폴더: {save_dir}")
        print(f"보고서: {report_path}")
        print("main_original.jpg:", report["main"]["main_original_jpg"])
        print("main_auto_brightness.jpg:", report["main"]["main_auto_brightness_jpg"])

        if report["raw"].get("raw_available"):
            print("raw_auto_brightness.jpg:", report["raw"]["raw_auto_brightness_jpg"])

        print("\n[판단 기준]")
        print("1) raw_auto_brightness.jpg가 보이는데 main_original.jpg가 검으면 → ISP/tuning 문제 가능성이 큽니다.")
        print("2) raw도 main도 전부 검고 max가 0에 가까우면 → 조명/렌즈/케이블/센서 입력 문제 가능성이 큽니다.")
        print("3) metadata의 ExposureTime, AnalogueGain이 요청값보다 낮으면 → 노출/게인 제어가 실제 적용되지 않은 겁니다.")

    except Exception as e:
        report["error"] = str(e)
        report_path = save_dir / "report_error.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print("[ERROR]", e)
        print("오류 보고서:", report_path)

    finally:
        if picam2 is not None:
            try:
                picam2.stop()
            except Exception:
                pass

        if relay is not None:
            try:
                print("[릴레이] OFF")
                relay.off()
            except Exception:
                pass


if __name__ == "__main__":
    main()
