from flask import Flask, jsonify, Response, request
from flask_cors import CORS
import cv2
import json
import threading
import re
import shutil
from pathlib import Path
from datetime import datetime
from time import sleep
from gpiozero import LED
from picamera2 import Picamera2

app = Flask(__name__)
CORS(app)

CAPTURE_WIDTH = 5120
CAPTURE_HEIGHT = 800
SINGLE_WIDTH = CAPTURE_WIDTH // 4

SAVE_ROOT = Path.home() / "Graduate_Project" / "Final_Project" / "captures"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

PROFILES_FILE = SAVE_ROOT / "profiles.json"

CURRENT_GAIN = 1.0
SAVE_AS_PNG = True
INITIAL_EXPOSURE_MS = 100
STREAM_FPS = 12

RELAY_PIN = 17
RELAY_ACTIVE_HIGH = False
RELAY_WARMUP_SEC = 0.3

relay = LED(RELAY_PIN, active_high=RELAY_ACTIVE_HIGH, initial_value=False)

CAMERA_INFO = {
    "cam2": {
        "label": "CAM 2 - NO FILTER",
        "folder": "cam2_no_filter",
        "filter": "no_filter",
        "x_start": SINGLE_WIDTH * 1,
        "x_end": SINGLE_WIDTH * 2,
        "sequence_order": 1
    },
    "cam3": {
        "label": "CAM 3 - 405nm FILTER",
        "folder": "cam3_405nm",
        "filter": "405nm_filter",
        "x_start": SINGLE_WIDTH * 2,
        "x_end": SINGLE_WIDTH * 3,
        "sequence_order": 2
    },
    "cam4": {
        "label": "CAM 4 - 660nm FILTER",
        "folder": "cam4_660nm",
        "filter": "660nm_filter",
        "x_start": SINGLE_WIDTH * 3,
        "x_end": SINGLE_WIDTH * 4,
        "sequence_order": 3
    }
}

camera_lock = threading.Lock()
camera_ready = False
picam2 = None
preview_config = None
still_config = None


def relay_on():
    relay.on()


def relay_off():
    relay.off()


def set_manual_controls(camera, exposure_ms, gain):
    camera.set_controls({
        "AeEnable": False,
        "ExposureTime": exposure_ms * 1000,
        "AnalogueGain": gain
    })


def extract_cam_frame(full_frame_bgr, cam_key):
    info = CAMERA_INFO[cam_key]
    return full_frame_bgr[:, info["x_start"]:info["x_end"]]


def save_image(path, img_bgr):
    if path.suffix.lower() == ".png":
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])


def sanitize_profile_name(profile_name: str) -> str:
    cleaned = profile_name.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    if not cleaned:
        raise ValueError("유효한 프로필 이름이 아닙니다.")

    return cleaned


def make_folder_id() -> str:
    return f"profile_{int(datetime.now().timestamp() * 1000)}"


def load_profiles():
    if not PROFILES_FILE.exists():
        return []

    try:
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_profiles(profiles):
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def ensure_profile_dirs(folder_id: str):
    profile_root = SAVE_ROOT / folder_id
    profile_root.mkdir(parents=True, exist_ok=True)

    for cam in CAMERA_INFO.values():
        (profile_root / cam["folder"]).mkdir(parents=True, exist_ok=True)

    return profile_root


def find_profile_by_id(profile_id: str):
    profiles = load_profiles()
    for profile in profiles:
        if profile["folderId"] == profile_id:
            return profile
    return None


def create_profile(profile_name: str):
    display_name = sanitize_profile_name(profile_name)
    profiles = load_profiles()

    # 표시 이름 중복 방지
    exists = any(p["name"] == display_name for p in profiles)
    if exists:
        raise ValueError("이미 존재하는 프로필입니다.")

    created_at = datetime.now().strftime("%Y.%m.%d")
    profile_id = int(datetime.now().timestamp() * 1000)
    folder_id = make_folder_id()

    new_profile = {
        "id": profile_id,
        "name": display_name,      # 화면 표시용
        "folderId": folder_id,     # 실제 디렉토리용
        "createdAt": created_at
    }

    profiles.append(new_profile)
    save_profiles(profiles)
    ensure_profile_dirs(folder_id)

    return new_profile


def delete_profile(profile_id: str):
    profiles = load_profiles()
    target = None

    for p in profiles:
        if p["folderId"] == profile_id:
            target = p
            break

    if target is None:
        raise ValueError("삭제할 프로필이 없습니다.")

    new_profiles = [p for p in profiles if p["folderId"] != profile_id]

    profile_root = SAVE_ROOT / target["folderId"]
    if profile_root.exists() and profile_root.is_dir():
        shutil.rmtree(profile_root)

    save_profiles(new_profiles)


def save_one_camera_image(cam_key, frame_bgr, profile_root, capture_id, timestamp, exposure_ms, gain, ext, profile_name, folder_id):
    info = CAMERA_INFO[cam_key]

    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

    target_dir = profile_root / info["folder"] / capture_id
    target_dir.mkdir(parents=True, exist_ok=True)

    image_path = target_dir / f"{cam_key}.{ext}"
    meta_path = target_dir / "metadata.json"

    save_image(image_path, frame_bgr)

    metadata = {
        "captured_at": timestamp.isoformat(),
        "profile_name": profile_name,
        "profile_folder_id": folder_id,
        "camera_name": cam_key,
        "camera_label": info["label"],
        "filter_type": info["filter"],
        "sequence_order": info["sequence_order"],
        "capture_type": "high_quality_still_fullframe_then_crop",
        "file_format": ext,
        "camera_mode": {
            "capture_width": CAPTURE_WIDTH,
            "capture_height": CAPTURE_HEIGHT,
            "single_width": SINGLE_WIDTH
        },
        "camera_control": {
            "AeEnable": False,
            "ExposureTime_ms": exposure_ms,
            "ExposureTime_us": exposure_ms * 1000,
            "AnalogueGain": gain
        },
        "saved_file": str(image_path),
        "rotation_applied": "ROTATE_90_CLOCKWISE"
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return {
        "camera": cam_key,
        "image_path": str(image_path),
        "metadata_path": str(meta_path)
    }


def init_camera():
    global picam2, preview_config, still_config, camera_ready

    camera_list = Picamera2.global_camera_info()
    if len(camera_list) == 0:
        raise RuntimeError("카메라가 감지되지 않았습니다.")

    picam2 = Picamera2(0)

    preview_config = picam2.create_preview_configuration(
        main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "XRGB8888"}
    )

    still_config = picam2.create_still_configuration(
        main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
    )

    picam2.configure(preview_config)
    picam2.options["quality"] = 100
    picam2.options["compress_level"] = 0
    picam2.start()
    set_manual_controls(picam2, INITIAL_EXPOSURE_MS, CURRENT_GAIN)

    camera_ready = True


def read_preview_frame():
    frame = picam2.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def capture_high_quality_full_frame(exposure_ms, gain):
    global picam2, preview_config, still_config

    relay_on()
    sleep(RELAY_WARMUP_SEC)

    try:
        picam2.stop()
        picam2.configure(still_config)
        picam2.start()
        set_manual_controls(picam2, exposure_ms, gain)
        sleep(1)

        still_frame = picam2.capture_array()
        full_frame_bgr = cv2.cvtColor(still_frame, cv2.COLOR_RGB2BGR)

    finally:
        relay_off()
        picam2.stop()
        picam2.configure(preview_config)
        picam2.start()
        set_manual_controls(picam2, exposure_ms, gain)

    return full_frame_bgr


@app.route("/health")
def health():
    return jsonify({"ok": True, "camera_ready": camera_ready})


@app.route("/profiles", methods=["GET"])
def get_profiles():
    return jsonify({
        "ok": True,
        "profiles": load_profiles()
    })


@app.route("/profiles", methods=["POST"])
def create_profile_api():
    try:
        body = request.get_json(silent=True) or {}
        profile_name = body.get("name", "")
        new_profile = create_profile(profile_name)

        return jsonify({
            "ok": True,
            "profile": new_profile
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400


@app.route("/profiles/<profile_id>", methods=["DELETE"])
def delete_profile_api(profile_id):
    try:
        delete_profile(profile_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400


def generate_cam4_stream():
    frame_delay = 1.0 / STREAM_FPS

    while True:
        try:
            with camera_lock:
                if not camera_ready:
                    sleep(0.2)
                    continue

                full_frame_bgr = read_preview_frame()
                cam4_frame = extract_cam_frame(full_frame_bgr, "cam4").copy()

                ok, buffer = cv2.imencode(".jpg", cam4_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            if not ok:
                sleep(frame_delay)
                continue

            jpg_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpg_bytes +
                b"\r\n"
            )

            sleep(frame_delay)

        except Exception as e:
            print("[스트림 오류]", e)
            sleep(0.3)


@app.route("/stream-cam4")
def stream_cam4():
    return Response(
        generate_cam4_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/capture-all", methods=["POST"])
def capture_all():
    try:
        body = request.get_json(silent=True) or {}
        profile_id = body.get("profileId", "")

        if not str(profile_id).strip():
            return jsonify({
                "ok": False,
                "error": "profileId가 필요합니다."
            }), 400

        profile = find_profile_by_id(profile_id)

        if profile is None:
            return jsonify({
                "ok": False,
                "error": "존재하지 않는 프로필입니다."
            }), 400

        profile_name = profile["name"]
        folder_id = profile["folderId"]
        profile_root = SAVE_ROOT / folder_id

        if not profile_root.exists():
            return jsonify({
                "ok": False,
                "error": "프로필 폴더가 존재하지 않습니다."
            }), 400

        exposure_ms = INITIAL_EXPOSURE_MS
        gain = CURRENT_GAIN
        ext = "png" if SAVE_AS_PNG else "jpg"

        with camera_lock:
            full_frame_bgr = capture_high_quality_full_frame(
                exposure_ms=exposure_ms,
                gain=gain
            )

        capture_timestamp = datetime.now()
        capture_id = capture_timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]

        saved_files = []

        for cam_key in ["cam2", "cam3", "cam4"]:
            target_frame = extract_cam_frame(full_frame_bgr, cam_key).copy()

            result = save_one_camera_image(
                cam_key=cam_key,
                frame_bgr=target_frame,
                profile_root=profile_root,
                capture_id=capture_id,
                timestamp=capture_timestamp,
                exposure_ms=exposure_ms,
                gain=gain,
                ext=ext,
                profile_name=profile_name,
                folder_id=folder_id
            )
            saved_files.append(result)

        return jsonify({
            "ok": True,
            "captured_at": capture_timestamp.isoformat(),
            "profile_name": profile_name,
            "profile_id": folder_id,
            "capture_id": capture_id,
            "files": saved_files
        })

    except Exception as e:
        print("[촬영 오류]", e)
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    relay_off()
    init_camera()
    app.run(host="0.0.0.0", port=8000, threaded=True)