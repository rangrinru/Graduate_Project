import cv2
import json

from config import CAMERA_INFO, CAPTURE_HEIGHT, CAPTURE_WIDTH, SINGLE_WIDTH
def extract_cam_frame(full_frame_bgr, cam_key):
    info = CAMERA_INFO[cam_key]
    return full_frame_bgr[:, info["x_start"]:info["x_end"]]


def save_image(path, img_bgr):
    if path.suffix.lower() == ".png":
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])


def save_one_camera_image(
    cam_key,
    frame_bgr,
    profile_root,
    capture_id,
    timestamp,
    exposure_ms,
    gain,
    ext,
    profile_name,
    folder_id,
    trigger_metadata=None,
):
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
        "display_name": info["display_name"],
        "sequence_order": info["sequence_order"],
        "capture_type": "picamera2_still_fullframe_then_crop",
        "file_format": ext,
        "camera_mode": {
            "capture_width": CAPTURE_WIDTH,
            "capture_height": CAPTURE_HEIGHT,
            "single_width": SINGLE_WIDTH,
        },
        "camera_control": {
            "AeEnable": False,
            "ExposureTime_ms": exposure_ms,
            "ExposureTime_us": exposure_ms * 1000,
            "AnalogueGain": gain,
        },
        "saved_file": str(image_path),
        "rotation_applied": "ROTATE_90_CLOCKWISE",
        "auto_capture_trigger": trigger_metadata,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return {
        "camera": cam_key,
        "filter_type": info["filter"],
        "display_name": info["display_name"],
        "image_path": str(image_path),
        "metadata_path": str(meta_path),
    }
