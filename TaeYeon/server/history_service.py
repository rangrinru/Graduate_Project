import json
import shutil
from pathlib import Path

from config import CAMERA_INFO, SAVE_ROOT
from profile_service import find_profile_by_id


def format_capture_id_to_text(capture_id: str) -> str:
    try:
        date_part, time_part, millis_part = capture_id.split("_")
        return (
            f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} "
            f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}.{millis_part}"
        )
    except Exception:
        return capture_id


def get_profile_root(profile_id: str) -> Path:
    profile = find_profile_by_id(profile_id)
    if profile is None:
        raise ValueError("존재하지 않는 프로필입니다.")
    return SAVE_ROOT / profile["folderId"]


def get_capture_history(profile_id: str):
    profile_root = get_profile_root(profile_id)
    profile = find_profile_by_id(profile_id)

    if not profile_root.exists():
        return []

    cam2_root = profile_root / CAMERA_INFO["cam2"]["folder"]
    if not cam2_root.exists():
        return []

    history = []
    for capture_dir in cam2_root.iterdir():
        if not capture_dir.is_dir():
            continue

        capture_id = capture_dir.name
        captured_at = None
        meta_path = capture_dir / "metadata.json"

        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    captured_at = meta.get("captured_at")
            except Exception:
                captured_at = None

        if not captured_at:
            captured_at = capture_id

        history.append({
            "captureId": capture_id,
            "capturedAt": captured_at,
            "displayTime": format_capture_id_to_text(capture_id),
            "profileId": profile["folderId"],
            "profileName": profile["name"],
        })

    history.sort(key=lambda x: x["captureId"], reverse=True)
    return history


def get_capture_detail(profile_id: str, capture_id: str):
    profile_root = get_profile_root(profile_id)
    profile = find_profile_by_id(profile_id)
    images = {}
    captured_at = None

    for cam_key, info in CAMERA_INFO.items():
        target_dir = profile_root / info["folder"] / capture_id
        candidates = [
            target_dir / f"{cam_key}.png",
            target_dir / f"{cam_key}.jpg",
            target_dir / f"{cam_key}.jpeg",
        ]

        image_path = None
        for candidate in candidates:
            if candidate.exists():
                image_path = candidate
                break

        meta_path = target_dir / "metadata.json"
        meta_data = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
            except Exception:
                meta_data = {}

        if captured_at is None:
            captured_at = meta_data.get("captured_at")

        images[info["filter"]] = {
            "camera": cam_key,
            "display_name": info["display_name"],
            "filter_type": info["filter"],
            "exists": image_path is not None,
            "image_url": (
                f"/profiles/{profile_id}/history/{capture_id}/image/{info['filter']}"
                if image_path else None
            ),
            "metadata": meta_data,
        }

    if captured_at is None:
        captured_at = capture_id

    has_any = any(item["exists"] for item in images.values())
    if not has_any:
        raise ValueError("해당 촬영 기록을 찾을 수 없습니다.")

    return {
        "captureId": capture_id,
        "capturedAt": captured_at,
        "displayTime": format_capture_id_to_text(capture_id),
        "profileId": profile["folderId"],
        "profileName": profile["name"],
        "images": images,
    }


def delete_capture_history(profile_id: str, capture_id: str):
    profile_root = get_profile_root(profile_id)
    deleted_any = False

    for _cam_key, info in CAMERA_INFO.items():
        target_dir = profile_root / info["folder"] / capture_id
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir)
            deleted_any = True

    if not deleted_any:
        raise ValueError("삭제할 촬영 기록이 없습니다.")


def resolve_image_path(profile_id: str, capture_id: str, filter_type: str) -> Path:
    profile_root = get_profile_root(profile_id)
    matched_cam_key = None
    matched_info = None

    for cam_key, info in CAMERA_INFO.items():
        if info["filter"] == filter_type:
            matched_cam_key = cam_key
            matched_info = info
            break

    if matched_cam_key is None:
        raise ValueError("유효하지 않은 필터 타입입니다.")

    target_dir = profile_root / matched_info["folder"] / capture_id
    candidates = [
        target_dir / f"{matched_cam_key}.png",
        target_dir / f"{matched_cam_key}.jpg",
        target_dir / f"{matched_cam_key}.jpeg",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise ValueError("이미지 파일이 존재하지 않습니다.")


def resolve_analysis_image_path(profile_id: str, capture_id: str, result_type: str) -> Path:
    profile_root = get_profile_root(profile_id)
    file_map = {
        "porphyrin-overlay": (CAMERA_INFO["cam4"]["folder"], "porphyrin_overlay.jpg"),
        "porphyrin-mask": (CAMERA_INFO["cam4"]["folder"], "porphyrin_mask.jpg"),
        "porphyrin-face-mask": (CAMERA_INFO["cam4"]["folder"], "porphyrin_face_mask.jpg"),
        "porphyrin-compare": (CAMERA_INFO["cam4"]["folder"], "porphyrin_compare.jpg"),
        "porphyrin-heatmap": (CAMERA_INFO["cam4"]["folder"], "porphyrin_heatmap.jpg"),
        "skin-aging-result": (CAMERA_INFO["cam3"]["folder"], "skin_aging_result.jpg"),
        "skin-aging-mask": (CAMERA_INFO["cam3"]["folder"], "skin_aging_mask.jpg"),
    }

    if result_type not in file_map:
        raise ValueError("유효하지 않은 분석 이미지 타입입니다.")

    folder, file_name = file_map[result_type]
    analysis_dir = profile_root / folder / capture_id / "analysis"
    image_path = analysis_dir / file_name
    if not image_path.exists() or not image_path.is_file():
        raise ValueError("분석 결과 이미지가 없습니다. 먼저 분석을 실행하세요.")

    return image_path
