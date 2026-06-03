import json
import re
import shutil
from datetime import datetime

from config import CAMERA_INFO, PROFILES_FILE
from storage_utils import safe_profile_path


def sanitize_profile_name(profile_name: str) -> str:
    cleaned = re.sub(r"\s+", " ", profile_name.strip())
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
    profile_root = safe_profile_path(folder_id)
    profile_root.mkdir(parents=True, exist_ok=True)
    for cam in CAMERA_INFO.values():
        (profile_root / cam["folder"]).mkdir(parents=True, exist_ok=True)
    return profile_root


def find_profile_by_id(profile_id: str):
    for profile in load_profiles():
        if profile["folderId"] == profile_id:
            return profile
    return None


def create_profile(profile_name: str):
    display_name = sanitize_profile_name(profile_name)
    profiles = load_profiles()
    if any(p["name"] == display_name for p in profiles):
        raise ValueError("이미 존재하는 프로필입니다.")

    new_profile = {
        "id": int(datetime.now().timestamp() * 1000),
        "name": display_name,
        "folderId": make_folder_id(),
        "createdAt": datetime.now().strftime("%Y.%m.%d"),
    }
    profiles.append(new_profile)
    save_profiles(profiles)
    ensure_profile_dirs(new_profile["folderId"])
    return new_profile


def delete_profile(profile_id: str):
    profiles = load_profiles()
    target = next((p for p in profiles if p["folderId"] == profile_id), None)
    if target is None:
        raise ValueError("삭제할 프로필이 없습니다.")

    profile_root = safe_profile_path(target["folderId"])
    if profile_root.exists() and profile_root.is_dir():
        shutil.rmtree(profile_root)

    save_profiles([p for p in profiles if p["folderId"] != profile_id])
