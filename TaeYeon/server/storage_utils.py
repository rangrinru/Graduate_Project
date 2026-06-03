from pathlib import Path

from config import SAVE_ROOT


def ensure_within_save_root(path: Path) -> Path:
    resolved_root = SAVE_ROOT.resolve()
    resolved_path = path.resolve()

    if resolved_path == resolved_root or resolved_root in resolved_path.parents:
        return resolved_path

    raise ValueError("저장 루트 밖의 경로에는 접근할 수 없습니다.")


def safe_profile_path(folder_id: str) -> Path:
    if not str(folder_id).strip():
        raise ValueError("프로필 폴더 ID가 필요합니다.")

    return ensure_within_save_root(SAVE_ROOT / str(folder_id))


def safe_capture_path(profile_root: Path, *parts: str) -> Path:
    root = ensure_within_save_root(profile_root)
    target = root

    for part in parts:
        if not str(part).strip():
            raise ValueError("경로 구성값이 비어 있습니다.")
        target = target / str(part)

    return ensure_within_save_root(target)
