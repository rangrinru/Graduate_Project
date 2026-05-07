from pathlib import Path

BASE_PATH = Path.home() / "Graduate_Project" / "captures" / "sessions"

def get_latest_result():
    sessions = sorted(BASE_PATH.glob("*"), reverse=True)

    if not sessions:
        return None

    latest = sessions[0]

    result_img = latest / "analysis" / "porphyrin_overlay.png"

    if result_img.exists():
        return str(result_img)
    else:
        return None