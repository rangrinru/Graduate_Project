from __future__ import annotations

"""
porphyrin_analysis.py

저장 구조가 아래처럼 "카메라별 / 날짜별"로 나뉜 경우를 지원하는 버전

예시:
Graduate_Project/captures/sessions/
 ├─ cam2_no_filter/2026-04-07/20260407_224057_658_cam2.png
 ├─ cam2_no_filter/2026-04-07/20260407_224057_658_cam2.metadata.json
 ├─ cam3_405nm/2026-04-07/20260407_224057_658_cam3.png
 └─ cam4_660nm/2026-04-07/20260407_224057_658_cam4.png

핵심:
- analyze_session_dir(...)
- analyze_porphyrin_session(...)
- run_analysis_for_session(...)

이 함수들은 이제 "대표 경로 하나"만 받아도 같은 capture_id의 cam2/cam3/cam4를 찾아낸다.
대표 경로 예시:
- cam4 이미지 경로
- cam2 이미지 경로
- .metadata.json 경로
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


CAM_FOLDER_MAP = {
    "cam2": "cam2_no_filter",
    "cam3": "cam3_405nm",
    "cam4": "cam4_660nm",
}

IMAGE_EXTS = [".png", ".jpg", ".jpeg"]


# -----------------------------
# 최신 이미지 찾기 (기존 유지)
# -----------------------------
def get_latest_image(base_path: str) -> str:
    folders = [
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
    ]

    latest_folder = sorted(folders)[-1]
    latest_path = os.path.join(base_path, latest_folder)

    files = [
        f for f in os.listdir(latest_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    latest_file = sorted(files)[-1]
    image_path = os.path.join(latest_path, latest_file)

    print("사용 이미지:", image_path)
    return image_path


def _ensure_analysis_dir(save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


# =========================================================
# 여기부터가 핵심 수정 포인트
# =========================================================
# 기존 코드에서는 _resolve_cam4_path_from_session(session_dir)가
# session_metadata.json 또는 session_dir/cam4.png 형태만 찾았음.
#
# 새 구조에서는:
#   sessions/cam2_no_filter/YYYY-MM-DD/<capture_id>_cam2.png
#   sessions/cam3_405nm/YYYY-MM-DD/<capture_id>_cam3.png
#   sessions/cam4_660nm/YYYY-MM-DD/<capture_id>_cam4.png
# 형태이므로, 대표 경로 하나에서 capture_id/date/root를 추출한 뒤
# cam2/cam3/cam4 파일을 다시 조합해서 찾아야 함.
# =========================================================

def _parse_capture_info_from_path(input_path: Path) -> Tuple[Path, str, str]:
    """
    대표 경로 하나에서
    - sessions 루트
    - 날짜 폴더명
    - capture prefix (예: 20260407_224057_658)
    를 추출한다.

    허용 입력:
    - .../cam4_660nm/2026-04-07/20260407_224057_658_cam4.png
    - .../cam2_no_filter/2026-04-07/20260407_224057_658_cam2.metadata.json
    """
    if not input_path.exists():
        raise FileNotFoundError(f"입력 경로가 존재하지 않습니다: {input_path}")

    if input_path.is_dir():
        raise ValueError(
            "새 저장 구조에서는 폴더 하나만으로 capture_id를 알 수 없습니다. "
            "cam2/cam3/cam4 이미지 또는 metadata 파일 경로를 대표 경로로 넘겨주세요."
        )

    date_str = input_path.parent.name
    cam_folder = input_path.parent.parent.name
    sessions_root = input_path.parent.parent.parent

    if cam_folder not in CAM_FOLDER_MAP.values():
        raise ValueError(
            f"카메라 폴더를 인식하지 못했습니다: {cam_folder}. "
            f"허용: {list(CAM_FOLDER_MAP.values())}"
        )

    name = input_path.name

    # .metadata.json 제거
    if name.endswith(".metadata.json"):
        name = name[:-len(".metadata.json")]

    # 확장자 제거
    stem = Path(name).stem

    # 예: 20260407_224057_658_cam4 -> 20260407_224057_658
    # suffix가 cam2/cam3/cam4 중 하나인 경우 prefix 분리
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"파일명에서 capture_id를 추출하지 못했습니다: {input_path.name}")

    if parts[-1] in ("cam2", "cam3", "cam4"):
        capture_prefix = "_".join(parts[:-1])
    else:
        # 혹시 cam suffix가 누락된 경우 대비
        capture_prefix = stem

    return sessions_root, date_str, capture_prefix


def _find_capture_file(
    sessions_root: Path,
    date_str: str,
    capture_prefix: str,
    cam_key: str,
    allow_metadata: bool = False,
) -> Path:
    """
    예:
    sessions_root/cam4_660nm/2026-04-07/20260407_224057_658_cam4.png
    """
    cam_folder = CAM_FOLDER_MAP[cam_key]
    target_dir = sessions_root / cam_folder / date_str

    if not target_dir.exists():
        raise FileNotFoundError(f"대상 폴더가 없습니다: {target_dir}")

    suffix_candidates = [f"{capture_prefix}_{cam_key}"]
    # 사용자가 예시에서 cam3/cam4도 _cam2.png라고 적은 경우를 대비한 fallback
    suffix_candidates.append(f"{capture_prefix}_cam2")
    suffix_candidates.append(capture_prefix)

    for prefix in suffix_candidates:
        if allow_metadata:
            meta_path = target_dir / f"{prefix}.metadata.json"
            if meta_path.exists():
                return meta_path

        for ext in IMAGE_EXTS:
            img_path = target_dir / f"{prefix}{ext}"
            if img_path.exists():
                return img_path

    # 완전 fallback: capture_prefix로 시작하는 첫 파일
    candidates = sorted(
        [
            p for p in target_dir.iterdir()
            if p.is_file() and p.stem.startswith(capture_prefix)
        ]
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"{cam_key} 파일을 찾지 못했습니다. "
        f"root={sessions_root}, date={date_str}, capture_prefix={capture_prefix}"
    )


def _resolve_capture_bundle_paths(reference_path: Path) -> Dict[str, Path]:
    sessions_root, date_str, capture_prefix = _parse_capture_info_from_path(reference_path)

    cam2_path = _find_capture_file(sessions_root, date_str, capture_prefix, "cam2")
    cam3_path = _find_capture_file(sessions_root, date_str, capture_prefix, "cam3")
    cam4_path = _find_capture_file(sessions_root, date_str, capture_prefix, "cam4")

    analysis_dir = sessions_root / "analysis" / date_str / capture_prefix
    analysis_dir.mkdir(parents=True, exist_ok=True)

    return {
        "sessions_root": sessions_root,
        "date_str": Path(date_str),
        "capture_prefix": Path(capture_prefix),
        "cam2": cam2_path,
        "cam3": cam3_path,
        "cam4": cam4_path,
        "analysis_dir": analysis_dir,
    }


# -----------------------------
# 포르피린 검출 (기존 함수명 유지)
# -----------------------------
def detect_porhyrin(
    image_path: str | Path,
    save_dir: Optional[str | Path] = None,
    show_result: bool = False,
) -> Dict[str, Any]:
    img = cv2.imread(str(image_path))

    if img is None:
        raise RuntimeError(f"이미지 로드 실패: {image_path}")

    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    threshold_value = float(np.percentile(blur, 97))
    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    total_area = 0.0
    mask = np.zeros_like(gray)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if 3 < radius < 15:
            center = (int(x), int(y))
            cv2.circle(output, center, int(radius), (0, 0, 255), 2)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
            count += 1
            total_area += float(area)

    if np.any(mask > 0):
        mean_intensity = float(np.mean(blur[mask > 0]))
    else:
        mean_intensity = 0.0

    if save_dir is None:
        save_dir = Path(".")
    else:
        save_dir = Path(save_dir)

    analysis_dir = _ensure_analysis_dir(save_dir)

    overlay_path = analysis_dir / "porphyrin_detect_result.jpg"
    compare_path = analysis_dir / "compare_result.jpg"
    mask_path = analysis_dir / "porphyrin_mask.png"
    report_path = analysis_dir / "report.json"

    cv2.imwrite(str(overlay_path), output)
    cv2.imwrite(str(mask_path), mask)

    h, w = img.shape[:2]
    output_resized = cv2.resize(output, (w, h))
    combined = np.hstack((img, output_resized))

    cv2.putText(combined, "Original", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Detection", (w + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(str(compare_path), combined)

    result = {
        "session_dir": str(analysis_dir),
        "source_image": str(image_path),
        "porphyrin_count": int(count),
        "porphyrin_area": float(total_area),
        "mean_intensity": float(mean_intensity),
        "threshold_value": threshold_value,
        "overlay_path": str(overlay_path),
        "mask_path": str(mask_path),
        "compare_path": str(compare_path),
        "report_path": str(report_path),
        "regional_distribution": {},
    }

    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("검출 개수:", count)
    print("총 면적:", total_area)
    print("평균 강도:", mean_intensity)

    if show_result:
        cv2.imshow("Porphyrin Compare", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def detect_porphyrin(
    image_path: str | Path,
    save_dir: Optional[str | Path] = None,
    show_result: bool = False,
) -> Dict[str, Any]:
    return detect_porhyrin(image_path=image_path, save_dir=save_dir, show_result=show_result)


# -----------------------------
# analysis_pipeline.py가 찾는 표준 함수
# -----------------------------
def analyze_session_dir(session_dir: str | Path) -> Dict[str, Any]:
    """
    이제 session_dir에는 "대표 파일 경로 하나"를 넣으면 된다.
    예:
    Graduate_Project/captures/sessions/cam4_660nm/2026-04-07/20260407_224057_658_cam4.png
    또는
    Graduate_Project/captures/sessions/cam2_no_filter/2026-04-07/20260407_224057_658_cam2.metadata.json
    """
    reference_path = Path(session_dir)
    bundle = _resolve_capture_bundle_paths(reference_path)

    result = detect_porhyrin(
        image_path=bundle["cam4"],
        save_dir=bundle["analysis_dir"],
        show_result=False,
    )

    # cam2/cam3/cam4 경로도 같이 넣어주면 후속 파이프라인에서 유용
    result["cam2_path"] = str(bundle["cam2"])
    result["cam3_path"] = str(bundle["cam3"])
    result["cam4_path"] = str(bundle["cam4"])
    result["capture_prefix"] = bundle["capture_prefix"].name
    result["capture_date"] = bundle["date_str"].name
    return result


def analyze_porphyrin_session(session_dir: str | Path) -> Dict[str, Any]:
    return analyze_session_dir(session_dir)


def run_analysis_for_session(session_dir: str | Path) -> Dict[str, Any]:
    return analyze_session_dir(session_dir)


# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        result = analyze_session_dir(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        base_path = "Graduate_Project/captures/cam4_660nm"
        img_path = get_latest_image(base_path)
        result = detect_porhyrin(img_path, save_dir=".", show_result=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
