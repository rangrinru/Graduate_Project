from __future__ import annotations

"""
porphyrin_analysis_multifolder_auto.py

목표
- 경로 인자를 넘기지 않아도 고정된 저장 구조를 자동 스캔
- cam4_660nm 폴더 아래 날짜별 이미지를 기준으로 전체 분석
- 같은 capture_prefix의 cam2/cam3/cam4를 자동 매칭
- 결과를 sessions/analysis/<date>/<capture_prefix>/ 아래 저장
- 날짜별 summary.json도 저장
- 기존 함수명(analyze_session_dir / analyze_porphyrin_session / run_analysis_for_session) 유지
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


# =============================
# 기본 경로 설정
# =============================
DEFAULT_SESSIONS_ROOT = Path.home() / "Graduate_Project" / "captures" / "sessions"
DEFAULT_CAM4_ROOT = DEFAULT_SESSIONS_ROOT / "cam4_660nm"

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


def _parse_capture_info_from_path(input_path: Path) -> Tuple[Path, str, str]:
    """
    대표 파일 경로에서:
    - sessions_root
    - date_str
    - capture_prefix
    추출

    예:
    .../cam4_660nm/2026-04-07/20260407_224057_658_cam4.png
    -> sessions_root, 2026-04-07, 20260407_224057_658
    """
    if not input_path.exists():
        raise FileNotFoundError(f"입력 경로가 존재하지 않습니다: {input_path}")

    if input_path.is_dir():
        raise ValueError("대표 파일 경로는 이미지 또는 metadata 파일이어야 합니다.")

    date_str = input_path.parent.name
    cam_folder = input_path.parent.parent.name
    sessions_root = input_path.parent.parent.parent

    if cam_folder not in CAM_FOLDER_MAP.values():
        raise ValueError(
            f"카메라 폴더를 인식하지 못했습니다: {cam_folder}. "
            f"허용: {list(CAM_FOLDER_MAP.values())}"
        )

    name = input_path.name

    if name.endswith(".metadata.json"):
        name = name[:-len(".metadata.json")]

    stem = Path(name).stem
    parts = stem.split("_")

    if len(parts) < 2:
        raise ValueError(f"파일명에서 capture_id를 추출하지 못했습니다: {input_path.name}")

    if parts[-1] in ("cam2", "cam3", "cam4"):
        capture_prefix = "_".join(parts[:-1])
    else:
        capture_prefix = stem

    return sessions_root, date_str, capture_prefix


def _find_capture_file(
    sessions_root: Path,
    date_str: str,
    capture_prefix: str,
    cam_key: str,
    allow_metadata: bool = False,
) -> Path:
    cam_folder = CAM_FOLDER_MAP[cam_key]
    target_dir = sessions_root / cam_folder / date_str

    if not target_dir.exists():
        raise FileNotFoundError(f"대상 폴더가 없습니다: {target_dir}")

    suffix_candidates = [f"{capture_prefix}_{cam_key}", f"{capture_prefix}_cam2", capture_prefix]

    for prefix in suffix_candidates:
        if allow_metadata:
            meta_path = target_dir / f"{prefix}.metadata.json"
            if meta_path.exists():
                return meta_path

        for ext in IMAGE_EXTS:
            img_path = target_dir / f"{prefix}{ext}"
            if img_path.exists():
                return img_path

    candidates = sorted(
        [p for p in target_dir.iterdir() if p.is_file() and p.stem.startswith(capture_prefix)]
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


def _find_all_cam4_images(cam4_root: Path) -> List[Path]:
    if not cam4_root.exists():
        return []

    found: List[Path] = []
    for date_dir in sorted([p for p in cam4_root.iterdir() if p.is_dir()]):
        for p in sorted(date_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                found.append(p)
    return found


def _save_date_summary(sessions_root: Path, date_str: str, items: List[Dict[str, Any]]) -> Path:
    date_analysis_dir = sessions_root / "analysis" / date_str
    date_analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_path = date_analysis_dir / "summary.json"
    payload = {
        "date": date_str,
        "count": len(items),
        "items": items,
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


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

    save_dir = Path(save_dir) if save_dir is not None else Path(".")
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

    print(f"[분석 완료] {image_path}")
    print(f"  - 검출 개수: {count}")
    print(f"  - 총 면적: {total_area}")
    print(f"  - 평균 강도: {mean_intensity:.2f}")
    print(f"  - 저장 위치: {analysis_dir}")

    if show_result:
        cv2.imshow("Porphyrin Compare", combined)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    return result


def detect_porphyrin(
    image_path: str | Path,
    save_dir: Optional[str | Path] = None,
    show_result: bool = False,
) -> Dict[str, Any]:
    return detect_porhyrin(image_path=image_path, save_dir=save_dir, show_result=show_result)


# -----------------------------
# 표준 함수들
# -----------------------------
def analyze_session_dir(session_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    1) session_dir가 주어지면: 대표 파일 1개만 분석
    2) session_dir가 None이면: DEFAULT_CAM4_ROOT 전체를 자동 분석
    """
    if session_dir is None:
        return run_analysis_for_session(None)

    reference_path = Path(session_dir)
    bundle = _resolve_capture_bundle_paths(reference_path)

    result = detect_porhyrin(
        image_path=bundle["cam4"],
        save_dir=bundle["analysis_dir"],
        show_result=False,
    )

    result["cam2_path"] = str(bundle["cam2"])
    result["cam3_path"] = str(bundle["cam3"])
    result["cam4_path"] = str(bundle["cam4"])
    result["capture_prefix"] = bundle["capture_prefix"].name
    result["capture_date"] = bundle["date_str"].name
    return result


def analyze_porphyrin_session(session_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    return analyze_session_dir(session_dir)


def run_analysis_for_session(session_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    사용자가 원하는 동작:
    - 경로 인자 없이 실행
    - cam4_660nm 아래 날짜별 사진 전체 분석
    - 결과를 sessions/analysis/<date>/<capture_prefix>/ 아래 저장
    - 날짜별 summary.json 생성
    - 콘솔에 결과 출력
    """
    if session_dir is not None:
        return analyze_session_dir(session_dir)

    cam4_images = _find_all_cam4_images(DEFAULT_CAM4_ROOT)
    if not cam4_images:
        raise FileNotFoundError(f"분석할 cam4 이미지가 없습니다: {DEFAULT_CAM4_ROOT}")

    all_results: List[Dict[str, Any]] = []
    grouped_by_date: Dict[str, List[Dict[str, Any]]] = {}

    print(f"[시작] 자동 분석")
    print(f"  - 기준 폴더: {DEFAULT_CAM4_ROOT}")
    print(f"  - 대상 이미지 수: {len(cam4_images)}")

    for cam4_img in cam4_images:
        try:
            result = analyze_session_dir(cam4_img)
            all_results.append(result)

            date_str = result.get("capture_date", cam4_img.parent.name)
            grouped_by_date.setdefault(date_str, []).append(result)

        except Exception as exc:
            print(f"[실패] {cam4_img}")
            print(f"  - 오류: {exc}")

    summary_paths = []
    for date_str, items in grouped_by_date.items():
        summary_path = _save_date_summary(DEFAULT_SESSIONS_ROOT, date_str, items)
        summary_paths.append(str(summary_path))

    final_result = {
        "success": True,
        "mode": "batch_auto_scan",
        "sessions_root": str(DEFAULT_SESSIONS_ROOT),
        "cam4_root": str(DEFAULT_CAM4_ROOT),
        "total_analyzed": len(all_results),
        "summary_paths": summary_paths,
        "results": all_results,
    }

    print(f"[완료] 총 분석 수: {len(all_results)}")
    for p in summary_paths:
        print(f"  - 날짜별 summary 저장: {p}")

    return final_result


# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        # 대표 파일 경로 1개를 넘겨서 단건 분석
        result = analyze_session_dir(sys.argv[1])
    else:
        # 인자 없이 실행하면 전체 자동 분석
        result = run_analysis_for_session()

    print(json.dumps(result, ensure_ascii=False, indent=2))
