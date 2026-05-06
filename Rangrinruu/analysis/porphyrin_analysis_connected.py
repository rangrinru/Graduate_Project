from __future__ import annotations

"""
porphyrin_analysis.py

기존 함수명(get_latest_image, detect_porhyrin)을 유지하면서
analysis_pipeline.py가 바로 연결될 수 있도록 수정한 버전

추가된 핵심:
- analyze_session_dir(session_dir)
- analyze_porphyrin_session(session_dir)
- run_analysis_for_session(session_dir)

즉, 기존 단독 실행도 가능하고,
analysis_pipeline.py에서도 바로 import해서 사용할 수 있다.
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


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
        if f.lower().endswith((".jpg", ".png"))
    ]

    latest_file = sorted(files)[-1]
    image_path = os.path.join(latest_path, latest_file)

    print("사용 이미지:", image_path)
    return image_path


def _ensure_analysis_dir(save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _resolve_cam4_path_from_session(session_dir: Path) -> Path:
    session_meta = session_dir / "session_metadata.json"

    if session_meta.exists():
        try:
            data = json.loads(session_meta.read_text(encoding="utf-8"))
            files = data.get("files", {})
            cam4_path = files.get("cam4")
            if cam4_path:
                path = Path(cam4_path)
                if path.exists():
                    return path
        except Exception:
            pass

    for name in ["cam4.png", "cam4.jpg", "cam4.jpeg"]:
        path = session_dir / name
        if path.exists():
            return path

    raise FileNotFoundError(f"cam4 이미지를 찾을 수 없습니다: {session_dir}")


# -----------------------------
# 포르피린 검출 (기존 함수명 유지)
# 기존에는 출력만 했지만, 이제 dict도 반환하도록 확장
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

    # -----------------------------
    # 그레이 + 대비 강화
    # -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # -----------------------------
    # 상위 밝기만 추출 (핵심)
    # -----------------------------
    threshold_value = float(np.percentile(blur, 97))
    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)

    # -----------------------------
    # 노이즈 제거
    # -----------------------------
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # -----------------------------
    # 컨투어 검출
    # -----------------------------
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    total_area = 0.0
    intensity_values = []

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
        intensity_values = blur[mask > 0]
        mean_intensity = float(np.mean(intensity_values))
    else:
        mean_intensity = 0.0

    # -----------------------------
    # 저장 경로 결정
    # -----------------------------
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
        "session_dir": str(analysis_dir.parent) if analysis_dir.name == "analysis" else str(analysis_dir),
        "source_image": str(image_path),
        "porphyrin_count": int(count),
        "porphyrin_area": float(total_area),
        "mean_intensity": float(mean_intensity),
        "threshold_value": threshold_value,
        "overlay_path": str(overlay_path),
        "mask_path": str(mask_path),
        "compare_path": str(compare_path),
        "report_path": str(report_path),
        "regional_distribution": {},  # 이후 얼굴 부위별 분포 로직 추가 예정
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


# 오탈자 보정용 alias
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
    session_dir = Path(session_dir)
    cam4_path = _resolve_cam4_path_from_session(session_dir)
    analysis_dir = session_dir / "analysis"
    return detect_porhyrin(cam4_path, save_dir=analysis_dir, show_result=False)


def analyze_porphyrin_session(session_dir: str | Path) -> Dict[str, Any]:
    return analyze_session_dir(session_dir)


def run_analysis_for_session(session_dir: str | Path) -> Dict[str, Any]:
    return analyze_session_dir(session_dir)


# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    # 1) 예전 방식 유지
    # base_path = "../captures/cam4_660nm"
    # img_path = get_latest_image(base_path)
    # detect_porhyrin(img_path, save_dir=".", show_result=True)

    # 2) 새 방식: 세션 폴더 직접 분석
    import sys

    if len(sys.argv) >= 2:
        result = analyze_session_dir(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        base_path = "../captures/cam4_660nm"
        img_path = get_latest_image(base_path)
        result = detect_porhyrin(img_path, save_dir=".", show_result=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
