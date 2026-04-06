import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CAM_KEYS = ("cam2", "cam3", "cam4")
CAM_FOLDERS = {
    "cam2": "cam2_no_filter",
    "cam3": "cam3_405nm",
    "cam4": "cam4_660nm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Cam2/Cam3/Cam4 capture sets and detect porphyrin candidate regions."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.home() / "Graduate_Project" / "captures",
        help="Root folder created by rpicam_02.py",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "Graduate_Project" / "analysis_results",
        help="Where analysis outputs will be saved",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=40,
        help="Minimum connected-component area (pixels)",
    )
    parser.add_argument(
        "--gaussian-ksize",
        type=int,
        default=5,
        help="Gaussian blur kernel size (odd number, 1 to disable blur)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=-1.0,
        help="Manual threshold for porphyrin score in [0,1]. If negative, auto-threshold is used.",
    )
    parser.add_argument(
        "--cam4-weight",
        type=float,
        default=0.70,
        help="Weight for cam4 intensity in score",
    )
    parser.add_argument(
        "--cam3-weight",
        type=float,
        default=0.20,
        help="Penalty weight for cam3 intensity in score",
    )
    parser.add_argument(
        "--cam2-weight",
        type=float,
        default=0.10,
        help="Penalty weight for cam2 intensity in score",
    )
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def extract_key(path: Path, cam_key: str) -> Optional[str]:
    suffix = f"_{cam_key}"
    stem = path.stem
    if not stem.endswith(suffix):
        return None
    return stem[: -len(suffix)]


def collect_sets(input_root: Path) -> Dict[str, Dict[str, Path]]:
    found: Dict[str, Dict[str, Path]] = {}

    for cam_key, folder_name in CAM_FOLDERS.items():
        folder = input_root / folder_name
        if not folder.exists():
            continue
        for img_path in list_images(folder):
            key = extract_key(img_path, cam_key)
            if key is None:
                continue
            found.setdefault(key, {})[cam_key] = img_path

    complete = {
        key: paths for key, paths in found.items() if all(cam in paths for cam in CAM_KEYS)
    }
    return dict(sorted(complete.items()))


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def percentile_normalize(gray: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    arr = gray.astype(np.float32)
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)


def maybe_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return img
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def normalize_map01(arr: np.ndarray) -> np.ndarray:
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def auto_threshold(score: np.ndarray) -> float:
    # Robust auto-threshold from high tail of distribution
    p97 = float(np.percentile(score, 97.0))
    p99 = float(np.percentile(score, 99.0))
    mu = float(score.mean())
    sigma = float(score.std())
    thr = max(mu + 1.5 * sigma, p97)
    thr = min(thr, p99)
    thr = max(0.20, min(0.95, thr))
    return thr


def build_mask(score: np.ndarray, thr: float, min_area: int) -> np.ndarray:
    mask = (score >= thr).astype(np.uint8) * 255

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label_idx] = 255
    return cleaned


def mask_to_overlay(base_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(base, 0.75, overlay, 0.25, 0)


def make_score_visual(score: np.ndarray) -> np.ndarray:
    vis = np.clip(score * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def component_stats(mask: np.ndarray, cam4_gray: np.ndarray, score: np.ndarray) -> List[dict]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: List[dict] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_idx]
        region = labels == label_idx
        mean_cam4 = float(cam4_gray[region].mean()) if np.any(region) else 0.0
        mean_score = float(score[region].mean()) if np.any(region) else 0.0
        components.append(
            {
                "id": label_idx,
                "area_px": area,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "centroid": {"x": float(cx), "y": float(cy)},
                "mean_cam4_intensity": mean_cam4,
                "mean_score": mean_score,
            }
        )
    components.sort(key=lambda c: c["area_px"], reverse=True)
    return components


def analyze_set(
    key: str,
    paths: Dict[str, Path],
    output_root: Path,
    min_area: int,
    gaussian_ksize: int,
    score_threshold: float,
    cam4_weight: float,
    cam3_weight: float,
    cam2_weight: float,
) -> dict:
    cam2_gray = read_gray(paths["cam2"])
    cam3_gray = read_gray(paths["cam3"])
    cam4_gray = read_gray(paths["cam4"])

    # Safe size alignment if needed
    h = min(cam2_gray.shape[0], cam3_gray.shape[0], cam4_gray.shape[0])
    w = min(cam2_gray.shape[1], cam3_gray.shape[1], cam4_gray.shape[1])
    cam2_gray = cam2_gray[:h, :w]
    cam3_gray = cam3_gray[:h, :w]
    cam4_gray = cam4_gray[:h, :w]

    cam2_n = maybe_blur(percentile_normalize(cam2_gray), gaussian_ksize)
    cam3_n = maybe_blur(percentile_normalize(cam3_gray), gaussian_ksize)
    cam4_n = maybe_blur(percentile_normalize(cam4_gray), gaussian_ksize)

    # Score idea:
    # porphyrin candidate = strong in cam4, weaker in cam3/cam2
    diff_score = cam4_weight * cam4_n - cam3_weight * cam3_n - cam2_weight * cam2_n
    diff_score = normalize_map01(diff_score)

    ratio = cam4_n / (cam3_n + 0.05)
    ratio = normalize_map01(ratio)

    final_score = normalize_map01(0.7 * diff_score + 0.3 * ratio)

    thr = auto_threshold(final_score) if score_threshold < 0 else float(score_threshold)
    mask = build_mask(final_score, thr, min_area)
    overlay = mask_to_overlay(cam4_gray, mask)
    score_vis = make_score_visual(final_score)

    components = component_stats(mask, cam4_gray, final_score)

    area_px = int(np.count_nonzero(mask))
    total_px = int(mask.shape[0] * mask.shape[1])
    area_ratio = float(area_px / total_px) if total_px else 0.0
    mean_cam4_all = float(cam4_gray.mean())
    mean_cam4_mask = float(cam4_gray[mask > 0].mean()) if area_px > 0 else 0.0
    mean_score_mask = float(final_score[mask > 0].mean()) if area_px > 0 else 0.0

    out_dir = output_root / key
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "cam2_gray.png"), cam2_gray)
    cv2.imwrite(str(out_dir / "cam3_gray.png"), cam3_gray)
    cv2.imwrite(str(out_dir / "cam4_gray.png"), cam4_gray)
    cv2.imwrite(str(out_dir / "porphyrin_score.png"), score_vis)
    cv2.imwrite(str(out_dir / "porphyrin_mask.png"), mask)
    cv2.imwrite(str(out_dir / "porphyrin_overlay.png"), overlay)

    report = {
        "set_key": key,
        "input_files": {cam_key: str(path) for cam_key, path in paths.items()},
        "parameters": {
            "min_area": min_area,
            "gaussian_ksize": gaussian_ksize,
            "score_threshold": thr,
            "cam4_weight": cam4_weight,
            "cam3_weight": cam3_weight,
            "cam2_weight": cam2_weight,
        },
        "summary": {
            "image_width": w,
            "image_height": h,
            "porphyrin_candidate_count": len(components),
            "porphyrin_area_px": area_px,
            "porphyrin_area_ratio": area_ratio,
            "cam4_mean_intensity_all": mean_cam4_all,
            "cam4_mean_intensity_mask": mean_cam4_mask,
            "score_mean_mask": mean_score_mask,
        },
        "components": components,
    }

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    return {
        "set_key": key,
        "candidate_count": len(components),
        "area_px": area_px,
        "area_ratio": area_ratio,
        "cam4_mean_all": mean_cam4_all,
        "cam4_mean_mask": mean_cam4_mask,
        "score_mean_mask": mean_score_mask,
        "threshold": thr,
    }


def write_summary_csv(output_root: Path, rows: List[dict]) -> None:
    csv_path = output_root / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "set_key",
                "candidate_count",
                "area_px",
                "area_ratio",
                "cam4_mean_all",
                "cam4_mean_mask",
                "score_mean_mask",
                "threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    sets = collect_sets(args.input_root)
    if not sets:
        print("분석할 cam2/cam3/cam4 세트를 찾지 못했습니다.")
        print(f"입력 경로를 확인하세요: {args.input_root}")
        return

    print(f"[INFO] 발견된 완전한 촬영 세트 수: {len(sets)}")

    summary_rows = []
    for idx, (key, paths) in enumerate(sets.items(), start=1):
        print(f"[INFO] ({idx}/{len(sets)}) 분석 중: {key}")
        row = analyze_set(
            key=key,
            paths=paths,
            output_root=args.output_root,
            min_area=args.min_area,
            gaussian_ksize=args.gaussian_ksize,
            score_threshold=args.score_threshold,
            cam4_weight=args.cam4_weight,
            cam3_weight=args.cam3_weight,
            cam2_weight=args.cam2_weight,
        )
        summary_rows.append(row)

    write_summary_csv(args.output_root, summary_rows)
    print(f"[완료] 결과 저장 경로: {args.output_root}")
    print(f"[완료] 요약 CSV: {args.output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
