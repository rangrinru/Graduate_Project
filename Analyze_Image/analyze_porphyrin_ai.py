#!/usr/bin/env python3
"""
Practical AI/ML porphyrin analyzer for UVA fluorescence capture sets.

Input structure (compatible with rpicam_03.py):
  captures/
    cam2_no_filter/YYYY-MM-DD/*_cam2.png
    cam3_405nm/YYYY-MM-DD/*_cam3.png
    cam4_660nm/YYYY-MM-DD/*_cam4.png

What it does:
- Finds triplets (cam2/cam3/cam4) with the same timestamp.
- Builds pixel-wise features from the three channels/images.
- Uses unsupervised ML (MiniBatchKMeans) to segment bright porphyrin-like areas.
- Saves mask, overlay, score map, and JSON report per capture set.
- Writes a summary CSV for all processed sets.

Why this approach:
- No labeled training dataset is required.
- Works as a practical first-stage "AI" analyzer.
- Easy to later replace with a U-Net or other deep model when labels exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


CAM_FOLDERS = {
    "cam2": "cam2_no_filter",
    "cam3": "cam3_405nm",
    "cam4": "cam4_660nm",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TRIPLET_RE = re.compile(r"^(?P<stem>.+)_cam(?P<cam>[234])$", re.IGNORECASE)


@dataclass
class CaptureTriplet:
    key: str
    cam2: Path
    cam3: Path
    cam4: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze bright porphyrin areas from UVA capture triplets.")
    parser.add_argument(
        "--captures-root",
        type=Path,
        default=Path.home() / "Graduate_Project" / "captures",
        help="Root folder containing cam2_no_filter, cam3_405nm, cam4_660nm",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "Graduate_Project" / "analysis_ai_results",
        help="Output folder for reports, masks and overlays",
    )
    parser.add_argument("--n-clusters", type=int, default=4, help="KMeans cluster count")
    parser.add_argument("--sample-pixels", type=int, default=60000, help="Maximum sampled pixels for KMeans fit")
    parser.add_argument("--min-area", type=int, default=40, help="Minimum blob area in pixels")
    parser.add_argument("--roi-size", type=int, default=0, help="Optional center square crop. 0 keeps the full image")
    parser.add_argument("--save-debug", action="store_true", help="Also save normalized intermediate images")
    return parser.parse_args()


def iter_images(folder: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not folder.exists():
        return mapping

    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        m = TRIPLET_RE.match(path.stem)
        if not m:
            continue
        stem = m.group("stem")
        mapping[stem] = path
    return mapping


def find_triplets(captures_root: Path) -> List[CaptureTriplet]:
    cam_maps = {cam: iter_images(captures_root / folder) for cam, folder in CAM_FOLDERS.items()}
    common_keys = sorted(set(cam_maps["cam2"]).intersection(cam_maps["cam3"]).intersection(cam_maps["cam4"]))

    triplets = [
        CaptureTriplet(
            key=k,
            cam2=cam_maps["cam2"][k],
            cam3=cam_maps["cam3"][k],
            cam4=cam_maps["cam4"][k],
        )
        for k in common_keys
    ]
    return triplets


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return img
    h, w = img.shape[:2]
    s = min(size, h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def normalize_u8(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    mn, mx = float(img_f.min()), float(img_f.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img_f - mn) / (mx - mn)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def clahe_u8(img: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def build_features(cam2: np.ndarray, cam3: np.ndarray, cam4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Normalize each channel to reduce exposure variation.
    n2 = normalize_u8(cam2)
    n3 = normalize_u8(cam3)
    n4 = normalize_u8(cam4)

    # Use CLAHE on cam4 because porphyrin target is bright in this channel.
    c4 = clahe_u8(n4)

    f2 = n2.astype(np.float32) / 255.0
    f3 = n3.astype(np.float32) / 255.0
    f4 = n4.astype(np.float32) / 255.0
    fc4 = c4.astype(np.float32) / 255.0

    diff43 = np.clip(f4 - f3, 0.0, 1.0)
    diff42 = np.clip(f4 - f2, 0.0, 1.0)
    ratio = (f4 + 1e-3) / (0.5 * (f2 + f3) + 1e-3)
    ratio = np.clip(ratio / 4.0, 0.0, 1.0)

    # Local bright spots are useful for follicular fluorescence.
    blur = cv2.GaussianBlur(c4, (0, 0), 3)
    top_hat = cv2.subtract(c4, blur).astype(np.float32) / 255.0
    top_hat = np.clip(top_hat * 2.0, 0.0, 1.0)

    h, w = cam4.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    x_norm = xx.astype(np.float32) / max(w - 1, 1)
    y_norm = yy.astype(np.float32) / max(h - 1, 1)

    feats = np.stack([f4, diff43, diff42, ratio, top_hat, fc4, x_norm, y_norm], axis=-1)
    return feats.reshape(-1, feats.shape[-1]), c4


def fit_kmeans(features: np.ndarray, n_clusters: int, sample_pixels: int) -> MiniBatchKMeans:
    rng = np.random.default_rng(42)
    n = features.shape[0]
    if n > sample_pixels:
        idx = rng.choice(n, size=sample_pixels, replace=False)
        train_x = features[idx]
    else:
        train_x = features

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=4096,
        n_init=10,
    )
    model.fit(train_x)
    return model


def choose_porhyrin_cluster(labels: np.ndarray, features: np.ndarray, n_clusters: int) -> int:
    # High cam4 brightness + positive (cam4-cam3)/(cam4-cam2) + local contrast
    scores: List[float] = []
    for k in range(n_clusters):
        mask = labels == k
        if not np.any(mask):
            scores.append(-1e9)
            continue
        f = features[mask]
        cam4_mean = float(f[:, 0].mean())
        diff43_mean = float(f[:, 1].mean())
        diff42_mean = float(f[:, 2].mean())
        ratio_mean = float(f[:, 3].mean())
        top_hat_mean = float(f[:, 4].mean())
        score = (2.2 * cam4_mean) + (1.5 * diff43_mean) + (1.5 * diff42_mean) + (1.0 * ratio_mean) + (1.0 * top_hat_mean)
        scores.append(score)
    return int(np.argmax(scores))


def clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255
    return clean


def score_map_from_cluster(labels: np.ndarray, features: np.ndarray, chosen_cluster: int, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    # Soft score: similarity to chosen cluster center-like mean.
    mask = labels == chosen_cluster
    target = features[mask].mean(axis=0)
    dist = np.linalg.norm(features - target, axis=1)
    dist = dist.reshape(h, w)
    score = 1.0 - (dist / (dist.max() + 1e-6))
    return np.clip(score * 255.0, 0, 255).astype(np.uint8)


def overlay_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(base, 0.72, overlay, 0.28, 0.0)


def summarize_blobs(mask: np.ndarray, score_img: np.ndarray, cam4: np.ndarray) -> Tuple[List[dict], dict]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs: List[dict] = []
    total_area = 0
    weighted_intensity_sum = 0.0

    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx, cy = centroids[i]
        region = labels == i
        mean_score = float(score_img[region].mean())
        mean_cam4 = float(cam4[region].mean())
        total_area += area
        weighted_intensity_sum += mean_cam4 * area
        blobs.append({
            "bbox": [x, y, w, h],
            "centroid": [float(cx), float(cy)],
            "area_px": area,
            "mean_score_0_255": mean_score,
            "mean_cam4_gray_0_255": mean_cam4,
        })

    summary = {
        "blob_count": len(blobs),
        "total_area_px": int(total_area),
        "mean_blob_area_px": float(total_area / len(blobs)) if blobs else 0.0,
        "mean_cam4_gray_weighted": float(weighted_intensity_sum / total_area) if total_area > 0 else 0.0,
        "mask_coverage_ratio": float(total_area / mask.size),
    }
    return blobs, summary


def save_debug_images(out_dir: Path, cam2: np.ndarray, cam3: np.ndarray, cam4: np.ndarray) -> None:
    cv2.imwrite(str(out_dir / "cam2_gray.png"), cam2)
    cv2.imwrite(str(out_dir / "cam3_gray.png"), cam3)
    cv2.imwrite(str(out_dir / "cam4_gray.png"), cam4)
    cv2.imwrite(str(out_dir / "cam2_norm.png"), normalize_u8(cam2))
    cv2.imwrite(str(out_dir / "cam3_norm.png"), normalize_u8(cam3))
    cv2.imwrite(str(out_dir / "cam4_norm.png"), normalize_u8(cam4))


def process_triplet(triplet: CaptureTriplet, output_root: Path, args: argparse.Namespace) -> dict:
    cam2 = center_crop(read_gray(triplet.cam2), args.roi_size)
    cam3 = center_crop(read_gray(triplet.cam3), args.roi_size)
    cam4 = center_crop(read_gray(triplet.cam4), args.roi_size)

    features, cam4_clahe = build_features(cam2, cam3, cam4)
    model = fit_kmeans(features, args.n_clusters, args.sample_pixels)
    labels = model.predict(features)
    chosen_cluster = choose_porhyrin_cluster(labels, features, args.n_clusters)

    h, w = cam4.shape[:2]
    raw_mask = (labels.reshape(h, w) == chosen_cluster).astype(np.uint8) * 255
    score_img = score_map_from_cluster(labels, features, chosen_cluster, (h, w))

    # Restrict segmentation to truly bright score regions for stability.
    _, score_thr = cv2.threshold(score_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidate_mask = cv2.bitwise_and(raw_mask, score_thr)
    mask = clean_mask(candidate_mask, args.min_area)
    overlay = overlay_mask(cam4_clahe, mask)

    blobs, summary = summarize_blobs(mask, score_img, cam4_clahe)

    out_dir = output_root / triplet.key
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_debug:
        save_debug_images(out_dir, cam2, cam3, cam4)
    else:
        cv2.imwrite(str(out_dir / "cam4_gray.png"), cam4)

    cv2.imwrite(str(out_dir / "porphyrin_score.png"), score_img)
    cv2.imwrite(str(out_dir / "porphyrin_mask.png"), mask)
    cv2.imwrite(str(out_dir / "porphyrin_overlay.png"), overlay)

    report = {
        "key": triplet.key,
        "source_images": {
            "cam2": str(triplet.cam2),
            "cam3": str(triplet.cam3),
            "cam4": str(triplet.cam4),
        },
        "kmeans": {
            "n_clusters": args.n_clusters,
            "chosen_cluster": int(chosen_cluster),
            "sample_pixels": args.sample_pixels,
        },
        "summary": summary,
        "blobs": blobs,
    }

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    row = {
        "key": triplet.key,
        "blob_count": summary["blob_count"],
        "total_area_px": summary["total_area_px"],
        "mean_blob_area_px": summary["mean_blob_area_px"],
        "mask_coverage_ratio": summary["mask_coverage_ratio"],
        "mean_cam4_gray_weighted": summary["mean_cam4_gray_weighted"],
        "report_json": str(out_dir / "report.json"),
        "overlay_png": str(out_dir / "porphyrin_overlay.png"),
    }
    return row


def write_summary(rows: List[dict], summary_path: Path) -> None:
    fieldnames = [
        "key",
        "blob_count",
        "total_area_px",
        "mean_blob_area_px",
        "mask_coverage_ratio",
        "mean_cam4_gray_weighted",
        "report_json",
        "overlay_png",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    triplets = find_triplets(args.captures_root)
    if not triplets:
        raise SystemExit(f"No cam2/cam3/cam4 triplets found under: {args.captures_root}")

    rows: List[dict] = []
    print(f"Found {len(triplets)} capture sets")
    for i, triplet in enumerate(triplets, start=1):
        print(f"[{i}/{len(triplets)}] analyzing {triplet.key}")
        row = process_triplet(triplet, args.output_root, args)
        rows.append(row)

    write_summary(rows, args.output_root / "summary.csv")
    print(f"Done. Results saved to: {args.output_root}")


if __name__ == "__main__":
    main()
