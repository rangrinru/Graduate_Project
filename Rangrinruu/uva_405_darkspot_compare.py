from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any

import cv2
import numpy as np

# =========================================================
# UVA / 405nm 어두운 부위 검출 설정값
# =========================================================
# DARK_PERCENTILE:
#   405nm 이미지 안에서 어두운 하위 몇 %를 후보로 볼지 결정
# DIFF_THRESHOLD:
#   일반 사진에서는 밝은데 405nm 사진에서 어두운 차이가 이 값 이상이면 후보
# MIN_AREA:
#   너무 작은 노이즈 제거
# MAX_AREA_RATIO:
#   너무 큰 영역은 조명 불균일/배경일 가능성이 있어 제거
# =========================================================
DARK_PERCENTILE = 22
DIFF_THRESHOLD = 28
MIN_AREA = 80
MAX_AREA_RATIO = 0.18

LOCAL_BLUR_SIZE = 61
MORPH_KERNEL_SIZE = 5
DEFAULT_OUTPUT_DIR_NAME = "uva_darkspot_analysis"
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


def load_image_korean_path(path: Path) -> np.ndarray | None:
    path = Path(path)
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def save_image_korean_path(path: Path, img: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        ok, encoded = cv2.imencode(".png", img)

    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")

    encoded.tofile(str(path))


def save_json(path: Path, data: dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def find_default_sessions_root() -> Path:
    candidates = [
        Path.home() / "Graduate_Project" / "captures" / "sessions",
        Path.cwd() / "captures" / "sessions",
        Path.cwd().parent / "captures" / "sessions",
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\captures\sessions"),
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\TaeYeon\captures"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def list_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []

    files = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(f"**/*{ext}"))
        files.extend(folder.glob(f"**/*{ext.upper()}"))

    return sorted(set(files), key=lambda p: p.stat().st_mtime)


def extract_timestamp_key(path: Path) -> str | None:
    """
    예:
    20260407_233614_041_cam2.png
    20260407_233614_041_cam3.png
    → 20260407_233614_041
    """
    m = re.search(r"(\d{8}_\d{6}_\d{3})", path.name)
    if m:
        return m.group(1)
    m = re.search(r"(\d{8}_\d{6})", path.name)
    if m:
        return m.group(1)
    return None


def find_latest_pair(sessions_root: Path) -> tuple[Path, Path]:
    """
    기본 폴더 구조:
    captures/sessions/cam2_no_filter/YYYY-MM-DD/*.png
    captures/sessions/cam3_405nm/YYYY-MM-DD/*.png

    가장 최근 405nm 이미지를 기준으로 timestamp가 같은 no_filter 이미지를 찾음.
    """
    sessions_root = Path(sessions_root)

    no_filter_root_candidates = [
        sessions_root / "cam2_no_filter",
        sessions_root / "no_filter",
        sessions_root / "cam2",
    ]
    uv405_root_candidates = [
        sessions_root / "cam3_405nm",
        sessions_root / "405nm",
        sessions_root / "cam3",
    ]

    no_filter_root = next((p for p in no_filter_root_candidates if p.exists()), no_filter_root_candidates[0])
    uv405_root = next((p for p in uv405_root_candidates if p.exists()), uv405_root_candidates[0])

    no_filter_images = list_images(no_filter_root)
    uv405_images = list_images(uv405_root)

    if not no_filter_images:
        raise FileNotFoundError(f"필터 없는 사진을 찾지 못했습니다: {no_filter_root}")
    if not uv405_images:
        raise FileNotFoundError(f"405nm 필터 사진을 찾지 못했습니다: {uv405_root}")

    no_filter_by_key = {}
    for path in no_filter_images:
        key = extract_timestamp_key(path)
        if key:
            no_filter_by_key[key] = path

    for uv_path in reversed(uv405_images):
        key = extract_timestamp_key(uv_path)
        if key and key in no_filter_by_key:
            return no_filter_by_key[key], uv_path

    return no_filter_images[-1], uv405_images[-1]


def resize_to_match(src: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    return cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance_gray(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def local_normalize(gray: np.ndarray, blur_size: int = LOCAL_BLUR_SIZE) -> np.ndarray:
    """
    조명 불균일 보정.
    전체적으로 어둡거나 가장자리가 어두운 이미지를 보정하기 위한 처리.
    """
    gray = gray.astype(np.float32)
    if blur_size % 2 == 0:
        blur_size += 1

    background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    corrected = gray - background + np.mean(background)
    corrected = np.clip(corrected, 0, 255)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


def align_405_to_no_filter(no_filter_bgr: np.ndarray, uv405_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    405nm 이미지를 no_filter 이미지 좌표계에 맞춤.
    ORB Homography를 우선 시도하고, 실패하면 단순 resize 사용.
    """
    h, w = no_filter_bgr.shape[:2]
    if uv405_bgr.shape[:2] != (h, w):
        uv405_resized = resize_to_match(uv405_bgr, (h, w))
    else:
        uv405_resized = uv405_bgr.copy()

    gray_ref = enhance_gray(to_gray(no_filter_bgr))
    gray_mov = enhance_gray(to_gray(uv405_resized))

    try:
        orb = cv2.ORB_create(nfeatures=1500)
        kp1, des1 = orb.detectAndCompute(gray_ref, None)
        kp2, des2 = orb.detectAndCompute(gray_mov, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return uv405_resized, {"method": "resize_only", "reason": "not_enough_features", "matched_points": 0}

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:80]

        if len(good) < 10:
            return uv405_resized, {"method": "resize_only", "reason": "not_enough_matches", "matched_points": len(good)}

        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return uv405_resized, {"method": "resize_only", "reason": "homography_failed", "matched_points": len(good)}

        aligned = cv2.warpPerspective(uv405_resized, H, (w, h), flags=cv2.INTER_LINEAR)
        inliers = int(mask.sum()) if mask is not None else 0
        return aligned, {"method": "orb_homography", "matched_points": len(good), "inliers": inliers, "homography": H.tolist()}

    except Exception as e:
        return uv405_resized, {"method": "resize_only", "reason": f"exception: {str(e)}"}


def make_face_like_roi_mask(no_filter_gray: np.ndarray) -> np.ndarray:
    """
    얼굴 검출 모델 없이 배경 영향을 줄이기 위한 기본 ROI.
    1) 일반 사진에서 너무 어두운 배경 제거
    2) 중앙 타원 마스크 적용
    """
    h, w = no_filter_gray.shape[:2]
    brightness_mask = no_filter_gray > np.percentile(no_filter_gray, 15)

    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, int(h * 0.52))
    axes = (int(w * 0.34), int(h * 0.44))
    cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 255, -1)

    roi = np.logical_and(brightness_mask, ellipse_mask > 0)
    return roi.astype(np.uint8) * 255


def detect_uva_dark_regions(
    no_filter_bgr: np.ndarray,
    uv405_bgr_aligned: np.ndarray,
    dark_percentile: float = DARK_PERCENTILE,
    diff_threshold: int = DIFF_THRESHOLD,
    min_area: int = MIN_AREA,
    max_area_ratio: float = MAX_AREA_RATIO,
) -> dict[str, Any]:
    h, w = no_filter_bgr.shape[:2]
    image_area = h * w

    no_gray_raw = to_gray(no_filter_bgr)
    uv_gray_raw = to_gray(uv405_bgr_aligned)

    no_gray = local_normalize(enhance_gray(no_gray_raw))
    uv_gray = local_normalize(enhance_gray(uv_gray_raw))

    roi_mask = make_face_like_roi_mask(no_gray_raw)
    roi_bool = roi_mask > 0
    if np.count_nonzero(roi_bool) == 0:
        roi_bool = np.ones_like(no_gray, dtype=bool)

    uv_values_in_roi = uv_gray[roi_bool]
    dark_threshold = int(np.percentile(uv_values_in_roi, dark_percentile))

    diff = cv2.subtract(no_gray, uv_gray)

    dark_mask = np.zeros_like(uv_gray, dtype=np.uint8)
    dark_mask[(uv_gray <= dark_threshold) & (diff >= diff_threshold) & roi_bool] = 255

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    max_area = image_area * max_area_ratio

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        if area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + bw // 2
            cy = y + bh // 2

        region_mask = np.zeros_like(dark_mask)
        cv2.drawContours(region_mask, [contour], -1, 255, -1)
        region_bool = region_mask > 0

        mean_no = float(np.mean(no_gray_raw[region_bool]))
        mean_405 = float(np.mean(uv_gray_raw[region_bool]))
        mean_diff = float(np.mean(diff[region_bool]))
        min_405 = int(np.min(uv_gray_raw[region_bool]))
        max_diff = int(np.max(diff[region_bool]))
        darkness_score = float(mean_diff * area / 1000.0)
        contour_points = contour.reshape(-1, 2).astype(int).tolist()

        regions.append({
            "id": len(regions) + 1,
            "center": {"x": int(cx), "y": int(cy)},
            "bbox": {"x": int(x), "y": int(y), "width": int(bw), "height": int(bh)},
            "area_px": round(area, 2),
            "area_ratio": round(area / image_area, 8),
            "mean_no_filter_brightness": round(mean_no, 4),
            "mean_405nm_brightness": round(mean_405, 4),
            "mean_darkness_difference": round(mean_diff, 4),
            "min_405nm_brightness": int(min_405),
            "max_darkness_difference": int(max_diff),
            "darkness_score": round(darkness_score, 4),
            "contour_points": contour_points,
        })

    regions = sorted(regions, key=lambda r: r["darkness_score"], reverse=True)
    for new_id, region in enumerate(regions, start=1):
        region["id"] = new_id

    total_area = sum(float(r["area_px"]) for r in regions)
    return {
        "no_gray": no_gray_raw,
        "uv_gray": uv_gray_raw,
        "no_norm": no_gray,
        "uv_norm": uv_gray,
        "diff": diff,
        "mask": dark_mask,
        "roi_mask": roi_mask,
        "dark_threshold": dark_threshold,
        "regions": regions,
        "summary": {
            "region_count": len(regions),
            "total_dark_area_px": round(total_area, 2),
            "total_dark_area_ratio": round(total_area / image_area, 8),
            "image_width": int(w),
            "image_height": int(h),
        },
    }


def apply_mask_overlay(base_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.45) -> np.ndarray:
    overlay = base_bgr.copy()
    color_layer = np.zeros_like(base_bgr)
    color_layer[:, :] = color
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(base_bgr[mask_bool], 1.0 - alpha, color_layer[mask_bool], alpha, 0)
    return overlay


def draw_regions(image_bgr: np.ndarray, regions: list[dict[str, Any]], draw_contour=True) -> np.ndarray:
    out = image_bgr.copy()
    for region in regions:
        x = int(region["bbox"]["x"])
        y = int(region["bbox"]["y"])
        bw = int(region["bbox"]["width"])
        bh = int(region["bbox"]["height"])
        cx = int(region["center"]["x"])
        cy = int(region["center"]["y"])
        rid = int(region["id"])

        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.circle(out, (cx, cy), 4, (0, 255, 255), -1)
        label = f"#{rid} area:{int(region['area_px'])}"
        cv2.putText(out, label, (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        if draw_contour and region.get("contour_points"):
            pts = np.array(region["contour_points"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(out, [pts], -1, (0, 255, 255), 2)
    return out


def make_heatmap(diff: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    outside = roi_mask <= 0
    heat[outside] = (heat[outside] * 0.25).astype(np.uint8)
    return heat


def make_comparison_canvas(
    no_filter_bgr: np.ndarray,
    uv405_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    summary: dict[str, Any],
) -> np.ndarray:
    target_h = 520

    def resize_keep(img):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)

    a = resize_keep(no_filter_bgr)
    b = resize_keep(uv405_bgr)
    c = resize_keep(overlay_bgr)
    d = resize_keep(heatmap_bgr)

    min_w = min(a.shape[1], b.shape[1], c.shape[1], d.shape[1])
    a = cv2.resize(a, (min_w, target_h))
    b = cv2.resize(b, (min_w, target_h))
    c = cv2.resize(c, (min_w, target_h))
    d = cv2.resize(d, (min_w, target_h))

    top = np.hstack([a, b])
    bottom = np.hstack([c, d])
    canvas = np.vstack([top, bottom])

    panel_h = 170
    panel = np.full((panel_h, canvas.shape[1], 3), 245, dtype=np.uint8)
    total_count = int(summary["region_count"])
    total_area = float(summary["total_dark_area_px"])
    total_ratio = float(summary["total_dark_area_ratio"])

    lines = [
        "UVA / 405nm Dark Area Comparison",
        f"Detected dark regions: {total_count}",
        f"Total dark area: {total_area:.1f} px  |  Area ratio: {total_ratio:.6f}",
        "Red/Yellow overlay: areas darker in 405nm image than no-filter image",
        "Note: rule-based image comparison result, not a medical diagnosis",
    ]

    y = 35
    for i, line in enumerate(lines):
        color = (20, 20, 20) if i == 0 else (70, 70, 70)
        scale = 0.85 if i == 0 else 0.62
        thickness = 2 if i == 0 else 1
        cv2.putText(panel, line, (25, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += 30

    return np.vstack([canvas, panel])


def analyze_uva_405_darkspots(
    no_filter_path: Path,
    uv405_path: Path,
    output_dir: Path | None = None,
    dark_percentile: float = DARK_PERCENTILE,
    diff_threshold: int = DIFF_THRESHOLD,
    min_area: int = MIN_AREA,
) -> dict[str, Any]:
    no_filter_path = Path(no_filter_path)
    uv405_path = Path(uv405_path)

    no_filter = load_image_korean_path(no_filter_path)
    uv405 = load_image_korean_path(uv405_path)

    if no_filter is None:
        raise RuntimeError(f"필터 없는 사진 로드 실패: {no_filter_path}")
    if uv405 is None:
        raise RuntimeError(f"405nm 필터 사진 로드 실패: {uv405_path}")

    if output_dir is None:
        output_dir = uv405_path.parent / DEFAULT_OUTPUT_DIR_NAME
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uv405_aligned, align_info = align_405_to_no_filter(no_filter, uv405)

    result = detect_uva_dark_regions(
        no_filter_bgr=no_filter,
        uv405_bgr_aligned=uv405_aligned,
        dark_percentile=dark_percentile,
        diff_threshold=diff_threshold,
        min_area=min_area,
    )

    mask = result["mask"]
    diff = result["diff"]
    roi_mask = result["roi_mask"]
    regions = result["regions"]
    summary = result["summary"]

    overlay = apply_mask_overlay(no_filter, mask, color=(0, 0, 255), alpha=0.43)
    overlay = draw_regions(overlay, regions)
    uv405_draw = draw_regions(uv405_aligned, regions)
    heatmap = make_heatmap(diff, roi_mask)
    comparison = make_comparison_canvas(no_filter, uv405_draw, overlay, heatmap, summary)

    overlay_path = output_dir / "uva_darkspot_overlay.jpg"
    compare_path = output_dir / "uva_405_vs_no_filter_compare.jpg"
    heatmap_path = output_dir / "uva_darkspot_heatmap.jpg"
    mask_path = output_dir / "uva_darkspot_mask.png"
    aligned_405_path = output_dir / "aligned_405nm.jpg"
    metadata_path = output_dir / "uva_darkspot_metadata.json"

    save_image_korean_path(overlay_path, overlay)
    save_image_korean_path(compare_path, comparison)
    save_image_korean_path(heatmap_path, heatmap)
    save_image_korean_path(mask_path, mask)
    save_image_korean_path(aligned_405_path, uv405_aligned)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "analysis_type": "no_filter_vs_405nm_darkspot_comparison",
        "description": "필터 없는 사진과 405nm 필터 사진을 비교하여 405nm 이미지에서 상대적으로 어둡게 나타나는 부위를 검출합니다.",
        "input_files": {"no_filter_image": str(no_filter_path), "uv405_image": str(uv405_path)},
        "output_files": {
            "overlay_image": str(overlay_path),
            "comparison_image": str(compare_path),
            "heatmap_image": str(heatmap_path),
            "mask_image": str(mask_path),
            "aligned_405nm_image": str(aligned_405_path),
            "metadata": str(metadata_path),
        },
        "image_info": {
            "width": int(no_filter.shape[1]),
            "height": int(no_filter.shape[0]),
            "no_filter_shape": list(no_filter.shape),
            "uv405_original_shape": list(uv405.shape),
            "uv405_aligned_shape": list(uv405_aligned.shape),
        },
        "alignment": align_info,
        "parameters": {
            "dark_percentile": dark_percentile,
            "diff_threshold": diff_threshold,
            "min_area": min_area,
            "max_area_ratio": MAX_AREA_RATIO,
            "local_blur_size": LOCAL_BLUR_SIZE,
            "morph_kernel_size": MORPH_KERNEL_SIZE,
        },
        "summary": summary,
        "dark_regions": regions,
        "coordinate_system": {
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
            "unit": "pixel",
            "note": "좌표는 no_filter 이미지 기준입니다. 405nm 이미지는 no_filter 기준으로 정합 또는 resize된 뒤 비교됩니다.",
        },
        "interpretation_note": (
            "405nm/UVA 형광 이미지에서 어두운 부위는 멜라닌 색소, 차광, 조명 불균일, 그림자, 머리카락, 반사 차이 등 여러 원인으로 나타날 수 있습니다. "
            "본 결과는 의료 진단이 아니라 졸업작품용 영상 비교 지표입니다."
        ),
    }
    save_json(metadata_path, metadata)

    print("\nUVA / 405nm 어두운 부위 비교 분석 완료")
    print("필터 없는 사진:", no_filter_path)
    print("405nm 사진:", uv405_path)
    print("검출 부위 수:", summary["region_count"])
    print("비교 결과:", compare_path)
    print("오버레이:", overlay_path)
    print("메타데이터:", metadata_path)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-filter", type=str, default=None, help="필터 없는 사진 경로")
    parser.add_argument("--uv405", type=str, default=None, help="405nm 필터 사진 경로")
    parser.add_argument("--sessions-root", type=str, default=None, help="captures/sessions 기준 폴더")
    parser.add_argument("--out", type=str, default=None, help="결과 저장 폴더")
    parser.add_argument("--dark-percentile", type=float, default=DARK_PERCENTILE, help="405nm 어두운 후보 percentile")
    parser.add_argument("--diff-threshold", type=int, default=DIFF_THRESHOLD, help="no-filter와 405nm 차이 임계값")
    parser.add_argument("--min-area", type=int, default=MIN_AREA, help="최소 영역 면적 px")
    args = parser.parse_args()

    if args.no_filter and args.uv405:
        no_filter_image = Path(args.no_filter).expanduser()
        uv405_image = Path(args.uv405).expanduser()
    else:
        sessions_root = Path(args.sessions_root).expanduser() if args.sessions_root else find_default_sessions_root()
        no_filter_image, uv405_image = find_latest_pair(sessions_root)

    out_dir = Path(args.out).expanduser() if args.out else None

    analyze_uva_405_darkspots(
        no_filter_path=no_filter_image,
        uv405_path=uv405_image,
        output_dir=out_dir,
        dark_percentile=args.dark_percentile,
        diff_threshold=args.diff_threshold,
        min_area=args.min_area,
    )
