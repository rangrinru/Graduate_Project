from pathlib import Path
import json

import cv2
import numpy as np

from porphyrin_analysis import (
    classify_face_region,
    classify_face_region_by_landmarks,
    get_landmark_metrics,
    make_landmark_face_mask,
    make_porphyrin_face_mask,
    normalize_region_scores,
    rescale_landmarks,
)


REGION_KEYS = [
    "forehead",
    "nose",
    "philtrum",
    "chin",
    "right_cheek",
    "left_cheek",
]


def _load_image(path: Path, label: str):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"{label} 이미지 로드 실패")
    return img


def _extract_landmarks(landmark_extractor, image):
    if landmark_extractor is None:
        return None
    try:
        return landmark_extractor(image)
    except Exception:
        return None


def _build_face_mask(gray, landmark_pts):
    landmark_mask = make_landmark_face_mask(gray.shape, landmark_pts)
    if landmark_mask is not None:
        return landmark_mask
    return make_porphyrin_face_mask(gray)


def _grade_from_rate(rate):
    if rate < 1.0:
        return "Low"
    if rate < 3.0:
        return "Medium"
    if rate < 7.0:
        return "High"
    return "Very High"


def _make_focus_areas(mask, heat_scaled, face_rect, landmark_pts, landmark_metrics):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = []

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < 60:
            continue

        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_idx]

        component = labels == label_idx
        mean_intensity = float(np.mean(heat_scaled[component])) if np.any(component) else 0.0

        if landmark_pts is not None:
            region_key = classify_face_region_by_landmarks(int(cx), int(cy), face_rect, landmark_metrics)
        else:
            region_key = classify_face_region(int(cx), int(cy), face_rect)

        areas.append({
            "region": region_key,
            "bbox": {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            },
            "center": {
                "x": int(round(cx)),
                "y": int(round(cy)),
            },
            "area": area,
            "mean_intensity": mean_intensity,
            "risk_score": float(area * (mean_intensity / 255.0)),
        })

    areas.sort(key=lambda item: item["risk_score"], reverse=True)

    for idx, item in enumerate(areas[:8], start=1):
        item["id"] = idx

    return areas[:8]


def _draw_focus_overlay(base_img, mask, heat_scaled, focus_areas):
    overlay = base_img.copy()
    color_layer = np.zeros_like(base_img)

    high = heat_scaled >= 210
    medium = (heat_scaled >= 165) & (heat_scaled < 210)
    color_layer[(mask > 0) & medium] = (0, 180, 255)
    color_layer[(mask > 0) & high] = (0, 40, 255)

    overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.42, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    for area in focus_areas:
        bbox = area["bbox"]
        center = area["center"]
        x = int(bbox["x"])
        y = int(bbox["y"])
        w = int(bbox["width"])
        h = int(bbox["height"])

        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 80, 255), 2)
        cv2.circle(overlay, (center["x"], center["y"]), 16, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            str(area["id"]),
            (center["x"] - 6, center["y"] + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def analyze_trouble_risk_map(
    porphyrin_image_path: Path,
    output_dir: Path,
    face_reference_path: Path = None,
    landmark_extractor=None,
):
    porphyrin_img = _load_image(porphyrin_image_path, "660nm")
    base_img = porphyrin_img

    if face_reference_path is not None:
        try:
            base_img = _load_image(face_reference_path, "no filter")
        except Exception:
            base_img = porphyrin_img

    gray = cv2.cvtColor(porphyrin_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    landmark_pts = _extract_landmarks(landmark_extractor, base_img)
    if landmark_pts is not None and base_img.shape[:2] != gray.shape[:2]:
        landmark_pts = rescale_landmarks(landmark_pts, base_img.shape, gray.shape)

    face_mask = _build_face_mask(gray, landmark_pts)
    face_pixels = int(np.count_nonzero(face_mask))
    if face_pixels <= 0:
        face_mask = np.full_like(gray, 255)
        face_pixels = gray.size

    face_points = cv2.findNonZero(face_mask)
    face_rect = cv2.boundingRect(face_points) if face_points is not None else (0, 0, gray.shape[1], gray.shape[0])
    landmark_metrics = get_landmark_metrics(landmark_pts, face_rect)

    face_values = blur[face_mask > 0]
    heat_low = np.percentile(face_values, 45)
    heat_high = np.percentile(face_values, 99.0)
    if heat_high <= heat_low:
        heat_high = heat_low + 1

    heat_scaled = np.clip(
        (blur.astype(np.float32) - heat_low) * 255.0 / (heat_high - heat_low),
        0,
        255
    ).astype(np.uint8)
    heat_scaled = cv2.bitwise_and(heat_scaled, heat_scaled, mask=face_mask)

    risk_threshold = 160
    _, risk_mask = cv2.threshold(heat_scaled, risk_threshold, 255, cv2.THRESH_BINARY)
    risk_mask = cv2.morphologyEx(risk_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    risk_mask = cv2.morphologyEx(risk_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(risk_mask, connectivity=8)
    clean_mask = np.zeros_like(risk_mask)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < 18:
            continue
        clean_mask[labels == label_idx] = 255

    region_score = {key: 0.0 for key in REGION_KEYS}
    ys, xs = np.where(clean_mask > 0)
    for x, y in zip(xs, ys):
        if landmark_pts is not None:
            region_key = classify_face_region_by_landmarks(int(x), int(y), face_rect, landmark_metrics)
        else:
            region_key = classify_face_region(int(x), int(y), face_rect)
        region_score[region_key] += float(heat_scaled[y, x]) / 255.0

    region_analysis = normalize_region_scores(region_score)
    risk_area = int(np.count_nonzero(clean_mask))
    risk_rate = (risk_area / float(face_pixels)) * 100.0 if face_pixels else 0.0
    grade = _grade_from_rate(risk_rate)

    risk_heatmap = cv2.applyColorMap(heat_scaled, cv2.COLORMAP_JET)
    dimmed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    risk_heatmap = cv2.addWeighted(dimmed, 0.20, risk_heatmap, 0.90, 0)
    risk_heatmap[face_mask == 0] = (0, 0, 0)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(risk_heatmap, contours, -1, (255, 255, 255), 1)

    if base_img.shape[:2] != clean_mask.shape[:2]:
        focus_mask = cv2.resize(clean_mask, (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        focus_heat = cv2.resize(heat_scaled, (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        focus_landmarks = _extract_landmarks(landmark_extractor, base_img)
        focus_face_mask = _build_face_mask(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), focus_landmarks)
        focus_points = cv2.findNonZero(focus_face_mask)
        focus_rect = cv2.boundingRect(focus_points) if focus_points is not None else (0, 0, base_img.shape[1], base_img.shape[0])
        focus_metrics = get_landmark_metrics(focus_landmarks, focus_rect)
    else:
        focus_mask = clean_mask
        focus_heat = heat_scaled
        focus_landmarks = landmark_pts
        focus_rect = face_rect
        focus_metrics = landmark_metrics

    focus_areas = _make_focus_areas(focus_mask, focus_heat, focus_rect, focus_landmarks, focus_metrics)
    focus_overlay = _draw_focus_overlay(base_img, focus_mask, focus_heat, focus_areas)

    top_region = max(region_analysis.items(), key=lambda item: item[1])[0] if region_analysis else None

    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_dir / "trouble_risk_heatmap.jpg"
    mask_path = output_dir / "trouble_risk_mask.jpg"
    overlay_path = output_dir / "focus_care_overlay.jpg"
    report_path = output_dir / "trouble_risk_report.json"

    cv2.imwrite(str(heatmap_path), risk_heatmap)
    cv2.imwrite(str(mask_path), clean_mask)
    cv2.imwrite(str(overlay_path), focus_overlay)

    report = {
        "risk_area": float(risk_area),
        "risk_rate_percent": float(risk_rate),
        "risk_grade": grade,
        "region_analysis": {
            key: float(value)
            for key, value in region_analysis.items()
        },
        "focus_areas": focus_areas,
        "top_region": top_region,
        "threshold_value": float(risk_threshold),
        "face_area_pixels": int(face_pixels),
        "face_landmarks_used": bool(landmark_pts is not None),
        "heatmap_path": str(heatmap_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "report_path": str(report_path),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report
