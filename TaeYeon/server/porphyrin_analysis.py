from pathlib import Path
import json

import cv2
import numpy as np


HEATMAP_MIN_VALUE = 25.0
HEATMAP_MAX_VALUE = 110.0
VISIBLE_THRESHOLD = 70
LOCAL_CONTRAST_THRESHOLD = 5
STRONG_ABSOLUTE_THRESHOLD = 100
LOW_VISIBLE_THRESHOLD = 45
LOW_LOCAL_CONTRAST_THRESHOLD = 3
LOW_STRONG_ABSOLUTE_THRESHOLD = 80
MIN_COMPONENT_AREA = 2
TOP_BRIGHT_PERCENT = 5.0
def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def calculate_skin_score_from_porphyrin_count(porphyrin_count: int):
    reference_bad_count = 80
    count = max(0, int(porphyrin_count))
    count_risk = min(count / reference_bad_count, 1.0)
    score = int(round(100 - (70 * count_risk)))

    if score >= 90:
        grade = "A"
        label = "매우 양호"
    elif score >= 80:
        grade = "B"
        label = "양호"
    elif score >= 70:
        grade = "C"
        label = "주의"
    elif score >= 60:
        grade = "D"
        label = "관리 필요"
    else:
        grade = "E"
        label = "집중 관리"

    return {
        "score": score,
        "grade": grade,
        "label": label,
        "basis": "porphyrin_count",
        "porphyrin_count": count,
        "reference_bad_count": reference_bad_count,
    }


def make_porphyrin_face_mask(gray):
    non_black = cv2.inRange(gray, 8, 255)
    kernel = np.ones((21, 21), np.uint8)
    non_black = cv2.morphologyEx(non_black, cv2.MORPH_CLOSE, kernel, iterations=2)
    non_black = cv2.morphologyEx(non_black, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(non_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)

    if contours:
        valid_contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) > gray.size * 0.01
        ]
        if valid_contours:
            cv2.drawContours(mask, valid_contours, -1, 255, thickness=cv2.FILLED)
            return mask

    return np.full_like(gray, 255)


def normalize_region_percentages(region_area, total_area):
    if total_area <= 0:
        return {
            key: 0
            for key in region_area
        }

    raw = {
        key: value / total_area * 100
        for key, value in region_area.items()
    }
    rounded = {
        key: int(round(value))
        for key, value in raw.items()
    }

    diff = 100 - sum(rounded.values())
    if diff != 0:
        order = sorted(
            raw,
            key=lambda key: abs(raw[key] - rounded[key]),
            reverse=True
        )
        step = 1 if diff > 0 else -1
        for idx in range(abs(diff)):
            rounded[order[idx % len(order)]] += step

    return rounded


def normalize_region_scores(region_score):
    total_score = float(sum(region_score.values()))
    if total_score <= 0:
        return {
            key: 0
            for key in region_score
        }

    raw = {
        key: value / total_score * 100.0
        for key, value in region_score.items()
    }
    rounded = {
        key: int(round(value))
        for key, value in raw.items()
    }

    diff = 100 - sum(rounded.values())
    if diff != 0:
        order = sorted(
            raw,
            key=lambda key: abs(raw[key] - rounded[key]),
            reverse=True
        )
        step = 1 if diff > 0 else -1
        for idx in range(abs(diff)):
            rounded[order[idx % len(order)]] += step

    return rounded


def clean_component_mask(mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    accepted_count = 0
    max_component_area = 0

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        clean_mask[labels == label_idx] = 255
        accepted_count += 1
        max_component_area = max(max_component_area, area)

    return clean_mask, accepted_count, max_component_area


def summarize_detected_brightness(heat_scaled, detected_mask):
    detected_values = heat_scaled[detected_mask > 0].astype(np.float32)
    if detected_values.size == 0:
        return {
            "mean": 0.0,
            "top5_max": 0.0,
        }

    percentile = 100.0 - TOP_BRIGHT_PERCENT
    top_threshold = float(np.percentile(detected_values, percentile))
    top_values = detected_values[detected_values >= top_threshold]

    return {
        "mean": float(detected_values.mean()),
        "top5_max": float(top_values.max()) if top_values.size else 0.0,
    }


def build_absolute_porphyrin_heatmap(heat_source, mask=None):
    if mask is None:
        mask = np.full_like(heat_source, 255)

    heat_scaled = cv2.bitwise_and(heat_source, heat_source, mask=mask)
    heatmap_source = np.clip(
        (heat_scaled.astype(np.float32) - HEATMAP_MIN_VALUE)
        * 255.0
        / (HEATMAP_MAX_VALUE - HEATMAP_MIN_VALUE),
        0,
        255,
    ).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap_source, cv2.COLORMAP_JET)

    local_background = cv2.GaussianBlur(heat_scaled, (21, 21), 0)
    bright_detail = cv2.subtract(heat_scaled, local_background)
    thresh = np.where(
        (
            (heat_scaled >= VISIBLE_THRESHOLD)
            & (bright_detail >= LOCAL_CONTRAST_THRESHOLD)
        )
        | (heat_scaled >= STRONG_ABSOLUTE_THRESHOLD),
        255,
        0,
    ).astype(np.uint8)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

    low_thresh = np.where(
        (
            (heat_scaled >= LOW_VISIBLE_THRESHOLD)
            & (bright_detail >= LOW_LOCAL_CONTRAST_THRESHOLD)
        )
        | (heat_scaled >= LOW_STRONG_ABSOLUTE_THRESHOLD),
        255,
        0,
    ).astype(np.uint8)
    low_thresh = cv2.bitwise_and(low_thresh, low_thresh, mask=mask)

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    low_thresh = cv2.morphologyEx(low_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    clean_mask, accepted_count, max_component_area = clean_component_mask(thresh, MIN_COMPONENT_AREA)
    low_mask, low_count, _low_max_component_area = clean_component_mask(low_thresh, MIN_COMPONENT_AREA)
    low_only_mask = cv2.bitwise_and(low_mask, cv2.bitwise_not(clean_mask))

    detected_overlay = np.zeros_like(heatmap)
    detected_overlay[low_only_mask > 0] = (0, 255, 255)
    detected_overlay[clean_mask > 0] = (0, 0, 255)
    heatmap = cv2.addWeighted(heatmap, 1.0, detected_overlay, 0.45, 0)
    low_contours, _ = cv2.findContours(low_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heatmap, low_contours, -1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.drawContours(heatmap, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    return heatmap, clean_mask, low_mask, heatmap_source, accepted_count, low_count, max_component_area


def build_mask_overlay(base_img, strong_mask, low_mask=None):
    mask = strong_mask
    if base_img.shape[:2] != mask.shape[:2]:
        base_img = cv2.resize(
            base_img,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    overlay = base_img.copy()
    if low_mask is not None:
        low_only_mask = cv2.bitwise_and(low_mask, cv2.bitwise_not(strong_mask))
        overlay[low_only_mask > 0] = (0, 255, 255)
    overlay[strong_mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(base_img, 0.72, overlay, 0.28, 0)

    if low_mask is not None:
        low_only_mask = cv2.bitwise_and(low_mask, cv2.bitwise_not(strong_mask))
        low_contours, _ = cv2.findContours(low_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, low_contours, -1, (0, 255, 255), 1, cv2.LINE_AA)
    contours, _ = cv2.findContours(strong_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)
    return overlay


def rescale_landmarks(pts, source_shape, target_shape):
    if not pts:
        return None

    source_h, source_w = source_shape[:2]
    target_h, target_w = target_shape[:2]
    if source_w <= 0 or source_h <= 0:
        return None

    scale_x = target_w / float(source_w)
    scale_y = target_h / float(source_h)

    return [
        (
            clamp(int(round(x * scale_x)), 0, target_w - 1),
            clamp(int(round(y * scale_y)), 0, target_h - 1),
        )
        for x, y in pts
    ]


def make_landmark_face_mask(shape, pts):
    mask = np.zeros(shape, dtype=np.uint8)
    if not pts or len(pts) < 20:
        return None

    points = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def get_landmark_metrics(pts, fallback_rect):
    fx, fy, fw, fh = fallback_rect
    metrics = {
        "eye_y": fy + fh * 0.32,
        "nose_y": fy + fh * 0.50,
        "mouth_y": fy + fh * 0.66,
        "face_mid_x": fx + fw * 0.50,
        "nose_x": fx + fw * 0.50,
    }

    if not pts or len(pts) < 469:
        return metrics

    def avg(indices, axis):
        values = [pts[index][axis] for index in indices if index < len(pts)]
        if not values:
            return None
        return sum(values) / float(len(values))

    eye_y = avg([33, 133, 159, 145, 263, 362, 386, 374], 1)
    mouth_y = avg([13, 14, 61, 291, 0, 17], 1)
    nose_y = avg([1, 2, 4, 5], 1)
    nose_x = avg([1, 2, 4, 5], 0)

    if eye_y is not None:
        metrics["eye_y"] = eye_y
    if mouth_y is not None:
        metrics["mouth_y"] = mouth_y
    if nose_y is not None:
        metrics["nose_y"] = nose_y
    if nose_x is not None:
        metrics["nose_x"] = nose_x

    return metrics


def classify_face_region(x, y, face_rect):
    fx, fy, fw, fh = face_rect
    rel_x = (x - fx) / float(fw) if fw else 0.5
    rel_y = (y - fy) / float(fh) if fh else 0.5

    if 0.39 <= rel_x <= 0.61 and 0.29 <= rel_y < 0.58:
        return "nose"

    if rel_y < 0.35:
        return "forehead"

    if 0.42 <= rel_x <= 0.58 and 0.58 <= rel_y < 0.70:
        return "philtrum"

    if rel_y >= 0.70:
        return "chin"

    if rel_x < 0.50:
        return "right_cheek"

    return "left_cheek"


def classify_face_region_by_landmarks(x, y, face_rect, metrics):
    fx, fy, fw, fh = face_rect
    rel_x = (x - fx) / float(fw) if fw else 0.5
    nose_rel_x = (metrics["nose_x"] - fx) / float(fw) if fw else 0.5

    forehead_bottom = max(fy + fh * 0.34, metrics["eye_y"] + fh * 0.04)
    mouth_y = metrics["mouth_y"]
    philtrum_top = metrics["nose_y"] + fh * 0.08
    philtrum_bottom = mouth_y + fh * 0.04

    if y < forehead_bottom:
        return "forehead"

    if abs(x - metrics["nose_x"]) <= fw * 0.14 and philtrum_top <= y < philtrum_bottom:
        return "philtrum"

    nose_half_width = fw * 0.13
    if abs(x - metrics["nose_x"]) <= nose_half_width and y < philtrum_top:
        return "nose"

    if y >= fy + fh * 0.72:
        return "chin"

    if rel_x < nose_rel_x:
        return "right_cheek"

    return "left_cheek"


def analyze_porphyrin_heatmap_v04(
    image_path: Path,
    output_dir: Path,
    face_reference_path: Path = None,
    landmark_extractor=None,
    white_reference_path: Path = None,
):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError("이미지 로드 실패")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    landmark_pts = None
    landmark_source_shape = None
    if landmark_extractor is not None and face_reference_path is not None:
        reference_img = cv2.imread(str(face_reference_path))
        if reference_img is not None:
            landmark_pts = landmark_extractor(reference_img)
            landmark_source_shape = reference_img.shape

    if landmark_extractor is not None and landmark_pts is None:
        landmark_pts = landmark_extractor(img)
        landmark_source_shape = img.shape

    if landmark_pts is not None and landmark_source_shape is not None:
        landmark_pts = rescale_landmarks(landmark_pts, landmark_source_shape, gray.shape)

    face_mask = make_landmark_face_mask(gray.shape, landmark_pts)
    if face_mask is None:
        face_mask = make_porphyrin_face_mask(gray)

    face_pixels = int(np.count_nonzero(face_mask))
    if face_pixels <= 0:
        face_mask = np.full_like(gray, 255)
        face_pixels = gray.size

    face_contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if face_contours:
        face_rect = cv2.boundingRect(max(face_contours, key=cv2.contourArea))
    else:
        face_rect = (0, 0, gray.shape[1], gray.shape[0])
    landmark_metrics = get_landmark_metrics(landmark_pts, face_rect)

    # Same absolute-threshold preview logic as porphyrin_threshold_viewer.py.
    heatmap, clean_mask, low_mask, heatmap_source, accepted_count, low_count, max_component_area = (
        build_absolute_porphyrin_heatmap(blur, face_mask)
    )

    region_score = {
        "forehead": 0.0,
        "nose": 0.0,
        "philtrum": 0.0,
        "chin": 0.0,
        "right_cheek": 0.0,
        "left_cheek": 0.0,
    }

    ys, xs = np.where(low_mask > 0)
    total_area = float(len(xs))
    low_total_area = float(np.count_nonzero(low_mask))
    intensity_values = heatmap_source[ys, xs].astype(np.float32) / 255.0 if len(xs) else []

    for x, y, intensity in zip(xs, ys, intensity_values):
        if landmark_pts is not None:
            region_key = classify_face_region_by_landmarks(int(x), int(y), face_rect, landmark_metrics)
        else:
            region_key = classify_face_region(int(x), int(y), face_rect)
        region_score[region_key] += float(intensity)

    detection_rate = (total_area / face_pixels) * 100 if face_pixels else 0.0
    brightness_summary = summarize_detected_brightness(blur, low_mask)
    if detection_rate < 1:
        grade = "Low"
    elif detection_rate < 3:
        grade = "Medium"
    else:
        grade = "High"

    region_analysis = normalize_region_scores(region_score)
    skin_score = calculate_skin_score_from_porphyrin_count(low_count)

    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_dir / "porphyrin_heatmap.jpg"
    white_overlay_path = output_dir / "porphyrin_overlay_white.jpg"
    mask_path = output_dir / "porphyrin_mask.jpg"
    face_mask_path = output_dir / "porphyrin_face_mask.jpg"
    report_path = output_dir / "porphyrin_report.json"

    cv2.imwrite(str(heatmap_path), heatmap)
    white_overlay_saved = False
    if white_reference_path is not None:
        white_reference_img = cv2.imread(str(white_reference_path))
        if white_reference_img is not None:
            white_overlay = build_mask_overlay(white_reference_img, clean_mask, low_mask)
            cv2.imwrite(str(white_overlay_path), white_overlay)
            white_overlay_saved = True

    cv2.imwrite(str(mask_path), clean_mask)
    cv2.imwrite(str(face_mask_path), face_mask)

    report = {
        "porphyrin_count": int(low_count),
        "porphyrin_area": float(total_area),
        "strong_porphyrin_count": int(accepted_count),
        "strong_porphyrin_area": float(np.count_nonzero(clean_mask)),
        "low_candidate_count": int(low_count),
        "low_candidate_area": float(low_total_area),
        "porphyrin_mean_brightness": brightness_summary["mean"],
        "porphyrin_top5_max_brightness": brightness_summary["top5_max"],
        "detection_rate_percent": float(detection_rate),
        "face_area_pixels": int(face_pixels),
        "grade": grade,
        "skin_score": skin_score,
        "region_analysis": {
            key: float(value)
            for key, value in region_analysis.items()
        },
        "threshold_percentile": 0,
        "threshold_value": float(VISIBLE_THRESHOLD),
        "local_contrast_threshold": float(LOCAL_CONTRAST_THRESHOLD),
        "strong_absolute_threshold": float(STRONG_ABSOLUTE_THRESHOLD),
        "low_visible_threshold": float(LOW_VISIBLE_THRESHOLD),
        "low_local_contrast_threshold": float(LOW_LOCAL_CONTRAST_THRESHOLD),
        "low_strong_absolute_threshold": float(LOW_STRONG_ABSOLUTE_THRESHOLD),
        "heatmap_scale": "fixed_absolute",
        "heatmap_min_value": HEATMAP_MIN_VALUE,
        "heatmap_max_value": HEATMAP_MAX_VALUE,
        "min_area": MIN_COMPONENT_AREA,
        "max_area": max_component_area,
        "face_landmarks_used": bool(landmark_pts is not None),
        "heatmap_path": str(heatmap_path),
        "white_reference_path": str(white_reference_path) if white_reference_path is not None else None,
        "white_overlay_path": str(white_overlay_path) if white_overlay_saved else None,
        "mask_path": str(mask_path),
        "face_mask_path": str(face_mask_path),
        "report_path": str(report_path),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report



