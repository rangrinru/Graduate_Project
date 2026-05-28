from pathlib import Path
import json

import cv2
import numpy as np


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def make_porphyrin_face_mask(gray):
    h, w = gray.shape
    non_black = cv2.inRange(gray, 8, 255)
    kernel = np.ones((21, 21), np.uint8)
    non_black = cv2.morphologyEx(non_black, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(non_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        if bw * bh > gray.size * 0.08:
            center = (x + bw // 2, y + bh // 2)
            axes = (max(1, int(bw * 0.36)), max(1, int(bh * 0.46)))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            return mask

    center = (w // 2, int(h * 0.52))
    axes = (int(w * 0.32), int(h * 0.42))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


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

    landmark_mask = make_landmark_face_mask(gray.shape, landmark_pts)
    face_mask = landmark_mask if landmark_mask is not None else make_porphyrin_face_mask(gray)
    face_pixels = int(np.count_nonzero(face_mask))
    if face_pixels <= 0:
        face_mask = np.full_like(gray, 255)
        face_pixels = gray.size

    face_points = cv2.findNonZero(face_mask)
    if face_points is not None:
        face_rect = cv2.boundingRect(face_points)
    else:
        face_rect = (0, 0, gray.shape[1], gray.shape[0])
    landmark_metrics = get_landmark_metrics(landmark_pts, face_rect)

    face_values = blur[face_mask > 0]
    heat_low = np.percentile(face_values, 50)
    heat_high = np.percentile(face_values, 98.2)
    if heat_high <= heat_low:
        heat_high = heat_low + 1

    heat_scaled = np.clip(
        (blur.astype(np.float32) - heat_low) * 255.0 / (heat_high - heat_low),
        0,
        255
    ).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat_scaled, cv2.COLORMAP_JET)

    visible_threshold = 155
    _, thresh = cv2.threshold(heat_scaled, visible_threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh, thresh, mask=face_mask)

    if landmark_pts is not None:
        philtrum_threshold = 105
        philtrum_mask = np.zeros_like(gray)
        face_ys, face_xs = np.where((face_mask > 0) & (heat_scaled >= philtrum_threshold))
        for px, py in zip(face_xs, face_ys):
            region_key = classify_face_region_by_landmarks(
                int(px),
                int(py),
                face_rect,
                landmark_metrics
            )
            if region_key == "philtrum":
                philtrum_mask[int(py), int(px)] = 255
        thresh = cv2.bitwise_or(thresh, philtrum_mask)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    clean_mask = np.zeros_like(gray)
    accepted_count = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < 12:
            continue

        clean_mask[labels == label_idx] = 255
        accepted_count += 1

    region_score = {
        "forehead": 0.0,
        "nose": 0.0,
        "philtrum": 0.0,
        "chin": 0.0,
        "right_cheek": 0.0,
        "left_cheek": 0.0,
    }

    ys, xs = np.where(clean_mask > 0)
    total_area = float(len(xs))
    intensity_values = heat_scaled[ys, xs].astype(np.float32) / 255.0 if len(xs) else []

    for x, y, intensity in zip(xs, ys, intensity_values):
        if landmark_pts is not None:
            region_key = classify_face_region_by_landmarks(int(x), int(y), face_rect, landmark_metrics)
        else:
            region_key = classify_face_region(int(x), int(y), face_rect)
        region_score[region_key] += float(intensity)

    detection_rate = (total_area / face_pixels) * 100 if face_pixels else 0.0
    if detection_rate < 1:
        grade = "Low"
    elif detection_rate < 3:
        grade = "Medium"
    else:
        grade = "High"

    region_analysis = normalize_region_scores(region_score)

    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_dir / "porphyrin_heatmap.jpg"
    mask_path = output_dir / "porphyrin_mask.jpg"
    face_mask_path = output_dir / "porphyrin_face_mask.jpg"
    report_path = output_dir / "porphyrin_report.json"

    cv2.imwrite(str(heatmap_path), heatmap)
    cv2.imwrite(str(mask_path), clean_mask)
    cv2.imwrite(str(face_mask_path), face_mask)

    report = {
        "porphyrin_count": int(accepted_count),
        "porphyrin_area": float(total_area),
        "detection_rate_percent": float(detection_rate),
        "face_area_pixels": int(face_pixels),
        "grade": grade,
        "region_analysis": {
            key: float(value)
            for key, value in region_analysis.items()
        },
        "threshold_percentile": 0,
        "threshold_value": float(visible_threshold),
        "min_area": 12,
        "max_area": 0,
        "face_landmarks_used": bool(landmark_pts is not None),
        "heatmap_path": str(heatmap_path),
        "mask_path": str(mask_path),
        "face_mask_path": str(face_mask_path),
        "report_path": str(report_path),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report



