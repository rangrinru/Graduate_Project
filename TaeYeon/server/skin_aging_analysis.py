from datetime import datetime
from pathlib import Path
import json

import cv2
import numpy as np


def clamp(value, low, high):
    return max(low, min(high, value))


def get_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load OpenCV face cascade.")
    return face_cascade


def detect_largest_face(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    return int(x), int(y), int(w), int(h), "haar_face"


def fallback_face_roi(img):
    img_h, img_w = img.shape[:2]
    w = int(img_w * 0.62)
    h = int(img_h * 0.72)
    x = (img_w - w) // 2
    y = int(img_h * 0.12)
    return x, y, w, h, "center_fallback"


def expand_bbox(bbox, img_shape, ratio_x=0.16, ratio_y=0.18):
    x, y, w, h, method = bbox
    img_h, img_w = img_shape[:2]
    ex = int(w * ratio_x)
    ey = int(h * ratio_y)

    x1 = max(0, x - ex)
    y1 = max(0, y - ey)
    x2 = min(img_w, x + w + ex)
    y2 = min(img_h, y + h + ey)
    return x1, y1, x2 - x1, y2 - y1, method


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


def bbox_from_landmarks(pts, img_shape):
    if not pts:
        return None

    img_h, img_w = img_shape[:2]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = clamp(min(xs), 0, img_w - 1)
    y1 = clamp(min(ys), 0, img_h - 1)
    x2 = clamp(max(xs), 0, img_w - 1)
    y2 = clamp(max(ys), 0, img_h - 1)
    return expand_bbox((x1, y1, x2 - x1, y2 - y1, "mediapipe_landmark"), img_shape, 0.06, 0.08)


def make_landmark_face_mask(shape, pts):
    if not pts or len(pts) < 20:
        return None

    mask = np.zeros(shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = np.ones((15, 15), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


def valid_face_mask(l_channel):
    low = np.percentile(l_channel, 18)
    high = np.percentile(l_channel, 99.4)
    mask = cv2.inRange(l_channel, int(low), int(high))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def anatomy_mask(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, int(h * 0.52))
    axes = (int(w * 0.43), int(h * 0.48))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    remove = np.zeros_like(mask)
    cv2.ellipse(remove, (int(w * 0.31), int(h * 0.35)), (int(w * 0.25), int(h * 0.18)), 0, 0, 360, 255, -1)
    cv2.ellipse(remove, (int(w * 0.69), int(h * 0.35)), (int(w * 0.25), int(h * 0.18)), 0, 0, 360, 255, -1)
    cv2.ellipse(remove, (int(w * 0.50), int(h * 0.76)), (int(w * 0.31), int(h * 0.16)), 0, 0, 360, 255, -1)
    cv2.ellipse(remove, (int(w * 0.50), int(h * 0.57)), (int(w * 0.13), int(h * 0.08)), 0, 0, 360, 255, -1)
    return cv2.bitwise_and(mask, cv2.bitwise_not(remove))


def cheek_focus_mask(shape, landmark_pts=None):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if landmark_pts and len(landmark_pts) >= 469:
        xs = [p[0] for p in landmark_pts]
        ys = [p[1] for p in landmark_pts]
        face_x1, face_x2 = min(xs), max(xs)
        face_y1, face_y2 = min(ys), max(ys)
        face_w = max(1, face_x2 - face_x1)
        face_h = max(1, face_y2 - face_y1)

        def avg(indices, axis, fallback):
            values = [landmark_pts[index][axis] for index in indices if index < len(landmark_pts)]
            return sum(values) / float(len(values)) if values else fallback

        nose_x = avg([1, 2, 4, 5], 0, face_x1 + face_w * 0.50)
        eye_y = avg([33, 133, 159, 145, 263, 362, 386, 374], 1, face_y1 + face_h * 0.34)
        mouth_y = avg([13, 14, 61, 291, 0, 17], 1, face_y1 + face_h * 0.68)
        cheek_y = eye_y + (mouth_y - eye_y) * 0.52

        cheek_axes = (int(face_w * 0.16), int(face_h * 0.14))
        left_center = (int(nose_x - face_w * 0.25), int(cheek_y))
        right_center = (int(nose_x + face_w * 0.25), int(cheek_y))
        cv2.ellipse(mask, left_center, cheek_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(mask, right_center, cheek_axes, 0, 0, 360, 255, -1)

        remove = np.zeros_like(mask)
        cv2.ellipse(remove, (int(nose_x), int(face_y1 + face_h * 0.53)), (int(face_w * 0.13), int(face_h * 0.20)), 0, 0, 360, 255, -1)
        cv2.rectangle(remove, (0, 0), (w, int(eye_y + face_h * 0.08)), 255, -1)
        cv2.rectangle(remove, (0, int(mouth_y - face_h * 0.04)), (w, h), 255, -1)

        face_mask = make_landmark_face_mask(shape, landmark_pts)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(remove))
        if face_mask is not None:
            mask = cv2.bitwise_and(mask, face_mask)
        return mask

    cv2.ellipse(mask, (int(w * 0.32), int(h * 0.58)), (int(w * 0.18), int(h * 0.19)), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (int(w * 0.68), int(h * 0.58)), (int(w * 0.18), int(h * 0.19)), 0, 0, 360, 255, -1)

    remove = np.zeros_like(mask)
    cv2.ellipse(remove, (int(w * 0.50), int(h * 0.54)), (int(w * 0.15), int(h * 0.17)), 0, 0, 360, 255, -1)
    cv2.ellipse(remove, (int(w * 0.50), int(h * 0.76)), (int(w * 0.30), int(h * 0.14)), 0, 0, 360, 255, -1)
    cv2.rectangle(remove, (0, 0), (w, int(h * 0.37)), 255, -1)
    cv2.rectangle(remove, (0, int(h * 0.76)), (w, h), 255, -1)

    return cv2.bitwise_and(mask, cv2.bitwise_not(remove))


def freckle_candidates(face_roi, landmark_pts=None):
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(l_channel)
    smooth = cv2.GaussianBlur(enhanced, (5, 5), 0)
    background = cv2.GaussianBlur(smooth, (31, 31), 0)
    dark_spots = cv2.subtract(background, smooth)

    valid_mask = cv2.bitwise_and(valid_face_mask(l_channel), cheek_focus_mask(face_roi.shape, landmark_pts))
    dark_spots = cv2.bitwise_and(dark_spots, dark_spots, mask=valid_mask)

    if np.any(valid_mask):
        threshold_value = max(4, np.percentile(dark_spots[valid_mask > 0], 96.5))
    else:
        threshold_value = 10

    _, mask = cv2.threshold(dark_spots, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.bitwise_and(mask, mask, mask=valid_mask)
    return mask, float(threshold_value), valid_mask


def calculate_predicted_skin_age(freckle_count, freckle_area_rate_percent):
    base_age = 22
    age_offset = (freckle_count * 0.35) + (freckle_area_rate_percent * 8.0)
    predicted_age = int(round(base_age + age_offset))
    predicted_age = max(18, min(70, predicted_age))

    if predicted_age < 30:
        level = "young"
        grade = "A"
        label = "young skin"
    elif predicted_age < 40:
        level = "normal"
        grade = "B"
        label = "normal skin age"
    elif predicted_age < 50:
        level = "aging_care"
        grade = "C"
        label = "aging care"
    else:
        level = "intensive_care"
        grade = "D"
        label = "intensive care"

    return {
        "predicted_skin_age": predicted_age,
        "skin_age_offset": round(float(age_offset), 2),
        "skin_age_level": level,
        "grade": grade,
        "label": label,
        "basis": "freckle_count",
        "base_age": base_age,
    }


def draw_info_panel(img, lines):
    overlay = img.copy()
    cv2.rectangle(overlay, (18, 18), (600, 190), (20, 20, 20), -1)
    img[:] = cv2.addWeighted(overlay, 0.62, img, 0.38, 0)

    y = 52
    for idx, line in enumerate(lines):
        color = (0, 255, 255) if idx == 0 else (255, 255, 255)
        scale = 0.78 if idx == 0 else 0.64
        thickness = 2 if idx == 0 else 1
        cv2.putText(img, line, (38, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += 34


def analyze_skin_aging_405nm(
    image_path: Path,
    output_dir: Path,
    face_reference_path: Path = None,
    landmark_extractor=None,
):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError("Failed to load 405nm image")

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
        landmark_pts = rescale_landmarks(landmark_pts, landmark_source_shape, img.shape)

    result = img.copy()
    face = bbox_from_landmarks(landmark_pts, img.shape)
    if face is None:
        face_cascade = get_face_cascade()
        face = detect_largest_face(img, face_cascade) or fallback_face_roi(img)
        face = expand_bbox(face, img.shape)

    fx, fy, fw, fh, face_method = face
    face_roi = img[fy : fy + fh, fx : fx + fw]
    roi_landmarks = None
    if landmark_pts is not None:
        roi_landmarks = [
            (x - fx, y - fy)
            for x, y in landmark_pts
            if fx <= x < fx + fw and fy <= y < fy + fh
        ]

    mask, threshold_value, cheek_mask = freckle_candidates(face_roi, roi_landmarks)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    freckles = []
    total_area = 0.0
    min_area = max(2, int((fw * fh) * 0.000006))
    max_area = max(80, int((fw * fh) * 0.00045))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if circularity < 0.18:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(h)
        if aspect < 0.35 or aspect > 2.60:
            continue

        gx = fx + x
        gy = fy + y
        cx = gx + w // 2
        cy = gy + h // 2
        rel_x = (cx - fx) / float(fw)
        rel_y = (cy - fy) / float(fh)

        if rel_y < 0.38 or rel_y > 0.76:
            continue

        in_left_cheek = 0.14 < rel_x < 0.46
        in_right_cheek = 0.54 < rel_x < 0.86
        if not (in_left_cheek or in_right_cheek):
            continue

        radius = max(5, int(max(w, h) * 0.65))
        cv2.circle(result, (cx, cy), radius, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(result, (cx, cy), 2, (0, 255, 255), -1, cv2.LINE_AA)

        freckles.append(
            {
                "x": int(gx),
                "y": int(gy),
                "width": int(w),
                "height": int(h),
                "center_x": int(cx),
                "center_y": int(cy),
                "area": round(float(area), 2),
            }
        )
        total_area += area

    face_area = fw * fh
    freckle_area_rate = (total_area / face_area) * 100 if face_area else 0.0
    age_result = calculate_predicted_skin_age(len(freckles), freckle_area_rate)

    cv2.rectangle(result, (fx, fy), (fx + fw, fy + fh), (0, 220, 0), 2, cv2.LINE_AA)
    cheek_overlay = np.zeros_like(face_roi)
    cheek_overlay[:, :, 1] = cheek_mask
    blended_roi = cv2.addWeighted(result[fy : fy + fh, fx : fx + fw], 1.0, cheek_overlay, 0.18, 0)
    result[fy : fy + fh, fx : fx + fw] = blended_roi
    draw_info_panel(
        result,
        [
            "405nm Freckle Skin Age Analysis",
            f"Freckle Count: {len(freckles)}",
            f"Predicted Skin Age: {age_result['predicted_skin_age']}",
            f"Age Offset: +{age_result['skin_age_offset']}",
            f"Level: {age_result['skin_age_level']}",
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "skin_aging_result.jpg"
    mask_path = output_dir / "skin_aging_mask.jpg"
    report_path = output_dir / "skin_aging_report.json"

    cv2.imwrite(str(result_path), result)
    cv2.imwrite(str(mask_path), mask)

    report = {
        "metadata_type": "skin_age_prediction",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_note": "Image-processing estimate from 405nm freckle-like regions. Not a medical diagnosis.",
        "freckle_count": len(freckles),
        "freckle_area": float(total_area),
        "freckle_area_rate_percent": round(float(freckle_area_rate), 4),
        "predicted_skin_age": age_result["predicted_skin_age"],
        "skin_age_offset": age_result["skin_age_offset"],
        "skin_age_level": age_result["skin_age_level"],
        "grade": age_result["grade"],
        "label": age_result["label"],
        "basis": age_result["basis"],
        "base_age": age_result["base_age"],
        "threshold_value": threshold_value,
        "min_area": int(min_area),
        "max_area": int(max_area),
        "face_detection_method": face_method,
        "detection_focus": "bilateral_cheeks",
        "face_landmarks_used": bool(landmark_pts is not None),
        "face_bbox": {
            "x": int(fx),
            "y": int(fy),
            "width": int(fw),
            "height": int(fh),
        },
        "freckle_coordinates": freckles,
        "result_path": str(result_path),
        "mask_path": str(mask_path),
        "report_path": str(report_path),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report
