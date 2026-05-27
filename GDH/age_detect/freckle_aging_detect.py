import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "captures" / "2026-05-27" / "no_filter_2026-05-27"
SAVE_DIR = PROJECT_ROOT / "GDH" / "age_detect" / "aging_info"
WINDOW_NAME = "Freckle Aging Viewer"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, img)
    if ok:
        encoded.tofile(str(path))
    return ok


def get_image_files(input_dir):
    return sorted(
        path
        for path in Path(input_dir).iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "metadata" not in path.name.lower()
    )


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

    # Eye and eyebrow zones are the most common false positives in these dark captures.
    cv2.ellipse(
        remove,
        (int(w * 0.31), int(h * 0.35)),
        (int(w * 0.25), int(h * 0.18)),
        0,
        0,
        360,
        255,
        -1,
    )
    cv2.ellipse(
        remove,
        (int(w * 0.69), int(h * 0.35)),
        (int(w * 0.25), int(h * 0.18)),
        0,
        0,
        360,
        255,
        -1,
    )

    # Lips and nostril-shadow zones tend to be detected as dark round spots.
    cv2.ellipse(
        remove,
        (int(w * 0.50), int(h * 0.76)),
        (int(w * 0.31), int(h * 0.16)),
        0,
        0,
        360,
        255,
        -1,
    )
    cv2.ellipse(
        remove,
        (int(w * 0.50), int(h * 0.57)),
        (int(w * 0.13), int(h * 0.08)),
        0,
        0,
        360,
        255,
        -1,
    )

    return cv2.bitwise_and(mask, cv2.bitwise_not(remove))


def freckle_candidates(face_roi):
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(l_channel)
    smooth = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Local dark-spot contrast: freckles become bright in this difference image.
    background = cv2.GaussianBlur(smooth, (31, 31), 0)
    dark_spots = cv2.subtract(background, smooth)

    valid_mask = cv2.bitwise_and(valid_face_mask(l_channel), anatomy_mask(face_roi.shape))
    dark_spots = cv2.bitwise_and(dark_spots, dark_spots, mask=valid_mask)

    threshold_value = max(7, np.percentile(dark_spots[valid_mask > 0], 98.2)) if np.any(valid_mask) else 14
    _, mask = cv2.threshold(dark_spots, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def classify_aging(count, freckle_rate):
    score = min(100, round((count * 0.75) + (freckle_rate * 18)))

    if score < 30:
        return "Low", score, "Small amount of freckle-like pigmentation detected."
    if score < 65:
        return "Medium", score, "Moderate freckle-like pigmentation detected."
    return "High", score, "High amount of freckle-like pigmentation detected."


def draw_info_panel(img, lines):
    overlay = img.copy()
    panel_h = 190
    cv2.rectangle(overlay, (18, 18), (560, panel_h), (20, 20, 20), -1)
    img[:] = cv2.addWeighted(overlay, 0.62, img, 0.38, 0)

    y = 52
    for idx, line in enumerate(lines):
        color = (0, 255, 255) if idx == 0 else (255, 255, 255)
        scale = 0.82 if idx == 0 else 0.68
        thickness = 2 if idx == 0 else 1
        cv2.putText(img, line, (38, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += 34


def analyze_image(img, face_cascade):
    result = img.copy()
    face = detect_largest_face(img, face_cascade)
    if face is None:
        face = fallback_face_roi(img)

    fx, fy, fw, fh, face_method = expand_bbox(face, img.shape)
    face_roi = img[fy : fy + fh, fx : fx + fw]
    mask = freckle_candidates(face_roi)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    freckles = []
    total_area = 0.0

    min_area = max(4, int((fw * fh) * 0.000012))
    max_area = max(60, int((fw * fh) * 0.00035))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if circularity < 0.32:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(h)
        if aspect < 0.55 or aspect > 1.85:
            continue

        gx = fx + x
        gy = fy + y
        cx = gx + w // 2
        cy = gy + h // 2
        rel_x = (cx - fx) / float(fw)
        rel_y = (cy - fy) / float(fh)

        if rel_y < 0.34 or rel_y > 0.80:
            continue

        # Keep the detector away from eyes, eyebrows, nostrils, and lips.
        in_left_eye_zone = 0.08 < rel_x < 0.50 and 0.16 < rel_y < 0.54
        in_right_eye_zone = 0.50 < rel_x < 0.92 and 0.16 < rel_y < 0.54
        in_nostril_zone = 0.30 < rel_x < 0.70 and 0.46 < rel_y < 0.66
        in_mouth_zone = 0.18 < rel_x < 0.82 and 0.64 < rel_y < 0.92
        if in_left_eye_zone or in_right_eye_zone or in_nostril_zone or in_mouth_zone:
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
    freckle_rate = (total_area / face_area) * 100 if face_area else 0.0
    aging_level, aging_score, note = classify_aging(len(freckles), freckle_rate)

    cv2.rectangle(result, (fx, fy), (fx + fw, fy + fh), (0, 220, 0), 2, cv2.LINE_AA)
    draw_info_panel(
        result,
        [
            "Freckle Aging Analysis",
            f"Freckle Count: {len(freckles)}",
            f"Freckle Area Rate: {freckle_rate:.3f}%",
            f"Aging Level: {aging_level}",
            f"Aging Score: {aging_score}/100",
        ],
    )

    metadata = {
        "metadata_type": "freckle_aging_analysis",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_note": "This is an image-processing estimate, not a medical diagnosis.",
        "analysis_result": {
            "freckle_count": len(freckles),
            "freckle_area_rate_percent": round(float(freckle_rate), 4),
            "aging_level": aging_level,
            "aging_score": aging_score,
            "summary": note,
            "face_detection_method": face_method,
            "face_bbox": {
                "x": int(fx),
                "y": int(fy),
                "width": int(fw),
                "height": int(fh),
            },
            "freckle_coordinates": freckles,
        },
    }
    return result, metadata


def save_result(image_path, result_img, metadata, save_dir):
    base_name = image_path.stem.replace("_cam2", "")
    result_path = save_dir / f"{base_name}_freckle_aging_result.png"
    metadata_path = save_dir / f"{base_name}_freckle_aging_metadata.json"

    metadata["input_image"] = str(image_path)
    metadata["output_image"] = str(result_path)
    metadata["metadata_file"] = str(metadata_path)

    if not imwrite_unicode(result_path, result_img):
        raise RuntimeError(f"Failed to save result image: {result_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return result_path, metadata_path


def run_analysis(input_dir=INPUT_DIR, save_dir=SAVE_DIR):
    input_dir = Path(input_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    face_cascade = get_face_cascade()
    image_files = get_image_files(input_dir)

    if not image_files:
        print(f"No image files found: {input_dir}")
        return []

    print(f"Input folder : {input_dir}")
    print(f"Save folder  : {save_dir}")
    print(f"Image count  : {len(image_files)}")

    results = []
    for image_path in image_files:
        img = imread_unicode(image_path)
        if img is None:
            print(f"[SKIP] Failed to read: {image_path.name}")
            continue

        try:
            result_img, metadata = analyze_image(img, face_cascade)
            result_path, metadata_path = save_result(image_path, result_img, metadata, save_dir)
        except Exception as exc:
            print(f"[FAIL] {image_path.name}: {exc}")
            continue

        result = metadata["analysis_result"]
        print(
            "[OK] "
            f"{image_path.name} -> count={result['freckle_count']}, "
            f"level={result['aging_level']}, score={result['aging_score']}"
        )
        print(f"     image: {result_path}")
        print(f"     json : {metadata_path}")
        results.append(result_path)

    return results


def load_result_images(save_dir=SAVE_DIR):
    return sorted(Path(save_dir).glob("*_freckle_aging_result.png"))


def fit_to_screen(img, max_width=1400, max_height=900):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def viewer(save_dir=SAVE_DIR):
    result_images = load_result_images(save_dir)
    if not result_images:
        print(f"No result images found: {save_dir}")
        return

    idx = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        img = imread_unicode(result_images[idx])
        if img is None:
            print(f"Failed to read result image: {result_images[idx]}")
            break

        display = img.copy()
        h = display.shape[0]
        cv2.putText(
            display,
            f"{idx + 1}/{len(result_images)}",
            (30, h - 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "A: Prev   D: Next   ESC/Q: Exit",
            (30, h - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, fit_to_screen(display))
        key = cv2.waitKey(0) & 0xFF

        if key == ord("d"):
            idx = (idx + 1) % len(result_images)
        elif key == ord("a"):
            idx = (idx - 1) % len(result_images)
        elif key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Detect freckle-like spots and estimate aging level.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--no-analysis", action="store_true", help="Skip analysis and only open the viewer.")
    parser.add_argument("--no-viewer", action="store_true", help="Run analysis without opening the viewer.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.no_analysis:
        run_analysis(args.input_dir, args.save_dir)
    if not args.no_viewer:
        viewer(args.save_dir)


if __name__ == "__main__":
    main()
