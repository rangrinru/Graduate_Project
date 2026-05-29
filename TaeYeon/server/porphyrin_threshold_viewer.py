from pathlib import Path
import argparse

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "captures" / "2026-05-27" / "660nm_2026-05-27"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

HEATMAP_MIN_VALUE = 25.0
HEATMAP_MAX_VALUE = 110.0
VISIBLE_THRESHOLD = 70
LOCAL_CONTRAST_THRESHOLD = 5
STRONG_ABSOLUTE_THRESHOLD = 100
MIN_COMPONENT_AREA = 2


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_images(input_dir: Path):
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "metadata" not in path.name.lower()
    )


def fit_to_screen(img, max_width=2600, max_height=1350):
    h, w = img.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def analyze_preview(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    heat_scaled = blur.copy()
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

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    clean_mask = np.zeros_like(gray)
    count = 0
    area_total = 0

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA:
            continue
        clean_mask[labels == label_idx] = 255
        count += 1
        area_total += area

    detected_overlay = np.zeros_like(heatmap)
    detected_overlay[clean_mask > 0] = (0, 0, 255)
    heatmap = cv2.addWeighted(heatmap, 1.0, detected_overlay, 0.45, 0)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heatmap, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    return heatmap, clean_mask, count, area_total


def make_display(original, heatmap, mask, image_path, index, total, count, area_total):
    original_view = original.copy()
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    h, w = heatmap.shape[:2]
    original_view = cv2.resize(original_view, (w, h), interpolation=cv2.INTER_AREA)
    mask_bgr = cv2.resize(mask_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

    combined = np.hstack([original_view, heatmap, mask_bgr])
    panel_h = 105
    panel = np.zeros((panel_h, combined.shape[1], 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)

    lines = [
        f"{index + 1}/{total}  {image_path.name}",
        f"count={count}  area_px={area_total}  threshold={VISIBLE_THRESHOLD}  local={LOCAL_CONTRAST_THRESHOLD}  strong={STRONG_ABSOLUTE_THRESHOLD}",
        "A: previous   D: next   R: reload current   Q/ESC: quit",
    ]
    y = 30
    for idx, line in enumerate(lines):
        color = (0, 255, 255) if idx == 0 else (230, 230, 230)
        cv2.putText(panel, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2 if idx == 0 else 1, cv2.LINE_AA)
        y += 30

    return np.vstack([panel, combined])


def run_viewer(input_dir: Path):
    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No image files found: {input_dir}")

    window_name = "Porphyrin Threshold Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    index = 0

    while True:
        image_path = images[index]
        img = imread_unicode(image_path)
        if img is None:
            print(f"[SKIP] failed to read: {image_path}")
            index = (index + 1) % len(images)
            continue

        heatmap, mask, count, area_total = analyze_preview(img)
        display = make_display(img, heatmap, mask, image_path, index, len(images), count, area_total)
        cv2.imshow(window_name, fit_to_screen(display))

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("d"):
            index = (index + 1) % len(images)
        elif key == ord("a"):
            index = (index - 1) % len(images)
        elif key == ord("r"):
            images = list_images(input_dir)
            index = min(index, len(images) - 1)

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Quick 660nm porphyrin threshold viewer.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing 660nm images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_viewer(Path(args.input_dir))
