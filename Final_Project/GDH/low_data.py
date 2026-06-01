from pathlib import Path
import argparse

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "captures" / "2026-05-27" / "660nm_2026-05-27"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
OUTPUT_SUFFIX = "_top5_summary"

HEATMAP_MIN_VALUE = 25.0
HEATMAP_MAX_VALUE = 110.0
VISIBLE_THRESHOLD = 70
LOCAL_CONTRAST_THRESHOLD = 5
STRONG_ABSOLUTE_THRESHOLD = 100
MIN_COMPONENT_AREA = 2
TOP_BRIGHT_PERCENT = 5.0


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    ext = path.suffix or ".png"
    ok, data = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    data.tofile(str(path))



def list_images(input_dir: Path):
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "metadata" not in path.name.lower()
        and OUTPUT_SUFFIX not in path.stem
    )


def fit_to_screen(img, max_width=2600, max_height=1350):
    h, w = img.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def summarize_strong_brightness(heat_scaled, clean_mask):
    detected_values = heat_scaled[clean_mask > 0]
    if detected_values.size == 0:
        return np.zeros_like(clean_mask), {
            "threshold": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "area_px": 0,
            "ratio": 0.0,
        }

    percentile = 100.0 - TOP_BRIGHT_PERCENT
    strong_threshold = float(np.percentile(detected_values, percentile))
    strong_mask = np.where(
        (clean_mask > 0) & (heat_scaled >= strong_threshold),
        255,
        0,
    ).astype(np.uint8)

    strong_values = heat_scaled[strong_mask > 0]
    if strong_values.size == 0:
        return strong_mask, {
            "threshold": strong_threshold,
            "mean": 0.0,
            "max": 0.0,
            "area_px": 0,
            "ratio": 0.0,
        }

    return strong_mask, {
        "threshold": strong_threshold,
        "mean": float(strong_values.mean()),
        "max": float(strong_values.max()),
        "area_px": int(strong_values.size),
        "ratio": float(strong_values.size / max(detected_values.size, 1) * 100.0),
    }


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

    strong_mask, strong_stats = summarize_strong_brightness(heat_scaled, clean_mask)

    detected_overlay = np.zeros_like(heatmap)
    detected_overlay[clean_mask > 0] = (0, 0, 255)
    heatmap = cv2.addWeighted(heatmap, 1.0, detected_overlay, 0.45, 0)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heatmap, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    return heatmap, clean_mask, strong_mask, strong_stats, count, area_total


def draw_top_center_label(img, label):
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    margin_top = 34
    text_size, _ = cv2.getTextSize(label, font, scale, thickness)
    x = max(0, (out.shape[1] - text_size[0]) // 2)
    y = margin_top
    cv2.putText(out, label, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def build_stats_panel(height, stats_width, image_path, index, total, strong_stats):
    panel = np.zeros((height, stats_width, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)

    lines = [
        (f"{index + 1}/{total}", 1.2, (0, 255, 255), 2),
        (image_path.name, 0.72, (0, 255, 255), 2),
        ("", 0.9, (230, 230, 230), 1),
        ("Top 5% brightness", 0.82, (180, 180, 180), 1),
        (f"Mean {strong_stats['mean']:.1f}/255", 1.0, (0, 255, 255), 2),
        (f"Max {strong_stats['max']:.0f}/255", 1.0, (0, 255, 255), 2),
        (f"Area {strong_stats['area_px']} px", 1.0, (0, 255, 255), 2),
    ]

    y = 44
    for text, scale, color, thickness in lines:
        if text:
            cv2.putText(panel, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += 30 if scale < 0.9 else 42

    return panel


def make_display(original, heatmap, mask, strong_mask, strong_stats, image_path, index, total, count, area_total):
    original_view = original.copy()
    h, w = heatmap.shape[:2]
    original_view = cv2.resize(original_view, (w, h), interpolation=cv2.INTER_AREA)
    strong_mask = cv2.resize(strong_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    top5_view = np.zeros_like(heatmap)
    top5_view[strong_mask > 0] = (0, 0, 255)
    strong_contours, _ = cv2.findContours(strong_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(top5_view, strong_contours, -1, (0, 255, 255), 1, cv2.LINE_AA)

    original_view = draw_top_center_label(original_view, "Original")
    heatmap = draw_top_center_label(heatmap, "Heatmap")
    top5_view = draw_top_center_label(top5_view, "Top 5% Porphyrin")

    stats_panel = build_stats_panel(
        height=h,
        stats_width=420,
        image_path=image_path,
        index=index,
        total=total,
        strong_stats=strong_stats,
    )

    return np.hstack([stats_panel, original_view, heatmap, top5_view])


def save_display_next_to_source(image_path: Path, display):
    output_path = image_path.with_name(f"{image_path.stem}{OUTPUT_SUFFIX}.png")
    imwrite_unicode(output_path, display)
    return output_path


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

        heatmap, mask, strong_mask, strong_stats, count, area_total = analyze_preview(img)
        display = make_display(
            img,
            heatmap,
            mask,
            strong_mask,
            strong_stats,
            image_path,
            index,
            len(images),
            count,
            area_total,
        )
        save_display_next_to_source(image_path, display)
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
