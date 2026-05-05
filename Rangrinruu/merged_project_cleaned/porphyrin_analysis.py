from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np

PORPHYRIN_MIN_AREA = 20
ROI_MARGIN_RATIO = 0.15
DEFAULT_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def load_session_metadata(session_dir: str | Path) -> Dict[str, Any]:
    session_meta = Path(session_dir) / 'session_metadata.json'
    if not session_meta.exists():
        return {}
    try:
        return json.loads(session_meta.read_text(encoding='utf-8'))
    except Exception:
        return {}


def resolve_session_capture_file(session_dir: str | Path, cam_key: str) -> Path:
    session_dir = Path(session_dir)
    meta = load_session_metadata(session_dir)
    files = meta.get('files') or {}
    meta_path = files.get(cam_key)
    if meta_path:
        candidate = Path(meta_path)
        if not candidate.is_absolute():
            candidate = session_dir / candidate
        if candidate.exists():
            return candidate

    for ext in DEFAULT_EXTENSIONS:
        candidate = session_dir / f'{cam_key}{ext}'
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f'{cam_key} 이미지 파일을 찾을 수 없습니다: {session_dir}')


def load_trigger_bbox(session_dir: str | Path):
    data = load_session_metadata(session_dir)
    trigger = data.get('trigger_detection') or {}
    bbox = trigger.get('bbox')
    if bbox and len(bbox) == 4:
        return tuple(int(v) for v in bbox)
    return None


def build_face_roi_mask(shape, bbox, margin_ratio: float = ROI_MARGIN_RATIO):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if bbox is None:
        mask[:] = 255
        return mask

    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    rx1 = max(0, x1 - mx)
    ry1 = max(0, y1 - my)
    rx2 = min(w, x2 + mx)
    ry2 = min(h, y2 + my)

    cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, thickness=-1)
    return cv2.GaussianBlur(mask, (31, 31), 0)


def save_image(path: Path, img):
    ok = False
    if path.suffix.lower() == '.png':
        ok = cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        ok = cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    if not ok:
        raise RuntimeError(f'이미지 저장 실패: {path}')


def analyze_session_dir(session_dir: str | Path) -> Dict[str, Any]:
    session_dir = Path(session_dir)

    cam2_path = resolve_session_capture_file(session_dir, 'cam2')
    cam3_path = resolve_session_capture_file(session_dir, 'cam3')
    cam4_path = resolve_session_capture_file(session_dir, 'cam4')

    cam2 = cv2.imread(str(cam2_path), cv2.IMREAD_GRAYSCALE)
    cam3 = cv2.imread(str(cam3_path), cv2.IMREAD_GRAYSCALE)
    cam4 = cv2.imread(str(cam4_path), cv2.IMREAD_GRAYSCALE)

    if cam4 is None:
        raise RuntimeError('cam4 이미지를 불러오지 못했습니다.')
    if cam2 is None:
        cam2 = cv2.GaussianBlur(cam4, (3, 3), 0)
    if cam3 is None:
        cam3 = cv2.GaussianBlur(cam4, (3, 3), 0)

    bbox = load_trigger_bbox(session_dir)
    roi_mask = build_face_roi_mask(cam4.shape, bbox)

    score = (
        0.65 * cam4.astype(np.float32)
        + 0.20 * np.maximum(cam4.astype(np.float32) - cam3.astype(np.float32), 0)
        + 0.15 * np.maximum(cam4.astype(np.float32) - cam2.astype(np.float32), 0)
    )
    score = score * (0.35 + 0.65 * (roi_mask.astype(np.float32) / 255.0))
    score_norm = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(score_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(mask, roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)

    porphyrin_count = 0
    porphyrin_area = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= PORPHYRIN_MIN_AREA:
            clean_mask[labels == label_idx] = 255
            porphyrin_count += 1
            porphyrin_area += area

    cam4_bgr = cv2.imread(str(cam4_path), cv2.IMREAD_COLOR)
    if cam4_bgr is None:
        raise RuntimeError('cam4 컬러 이미지를 불러오지 못했습니다.')

    overlay = cam4_bgr.copy()
    overlay[clean_mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(cam4_bgr, 0.7, overlay, 0.3, 0)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)

    analysis_dir = session_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    mask_path = analysis_dir / 'porphyrin_mask.png'
    overlay_path = analysis_dir / 'porphyrin_overlay.png'
    roi_mask_path = analysis_dir / 'face_roi_mask.png'
    report_path = analysis_dir / 'report.json'

    save_image(mask_path, clean_mask)
    save_image(overlay_path, overlay)
    save_image(roi_mask_path, roi_mask)

    mean_intensity = float(score_norm[clean_mask > 0].mean()) if (clean_mask > 0).any() else 0.0

    report = {
        'exists': True,
        'session_dir': str(session_dir),
        'porphyrin_count': porphyrin_count,
        'porphyrin_area': porphyrin_area,
        'mean_intensity': mean_intensity,
        'face_bbox': bbox,
        'cam2_path': str(cam2_path),
        'cam3_path': str(cam3_path),
        'cam4_path': str(cam4_path),
        'mask_path': str(mask_path),
        'overlay_path': str(overlay_path),
        'roi_mask_path': str(roi_mask_path),
        'report_path': str(report_path),
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def load_analysis_report(session_dir: str | Path) -> Optional[Dict[str, Any]]:
    report_path = Path(session_dir) / 'analysis' / 'report.json'
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding='utf-8'))
    except Exception:
        return None
