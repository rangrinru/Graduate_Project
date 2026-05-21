from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None


# =========================================================
# 필터 없는 일반 사진 기반 여드름 검출/분류 설정
# =========================================================
# 이 코드는 의료 진단용이 아니라 졸업작품용 규칙 기반 영상처리 코드입니다.
# 실제 피부색, 조명, 카메라 노출에 따라 threshold는 조정해야 합니다.
# =========================================================

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]

DEFAULT_OUTPUT_DIR_NAME = "visible_acne_analysis"

# 얼굴 ROI가 없을 때 사용하는 중앙 타원 ROI
FACE_ELLIPSE_CENTER_Y_RATIO = 0.52
FACE_ELLIPSE_AXIS_X_RATIO = 0.34
FACE_ELLIPSE_AXIS_Y_RATIO = 0.43

# 공통 contour 필터
MIN_ACNE_AREA = 18
MAX_ACNE_AREA_RATIO = 0.025

# 염증성 붉은 여드름 threshold
REDNESS_THRESHOLD = 22
LAB_A_PERCENTILE = 72

# 화농성 중심 후보 threshold
PUSTULE_CENTER_MIN_V = 145
PUSTULE_CENTER_MAX_S = 95
PUSTULE_CENTER_MIN_AREA = 4

# 블랙헤드 후보 threshold
BLACKHEAD_MAX_AREA = 120
BLACKHEAD_MIN_AREA = 5
BLACKHEAD_LOCAL_DARK_DIFF = 18

# 화이트헤드 후보 threshold
WHITEHEAD_MAX_AREA = 160
WHITEHEAD_MIN_AREA = 8
WHITEHEAD_LOCAL_BRIGHT_DIFF = 22

# 중복 제거 IoU threshold
NMS_IOU_THRESHOLD = 0.25


ACNE_TYPE_KO = {
    "pustule": "화농성 여드름",
    "papule": "구진성/염증성 여드름",
    "blackhead": "블랙헤드/개방면포",
    "whitehead": "화이트헤드/폐쇄면포",
    "redness": "붉은 염증 의심 부위",
    "unknown": "분류 어려움",
}


ACNE_SOLUTIONS = {
    "pustule": {
        "summary": "고름처럼 보이는 밝은 중심과 붉은 주변부가 함께 관찰되는 유형입니다.",
        "care": [
            "손으로 짜거나 압출하지 마세요. 염증과 흉터 위험이 커질 수 있습니다.",
            "자극적인 스크럽, 강한 필링 제품 사용을 피하세요.",
            "세안은 하루 2회 정도로 부드럽게 유지하고, 염증 부위는 문지르지 마세요.",
            "붉어짐, 통증, 고름이 반복되거나 넓어지면 피부과 상담을 권장합니다.",
        ],
    },
    "papule": {
        "summary": "붉게 올라온 염증성 여드름으로 보이는 유형입니다.",
        "care": [
            "해당 부위를 손으로 만지거나 누르지 마세요.",
            "마스크, 베개, 휴대폰처럼 얼굴에 닿는 물건을 청결하게 유지하세요.",
            "유분감이 강한 제품을 줄이고, 논코메도제닉 제품을 고려하세요.",
            "수면 부족과 스트레스가 반복되면 염증성 트러블이 심해질 수 있으니 생활 습관을 점검하세요.",
        ],
    },
    "blackhead": {
        "summary": "작고 어두운 점 형태의 개방면포 가능성이 있는 유형입니다.",
        "care": [
            "무리한 코팩이나 손 압출은 피하세요.",
            "저녁 세안 시 코 주변, 볼 중심부 등 피지 부위를 부드럽게 관리하세요.",
            "피지와 각질 관리 제품은 낮은 자극 제품부터 천천히 사용하세요.",
            "반복적으로 같은 부위에 많다면 피지 분비와 모공 관리 루틴을 점검하세요.",
        ],
    },
    "whitehead": {
        "summary": "작고 밝은 돌기 형태의 폐쇄면포 가능성이 있는 유형입니다.",
        "care": [
            "두꺼운 크림, 유분감이 강한 선크림이나 베이스 제품 사용량을 점검하세요.",
            "세안 후 잔여물이 남지 않도록 충분히 헹구세요.",
            "강하게 문지르기보다 순한 세안과 적절한 보습을 유지하세요.",
            "좁쌀처럼 반복되면 사용 중인 화장품이 모공을 막는지 확인하세요.",
        ],
    },
    "redness": {
        "summary": "붉은기 중심으로 검출된 염증 의심 부위입니다.",
        "care": [
            "자극적인 세안, 스크럽, 강한 마찰을 줄이세요.",
            "피부 장벽이 약해졌을 수 있으므로 보습과 자극 회피를 우선하세요.",
            "붉은기가 오래 지속되거나 통증이 있으면 피부과 상담을 고려하세요.",
        ],
    },
    "unknown": {
        "summary": "영상 조건상 명확한 여드름 종류로 분류하기 어려운 부위입니다.",
        "care": [
            "동일한 조명과 거리에서 다시 촬영해 확인하세요.",
            "초점이 흐리거나 그림자가 있으면 오검출될 수 있습니다.",
        ],
    },
}


# =========================================================
# 파일 유틸
# =========================================================
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


def list_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []

    files = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(f"**/*{ext}"))
        files.extend(folder.glob(f"**/*{ext.upper()}"))

    return sorted(set(files), key=lambda p: p.stat().st_mtime)


def find_default_no_filter_root() -> Path:
    candidates = [
        Path.home() / "Graduate_Project" / "captures" / "sessions" / "cam2_no_filter",
        Path.home() / "Graduate_Project" / "TaeYeon" / "captures",
        Path.cwd() / "captures" / "sessions" / "cam2_no_filter",
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\captures\sessions\cam2_no_filter"),
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\TaeYeon\captures"),
    ]

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]


def find_latest_no_filter_image(base_path: Path) -> Path:
    images = list_images(base_path)

    if not images:
        raise FileNotFoundError(f"필터 없는 일반 사진을 찾지 못했습니다: {base_path}")

    return images[-1]


# =========================================================
# 전처리 / ROI
# =========================================================
def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def enhance_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray.astype(np.uint8))


def make_ellipse_face_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (w // 2, int(h * FACE_ELLIPSE_CENTER_Y_RATIO))
    axes = (int(w * FACE_ELLIPSE_AXIS_X_RATIO), int(h * FACE_ELLIPSE_AXIS_Y_RATIO))

    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def detect_face_mask_with_mediapipe(img_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    MediaPipe FaceMesh가 있으면 얼굴 윤곽 기반 mask 생성.
    실패하면 중앙 타원 mask 사용.
    """
    h, w = img_bgr.shape[:2]

    if mp is None:
        return make_ellipse_face_mask((h, w)), {
            "method": "ellipse_fallback",
            "reason": "mediapipe_not_installed"
        }

    try:
        face_mesh = mp.solutions.face_mesh

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.45
        ) as fm:
            result = fm.process(img_rgb)

        if not result.multi_face_landmarks:
            return make_ellipse_face_mask((h, w)), {
                "method": "ellipse_fallback",
                "reason": "face_not_detected"
            }

        # 얼굴 외곽에 가까운 landmark index 일부
        oval_idx = [
            10, 338, 297, 332, 284, 251, 389, 356,
            454, 323, 361, 288, 397, 365, 379, 378,
            400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21,
            54, 103, 67, 109
        ]

        landmarks = result.multi_face_landmarks[0].landmark
        pts = []

        for idx in oval_idx:
            lm = landmarks[idx]
            pts.append([int(lm.x * w), int(lm.y * h)])

        pts = np.array(pts, dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # 눈/입 등 너무 어두운 내부 구조 오검출을 완전히 제거하지는 않음.
        # 여드름이 눈썹/입술에 잡히는 문제만 후처리에서 줄임.
        return mask, {
            "method": "mediapipe_facemesh",
            "reason": "success"
        }

    except Exception as e:
        return make_ellipse_face_mask((h, w)), {
            "method": "ellipse_fallback",
            "reason": f"exception: {str(e)}"
        }


def make_skin_mask(img_bgr: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    """
    피부색에 가까운 영역 + 얼굴 ROI.
    조명에 따라 민감하므로 너무 강하게 제한하지 않음.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    h, s, v = cv2.split(hsv)
    y, cr, cb = cv2.split(ycrcb)

    # 일반적인 피부색 후보. 다양한 피부톤을 위해 넓게 설정.
    skin_hsv = ((h <= 35) | (h >= 160)) & (s >= 15) & (s <= 190) & (v >= 35)
    skin_ycrcb = (cr >= 125) & (cr <= 180) & (cb >= 70) & (cb <= 145) & (y >= 35)

    skin = (skin_hsv | skin_ycrcb) & (face_mask > 0)
    skin = skin.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)

    return skin


# =========================================================
# 검출 함수
# =========================================================
def contour_to_detection(
    contour: np.ndarray,
    acne_type: str,
    img_bgr: np.ndarray,
    score: float,
    extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    h, w = img_bgr.shape[:2]
    area = float(cv2.contourArea(contour))
    x, y, bw, bh = cv2.boundingRect(contour)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx = x + bw // 2
        cy = y + bh // 2

    if extra is None:
        extra = {}

    return {
        "id": -1,
        "type": acne_type,
        "type_ko": ACNE_TYPE_KO.get(acne_type, acne_type),
        "confidence_score": round(float(score), 4),
        "center": {"x": int(cx), "y": int(cy)},
        "bbox": {"x": int(x), "y": int(y), "width": int(bw), "height": int(bh)},
        "area_px": round(area, 2),
        "area_ratio": round(area / (h * w), 8),
        "contour_points": contour.reshape(-1, 2).astype(int).tolist(),
        "extra": extra,
        "solution": ACNE_SOLUTIONS.get(acne_type, ACNE_SOLUTIONS["unknown"]),
    }


def get_contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contour_area(contour: np.ndarray, img_shape: tuple[int, int], min_area: int = MIN_ACNE_AREA) -> bool:
    h, w = img_shape[:2]
    area = cv2.contourArea(contour)
    if area < min_area:
        return False
    if area > h * w * MAX_ACNE_AREA_RATIO:
        return False

    x, y, bw, bh = cv2.boundingRect(contour)
    if bw <= 2 or bh <= 2:
        return False

    # 길쭉한 선/머리카락 같은 후보 제거
    aspect = max(bw / max(bh, 1), bh / max(bw, 1))
    if aspect > 5.5:
        return False

    return True


def detect_red_inflammatory_acne(img_bgr: np.ndarray, skin_mask: np.ndarray) -> list[dict[str, Any]]:
    """
    붉은 염증성 부위 검출.
    이후 화농성 중심이 있으면 pustule, 없으면 papule로 분류.
    """
    h, w = img_bgr.shape[:2]
    b, g, r = cv2.split(img_bgr.astype(np.int16))

    redness = r - ((g + b) // 2)
    redness = np.clip(redness, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    _, lab_a, _ = cv2.split(lab)

    skin_bool = skin_mask > 0

    if np.count_nonzero(skin_bool) > 0:
        lab_a_th = max(138, int(np.percentile(lab_a[skin_bool], LAB_A_PERCENTILE)))
    else:
        lab_a_th = 145

    red_mask = np.zeros((h, w), dtype=np.uint8)
    red_mask[(redness >= REDNESS_THRESHOLD) & (lab_a >= lab_a_th) & skin_bool] = 255

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)

    detections = []

    for contour in get_contours_from_mask(red_mask):
        if not filter_contour_area(contour, img_bgr.shape):
            continue

        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, -1)

        # 붉은 영역 bbox를 약간 확장해서 중심 고름 후보 탐색
        x, y, bw, bh = cv2.boundingRect(contour)
        pad = int(max(bw, bh) * 0.35) + 3
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        patch_hsv = hsv[y1:y2, x1:x2]
        patch_skin = skin_mask[y1:y2, x1:x2]

        ph, ps, pv = cv2.split(patch_hsv)

        # 화농성 중심: 밝고 채도가 낮거나 노란빛이 약간 있는 작은 중심
        white_center = ((pv >= PUSTULE_CENTER_MIN_V) & (ps <= PUSTULE_CENTER_MAX_S) & (patch_skin > 0))
        yellow_center = ((ph >= 15) & (ph <= 42) & (ps >= 35) & (pv >= 120) & (patch_skin > 0))
        center_mask = np.logical_or(white_center, yellow_center).astype(np.uint8) * 255

        center_mask = cv2.morphologyEx(center_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        center_area = 0
        for cc in get_contours_from_mask(center_mask):
            center_area += int(cv2.contourArea(cc))

        area = float(cv2.contourArea(contour))
        mean_redness = float(np.mean(redness[region_mask > 0]))

        if center_area >= PUSTULE_CENTER_MIN_AREA:
            acne_type = "pustule"
            score = min(1.0, 0.45 + mean_redness / 120.0 + center_area / max(area, 1) * 0.5)
        else:
            acne_type = "papule"
            score = min(1.0, 0.35 + mean_redness / 110.0)

        detections.append(
            contour_to_detection(
                contour=contour,
                acne_type=acne_type,
                img_bgr=img_bgr,
                score=score,
                extra={
                    "mean_redness": round(mean_redness, 4),
                    "lab_a_threshold": int(lab_a_th),
                    "pustule_center_area_px": int(center_area),
                }
            )
        )

    return detections


def detect_blackheads(img_bgr: np.ndarray, skin_mask: np.ndarray) -> list[dict[str, Any]]:
    """
    작은 어두운 점 후보 검출.
    눈썹/머리카락/콧구멍/입술 오검출이 있을 수 있으므로 결과 확인 필요.
    """
    gray = to_gray(img_bgr)
    gray_eq = enhance_gray(gray)

    # 지역 평균보다 어두운 작은 점
    blur = cv2.GaussianBlur(gray_eq, (31, 31), 0)
    dark_diff = cv2.subtract(blur, gray_eq)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[(dark_diff >= BLACKHEAD_LOCAL_DARK_DIFF) & (skin_mask > 0)] = 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    detections = []

    for contour in get_contours_from_mask(mask):
        area = cv2.contourArea(contour)
        if area < BLACKHEAD_MIN_AREA or area > BLACKHEAD_MAX_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > 2.8:
            continue

        region_mask = np.zeros_like(mask)
        cv2.drawContours(region_mask, [contour], -1, 255, -1)
        mean_dark = float(np.mean(dark_diff[region_mask > 0]))

        score = min(1.0, 0.3 + mean_dark / 80.0)

        detections.append(
            contour_to_detection(
                contour=contour,
                acne_type="blackhead",
                img_bgr=img_bgr,
                score=score,
                extra={"mean_local_dark_difference": round(mean_dark, 4)}
            )
        )

    return detections


def detect_whiteheads(img_bgr: np.ndarray, skin_mask: np.ndarray) -> list[dict[str, Any]]:
    """
    작은 밝은 돌기 후보 검출.
    반사광/하이라이트를 오검출할 수 있으므로 면적과 채도 조건으로 제한.
    """
    gray = to_gray(img_bgr)
    gray_eq = enhance_gray(gray)

    blur = cv2.GaussianBlur(gray_eq, (31, 31), 0)
    bright_diff = cv2.subtract(gray_eq, blur)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[(bright_diff >= WHITEHEAD_LOCAL_BRIGHT_DIFF) & (s <= 90) & (v >= 110) & (skin_mask > 0)] = 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    detections = []

    for contour in get_contours_from_mask(mask):
        area = cv2.contourArea(contour)
        if area < WHITEHEAD_MIN_AREA or area > WHITEHEAD_MAX_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > 3.0:
            continue

        region_mask = np.zeros_like(mask)
        cv2.drawContours(region_mask, [contour], -1, 255, -1)
        mean_bright = float(np.mean(bright_diff[region_mask > 0]))

        score = min(1.0, 0.3 + mean_bright / 90.0)

        detections.append(
            contour_to_detection(
                contour=contour,
                acne_type="whitehead",
                img_bgr=img_bgr,
                score=score,
                extra={"mean_local_bright_difference": round(mean_bright, 4)}
            )
        )

    return detections


# =========================================================
# 중복 제거 / 통계 / 솔루션
# =========================================================
def bbox_iou(a: dict[str, int], b: dict[str, int]) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0

    return inter / denom


def type_priority(acne_type: str) -> int:
    # 같은 부위가 여러 유형으로 잡히면 더 구체적인 유형 우선
    priority = {
        "pustule": 5,
        "papule": 4,
        "blackhead": 3,
        "whitehead": 3,
        "redness": 2,
        "unknown": 1,
    }
    return priority.get(acne_type, 1)


def non_max_suppression(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_dets = sorted(
        detections,
        key=lambda d: (type_priority(d["type"]), d["confidence_score"], d["area_px"]),
        reverse=True
    )

    kept = []

    for det in sorted_dets:
        duplicate = False

        for k in kept:
            if bbox_iou(det["bbox"], k["bbox"]) >= NMS_IOU_THRESHOLD:
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    kept = sorted(kept, key=lambda d: d["confidence_score"], reverse=True)

    for idx, det in enumerate(kept, start=1):
        det["id"] = idx

    return kept


def build_summary(detections: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: 0 for key in ACNE_TYPE_KO.keys() if key != "unknown"}

    total_area = 0.0

    for det in detections:
        counts[det["type"]] = counts.get(det["type"], 0) + 1
        total_area += float(det.get("area_px", 0))

    main_type = "unknown"
    if detections:
        main_type = max(counts, key=counts.get)

    severity_score = (
        counts.get("pustule", 0) * 8
        + counts.get("papule", 0) * 5
        + counts.get("blackhead", 0) * 2
        + counts.get("whitehead", 0) * 2
    )

    if severity_score <= 5:
        severity_level = "낮음"
    elif severity_score <= 20:
        severity_level = "보통"
    elif severity_score <= 45:
        severity_level = "주의"
    else:
        severity_level = "높음"

    return {
        "total_count": len(detections),
        "type_counts": counts,
        "main_type": main_type,
        "main_type_ko": ACNE_TYPE_KO.get(main_type, main_type),
        "total_area_px": round(total_area, 2),
        "severity_score": int(severity_score),
        "severity_level": severity_level,
    }


def build_management_solution(summary: dict[str, Any]) -> dict[str, Any]:
    type_counts = summary["type_counts"]

    priority_order = ["pustule", "papule", "whitehead", "blackhead"]

    priority_types = [
        t for t in priority_order
        if type_counts.get(t, 0) > 0
    ]

    overall = []

    if summary["total_count"] == 0:
        overall.append("검출된 여드름 후보가 거의 없습니다. 현재 상태를 유지하되 동일 조건에서 주기적으로 촬영하세요.")
    else:
        overall.append(f"총 {summary['total_count']}개의 여드름 후보가 검출되었습니다.")
        overall.append(f"전체 심각도 단계는 '{summary['severity_level']}'입니다.")

    if type_counts.get("pustule", 0) > 0:
        overall.append("화농성 여드름 후보가 있으므로 압출이나 강한 자극을 피하는 것이 가장 중요합니다.")

    if type_counts.get("papule", 0) > 0:
        overall.append("붉은 염증성 후보가 있으므로 마찰, 수면 부족, 스트레스, 마스크 자극을 함께 관리하세요.")

    if type_counts.get("whitehead", 0) > 0:
        overall.append("화이트헤드 후보가 있어 세안 잔여물과 유분감 있는 제품 사용을 점검하는 것이 좋습니다.")

    if type_counts.get("blackhead", 0) > 0:
        overall.append("블랙헤드 후보가 있어 코 주변과 피지 분비 부위의 부드러운 피지 관리가 필요합니다.")

    type_solutions = {}

    for t in priority_order:
        if type_counts.get(t, 0) > 0:
            type_solutions[t] = {
                "type_ko": ACNE_TYPE_KO[t],
                "count": type_counts[t],
                "summary": ACNE_SOLUTIONS[t]["summary"],
                "care": ACNE_SOLUTIONS[t]["care"],
            }

    return {
        "overall_message": overall,
        "priority_types": [
            {
                "type": t,
                "type_ko": ACNE_TYPE_KO[t],
                "count": type_counts[t],
            }
            for t in priority_types
        ],
        "type_solutions": type_solutions,
        "common_notice": [
            "이 결과는 필터 없는 일반 사진 기반 규칙 영상처리 결과이며 의료 진단이 아닙니다.",
            "조명, 초점, 피부색, 그림자, 수염, 점, 잡티에 따라 오검출될 수 있습니다.",
            "통증, 고름, 붉은기 확산, 흉터 위험이 있으면 피부과 상담을 권장합니다.",
        ],
    }


# =========================================================
# 결과 이미지 생성
# =========================================================
def color_for_type(acne_type: str) -> tuple[int, int, int]:
    colors = {
        "pustule": (0, 0, 255),      # red
        "papule": (0, 80, 255),      # orange-red
        "blackhead": (40, 40, 40),   # dark
        "whitehead": (255, 255, 255),
        "redness": (0, 140, 255),
        "unknown": (180, 180, 180),
    }
    return colors.get(acne_type, (180, 180, 180))


def draw_detections(img_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    out = img_bgr.copy()

    for det in detections:
        x = int(det["bbox"]["x"])
        y = int(det["bbox"]["y"])
        w = int(det["bbox"]["width"])
        h = int(det["bbox"]["height"])
        cx = int(det["center"]["x"])
        cy = int(det["center"]["y"])
        label = f"#{det['id']} {det['type_ko']}"

        color = color_for_type(det["type"])

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.circle(out, (cx, cy), 4, color, -1)

        if det.get("contour_points"):
            pts = np.array(det["contour_points"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(out, [pts], -1, color, 2)

        text_bg1 = (x, max(0, y - 24))
        text_bg2 = (min(out.shape[1] - 1, x + 220), y)
        cv2.rectangle(out, text_bg1, text_bg2, color, -1)

        text_color = (0, 0, 0) if det["type"] == "whitehead" else (255, 255, 255)
        cv2.putText(
            out,
            label,
            (x + 4, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            text_color,
            1,
            cv2.LINE_AA
        )

    return out


def make_dashboard(
    original: np.ndarray,
    overlay: np.ndarray,
    skin_mask: np.ndarray,
    summary: dict[str, Any],
    solution: dict[str, Any]
) -> np.ndarray:
    h, w = original.shape[:2]
    target_h = 620

    def resize_keep(img):
        ih, iw = img.shape[:2]
        scale = target_h / ih
        return cv2.resize(img, (int(iw * scale), target_h), interpolation=cv2.INTER_AREA)

    original_r = resize_keep(original)
    overlay_r = resize_keep(overlay)

    skin_mask_bgr = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
    skin_mask_bgr[skin_mask > 0] = (90, 180, 90)
    skin_r = resize_keep(cv2.addWeighted(original, 0.65, skin_mask_bgr, 0.35, 0))

    min_w = min(original_r.shape[1], overlay_r.shape[1], skin_r.shape[1])
    original_r = cv2.resize(original_r, (min_w, target_h))
    overlay_r = cv2.resize(overlay_r, (min_w, target_h))
    skin_r = cv2.resize(skin_r, (min_w, target_h))

    images_panel = np.hstack([original_r, overlay_r, skin_r])

    text_h = 260
    text_panel = np.full((text_h, images_panel.shape[1], 3), 245, dtype=np.uint8)

    lines = [
        "Visible Acne Detection Result",
        f"Total: {summary['total_count']}  |  Severity: {summary['severity_level']}  |  Score: {summary['severity_score']}",
        f"Pustule: {summary['type_counts'].get('pustule', 0)}  Papule: {summary['type_counts'].get('papule', 0)}  Whitehead: {summary['type_counts'].get('whitehead', 0)}  Blackhead: {summary['type_counts'].get('blackhead', 0)}",
    ]

    y = 38
    for i, line in enumerate(lines):
        scale = 0.86 if i == 0 else 0.62
        thickness = 2 if i == 0 else 1
        cv2.putText(
            text_panel,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (30, 30, 30),
            thickness,
            cv2.LINE_AA
        )
        y += 32

    y += 10
    cv2.putText(
        text_panel,
        "Management Solution:",
        (24, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (30, 30, 30),
        2,
        cv2.LINE_AA
    )
    y += 30

    for msg in solution["overall_message"][:4]:
        safe_msg = msg.encode("ascii", "ignore").decode("ascii")
        # OpenCV 기본 폰트는 한글 표시가 안 되므로 dashboard에는 영어/숫자 위주만 안정적으로 표시.
        # 자세한 한글 솔루션은 JSON metadata에 저장됨.
        if not safe_msg.strip():
            safe_msg = "- See JSON metadata for Korean solution text."
        cv2.putText(
            text_panel,
            safe_msg[:120],
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (70, 70, 70),
            1,
            cv2.LINE_AA
        )
        y += 24

    cv2.putText(
        text_panel,
        "Detailed Korean care guide is saved in acne_visible_metadata.json",
        (24, text_h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (90, 90, 90),
        1,
        cv2.LINE_AA
    )

    return np.vstack([images_panel, text_panel])


# =========================================================
# 메인 분석 함수
# =========================================================
def analyze_visible_acne(
    image_path: Path,
    output_dir: Path | None = None,
    use_mediapipe: bool = True
) -> dict[str, Any]:
    image_path = Path(image_path)
    img = load_image_korean_path(image_path)

    if img is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")

    if output_dir is None:
        output_dir = image_path.parent / DEFAULT_OUTPUT_DIR_NAME
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if use_mediapipe:
        face_mask, face_info = detect_face_mask_with_mediapipe(img)
    else:
        face_mask = make_ellipse_face_mask(img.shape[:2])
        face_info = {"method": "ellipse_forced", "reason": "use_mediapipe_false"}

    skin_mask = make_skin_mask(img, face_mask)

    detections: list[dict[str, Any]] = []
    detections.extend(detect_red_inflammatory_acne(img, skin_mask))
    detections.extend(detect_blackheads(img, skin_mask))
    detections.extend(detect_whiteheads(img, skin_mask))

    detections = non_max_suppression(detections)

    summary = build_summary(detections)
    solution = build_management_solution(summary)

    overlay = draw_detections(img, detections)
    dashboard = make_dashboard(img, overlay, skin_mask, summary, solution)

    overlay_path = output_dir / "acne_visible_overlay.jpg"
    dashboard_path = output_dir / "acne_visible_dashboard.jpg"
    skin_mask_path = output_dir / "acne_visible_skin_mask.png"
    metadata_path = output_dir / "acne_visible_metadata.json"

    save_image_korean_path(overlay_path, overlay)
    save_image_korean_path(dashboard_path, dashboard)
    save_image_korean_path(skin_mask_path, skin_mask)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "analysis_type": "visible_light_acne_detection_and_classification",
        "input_file": str(image_path),
        "output_files": {
            "overlay_image": str(overlay_path),
            "dashboard_image": str(dashboard_path),
            "skin_mask_image": str(skin_mask_path),
            "metadata": str(metadata_path),
        },
        "image_info": {
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "shape": list(img.shape),
        },
        "face_roi": face_info,
        "parameters": {
            "min_acne_area": MIN_ACNE_AREA,
            "max_acne_area_ratio": MAX_ACNE_AREA_RATIO,
            "redness_threshold": REDNESS_THRESHOLD,
            "lab_a_percentile": LAB_A_PERCENTILE,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
        },
        "summary": summary,
        "detections": detections,
        "management_solution": solution,
        "coordinate_system": {
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
            "unit": "pixel",
            "note": "좌표는 입력된 필터 없는 일반 사진 기준입니다."
        },
        "caution": (
            "이 코드는 규칙 기반 영상처리로 여드름 후보를 검출합니다. "
            "점, 주근깨, 수염, 그림자, 반사광, 입술, 콧구멍, 머리카락 등이 오검출될 수 있습니다. "
            "의료 진단 목적이 아니라 졸업작품의 보조 지표로 사용하세요."
        )
    }

    save_json(metadata_path, metadata)

    print("\n필터 없는 일반 사진 기반 여드름 분석 완료")
    print("입력 사진:", image_path)
    print("총 검출 개수:", summary["total_count"])
    print("화농성:", summary["type_counts"].get("pustule", 0))
    print("구진성/염증성:", summary["type_counts"].get("papule", 0))
    print("화이트헤드:", summary["type_counts"].get("whitehead", 0))
    print("블랙헤드:", summary["type_counts"].get("blackhead", 0))
    print("결과 이미지:", overlay_path)
    print("대시보드:", dashboard_path)
    print("메타데이터:", metadata_path)

    return metadata


# =========================================================
# camera_server.py 연동용 함수
# =========================================================
def run_visible_acne_analysis_for_capture(
    no_filter_image_path: str | Path,
    output_dir: str | Path | None = None
) -> dict[str, Any]:
    """
    camera_server.py에서 import해서 사용하기 위한 함수.

    사용 예:
        from visible_acne_analysis import run_visible_acne_analysis_for_capture

        acne_result = run_visible_acne_analysis_for_capture(
            no_filter_image_path=cam2_saved_path,
            output_dir=analysis_dir / "visible_acne"
        )
    """
    return analyze_visible_acne(
        image_path=Path(no_filter_image_path),
        output_dir=Path(output_dir) if output_dir else None,
        use_mediapipe=True
    )


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="필터 없는 일반 얼굴 사진 경로")
    parser.add_argument("--base", type=str, default=None, help="최신 no-filter 이미지를 찾을 기준 폴더")
    parser.add_argument("--out", type=str, default=None, help="결과 저장 폴더")
    parser.add_argument("--no-mediapipe", action="store_true", help="MediaPipe 얼굴 ROI를 쓰지 않고 중앙 타원 ROI 사용")
    args = parser.parse_args()

    if args.image:
        target_image = Path(args.image).expanduser()
    else:
        base = Path(args.base).expanduser() if args.base else find_default_no_filter_root()
        target_image = find_latest_no_filter_image(base)

    out_dir = Path(args.out).expanduser() if args.out else None

    analyze_visible_acne(
        image_path=target_image,
        output_dir=out_dir,
        use_mediapipe=not args.no_mediapipe
    )
