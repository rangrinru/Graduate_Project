from __future__ import annotations

import argparse
import json
import re
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
# 405nm 피부 노화 의심 어두운 부위 배치 분석 코드
# =========================================================
# 목적:
# - 필터 없는 사진(cam2_no_filter)과 405nm 필터 사진(cam3_405nm)을 1:1로 비교
# - 필터 없는 사진에서는 잘 안 보이지만 405nm 사진에서 어둡게 보이는 얼굴 부위 검출
# - 검출 위치를 결과 이미지에 표시
# - captures/skin_old_01 폴더에 결과 이미지와 metadata 저장
#
# 주의:
# - 이 코드는 졸업작품용 규칙 기반 영상처리 코드입니다.
# - 의료 진단용이 아닙니다.
# - 405nm 영상의 어두운 부위는 자외선 손상/광노화 관련 색소 침착 가능성을 보여주는 참고 지표일 수 있지만,
#   조명, 그림자, 머리카락, 눈썹, 입술, 촬영 각도, 노출 차이 때문에 오검출될 수 있습니다.
# =========================================================


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]

# 405nm 영상에서 어두운 하위 몇 %를 후보로 볼지
DARK_PERCENTILE = 24

# no-filter 대비 405nm에서 얼마나 더 어두워야 후보로 볼지
DIFF_THRESHOLD = 24

# 너무 작은 노이즈 제거
MIN_AREA = 90

# 너무 큰 그림자/배경 영역 제거
MAX_AREA_RATIO = 0.12

# 조명 보정용 blur size
LOCAL_BLUR_SIZE = 61

# 형태학 처리 kernel
MORPH_KERNEL_SIZE = 5

# 결과 폴더명
OUTPUT_FOLDER_NAME = "skin_old_01"

# cam2 / cam3 폴더명
NO_FILTER_FOLDER_NAMES = ["cam2_no_filter", "no_filter", "cam2"]
UV405_FOLDER_NAMES = ["cam3_405nm", "405nm", "cam3"]


# =========================================================
# 파일 입출력
# =========================================================
def load_image(path: Path) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None

    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def save_image(path: Path, img: np.ndarray):
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


def list_images(root: Path) -> list[Path]:
    root = Path(root)

    if not root.exists():
        return []

    files: list[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(root.glob(f"**/*{ext}"))
        files.extend(root.glob(f"**/*{ext.upper()}"))

    # analysis, skin_old_01 같은 결과 폴더 안의 이미지는 제외
    excluded_keywords = [
        "analysis",
        "skin_old_01",
        "visible_acne_analysis",
        "uva_darkspot_analysis",
        "old_skin_analysis"
    ]

    filtered = []
    for p in files:
        path_lower = str(p).lower()
        if any(k in path_lower for k in excluded_keywords):
            continue
        filtered.append(p)

    return sorted(set(filtered), key=lambda p: (extract_timestamp_key(p) or "", p.stat().st_mtime, str(p)))


# =========================================================
# 경로 탐색
# =========================================================
def find_project_root() -> Path:
    """
    현재 실행 위치에서 Graduate_Project 루트를 최대한 찾음.
    """
    cwd = Path.cwd().resolve()

    # 현재 위치 또는 상위 폴더 중 Graduate_Project 찾기
    for p in [cwd] + list(cwd.parents):
        if p.name == "Graduate_Project":
            return p

    # 자주 쓰는 경로 후보
    candidates = [
        Path.home() / "Graduate_Project",
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project"),
        cwd,
    ]

    for p in candidates:
        if p.exists():
            return p

    return cwd


def find_captures_root(user_captures_root: str | None = None) -> Path:
    if user_captures_root:
        return Path(user_captures_root).expanduser().resolve()

    project_root = find_project_root()

    candidates = [
        project_root / "captures",
        project_root / "TaeYeon" / "captures",
        Path.home() / "Graduate_Project" / "captures",
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\captures"),
    ]

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]


def find_camera_folder(captures_root: Path, folder_names: list[str]) -> Path | None:
    captures_root = Path(captures_root)

    candidates = []

    for name in folder_names:
        candidates.append(captures_root / name)
        candidates.append(captures_root / "sessions" / name)

    for p in candidates:
        if p.exists():
            return p

    return None


# =========================================================
# 페어 매칭
# =========================================================
def extract_timestamp_key(path: Path) -> str | None:
    """
    파일명에서 공통 timestamp 추출.
    예:
    20260407_233614_041_cam2.png -> 20260407_233614_041
    20260407_233614_cam3.png     -> 20260407_233614
    """
    name = path.name

    m = re.search(r"(\d{8}_\d{6}_\d{3})", name)
    if m:
        return m.group(1)

    m = re.search(r"(\d{8}_\d{6})", name)
    if m:
        return m.group(1)

    # 날짜 폴더 + 파일 stem 조합도 fallback으로 사용
    return None


def build_image_pairs(no_filter_images: list[Path], uv405_images: list[Path]) -> list[tuple[Path, Path, str]]:
    """
    1순위: timestamp 같은 파일끼리 매칭
    2순위: 개수가 같거나 비슷하면 정렬 순서대로 매칭
    """
    pairs: list[tuple[Path, Path, str]] = []

    no_by_key: dict[str, Path] = {}
    used_no: set[Path] = set()
    used_uv: set[Path] = set()

    for p in no_filter_images:
        key = extract_timestamp_key(p)
        if key:
            no_by_key[key] = p

    for uv in uv405_images:
        key = extract_timestamp_key(uv)
        if key and key in no_by_key:
            no = no_by_key[key]
            pairs.append((no, uv, key))
            used_no.add(no)
            used_uv.add(uv)

    # timestamp 매칭 안 된 파일은 정렬 순서대로 fallback 매칭
    remain_no = [p for p in no_filter_images if p not in used_no]
    remain_uv = [p for p in uv405_images if p not in used_uv]

    for idx, (no, uv) in enumerate(zip(remain_no, remain_uv), start=1):
        key = extract_timestamp_key(no) or extract_timestamp_key(uv) or f"pair_{idx:04d}"
        pairs.append((no, uv, key))

    # 첫 사진부터 최신 사진 순서로 정렬
    pairs = sorted(
        pairs,
        key=lambda item: (
            extract_timestamp_key(item[0]) or extract_timestamp_key(item[1]) or "",
            item[0].stat().st_mtime,
            item[1].stat().st_mtime,
        )
    )

    return pairs


# =========================================================
# 이미지 전처리
# =========================================================
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
    """
    gray = gray.astype(np.float32)

    if blur_size % 2 == 0:
        blur_size += 1

    background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    corrected = gray - background + np.mean(background)
    corrected = np.clip(corrected, 0, 255)

    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


def resize_to_match(src: np.ndarray, target_shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_shape_hw
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)


def align_405_to_no_filter(no_filter_bgr: np.ndarray, uv405_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    405nm 이미지를 no-filter 좌표계로 정합.
    ORB Homography를 시도하고 실패하면 resize만 사용.
    """
    h, w = no_filter_bgr.shape[:2]

    if uv405_bgr.shape[:2] != (h, w):
        uv_resized = resize_to_match(uv405_bgr, (h, w))
    else:
        uv_resized = uv405_bgr.copy()

    ref_gray = enhance_gray(to_gray(no_filter_bgr))
    mov_gray = enhance_gray(to_gray(uv_resized))

    try:
        orb = cv2.ORB_create(nfeatures=1800)
        kp1, des1 = orb.detectAndCompute(ref_gray, None)
        kp2, des2 = orb.detectAndCompute(mov_gray, None)

        if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
            return uv_resized, {
                "method": "resize_only",
                "reason": "not_enough_features",
                "matched_points": 0
            }

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        good = matches[:100]

        if len(good) < 12:
            return uv_resized, {
                "method": "resize_only",
                "reason": "not_enough_matches",
                "matched_points": len(good)
            }

        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return uv_resized, {
                "method": "resize_only",
                "reason": "homography_failed",
                "matched_points": len(good)
            }

        aligned = cv2.warpPerspective(uv_resized, H, (w, h), flags=cv2.INTER_LINEAR)

        return aligned, {
            "method": "orb_homography",
            "matched_points": len(good),
            "inliers": int(mask.sum()) if mask is not None else 0,
            "homography": H.tolist()
        }

    except Exception as e:
        return uv_resized, {
            "method": "resize_only",
            "reason": f"exception: {str(e)}"
        }


# =========================================================
# 얼굴 ROI 생성
# =========================================================
def make_ellipse_face_mask(shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (w // 2, int(h * 0.52))
    axes = (int(w * 0.34), int(h * 0.43))

    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def detect_face_mask(no_filter_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    MediaPipe가 있으면 얼굴 윤곽을 사용하고, 실패하면 중앙 타원 사용.
    눈/입술/눈썹 주변 오검출을 줄이기 위해 결과적으로 얼굴 전체 마스크만 사용.
    """
    h, w = no_filter_bgr.shape[:2]

    if mp is None:
        return make_ellipse_face_mask((h, w)), {
            "method": "ellipse_fallback",
            "reason": "mediapipe_not_installed"
        }

    try:
        img_rgb = cv2.cvtColor(no_filter_bgr, cv2.COLOR_BGR2RGB)

        with mp.solutions.face_mesh.FaceMesh(
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

        return mask, {
            "method": "mediapipe_facemesh",
            "reason": "success"
        }

    except Exception as e:
        return make_ellipse_face_mask((h, w)), {
            "method": "ellipse_fallback",
            "reason": f"exception: {str(e)}"
        }


def remove_eye_mouth_like_regions(mask: np.ndarray, no_filter_bgr: np.ndarray) -> np.ndarray:
    """
    405nm 어두운 영역 검출 시 눈/입술이 크게 잡히는 것을 줄이기 위한 대략적 제외.
    정밀 landmark 제외는 아니고, 얼굴 ROI 내 상단 눈 영역/하단 입 영역의 너무 어두운 큰 구조를 완화.
    """
    h, w = mask.shape[:2]
    out = mask.copy()

    # 눈썹/눈 부근은 노화 반점이 아니라 눈 자체가 어둡게 잡힐 수 있으므로 일부 약화
    # 단, 완전히 제거하면 이마/눈가 분석이 어려워져 좁은 가로 띠만 제외
    eye_y1 = int(h * 0.36)
    eye_y2 = int(h * 0.48)
    mouth_y1 = int(h * 0.70)
    mouth_y2 = int(h * 0.82)

    # 중앙 타원 기준에서 눈/입 주변 중앙부 제외
    out[eye_y1:eye_y2, int(w * 0.18):int(w * 0.82)] = 0
    out[mouth_y1:mouth_y2, int(w * 0.28):int(w * 0.72)] = 0

    return out


# =========================================================
# 405nm 어두운 노화 의심 영역 검출
# =========================================================
def classify_region_by_position(cx: int, cy: int, width: int, height: int) -> str:
    """
    간단한 얼굴 위치 기반 부위 분류.
    """
    if cy < height * 0.33:
        return "forehead"
    if cy > height * 0.70:
        return "chin_or_mouth_area"
    if cx < width * 0.36:
        return "left_cheek"
    if cx > width * 0.64:
        return "right_cheek"
    return "nose_center"


def region_name_ko(region: str) -> str:
    mapping = {
        "forehead": "이마",
        "left_cheek": "왼쪽 볼",
        "right_cheek": "오른쪽 볼",
        "nose_center": "코/중앙",
        "chin_or_mouth_area": "턱/입 주변",
        "unknown": "알 수 없음",
    }
    return mapping.get(region, region)


def detect_skin_old_dark_regions(
    no_filter_bgr: np.ndarray,
    uv405_aligned_bgr: np.ndarray,
    dark_percentile: float = DARK_PERCENTILE,
    diff_threshold: int = DIFF_THRESHOLD,
    min_area: int = MIN_AREA,
    max_area_ratio: float = MAX_AREA_RATIO
) -> dict[str, Any]:
    h, w = no_filter_bgr.shape[:2]
    image_area = h * w

    no_raw_gray = to_gray(no_filter_bgr)
    uv_raw_gray = to_gray(uv405_aligned_bgr)

    no_norm = local_normalize(enhance_gray(no_raw_gray))
    uv_norm = local_normalize(enhance_gray(uv_raw_gray))

    face_mask, face_info = detect_face_mask(no_filter_bgr)
    analysis_mask = remove_eye_mouth_like_regions(face_mask, no_filter_bgr)

    roi_bool = analysis_mask > 0
    if np.count_nonzero(roi_bool) == 0:
        roi_bool = face_mask > 0

    if np.count_nonzero(roi_bool) == 0:
        roi_bool = np.ones((h, w), dtype=bool)

    # 405nm에서 어두운 하위 영역
    dark_threshold = int(np.percentile(uv_norm[roi_bool], dark_percentile))

    # 일반 사진 대비 405nm가 더 어둡게 나타나는 정도
    diff = cv2.subtract(no_norm, uv_norm)

    dark_mask = np.zeros((h, w), dtype=np.uint8)
    dark_mask[(uv_norm <= dark_threshold) & (diff >= diff_threshold) & roi_bool] = 255

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: list[dict[str, Any]] = []
    max_area = image_area * max_area_ratio

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < min_area:
            continue

        if area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        # 너무 길쭉한 그림자/경계선 제거
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > 5.5:
            continue

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + bw // 2
            cy = y + bh // 2

        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, -1)
        region_bool = region_mask > 0

        mean_no = float(np.mean(no_raw_gray[region_bool]))
        mean_405 = float(np.mean(uv_raw_gray[region_bool]))
        mean_diff = float(np.mean(diff[region_bool]))
        min_405 = int(np.min(uv_raw_gray[region_bool]))
        max_diff = int(np.max(diff[region_bool]))

        position_region = classify_region_by_position(cx, cy, w, h)

        severity_score = float((mean_diff * 0.6) + (area / 150.0) + max(0, 80 - mean_405) * 0.2)

        if severity_score < 25:
            severity = "mild"
            severity_ko = "약함"
        elif severity_score < 50:
            severity = "moderate"
            severity_ko = "보통"
        else:
            severity = "strong"
            severity_ko = "강함"

        regions.append({
            "id": len(regions) + 1,
            "region": position_region,
            "region_ko": region_name_ko(position_region),
            "center": {
                "x": int(cx),
                "y": int(cy)
            },
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(bw),
                "height": int(bh)
            },
            "area_px": round(area, 2),
            "area_ratio": round(area / image_area, 8),
            "mean_no_filter_brightness": round(mean_no, 4),
            "mean_405nm_brightness": round(mean_405, 4),
            "mean_darkness_difference": round(mean_diff, 4),
            "min_405nm_brightness": min_405,
            "max_darkness_difference": max_diff,
            "severity_score": round(severity_score, 4),
            "severity": severity,
            "severity_ko": severity_ko,
            "contour_points": contour.reshape(-1, 2).astype(int).tolist(),
        })

    # 강한 부위 우선 정렬
    regions = sorted(regions, key=lambda r: (r["severity_score"], r["area_px"]), reverse=True)

    for idx, region in enumerate(regions, start=1):
        region["id"] = idx

    total_area = sum(float(r["area_px"]) for r in regions)

    region_count_by_area: dict[str, int] = {}
    for region in regions:
        key = region["region"]
        region_count_by_area[key] = region_count_by_area.get(key, 0) + 1

    return {
        "face_mask": face_mask,
        "analysis_mask": analysis_mask,
        "dark_mask": dark_mask,
        "diff": diff,
        "no_norm": no_norm,
        "uv_norm": uv_norm,
        "dark_threshold": dark_threshold,
        "regions": regions,
        "face_info": face_info,
        "summary": {
            "dark_region_count": int(len(regions)),
            "total_dark_area_px": round(total_area, 2),
            "total_dark_area_ratio": round(total_area / image_area, 8),
            "region_count_by_area": region_count_by_area,
            "image_width": int(w),
            "image_height": int(h)
        }
    }


# =========================================================
# 결과 이미지 생성
# =========================================================
def make_color_mask_overlay(base_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.45) -> np.ndarray:
    out = base_bgr.copy()
    color_layer = np.zeros_like(base_bgr)
    color_layer[:, :] = color

    mask_bool = mask > 0
    out[mask_bool] = cv2.addWeighted(base_bgr[mask_bool], 1 - alpha, color_layer[mask_bool], alpha, 0)
    return out


def draw_regions(image_bgr: np.ndarray, regions: list[dict[str, Any]]) -> np.ndarray:
    out = image_bgr.copy()

    for region in regions:
        x = int(region["bbox"]["x"])
        y = int(region["bbox"]["y"])
        bw = int(region["bbox"]["width"])
        bh = int(region["bbox"]["height"])
        cx = int(region["center"]["x"])
        cy = int(region["center"]["y"])
        rid = int(region["id"])

        if region["severity"] == "strong":
            color = (0, 0, 255)
        elif region["severity"] == "moderate":
            color = (0, 140, 255)
        else:
            color = (0, 255, 255)

        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
        cv2.circle(out, (cx, cy), 4, color, -1)

        if region.get("contour_points"):
            pts = np.array(region["contour_points"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(out, [pts], -1, color, 2)

        # OpenCV 기본 폰트는 한글을 지원하지 않으므로 영어/숫자 중심으로 표시
        label = f"#{rid} {region['region']} {region['severity']}"
        cv2.putText(
            out,
            label,
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA
        )

    return out


def make_heatmap(diff: np.ndarray, analysis_mask: np.ndarray) -> np.ndarray:
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    outside = analysis_mask <= 0
    heat[outside] = (heat[outside] * 0.25).astype(np.uint8)

    return heat


def resize_keep_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)


def make_comparison_image(
    no_filter_bgr: np.ndarray,
    uv405_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    summary: dict[str, Any],
    pair_key: str
) -> np.ndarray:
    target_h = 520

    a = resize_keep_height(no_filter_bgr, target_h)
    b = resize_keep_height(uv405_bgr, target_h)
    c = resize_keep_height(overlay_bgr, target_h)
    d = resize_keep_height(heatmap_bgr, target_h)

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

    lines = [
        "Skin Aging Dark Region Analysis: No-filter vs 405nm",
        f"Pair: {pair_key}",
        f"Detected dark regions: {summary['dark_region_count']}",
        f"Total dark area: {summary['total_dark_area_px']} px | ratio: {summary['total_dark_area_ratio']}",
        "Red/Orange/Yellow boxes: darker regions visible in 405nm image",
    ]

    y = 35
    for i, line in enumerate(lines):
        scale = 0.8 if i == 0 else 0.58
        thickness = 2 if i == 0 else 1
        color = (30, 30, 30) if i == 0 else (70, 70, 70)

        cv2.putText(panel, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += 30

    return np.vstack([canvas, panel])


# =========================================================
# 단일 페어 분석
# =========================================================
def analyze_one_pair(
    no_filter_path: Path,
    uv405_path: Path,
    pair_output_dir: Path,
    pair_key: str,
    dark_percentile: float = DARK_PERCENTILE,
    diff_threshold: int = DIFF_THRESHOLD,
    min_area: int = MIN_AREA
) -> dict[str, Any]:
    no_filter = load_image(no_filter_path)
    uv405 = load_image(uv405_path)

    if no_filter is None:
        raise RuntimeError(f"필터 없는 사진 로드 실패: {no_filter_path}")

    if uv405 is None:
        raise RuntimeError(f"405nm 사진 로드 실패: {uv405_path}")

    pair_output_dir.mkdir(parents=True, exist_ok=True)

    uv405_aligned, align_info = align_405_to_no_filter(no_filter, uv405)

    result = detect_skin_old_dark_regions(
        no_filter_bgr=no_filter,
        uv405_aligned_bgr=uv405_aligned,
        dark_percentile=dark_percentile,
        diff_threshold=diff_threshold,
        min_area=min_area
    )

    mask = result["dark_mask"]
    diff = result["diff"]
    analysis_mask = result["analysis_mask"]
    regions = result["regions"]
    summary = result["summary"]

    overlay = make_color_mask_overlay(no_filter, mask, color=(0, 0, 255), alpha=0.42)
    overlay = draw_regions(overlay, regions)

    uv405_marked = draw_regions(uv405_aligned, regions)
    heatmap = make_heatmap(diff, analysis_mask)

    comparison = make_comparison_image(
        no_filter_bgr=no_filter,
        uv405_bgr=uv405_marked,
        overlay_bgr=overlay,
        heatmap_bgr=heatmap,
        summary=summary,
        pair_key=pair_key
    )

    overlay_path = pair_output_dir / "skin_old_overlay.jpg"
    compare_path = pair_output_dir / "skin_old_compare.jpg"
    heatmap_path = pair_output_dir / "skin_old_heatmap.jpg"
    mask_path = pair_output_dir / "skin_old_mask.png"
    aligned_405_path = pair_output_dir / "aligned_405nm.jpg"
    metadata_path = pair_output_dir / "skin_old_metadata.json"

    save_image(overlay_path, overlay)
    save_image(compare_path, comparison)
    save_image(heatmap_path, heatmap)
    save_image(mask_path, mask)
    save_image(aligned_405_path, uv405_aligned)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "analysis_type": "skin_autofluorescence_405nm_dark_region_aging_check",
        "pair_key": pair_key,
        "description": (
            "필터 없는 사진과 405nm 필터 사진을 비교하여, 일반 사진에서는 잘 보이지 않지만 "
            "405nm 사진에서 상대적으로 어둡게 나타나는 얼굴 부위를 검출합니다."
        ),
        "input_files": {
            "no_filter_image": str(no_filter_path),
            "uv405_image": str(uv405_path)
        },
        "output_files": {
            "overlay_image": str(overlay_path),
            "comparison_image": str(compare_path),
            "heatmap_image": str(heatmap_path),
            "mask_image": str(mask_path),
            "aligned_405nm_image": str(aligned_405_path),
            "metadata": str(metadata_path)
        },
        "image_info": {
            "width": int(no_filter.shape[1]),
            "height": int(no_filter.shape[0]),
            "no_filter_shape": list(no_filter.shape),
            "uv405_original_shape": list(uv405.shape),
            "uv405_aligned_shape": list(uv405_aligned.shape)
        },
        "alignment": align_info,
        "face_roi": result["face_info"],
        "parameters": {
            "dark_percentile": dark_percentile,
            "diff_threshold": diff_threshold,
            "min_area": min_area,
            "max_area_ratio": MAX_AREA_RATIO,
            "local_blur_size": LOCAL_BLUR_SIZE,
            "morph_kernel_size": MORPH_KERNEL_SIZE
        },
        "summary": summary,
        "skin_old_regions": regions,
        "coordinate_system": {
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
            "unit": "pixel",
            "basis": "no_filter_image",
            "note": "좌표는 no-filter 이미지 기준입니다. 405nm 이미지는 no-filter 이미지 기준으로 정합 또는 resize된 뒤 비교됩니다."
        },
        "interpretation_note": (
            "405nm/UVA 자가형광 이미지에서 어둡게 나타나는 부위는 색소 침착, 광노화 의심 부위, "
            "그림자, 머리카락, 눈썹, 입술, 조명 불균일, 촬영 각도 차이 등 다양한 원인으로 발생할 수 있습니다. "
            "본 결과는 의료 진단이 아니라 졸업작품용 영상 비교 지표입니다."
        )
    }

    save_json(metadata_path, metadata)

    return metadata


# =========================================================
# 전체 배치 분석
# =========================================================
def analyze_all_skin_old_pairs(
    captures_root: Path,
    no_filter_root: Path | None = None,
    uv405_root: Path | None = None,
    output_root: Path | None = None,
    dark_percentile: float = DARK_PERCENTILE,
    diff_threshold: int = DIFF_THRESHOLD,
    min_area: int = MIN_AREA
) -> dict[str, Any]:
    captures_root = Path(captures_root)

    if no_filter_root is None:
        no_filter_root = find_camera_folder(captures_root, NO_FILTER_FOLDER_NAMES)

    if uv405_root is None:
        uv405_root = find_camera_folder(captures_root, UV405_FOLDER_NAMES)

    if no_filter_root is None or not Path(no_filter_root).exists():
        raise FileNotFoundError(f"cam2_no_filter 폴더를 찾지 못했습니다. captures_root={captures_root}")

    if uv405_root is None or not Path(uv405_root).exists():
        raise FileNotFoundError(f"cam3_405nm 폴더를 찾지 못했습니다. captures_root={captures_root}")

    no_filter_root = Path(no_filter_root)
    uv405_root = Path(uv405_root)

    if output_root is None:
        output_root = captures_root / OUTPUT_FOLDER_NAME
    else:
        output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    no_filter_images = list_images(no_filter_root)
    uv405_images = list_images(uv405_root)

    if not no_filter_images:
        raise FileNotFoundError(f"필터 없는 사진이 없습니다: {no_filter_root}")

    if not uv405_images:
        raise FileNotFoundError(f"405nm 사진이 없습니다: {uv405_root}")

    pairs = build_image_pairs(no_filter_images, uv405_images)

    if not pairs:
        raise RuntimeError("비교할 사진 페어를 만들지 못했습니다.")

    print("[분석 시작]")
    print("captures_root:", captures_root)
    print("no_filter_root:", no_filter_root)
    print("uv405_root:", uv405_root)
    print("output_root:", output_root)
    print("no_filter 이미지 수:", len(no_filter_images))
    print("405nm 이미지 수:", len(uv405_images))
    print("분석 페어 수:", len(pairs))

    pair_results = []
    total_regions = 0
    total_area = 0.0
    failed_pairs = []

    for idx, (no_path, uv_path, key) in enumerate(pairs, start=1):
        safe_key = key.replace(":", "_").replace("/", "_").replace("\\", "_")
        pair_dir = output_root / f"{idx:04d}_{safe_key}"

        print(f"\n[{idx}/{len(pairs)}] 분석 중")
        print("no_filter:", no_path)
        print("405nm:", uv_path)

        try:
            metadata = analyze_one_pair(
                no_filter_path=no_path,
                uv405_path=uv_path,
                pair_output_dir=pair_dir,
                pair_key=safe_key,
                dark_percentile=dark_percentile,
                diff_threshold=diff_threshold,
                min_area=min_area
            )

            summary = metadata["summary"]
            total_regions += int(summary["dark_region_count"])
            total_area += float(summary["total_dark_area_px"])

            pair_results.append({
                "index": idx,
                "pair_key": safe_key,
                "no_filter_image": str(no_path),
                "uv405_image": str(uv_path),
                "output_dir": str(pair_dir),
                "metadata": metadata["output_files"]["metadata"],
                "comparison_image": metadata["output_files"]["comparison_image"],
                "overlay_image": metadata["output_files"]["overlay_image"],
                "dark_region_count": summary["dark_region_count"],
                "total_dark_area_px": summary["total_dark_area_px"],
                "total_dark_area_ratio": summary["total_dark_area_ratio"],
            })

            print("검출 개수:", summary["dark_region_count"])
            print("저장:", metadata["output_files"]["comparison_image"])

        except Exception as e:
            failed_pairs.append({
                "index": idx,
                "pair_key": safe_key,
                "no_filter_image": str(no_path),
                "uv405_image": str(uv_path),
                "error": str(e)
            })
            print("[실패]", e)

    batch_summary = {
        "generated_at": datetime.now().isoformat(),
        "analysis_type": "batch_skin_old_405nm_dark_region_analysis",
        "captures_root": str(captures_root),
        "no_filter_root": str(no_filter_root),
        "uv405_root": str(uv405_root),
        "output_root": str(output_root),
        "parameters": {
            "dark_percentile": dark_percentile,
            "diff_threshold": diff_threshold,
            "min_area": min_area,
            "max_area_ratio": MAX_AREA_RATIO
        },
        "input_counts": {
            "no_filter_image_count": len(no_filter_images),
            "uv405_image_count": len(uv405_images),
            "pair_count": len(pairs)
        },
        "summary": {
            "processed_pair_count": len(pair_results),
            "failed_pair_count": len(failed_pairs),
            "total_detected_old_region_count": int(total_regions),
            "total_detected_old_area_px": round(total_area, 2),
            "average_region_count_per_pair": round(total_regions / max(len(pair_results), 1), 4)
        },
        "pair_results": pair_results,
        "failed_pairs": failed_pairs,
        "output_files": {
            "batch_metadata": str(output_root / "skin_old_batch_metadata.json")
        },
        "interpretation_note": (
            "전체 결과는 captures/skin_old_01 아래에 페어별 폴더로 저장됩니다. "
            "각 페어의 skin_old_metadata.json에는 검출 부위 좌표, 개수, 면적, 밝기 차이, 심각도 정보가 저장됩니다."
        )
    }

    batch_metadata_path = output_root / "skin_old_batch_metadata.json"
    save_json(batch_metadata_path, batch_summary)

    print("\n[전체 분석 완료]")
    print("처리 페어 수:", len(pair_results))
    print("실패 페어 수:", len(failed_pairs))
    print("전체 검출 개수:", total_regions)
    print("batch metadata:", batch_metadata_path)

    return batch_summary


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--captures-root",
        type=str,
        default=None,
        help="captures 폴더 경로. 기본값은 현재 프로젝트의 captures 자동 탐색"
    )

    parser.add_argument(
        "--no-filter-root",
        type=str,
        default=None,
        help="cam2_no_filter 폴더 직접 지정"
    )

    parser.add_argument(
        "--uv405-root",
        type=str,
        default=None,
        help="cam3_405nm 폴더 직접 지정"
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="결과 저장 폴더. 기본값은 captures/skin_old_01"
    )

    parser.add_argument(
        "--dark-percentile",
        type=float,
        default=DARK_PERCENTILE,
        help="405nm 이미지에서 어두운 후보 percentile. 작을수록 엄격"
    )

    parser.add_argument(
        "--diff-threshold",
        type=int,
        default=DIFF_THRESHOLD,
        help="no-filter 대비 405nm 어두움 차이 임계값. 클수록 엄격"
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=MIN_AREA,
        help="검출 최소 면적 px. 클수록 작은 잡음 제거"
    )

    args = parser.parse_args()

    captures_root = find_captures_root(args.captures_root)

    no_filter_root = Path(args.no_filter_root).expanduser().resolve() if args.no_filter_root else None
    uv405_root = Path(args.uv405_root).expanduser().resolve() if args.uv405_root else None
    output_root = Path(args.out).expanduser().resolve() if args.out else None

    analyze_all_skin_old_pairs(
        captures_root=captures_root,
        no_filter_root=no_filter_root,
        uv405_root=uv405_root,
        output_root=output_root,
        dark_percentile=args.dark_percentile,
        diff_threshold=args.diff_threshold,
        min_area=args.min_area
    )
