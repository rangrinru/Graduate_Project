import cv2
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(r"C:\Users\고대현\Desktop\Graduate_Project")

CAM2_DIR = PROJECT_ROOT / "captures" / "cam2_no_filter"
CAM3_DIR = PROJECT_ROOT / "captures" / "cam3_405nm"

SAVE_DIR = PROJECT_ROOT / "GDH" / "age_detect" / "info"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 한글 경로 이미지 읽기/쓰기
# -----------------------------
def imread_korean(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_korean(path, img):
    ext = Path(path).suffix
    ok, encoded = cv2.imencode(ext, img)
    if ok:
        encoded.tofile(str(path))
    return ok


# -----------------------------
# Haar Cascade 얼굴 검출
# -----------------------------
def get_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError("얼굴 Cascade 로드 실패")

    return face_cascade


def detect_largest_face(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    return int(x), int(y), int(w), int(h)


def expand_bbox(bbox, img_shape, ratio_x=0.15, ratio_y=0.18):
    x, y, w, h = bbox
    img_h, img_w = img_shape[:2]

    ex = int(w * ratio_x)
    ey = int(h * ratio_y)

    x1 = max(0, x - ex)
    y1 = max(0, y - ey)
    x2 = min(img_w, x + w + ex)
    y2 = min(img_h, y + h + ey)

    return x1, y1, x2 - x1, y2 - y1


# -----------------------------
# 날짜 폴더 찾기
# -----------------------------
def get_date_folders(base_dir):
    folders = []

    for p in base_dir.iterdir():
        if p.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}$", p.name):
            folders.append(p)

    return sorted(folders)


def get_latest_common_date_folder():
    cam2_dates = {p.name: p for p in get_date_folders(CAM2_DIR)}
    cam3_dates = {p.name: p for p in get_date_folders(CAM3_DIR)}

    common = sorted(set(cam2_dates.keys()) & set(cam3_dates.keys()))

    if not common:
        raise FileNotFoundError("cam2/cam3 공통 날짜 폴더가 없습니다.")

    latest = common[-1]

    return cam2_dates[latest], cam3_dates[latest], latest


# -----------------------------
# 이미지 매칭
# -----------------------------
def get_capture_key(path):
    stem = path.stem
    stem = stem.replace("_cam2", "")
    stem = stem.replace("_cam3", "")
    return stem


def get_image_files(folder):
    return sorted([
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        and "metadata" not in p.name.lower()
    ])


def get_image_pairs():
    cam2_folder, cam3_folder, date_name = get_latest_common_date_folder()

    cam2_files = get_image_files(cam2_folder)
    cam3_files = get_image_files(cam3_folder)

    cam2_map = {get_capture_key(p): p for p in cam2_files}
    cam3_map = {get_capture_key(p): p for p in cam3_files}

    common_keys = sorted(set(cam2_map.keys()) & set(cam3_map.keys()))

    pairs = []

    for i, key in enumerate(common_keys):
        pairs.append({
            "index": i,
            "key": key,
            "date_folder": date_name,
            "cam2_path": cam2_map[key],
            "cam3_path": cam3_map[key]
        })

    if len(pairs) == 0:
        pair_count = min(len(cam2_files), len(cam3_files))

        for i in range(pair_count):
            pairs.append({
                "index": i,
                "key": cam2_files[i].stem.replace("_cam2", ""),
                "date_folder": date_name,
                "cam2_path": cam2_files[i],
                "cam3_path": cam3_files[i]
            })

    return pairs


# -----------------------------
# UV 노화 의심 영역 분석
# -----------------------------
def detect_uv_aging(img_normal, img_uv, face_cascade):
    normal_face = detect_largest_face(img_normal, face_cascade)
    uv_face = detect_largest_face(img_uv, face_cascade)

    if normal_face is None:
        raise RuntimeError("cam2 일반 이미지에서 얼굴을 찾지 못했습니다.")

    if uv_face is None:
        raise RuntimeError("cam3 405nm 이미지에서 얼굴을 찾지 못했습니다.")

    normal_face = expand_bbox(normal_face, img_normal.shape)
    uv_face = expand_bbox(uv_face, img_uv.shape)

    nx, ny, nw, nh = normal_face
    ux, uy, uw, uh = uv_face

    normal_roi = img_normal[ny:ny + nh, nx:nx + nw]
    uv_roi = img_uv[uy:uy + uh, ux:ux + uw]

    uv_roi = cv2.resize(
        uv_roi,
        (normal_roi.shape[1], normal_roi.shape[0])
    )

    normal_gray = cv2.cvtColor(normal_roi, cv2.COLOR_BGR2GRAY)
    uv_gray = cv2.cvtColor(uv_roi, cv2.COLOR_BGR2GRAY)

    normal_gray = cv2.equalizeHist(normal_gray)
    uv_gray = cv2.equalizeHist(uv_gray)

    diff = cv2.absdiff(uv_gray, normal_gray)

    blur = cv2.GaussianBlur(diff, (5, 5), 0)

    threshold_value = np.percentile(blur, 97)

    _, mask = cv2.threshold(
        blur,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = img_normal.copy()

    coordinates = []
    total_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 80:
            continue

        if area > 5000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        global_x = nx + x
        global_y = ny + y

        cv2.rectangle(
            result,
            (global_x, global_y),
            (global_x + w, global_y + h),
            (0, 0, 255),
            2
        )

        cv2.circle(
            result,
            (global_x + w // 2, global_y + h // 2),
            4,
            (0, 255, 255),
            -1
        )

        coordinates.append({
            "x": int(global_x),
            "y": int(global_y),
            "width": int(w),
            "height": int(h),
            "center_x": int(global_x + w / 2),
            "center_y": int(global_y + h / 2),
            "area": float(area)
        })

        total_area += area

    face_area = nw * nh
    detection_rate = (total_area / face_area) * 100 if face_area > 0 else 0

    if detection_rate < 0.5:
        grade = "Low"
    elif detection_rate < 2.0:
        grade = "Medium"
    else:
        grade = "High"

    return result, coordinates, detection_rate, grade, {
        "cam2_face_bbox": {
            "x": int(nx),
            "y": int(ny),
            "width": int(nw),
            "height": int(nh)
        },
        "cam3_face_bbox": {
            "x": int(ux),
            "y": int(uy),
            "width": int(uw),
            "height": int(uh)
        },
        "threshold_percentile": 97,
        "threshold_value": float(threshold_value),
        "analysis_method": "face_roi_aligned_cam3_to_cam2"
    }


# -----------------------------
# 솔루션 생성
# -----------------------------
def create_solution(grade):
    if grade == "Low":
        return {
            "summary": "자외선 노화 의심 영역이 적게 검출되었습니다.",
            "recommendation": [
                "현재 관리 상태를 유지하세요.",
                "외출 전 자외선 차단제를 꾸준히 사용하세요.",
                "기본 보습 관리를 유지하세요."
            ]
        }

    if grade == "Medium":
        return {
            "summary": "일부 부위에서 자외선 노화 의심 영역이 검출되었습니다.",
            "recommendation": [
                "자외선 차단제를 더 꼼꼼히 사용하세요.",
                "건조한 부위에는 보습 관리를 강화하세요.",
                "비타민 C 또는 항산화 성분 기반 관리를 고려하세요."
            ]
        }

    return {
        "summary": "자외선 노화 의심 영역이 비교적 많이 검출되었습니다.",
        "recommendation": [
            "자외선 노출 시간을 줄이세요.",
            "외출 시 자외선 차단제와 모자 사용을 권장합니다.",
            "피부 진정 및 재생 관리가 필요할 수 있습니다.",
            "지속적으로 악화되면 피부 전문가 상담을 권장합니다."
        ]
    }


# -----------------------------
# 결과 이미지 생성
# -----------------------------
def make_result_image(img_normal, result, info_text):
    h, w = img_normal.shape[:2]

    panel = np.zeros((h, 600, 3), dtype=np.uint8)

    y = 60

    for text in [
        "UV Aging Analysis",
        "",
        info_text,
        "",
        "Result image is based on",
        "CAM2 No Filter only.",
        "",
        "CAM3 face ROI is aligned",
        "to CAM2 face ROI before",
        "difference analysis."
    ]:
        cv2.putText(
            panel,
            text,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        y += 45

    final = np.hstack((result, panel))

    return final


# -----------------------------
# 메타데이터 저장
# -----------------------------
def save_metadata(
    metadata_path,
    pair_info,
    result_image_path,
    coordinates,
    detection_rate,
    grade,
    solution,
    debug_info
):
    metadata = {
        "metadata_type": "uv_aging_analysis",
        "created_at": datetime.now().isoformat(),
        "date_folder": pair_info["date_folder"],
        "capture_key": pair_info["key"],
        "input_images": {
            "cam2_no_filter": str(pair_info["cam2_path"]),
            "cam3_405nm": str(pair_info["cam3_path"])
        },
        "output": {
            "result_image": str(result_image_path),
            "metadata_file": str(metadata_path)
        },
        "analysis_result": {
            "aging_region_count": len(coordinates),
            "detection_rate_percent": round(float(detection_rate), 4),
            "grade": grade,
            "coordinates": coordinates
        },
        "alignment_info": debug_info,
        "solution": solution
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)


# -----------------------------
# 실행
# -----------------------------
def run():
    face_cascade = get_face_cascade()

    pairs = get_image_pairs()

    print("매칭된 이미지 수:", len(pairs))

    for pair in pairs:
        img_normal = imread_korean(pair["cam2_path"])
        img_uv = imread_korean(pair["cam3_path"])

        if img_normal is None:
            print("cam2 읽기 실패:", pair["cam2_path"])
            continue

        if img_uv is None:
            print("cam3 읽기 실패:", pair["cam3_path"])
            continue

        try:
            result, coordinates, detection_rate, grade, debug_info = detect_uv_aging(
                img_normal,
                img_uv,
                face_cascade
            )

        except Exception as e:
            print("분석 실패:", pair["key"], e)
            continue

        solution = create_solution(grade)

        base_name = pair["key"]

        result_image_path = SAVE_DIR / f"{base_name}_uv_aging_result.png"
        metadata_path = SAVE_DIR / f"{base_name}_uv_aging_metadata.json"

        info_text = (
            f"Count: {len(coordinates)} | "
            f"Rate: {detection_rate:.2f}% | "
            f"Grade: {grade}"
        )

        final_img = make_result_image(
            img_normal,
            result,
            info_text
        )

        imwrite_korean(result_image_path, final_img)

        save_metadata(
            metadata_path=metadata_path,
            pair_info=pair,
            result_image_path=result_image_path,
            coordinates=coordinates,
            detection_rate=detection_rate,
            grade=grade,
            solution=solution,
            debug_info=debug_info
        )

        print("저장 완료:", result_image_path)
        print("저장 완료:", metadata_path)

    print("전체 완료:", SAVE_DIR)


if __name__ == "__main__":
    run()