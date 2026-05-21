import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path


# =========================================================
# 포르피린 강한 형광 검출 + metadata 저장 버전
# =========================================================
# 기능:
# 1. cam4_660nm 폴더의 날짜별 이미지 로드
# 2. 포르피린 강한 형광 검출
# 3. 원본/검출결과/히트맵/마스크/비교이미지 저장
# 4. metadata.json에 검출 개수, 검출률, 등급, 부위별 비율, 좌표 저장
# 5. Viewer에서 a/d 키로 이전/다음 이미지 확인
#
# 주의:
# 이 코드는 졸업작품용 영상처리 지표입니다.
# 의료 진단용 코드가 아닙니다.
# =========================================================

IMAGE_EXTS = [".png", ".jpg", ".jpeg"]


# -----------------------------
# 한글/공백 경로 대응 이미지 저장
# -----------------------------
def save_image(image_path, img):
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    ext = image_path.suffix.lower()

    if ext in [".jpg", ".jpeg"]:
        ok, encoded = cv2.imencode(
            ".jpg",
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
    elif ext == ".png":
        ok, encoded = cv2.imencode(
            ".png",
            img,
            [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )
    else:
        ok, encoded = cv2.imencode(".jpg", img)

    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {image_path}")

    encoded.tofile(str(image_path))


# -----------------------------
# 한글/공백 경로 대응 이미지 로드
# -----------------------------
def load_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        return None

    data = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    return img


# -----------------------------
# JSON 저장
# -----------------------------
def save_json(json_path, data):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )


# -----------------------------
# 포르피린 강한 형광 검출
# -----------------------------
def detect_porphyrin(img):
    if img is None:
        empty_result = {
            "detected_points": [],
            "detected_regions": [],
            "region_data": {},
        }

        return (
            None,
            None,
            None,
            0,
            0.0,
            "Low",
            {},
            empty_result
        )

    output = img.copy()

    # -----------------------------
    # 그레이 변환
    # -----------------------------
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------
    # 대비 강화
    # -----------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # -----------------------------
    # 블러
    # -----------------------------
    blur = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0
    )

    # -----------------------------
    # 강한 형광만 추출
    # 상위 1% 밝기 영역 기준
    # -----------------------------
    threshold_value = np.percentile(
        blur,
        99
    )

    _, thresh = cv2.threshold(
        blur,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------------
    # 노이즈 제거
    # -----------------------------
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.dilate(
        thresh,
        kernel,
        iterations=1
    )

    # -----------------------------
    # 컨투어 검출
    # -----------------------------
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    count = 0
    total_area = 0.0

    detected_points = []
    detected_regions = []

    heatmap = cv2.applyColorMap(
        blur,
        cv2.COLORMAP_JET
    )

    h, w = gray.shape

    upper = 0.0
    middle = 0.0
    lower = 0.0

    left = 0.0
    center_area = 0.0
    right = 0.0

    face_pixels = gray.size

    # -----------------------------
    # 검출
    # -----------------------------
    for cnt in contours:
        area = float(cv2.contourArea(cnt))

        # 너무 작은 점 제거
        if area < 20:
            continue

        # 너무 거대한 반사 제거
        if area > 4000:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        cx = x + cw // 2
        cy = y + ch // 2

        # -----------------------------
        # 검출 표시
        # -----------------------------
        cv2.drawContours(
            output,
            [cnt],
            -1,
            (0, 0, 255),
            2
        )

        cv2.circle(
            output,
            (cx, cy),
            4,
            (0, 255, 255),
            -1
        )

        count += 1
        total_area += area

        contour_points = cnt.reshape(-1, 2).astype(int).tolist()

        patch = blur[y:y + ch, x:x + cw]

        detected_points.append({
            "x": int(cx),
            "y": int(cy)
        })

        detected_regions.append({
            "id": int(count),
            "center": {
                "x": int(cx),
                "y": int(cy)
            },
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(cw),
                "height": int(ch)
            },
            "area_px": round(area, 2),
            "contour_points": contour_points,
            "mean_brightness": round(float(np.mean(patch)), 4),
            "max_brightness": int(np.max(patch))
        })

        # -----------------------------
        # 영역 분석: 상/중/하
        # -----------------------------
        if cy < h // 3:
            upper += area
        elif cy < h * 2 // 3:
            middle += area
        else:
            lower += area

        # -----------------------------
        # 영역 분석: 좌/중앙/우
        # -----------------------------
        if cx < w // 3:
            left += area
        elif cx < w * 2 // 3:
            center_area += area
        else:
            right += area

    # -----------------------------
    # 검출률
    # -----------------------------
    detection_rate = (
        total_area / face_pixels
    ) * 100

    # -----------------------------
    # 등급
    # -----------------------------
    if detection_rate < 1:
        grade = "Low"
    elif detection_rate < 3:
        grade = "Medium"
    else:
        grade = "High"

    # -----------------------------
    # 부위별 분석
    # -----------------------------
    region_data = {
        "Upper": round(upper / face_pixels * 100, 6),
        "Middle": round(middle / face_pixels * 100, 6),
        "Lower": round(lower / face_pixels * 100, 6),
        "Left": round(left / face_pixels * 100, 6),
        "Center": round(center_area / face_pixels * 100, 6),
        "Right": round(right / face_pixels * 100, 6),
    }

    detection_detail = {
        "detected_points": detected_points,
        "detected_regions": detected_regions,
        "threshold_value": round(float(threshold_value), 4),
        "total_area_px": round(float(total_area), 2),
        "image_width": int(w),
        "image_height": int(h),
    }

    return (
        output,
        heatmap,
        thresh,
        count,
        detection_rate,
        grade,
        region_data,
        detection_detail
    )


# -----------------------------
# 결과 비교 이미지 생성
# -----------------------------
def make_combined_result(img, result, heatmap, count, rate, grade, region_data, image_name, date):
    h, w = img.shape[:2]

    result_resized = cv2.resize(
        result,
        (w, h)
    )

    heatmap_resized = cv2.resize(
        heatmap,
        (w, h)
    )

    combined = np.hstack((
        img,
        result_resized,
        heatmap_resized
    ))

    display_width = 1800
    display_height = 800

    combined = cv2.resize(
        combined,
        (display_width, display_height)
    )

    cv2.putText(
        combined,
        "Original",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        combined,
        "Detection",
        (620, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.putText(
        combined,
        "Heatmap",
        (1220, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    text_x = 1220
    y = 90

    text_lines = [
        f"File: {image_name}",
        f"Date: {date}",
        f"Detected Count: {count}",
        f"Detection Rate: {rate:.2f}%",
        f"Grade: {grade}",
        "Region Analysis",
    ]

    for line in text_lines:
        color = (0, 255, 255) if "Region" in line else (255, 255, 255)

        if "Detected" in line or "Rate" in line:
            color = (0, 0, 255)

        if "Grade" in line:
            color = (0, 255, 0)

        cv2.putText(
            combined,
            line,
            (text_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        y += 35

    y += 10

    for k, v in region_data.items():
        cv2.putText(
            combined,
            f"{k}: {v:.2f}%",
            (text_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        y += 32

    return combined


# -----------------------------
# metadata payload 생성
# -----------------------------
def build_metadata(
    input_path,
    date,
    name,
    img,
    count,
    rate,
    grade,
    region_data,
    detection_detail,
    output_files
):
    input_path = Path(input_path)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "analysis_type": "porphyrin_strong_fluorescence_detection",
        "input_file": str(input_path),
        "date_folder": str(date),
        "image_name": str(name),
        "output_files": output_files,
        "image_info": {
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "channels": int(img.shape[2]) if img.ndim == 3 else 1,
            "shape": list(img.shape)
        },
        "analysis_result": {
            "porphyrin_count": int(count),
            "detection_rate_percent": round(float(rate), 6),
            "grade": str(grade),
            "region_data": region_data,
            "detected_points": detection_detail["detected_points"],
            "detected_regions": detection_detail["detected_regions"],
            "porphyrin_total_area_px": detection_detail["total_area_px"],
            "threshold_value": detection_detail["threshold_value"]
        },
        "coordinate_system": {
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
            "unit": "pixel",
            "note": "좌표는 입력 이미지 기준입니다."
        },
        "interpretation_note": (
            "이 결과는 660nm 필터 영상에서 밝게 검출되는 포르피린 후보를 규칙 기반으로 분석한 값입니다. "
            "조명, 노출, 필터, 피부 반사, 이미지 방향에 따라 결과가 달라질 수 있으며 의료 진단용이 아닙니다."
        )
    }

    return metadata


# -----------------------------
# 결과 저장
# -----------------------------
def save_analysis_outputs(
    date,
    name,
    input_path,
    img,
    result,
    heatmap,
    thresh,
    count,
    rate,
    grade,
    region_data,
    detection_detail,
    combined
):
    input_path = Path(input_path)

    # 이미지마다 별도 분석 폴더 생성
    # 예:
    # 2026-04-07/
    # ├─ 20260407_233614_041_cam4.png
    # └─ analysis/
    #    └─ 20260407_233614_041_cam4/
    #       ├─ metadata.json
    #       ├─ porphyrin_result.jpg
    #       ├─ porphyrin_heatmap.jpg
    #       ├─ porphyrin_mask.png
    #       └─ porphyrin_compare.jpg
    save_dir = input_path.parent / "analysis" / input_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    result_path = save_dir / "porphyrin_result.jpg"
    heatmap_path = save_dir / "porphyrin_heatmap.jpg"
    mask_path = save_dir / "porphyrin_mask.png"
    compare_path = save_dir / "porphyrin_compare.jpg"
    metadata_path = save_dir / "metadata.json"

    save_image(result_path, result)
    save_image(heatmap_path, heatmap)
    save_image(mask_path, thresh)
    save_image(compare_path, combined)

    output_files = {
        "result_image": str(result_path),
        "heatmap_image": str(heatmap_path),
        "mask_image": str(mask_path),
        "compare_image": str(compare_path),
        "metadata": str(metadata_path)
    }

    metadata = build_metadata(
        input_path=input_path,
        date=date,
        name=name,
        img=img,
        count=count,
        rate=rate,
        grade=grade,
        region_data=region_data,
        detection_detail=detection_detail,
        output_files=output_files
    )

    save_json(metadata_path, metadata)

    return metadata_path, save_dir


# -----------------------------
# 이미지 리스트
# -----------------------------
def load_images(base_path):
    image_list = []

    base_path = Path(base_path)

    if not base_path.exists():
        print("기준 폴더가 없습니다:", base_path)
        return image_list

    date_folders = sorted([
        p for p in base_path.iterdir()
        if p.is_dir()
    ])

    for date_folder in date_folders:
        # analysis 폴더는 다시 읽지 않음
        if date_folder.name.lower() == "analysis":
            continue

        files = sorted([
            p for p in date_folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ])

        for file_path in files:
            image_list.append((
                date_folder.name,
                file_path.name,
                str(file_path)
            ))

    return image_list


# -----------------------------
# Viewer
# -----------------------------
def viewer(base_path, auto_save=True):
    images = load_images(base_path)

    if len(images) == 0:
        print("이미지 없음")
        return

    idx = 0
    total = len(images)

    cv2.namedWindow(
        "Porphyrin Viewer",
        cv2.WINDOW_NORMAL
    )

    print("조작 방법")
    print("d: 다음 이미지")
    print("a: 이전 이미지")
    print("s: 현재 이미지 분석 결과 다시 저장")
    print("ESC: 종료")

    while True:
        date, name, path = images[idx]

        img = load_image(path)

        if img is None:
            idx = (idx + 1) % total
            continue

        (
            result,
            heatmap,
            thresh,
            count,
            rate,
            grade,
            region_data,
            detection_detail
        ) = detect_porphyrin(img)

        combined = make_combined_result(
            img=img,
            result=result,
            heatmap=heatmap,
            count=count,
            rate=rate,
            grade=grade,
            region_data=region_data,
            image_name=name,
            date=date
        )

        if auto_save:
            metadata_path, save_dir = save_analysis_outputs(
                date=date,
                name=name,
                input_path=path,
                img=img,
                result=result,
                heatmap=heatmap,
                thresh=thresh,
                count=count,
                rate=rate,
                grade=grade,
                region_data=region_data,
                detection_detail=detection_detail,
                combined=combined
            )

            print(f"[저장 완료] {metadata_path}")

        cv2.imshow(
            "Porphyrin Viewer",
            combined
        )

        key = cv2.waitKey(0) & 0xFF

        if key == ord("d"):
            idx = (idx + 1) % total

        elif key == ord("a"):
            idx = (idx - 1) % total

        elif key == ord("s"):
            metadata_path, save_dir = save_analysis_outputs(
                date=date,
                name=name,
                input_path=path,
                img=img,
                result=result,
                heatmap=heatmap,
                thresh=thresh,
                count=count,
                rate=rate,
                grade=grade,
                region_data=region_data,
                detection_detail=detection_detail,
                combined=combined
            )

            print(f"[수동 저장 완료] {metadata_path}")

        elif key == 27:
            break

    cv2.destroyAllWindows()


# -----------------------------
# 단일 이미지 분석
# -----------------------------
def analyze_single_image(image_path, output_dir=None):
    image_path = Path(image_path)

    img = load_image(image_path)

    if img is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")

    (
        result,
        heatmap,
        thresh,
        count,
        rate,
        grade,
        region_data,
        detection_detail
    ) = detect_porphyrin(img)

    combined = make_combined_result(
        img=img,
        result=result,
        heatmap=heatmap,
        count=count,
        rate=rate,
        grade=grade,
        region_data=region_data,
        image_name=image_path.name,
        date=image_path.parent.name
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result_path = output_dir / "porphyrin_result.jpg"
        heatmap_path = output_dir / "porphyrin_heatmap.jpg"
        mask_path = output_dir / "porphyrin_mask.png"
        compare_path = output_dir / "porphyrin_compare.jpg"
        metadata_path = output_dir / "metadata.json"

        save_image(result_path, result)
        save_image(heatmap_path, heatmap)
        save_image(mask_path, thresh)
        save_image(compare_path, combined)

        output_files = {
            "result_image": str(result_path),
            "heatmap_image": str(heatmap_path),
            "mask_image": str(mask_path),
            "compare_image": str(compare_path),
            "metadata": str(metadata_path)
        }

        metadata = build_metadata(
            input_path=image_path,
            date=image_path.parent.name,
            name=image_path.name,
            img=img,
            count=count,
            rate=rate,
            grade=grade,
            region_data=region_data,
            detection_detail=detection_detail,
            output_files=output_files
        )

        save_json(metadata_path, metadata)

    else:
        metadata_path, output_dir = save_analysis_outputs(
            date=image_path.parent.name,
            name=image_path.name,
            input_path=image_path,
            img=img,
            result=result,
            heatmap=heatmap,
            thresh=thresh,
            count=count,
            rate=rate,
            grade=grade,
            region_data=region_data,
            detection_detail=detection_detail,
            combined=combined
        )

    print("분석 완료")
    print("포르피린 개수:", count)
    print("검출률:", f"{rate:.4f}%")
    print("등급:", grade)
    print("metadata:", metadata_path)

    return metadata_path


# -----------------------------
# 기본 경로 탐색
# -----------------------------
def find_default_base_path():
    candidates = [
        Path.home() / "Graduate_Project" / "captures" / "cam4_660nm",
        Path.home() / "Graduate_Project" / "captures" / "sessions" / "cam4_660nm",
        Path.cwd() / "captures" / "cam4_660nm",
        Path.cwd() / "captures" / "sessions" / "cam4_660nm",
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\captures\cam4_660nm"),
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\captures\sessions\cam4_660nm"),
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="날짜 폴더들이 들어있는 cam4_660nm 기준 폴더"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="단일 이미지 경로"
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="단일 이미지 분석 결과 저장 폴더"
    )

    parser.add_argument(
        "--no-auto-save",
        action="store_true",
        help="viewer에서 자동 저장하지 않음. s 키로 수동 저장 가능"
    )

    args = parser.parse_args()

    if args.image is not None:
        analyze_single_image(
            image_path=args.image,
            output_dir=args.out
        )

    else:
        if args.base is not None:
            base_path = Path(args.base)
        else:
            base_path = find_default_base_path()

        print("기준 폴더:", base_path)

        viewer(
            base_path=base_path,
            auto_save=not args.no_auto_save
        )
