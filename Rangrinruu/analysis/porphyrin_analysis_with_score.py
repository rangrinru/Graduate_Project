import argparse
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime


# =============================
# 점수화 기준값
# 나중에 실제 데이터가 쌓이면 이 값들을 조정하면 됨
# =============================
BAD_COUNT = 120                  # 이 개수 이상이면 개수 위험도 1.0
BAD_AREA_RATIO = 0.015           # 전체 이미지 면적 대비 1.5% 이상이면 면적 위험도 1.0
AVG_BRIGHTNESS_LOW = 120         # 평균 밝기 위험도 시작점
AVG_BRIGHTNESS_HIGH = 230        # 평균 밝기 위험도 1.0 기준
MAX_BRIGHTNESS_LOW = 160         # 최대 밝기 위험도 시작점
MAX_BRIGHTNESS_HIGH = 255        # 최대 밝기 위험도 1.0 기준
CONCENTRATION_SAFE = 0.35        # 한 부위 집중도 안전 기준
CONCENTRATION_BAD = 0.65         # 한 부위 집중도 위험 기준


# =============================
# 최신 이미지 찾기
# =============================
def get_latest_image(base_path):
    base_path = Path(base_path)

    if not base_path.exists():
        print("기준 폴더가 없습니다:", base_path)
        return None

    date_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])

    if not date_folders:
        print("날짜 폴더 없음")
        return None

    latest_date_folder = date_folders[-1]
    print("최신 날짜 폴더:", latest_date_folder)

    image_files = []
    image_files.extend(latest_date_folder.glob("*.png"))
    image_files.extend(latest_date_folder.glob("*.jpg"))
    image_files.extend(latest_date_folder.glob("*.jpeg"))

    if not image_files:
        print("이미지 파일 없음")
        return None

    latest_image = sorted(image_files)[-1]
    print("최신 이미지:", latest_image)
    return latest_image


# =============================
# 이미지 로드/저장: 한글 경로 대응
# =============================
def load_image_korean_path(image_path):
    img_array = np.fromfile(str(image_path), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def save_image_korean_path(path, img):
    path = Path(path)
    ext = path.suffix.lower()

    if ext in [".jpg", ".jpeg"]:
        success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    elif ext == ".png":
        success, encoded = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    if success:
        encoded.tofile(str(path))
    else:
        raise RuntimeError(f"이미지 저장 실패: {path}")


# =============================
# 점수화 유틸
# =============================
def clamp_float(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def normalize_linear(value, low, high):
    """
    value가 low 이하이면 0, high 이상이면 1.
    그 사이면 선형 정규화.
    """
    if high <= low:
        return 0.0
    return clamp_float((value - low) / (high - low))


def get_grade(score):
    if score >= 85:
        return "양호"
    if score >= 70:
        return "보통"
    if score >= 55:
        return "주의"
    return "위험"


def get_main_region(distribution):
    if not distribution:
        return "unknown", 0

    region = max(distribution, key=distribution.get)
    return region, distribution[region]


def make_recommendations(score, distribution, count, avg_brightness, area_ratio, concentration_ratio):
    recommendations = []

    main_region, main_region_count = get_main_region(distribution)

    region_name_ko = {
        "left_cheek": "왼쪽 볼",
        "right_cheek": "오른쪽 볼",
        "forehead": "이마",
        "nose": "코",
        "chin": "턱",
        "unknown": "알 수 없음"
    }

    if count == 0:
        recommendations.append("검출된 포르피린이 없거나 매우 적습니다. 단, 노출이 너무 낮은 촬영인지 확인이 필요합니다.")
        return recommendations

    if score < 70:
        recommendations.append("포르피린 지표가 높게 나타났습니다. 동일 조건에서 재촬영 후 변화 추적을 권장합니다.")

    if main_region_count > 0:
        recommendations.append(f"가장 많이 검출된 부위는 {region_name_ko.get(main_region, main_region)}입니다.")

    if concentration_ratio >= 0.65:
        recommendations.append("특정 부위에 포르피린이 집중되어 있어 부위별 관리가 필요합니다.")

    if avg_brightness >= 200:
        recommendations.append("평균 형광 강도가 높아 밝게 반응하는 포르피린 영역이 많을 수 있습니다.")

    if area_ratio >= 0.01:
        recommendations.append("포르피린 총 면적 비율이 높아 넓은 영역의 분포를 확인해야 합니다.")

    if not recommendations:
        recommendations.append("현재 기준에서는 큰 이상 패턴은 보이지 않습니다. 주기적으로 변화량을 비교하면 더 정확합니다.")

    return recommendations


def calculate_skin_score(
    porphyrin_count,
    total_area,
    image_area,
    avg_brightness,
    max_brightness,
    distribution
):
    """
    0~100 피부 상태 점수 계산.
    점수가 높을수록 좋은 상태로 해석.

    현재는 임상 모델이 아니라 프로젝트용 규칙 기반 점수화임.
    실제 사용 시에는 사용자별/촬영조건별 데이터로 기준값 보정 필요.
    """

    if image_area <= 0:
        image_area = 1

    area_ratio = float(total_area) / float(image_area)

    total_distribution_count = sum(distribution.values()) if distribution else 0

    if total_distribution_count > 0:
        max_region_count = max(distribution.values())
        concentration_ratio = max_region_count / total_distribution_count
    else:
        max_region_count = 0
        concentration_ratio = 0.0

    count_risk = clamp_float(porphyrin_count / BAD_COUNT)
    area_risk = clamp_float(area_ratio / BAD_AREA_RATIO)
    avg_intensity_risk = normalize_linear(avg_brightness, AVG_BRIGHTNESS_LOW, AVG_BRIGHTNESS_HIGH)
    max_intensity_risk = normalize_linear(max_brightness, MAX_BRIGHTNESS_LOW, MAX_BRIGHTNESS_HIGH)
    concentration_risk = normalize_linear(concentration_ratio, CONCENTRATION_SAFE, CONCENTRATION_BAD)

    # 가중치 합 = 1.0
    weighted_risk = (
        0.35 * count_risk +
        0.25 * area_risk +
        0.20 * avg_intensity_risk +
        0.10 * max_intensity_risk +
        0.10 * concentration_risk
    )

    score = int(round(100 * (1.0 - weighted_risk)))
    score = max(0, min(100, score))

    grade = get_grade(score)
    main_region, main_region_count = get_main_region(distribution)

    recommendations = make_recommendations(
        score=score,
        distribution=distribution,
        count=porphyrin_count,
        avg_brightness=avg_brightness,
        area_ratio=area_ratio,
        concentration_ratio=concentration_ratio
    )

    return {
        "skin_score": score,
        "grade": grade,
        "risk_components": {
            "count_risk": round(count_risk, 4),
            "area_risk": round(area_risk, 4),
            "avg_intensity_risk": round(avg_intensity_risk, 4),
            "max_intensity_risk": round(max_intensity_risk, 4),
            "concentration_risk": round(concentration_risk, 4),
            "weighted_risk": round(weighted_risk, 4)
        },
        "area_metrics": {
            "total_area_px": float(total_area),
            "image_area_px": int(image_area),
            "area_ratio": round(area_ratio, 6)
        },
        "distribution_metrics": {
            "main_region": main_region,
            "main_region_count": int(main_region_count),
            "concentration_ratio": round(concentration_ratio, 4)
        },
        "recommendations": recommendations,
        "scoring_note": "규칙 기반 점수입니다. 의료 진단용이 아니라 졸업작품용 정량 지표로 사용하세요."
    }


# =============================
# 포르피린 분석 + 피부 점수화
# =============================
def detect_porphyrin(image_path, show_window=True):
    image_path = Path(image_path)
    img = load_image_korean_path(image_path)

    if img is None:
        print("이미지 로드 실패")
        return None

    output = img.copy()

    analysis_dir = image_path.parent / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0
    )

    threshold_value = np.percentile(
        blur,
        97
    )

    _, thresh = cv2.threshold(
        blur,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    count = 0
    total_area = 0.0
    brightness_values = []
    points = []

    h, w = img.shape[:2]
    image_area = h * w

    distribution = {
        "left_cheek": 0,
        "right_cheek": 0,
        "forehead": 0,
        "nose": 0,
        "chin": 0
    }

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 10:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        if 3 < radius < 15:
            center = (int(x), int(y))

            cv2.circle(
                output,
                center,
                int(radius),
                (0, 0, 255),
                2
            )

            count += 1
            total_area += float(area)
            points.append([int(x), int(y)])

            brightness = int(blur[int(y), int(x)])
            brightness_values.append(brightness)

            cx, cy = center

            if cy < h * 0.3:
                distribution["forehead"] += 1
            elif cy > h * 0.75:
                distribution["chin"] += 1
            else:
                if cx < w * 0.35:
                    distribution["left_cheek"] += 1
                elif cx > w * 0.65:
                    distribution["right_cheek"] += 1
                else:
                    distribution["nose"] += 1

    print("검출 개수:", count)

    if len(brightness_values) > 0:
        avg_brightness = float(np.mean(brightness_values))
        max_brightness = int(np.max(brightness_values))
    else:
        avg_brightness = 0.0
        max_brightness = 0

    score_result = calculate_skin_score(
        porphyrin_count=count,
        total_area=total_area,
        image_area=image_area,
        avg_brightness=avg_brightness,
        max_brightness=max_brightness,
        distribution=distribution
    )

    detect_path = analysis_dir / "porphyrin_detect_result.jpg"
    save_image_korean_path(detect_path, output)

    output_resized = cv2.resize(output, (w, h))
    combined = np.hstack((img, output_resized))

    cv2.putText(
        combined,
        "Original",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        combined,
        "Detection",
        (w + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        combined,
        f"Porphyrin Count : {count}",
        (w + 20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.putText(
        combined,
        f"Skin Score : {score_result['skin_score']} / 100",
        (w + 20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        combined,
        f"Grade : {score_result['grade']}",
        (w + 20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    compare_path = analysis_dir / "compare_result.jpg"
    save_image_korean_path(compare_path, combined)

    heatmap = output.copy()

    for p in points:
        cv2.circle(
            heatmap,
            tuple(p),
            20,
            (0, 0, 255),
            -1
        )

    heatmap = cv2.GaussianBlur(
        heatmap,
        (51, 51),
        0
    )

    heatmap_path = analysis_dir / "distribution_map.jpg"
    save_image_korean_path(heatmap_path, heatmap)

    timestamp = datetime.now()

    metadata = {
        "captured_at": timestamp.isoformat(),
        "camera_name": "cam4",
        "camera_label": "CAM 4 - 660nm FILTER",
        "filter_type": "660nm_filter",
        "analysis_type": "porphyrin_detection_with_skin_score",
        "saved_file": str(image_path),
        "analysis_result": {
            "porphyrin_count": count,
            "porphyrin_total_area_px": float(total_area),
            "porphyrin_area_ratio": score_result["area_metrics"]["area_ratio"],
            "average_brightness": avg_brightness,
            "max_brightness": max_brightness,
            "distribution": distribution,
            "threshold_percentile": 97,
            "detected_points": points,
            "skin_scoring": score_result
        },
        "analysis_files": {
            "detect_image": str(detect_path),
            "compare_image": str(compare_path),
            "distribution_map": str(heatmap_path),
            "metadata": str(analysis_dir / "metadata.json")
        }
    }

    metadata_path = analysis_dir / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("분석 완료")
    print("analysis 저장 경로:", analysis_dir)
    print("피부 상태 점수:", score_result["skin_score"])
    print("등급:", score_result["grade"])
    print("주요 부위:", score_result["distribution_metrics"]["main_region"])

    for rec in score_result["recommendations"]:
        print("-", rec)

    if show_window:
        display = cv2.resize(
            combined,
            (1400, 500)
        )

        cv2.imshow(
            "Porphyrin Compare + Skin Score",
            display
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return metadata


# =============================
# 기본 경로 자동 탐색
# =============================
def find_default_base_path():
    candidates = [
        Path.home() / "Graduate_Project" / "captures" / "sessions" / "cam4_660nm",
        Path.home() / "Graduate_Project" / "captures" / "cam4_660nm",
        Path.cwd() / "captures" / "sessions" / "cam4_660nm",
        Path.cwd() / "captures" / "cam4_660nm",
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\captures\sessions\cam4_660nm"),
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\captures\cam4_660nm"),
    ]

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]


# =============================
# 실행
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="분석할 이미지 경로")
    parser.add_argument("--base", type=str, default=None, help="cam4_660nm 기준 폴더")
    parser.add_argument("--no-window", action="store_true", help="결과 창을 띄우지 않음")
    args = parser.parse_args()

    if args.image is not None:
        target_image = Path(args.image)
    else:
        base_path = Path(args.base) if args.base is not None else find_default_base_path()
        print("기준 폴더:", base_path)
        target_image = get_latest_image(base_path)

    if target_image is not None:
        print("사용 이미지:", target_image)
        detect_porphyrin(target_image, show_window=not args.no_window)
