import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime


# =============================
# 최신 이미지 찾기
# =============================
def get_latest_image(base_path):

    base_path = Path(base_path)

    # 날짜 폴더 찾기
    date_folders = sorted(
        [
            f for f in base_path.iterdir()
            if f.is_dir()
        ]
    )

    if not date_folders:
        print("날짜 폴더 없음")
        return None

    latest_date_folder = date_folders[-1]

    print("최신 날짜 폴더:", latest_date_folder)

    # 이미지 파일 찾기
    image_files = []

    image_files.extend(
        latest_date_folder.glob("*.png")
    )

    image_files.extend(
        latest_date_folder.glob("*.jpg")
    )

    image_files.extend(
        latest_date_folder.glob("*.jpeg")
    )

    if not image_files:
        print("이미지 파일 없음")
        return None

    latest_image = sorted(image_files)[-1]

    print("최신 이미지:", latest_image)

    return latest_image


# =============================
# 이미지 로드 (한글 경로 대응)
# =============================
def load_image_korean_path(image_path):

    img_array = np.fromfile(
        str(image_path),
        np.uint8
    )

    img = cv2.imdecode(
        img_array,
        cv2.IMREAD_COLOR
    )

    return img


# =============================
# 포르피린 분석
# =============================
def detect_porphyrin(image_path):

    image_path = Path(image_path)

    img = load_image_korean_path(image_path)

    if img is None:
        print("이미지 로드 실패")
        return

    output = img.copy()

    # =============================
    # analysis 폴더 생성
    # =============================
    analysis_dir = image_path.parent / "analysis"

    analysis_dir.mkdir(exist_ok=True)

    # =============================
    # 그레이 + 대비 강화
    # =============================
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

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

    # =============================
    # 상위 밝기만 추출
    # =============================
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

    # =============================
    # 노이즈 제거
    # =============================
    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    # =============================
    # 컨투어 검출
    # =============================
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    count = 0

    brightness_values = []

    points = []

    h, w = img.shape[:2]

    # =============================
    # 분포도 영역
    # =============================
    distribution = {
        "left_cheek": 0,
        "right_cheek": 0,
        "forehead": 0,
        "nose": 0,
        "chin": 0
    }

    # =============================
    # 포르피린 검출
    # =============================
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

            points.append(center)

            brightness = int(
                blur[int(y), int(x)]
            )

            brightness_values.append(
                brightness
            )

            # =============================
            # 얼굴 분포 계산
            # =============================
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

    # =============================
    # 평균 밝기 계산
    # =============================
    if len(brightness_values) > 0:

        avg_brightness = float(
            np.mean(brightness_values)
        )

        max_brightness = int(
            np.max(brightness_values)
        )

    else:

        avg_brightness = 0
        max_brightness = 0

    # =============================
    # 결과 이미지 저장
    # =============================
    detect_path = analysis_dir / "porphyrin_detect_result.jpg"

    cv2.imwrite(
        str(detect_path),
        output
    )

    # =============================
    # 비교 이미지 생성
    # =============================
    output_resized = cv2.resize(
        output,
        (w, h)
    )

    combined = np.hstack(
        (img, output_resized)
    )

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

    compare_path = analysis_dir / "compare_result.jpg"

    cv2.imwrite(
        str(compare_path),
        combined
    )

    # =============================
    # 분포도 이미지 생성
    # =============================
    heatmap = output.copy()

    for p in points:

        cv2.circle(
            heatmap,
            p,
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

    cv2.imwrite(
        str(heatmap_path),
        heatmap
    )

    # =============================
    # 메타데이터 생성
    # =============================
    timestamp = datetime.now()

    metadata = {

        "captured_at": timestamp.isoformat(),

        "camera_name": "cam4",

        "camera_label": "CAM 4 - 660nm FILTER",

        "filter_type": "660nm_filter",

        "analysis_type": "porphyrin_detection",

        "saved_file": str(image_path),

        "analysis_result": {

            "porphyrin_count": count,

            "average_brightness": avg_brightness,

            "max_brightness": max_brightness,

            "distribution": distribution,

            "threshold_percentile": 97,

            "detected_points": points
        },

        "analysis_files": {

            "detect_image": str(detect_path),

            "compare_image": str(compare_path),

            "distribution_map": str(heatmap_path)
        }
    }

    # =============================
    # 메타데이터 저장
    # =============================
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

    # =============================
    # 화면 출력
    # =============================
    display = cv2.resize(
        combined,
        (1400, 500)
    )

    cv2.imshow(
        "Porphyrin Compare",
        display
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return metadata


# =============================
# 실행
# =============================
if __name__ == "__main__":

    base_path = r"C:\Users\고대현\Desktop\Graduate_Project\captures\cam4_660nm"

    latest_image = get_latest_image(base_path)

    if latest_image is not None:

        print("사용 이미지:", latest_image)

        detect_porphyrin(latest_image)