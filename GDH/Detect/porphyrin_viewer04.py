import cv2
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime


# -----------------------------
# 포르피린 강한 형광 검출
# -----------------------------
def detect_porphyrin(img):

    if img is None:
        return None, None, 0, 0, "Low", {}

    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    threshold_value = np.percentile(blur, 99)

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

    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    count = 0
    total_area = 0

    heatmap = cv2.applyColorMap(
        blur,
        cv2.COLORMAP_JET
    )

    h, w = gray.shape

    upper = middle = lower = 0
    left = center_area = right = 0

    face_pixels = gray.size

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 20:
            continue

        if area > 4000:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        cx = x + cw // 2
        cy = y + ch // 2

        cv2.drawContours(
            output,
            [cnt],
            -1,
            (0, 0, 255),
            2
        )

        count += 1
        total_area += area

        if cy < h // 3:
            upper += area
        elif cy < h * 2 // 3:
            middle += area
        else:
            lower += area

        if cx < w // 3:
            left += area
        elif cx < w * 2 // 3:
            center_area += area
        else:
            right += area

    detection_rate = (total_area / face_pixels) * 100

    if detection_rate < 1:
        grade = "Low"
    elif detection_rate < 3:
        grade = "Medium"
    else:
        grade = "High"

    region_data = {
        "Upper": upper / face_pixels * 100,
        "Middle": middle / face_pixels * 100,
        "Lower": lower / face_pixels * 100,
        "Left": left / face_pixels * 100,
        "Center": center_area / face_pixels * 100,
        "Right": right / face_pixels * 100,
    }

    analysis_data = {
        "porphyrin_count": int(count),
        "detection_rate_percent": float(detection_rate),
        "grade": grade,
        "region_analysis": {
            k: float(v) for k, v in region_data.items()
        },
        "threshold_percentile": 99,
        "threshold_value": float(threshold_value),
        "min_area": 20,
        "max_area": 4000
    }

    return (
        output,
        heatmap,
        count,
        detection_rate,
        grade,
        region_data,
        analysis_data
    )


# -----------------------------
# 분석 메타데이터 저장
# -----------------------------
def save_analysis_metadata(date, name, path, analysis_data):

    image_path = Path(path)
    save_dir = image_path.parent

    meta_path = save_dir / f"{image_path.stem}_analysis_metadata.json"

    metadata = {
        "metadata_type": "porphyrin_analysis",
        "analyzed_at": datetime.now().isoformat(),
        "date_folder": date,
        "image_name": name,
        "image_path": str(image_path),
        "saved_metadata_file": str(meta_path),
        "analysis_result": {
            "porphyrin_count": analysis_data["porphyrin_count"],
            "detection_rate_percent": round(
                analysis_data["detection_rate_percent"], 4
            ),
            "grade": analysis_data["grade"],
            "region_analysis": {
                k: round(float(v), 4)
                for k, v in analysis_data["region_analysis"].items()
            },
            "threshold_percentile": analysis_data["threshold_percentile"],
            "threshold_value": round(
                analysis_data["threshold_value"], 4
            ),
            "min_area": analysis_data["min_area"],
            "max_area": analysis_data["max_area"]
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return meta_path


# -----------------------------
# 이미지 리스트
# -----------------------------
def load_images(base_path):

    image_list = []

    date_folders = sorted(os.listdir(base_path))

    for date in date_folders:

        folder_path = os.path.join(base_path, date)

        if not os.path.isdir(folder_path):
            continue

        files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        for f in files:

            # 분석 결과 이미지나 메타데이터 파일 제외
            if "analysis" in f.lower() or "metadata" in f.lower():
                continue

            full_path = os.path.join(
                folder_path,
                f
            )

            image_list.append(
                (date, f, full_path)
            )

    return image_list


# -----------------------------
# Viewer
# -----------------------------
def viewer(base_path):

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

    while True:

        date, name, path = images[idx]

        img = cv2.imread(path)

        if img is None:
            idx = (idx + 1) % total
            continue

        (
            result,
            heatmap,
            count,
            rate,
            grade,
            region_data,
            analysis_data
        ) = detect_porphyrin(img)

        meta_path = save_analysis_metadata(
            date=date,
            name=name,
            path=path,
            analysis_data=analysis_data
        )

        print("메타데이터 저장:", meta_path)

        combined = np.hstack((
            img,
            heatmap
        ))

        combined = cv2.resize(
            combined,
            (1600, 800)
        )

        cv2.putText(
            combined,
            name,
            (820, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            combined,
            f"{idx+1} / {total}",
            (820, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            combined,
            f"Date: {date}",
            (820, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            combined,
            f"Detected Count: {count}",
            (820, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

        cv2.putText(
            combined,
            f"Detection Rate: {rate:.2f}%",
            (820, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

        cv2.putText(
            combined,
            f"Grade: {grade}",
            (820, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3
        )

        cv2.putText(
            combined,
            "Region Analysis",
            (820, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        y = 390

        for k, v in region_data.items():

            cv2.putText(
                combined,
                f"{k}: {v:.2f}%",
                (820, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            y += 40

        cv2.putText(
            combined,
            "Original",
            (20, 760),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3
        )

        cv2.putText(
            combined,
            "Heatmap",
            (820, 760),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3
        )

        cv2.imshow(
            "Porphyrin Viewer",
            combined
        )

        key = cv2.waitKey(0) & 0xFF

        if key == ord('d'):
            idx = (idx + 1) % total

        elif key == ord('a'):
            idx = (idx - 1) % total

        elif key == 27:
            break

    cv2.destroyAllWindows()


# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":

    base_path = "../../captures/cam4_660nm"

    viewer(base_path)