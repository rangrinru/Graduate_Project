import cv2
import numpy as np
import os

# -----------------------------
# 포르피린 강한 형광 검출
# -----------------------------
def detect_porphyrin(img):

    if img is None:
        return None, None, 0, 0, "Low", {}

    output = img.copy()

    # -----------------------------
    # 그레이 변환
    # -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 대비 강화
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # 블러
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # -----------------------------
    # 강한 형광만 추출
    # -----------------------------
    threshold_value = np.percentile(blur, 99)

    _, thresh = cv2.threshold(
        blur,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------------
    # 노이즈 제거
    # -----------------------------
    kernel = np.ones((3,3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # -----------------------------
    # 컨투어 검출
    # -----------------------------
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

    # -----------------------------
    # 검출
    # -----------------------------
    for cnt in contours:

        area = cv2.contourArea(cnt)

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

        count += 1

        total_area += area

        # -----------------------------
        # 영역 분석
        # -----------------------------
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
        "Upper": upper / face_pixels * 100,
        "Middle": middle / face_pixels * 100,
        "Lower": lower / face_pixels * 100,
        "Left": left / face_pixels * 100,
        "Center": center_area / face_pixels * 100,
        "Right": right / face_pixels * 100,
    }

    return (
        output,
        heatmap,
        count,
        detection_rate,
        grade,
        region_data
    )


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
            if f.lower().endswith(('.png', '.jpg'))
        ])

        for f in files:

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
            region_data
        ) = detect_porphyrin(img)

        # -----------------------------
        # 원본 + 히트맵
        # -----------------------------
        combined = np.hstack((
            img,
            heatmap
        ))

        combined = cv2.resize(
            combined,
            (1600, 800)
        )

        # -----------------------------
        # 텍스트
        # -----------------------------
        cv2.putText(
            combined,
            name,
            (820, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            combined,
            f"{idx+1} / {total}",
            (820, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            combined,
            f"Date: {date}",
            (820, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        cv2.putText(
            combined,
            f"Detected Count: {count}",
            (820, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,0,255),
            3
        )

        cv2.putText(
            combined,
            f"Detection Rate: {rate:.2f}%",
            (820, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,0,255),
            3
        )

        cv2.putText(
            combined,
            f"Grade: {grade}",
            (820, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            3
        )

        cv2.putText(
            combined,
            "Region Analysis",
            (820, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
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
                (255,255,0),
                2
            )

            y += 40

        cv2.putText(
            combined,
            "Original",
            (20, 760),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255,255,255),
            3
        )

        cv2.putText(
            combined,
            "Heatmap",
            (820, 760),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255,255,255),
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