import cv2
import numpy as np
import os

# -----------------------------
# 등급 계산
# -----------------------------
def get_grade(rate):
    if rate < 1:
        return "Low"
    elif rate < 3:
        return "Normal"
    elif rate < 7:
        return "High"
    else:
        return "Very High"


# -----------------------------
# 얼굴 실루엣 마스크 생성
# -----------------------------
# -----------------------------
# 얼굴 실루엣 마스크 생성
# -----------------------------
def create_face_mask(img):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    mask = np.zeros((h, w), dtype=np.uint8)

    if len(faces) == 0:
        print("[경고] 얼굴을 찾지 못해서 전체 이미지 기준으로 분석함")
        mask[:, :] = 255
        return mask

    # 가장 큰 얼굴 선택
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    # 중심점 보정
    cx = x + fw // 2
    cy = y + int(fh * 0.55)

    # 기존보다 훨씬 타이트하게
    axis_x = int(fw * 0.48)
    axis_y = int(fh * 0.62)

    cv2.ellipse(
        mask,
        (cx, cy),
        (axis_x, axis_y),
        0,
        0,
        360,
        255,
        -1
    )

    # 상단 머리카락/헤어밴드 제거
    top_cut = y + int(fh * 0.08)
    mask[:top_cut, :] = 0

    # 하단 목 영역 제거
    bottom_cut = y + int(fh * 0.95)
    mask[bottom_cut:, :] = 0

    # 좌우 배경 줄이기
    left_cut = x + int(fw * 0.08)
    right_cut = x + int(fw * 0.92)
    mask[:, :left_cut] = 0
    mask[:, right_cut:] = 0

    return mask


# -----------------------------
# 부위별 분석
# -----------------------------
def analyze_regions(mask, face_mask):
    h, w = mask.shape

    regions = {
        "Upper": (mask[0:h//3, :], face_mask[0:h//3, :]),
        "Middle": (mask[h//3:2*h//3, :], face_mask[h//3:2*h//3, :]),
        "Lower": (mask[2*h//3:h, :], face_mask[2*h//3:h, :]),
        "Left": (mask[:, 0:w//3], face_mask[:, 0:w//3]),
        "Center": (mask[:, w//3:2*w//3], face_mask[:, w//3:2*w//3]),
        "Right": (mask[:, 2*w//3:w], face_mask[:, 2*w//3:w])
    }

    result = {}

    for name, (roi_mask, roi_face) in regions.items():
        face_pixels = np.count_nonzero(roi_face)
        detected_pixels = np.count_nonzero(roi_mask)

        if face_pixels == 0:
            rate = 0
        else:
            rate = (detected_pixels / face_pixels) * 100

        result[name] = rate

    return result


# -----------------------------
# 히트맵 생성
# -----------------------------
def create_heatmap(img, mask, face_mask):
    # 얼굴 안쪽 검출 영역만 사용
    mask = cv2.bitwise_and(mask, mask, mask=face_mask)

    mask_blur = cv2.GaussianBlur(mask, (31, 31), 0)
    mask_blur = cv2.bitwise_and(mask_blur, mask_blur, mask=face_mask)

    heatmap = cv2.applyColorMap(mask_blur, cv2.COLORMAP_JET)

    # 얼굴 바깥은 원본처럼 어둡게 유지
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    dark = (img * 0.4).astype(np.uint8)
    result = np.where(face_mask[:, :, None] == 255, overlay, dark)

    return result


# -----------------------------
# 포르피린 검출
# -----------------------------
def detect_porhyrin(img):
    if img is None:
        return None, None, 0, 0, "None", {}

    output = img.copy()

    # 얼굴 마스크 생성
    face_mask = create_face_mask(img)

    # 그레이 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 대비 강화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 얼굴 영역 바깥 제거
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=face_mask)

    # 블러 처리
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 얼굴 영역 안의 픽셀만 기준으로 threshold 계산
    face_pixels = blur[face_mask == 255]

    if len(face_pixels) == 0:
        threshold_value = np.percentile(blur, 97)
    else:
        threshold_value = np.percentile(face_pixels, 97)

    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)

    # 얼굴 영역 안에서만 검출
    thresh = cv2.bitwise_and(thresh, thresh, mask=face_mask)

    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 컨투어 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(thresh)

    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 10:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        if 3 < radius < 15:
            center = (int(x), int(y))

            cv2.circle(output, center, int(radius), (0, 0, 255), 2)
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

            count += 1

    # 얼굴 바깥 검출 제거
    clean_mask = cv2.bitwise_and(clean_mask, clean_mask, mask=face_mask)

    # 얼굴 영역 기준 검출률 계산
    total_face_pixels = np.count_nonzero(face_mask)
    detected_pixels = np.count_nonzero(clean_mask)

    if total_face_pixels == 0:
        detection_rate = 0
    else:
        detection_rate = (detected_pixels / total_face_pixels) * 100

    # 등급 계산
    grade = get_grade(detection_rate)

    # 부위별 분석
    region_result = analyze_regions(clean_mask, face_mask)

    # 히트맵 생성
    heatmap = create_heatmap(img, clean_mask, face_mask)

    # 얼굴 실루엣 표시
    face_contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, face_contours, -1, (0, 255, 0), 2)

    return output, heatmap, count, detection_rate, grade, region_result


# -----------------------------
# 이미지 리스트 가져오기
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
            full_path = os.path.join(folder_path, f)
            image_list.append((date, f, full_path))

    return image_list


# -----------------------------
# 텍스트 출력 함수
# -----------------------------
def put_info_text(img, name, idx, total, date, count, rate, grade, region_result):
    cv2.putText(img, name, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(img, f"{idx+1} / {total}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(img, f"Date: {date}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(img, f"Detected Count: {count}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv2.putText(img, f"Detection Rate: {rate:.2f}%", (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv2.putText(img, f"Grade: {grade}", (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    y = 250
    cv2.putText(img, "Region Analysis", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    y += 30

    for region, value in region_result.items():
        cv2.putText(img, f"{region}: {value:.2f}%", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += 25

    return img


# -----------------------------
# 뷰어
# -----------------------------
def viewer(base_path):
    images = load_images(base_path)

    if len(images) == 0:
        print("이미지 없음")
        return

    idx = 0
    total = len(images)

    cv2.namedWindow("Porphyrin Viewer", cv2.WINDOW_NORMAL)

    while True:
        date, name, path = images[idx]

        img = cv2.imread(path)

        if img is None:
            idx = (idx + 1) % total
            continue

        result, heatmap, count, rate, grade, region_result = detect_porhyrin(img)

        # 크기 통일
        view_w = 500
        view_h = 500

        img_view = cv2.resize(img, (view_w, view_h))
        result_view = cv2.resize(result, (view_w, view_h))
        heatmap_view = cv2.resize(heatmap, (view_w, view_h))

        result_view = put_info_text(
            result_view,
            name,
            idx,
            total,
            date,
            count,
            rate,
            grade,
            region_result
        )

        cv2.putText(img_view, "Original", (20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(result_view, "Detection Result", (20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(heatmap_view, "Heatmap", (20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        combined = np.hstack((img_view, result_view, heatmap_view))

        cv2.imshow("Porphyrin Viewer", combined)

        key = cv2.waitKey(0) & 0xFF

        # 다음
        if key == ord('d'):
            idx = (idx + 1) % total

        # 이전
        elif key == ord('a'):
            idx = (idx - 1) % total

        # 종료
        elif key == 27:
            break

    cv2.destroyAllWindows()


# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    base_path = "../../captures/cam4_660nm"
    viewer(base_path)