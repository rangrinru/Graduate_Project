import cv2
import numpy as np
import os

# -----------------------------
# 포르피린 검출 (percentile 기반)
# -----------------------------
def detect_porhyrin(img):
    if img is None:
        return None, 0

    output = img.copy()

    # -----------------------------
    # 그레이 + 대비 강화
    # -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (5,5), 0)

    # -----------------------------
    # 상위 밝기 추출 (핵심)
    # -----------------------------
    threshold_value = np.percentile(blur, 97)

    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)

    # -----------------------------
    # 노이즈 제거
    # -----------------------------
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # -----------------------------
    # 컨투어 검출
    # -----------------------------
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 10:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        if 3 < radius < 15:
            center = (int(x), int(y))
            cv2.circle(output, center, int(radius), (0, 0, 255), 2)
            count += 1

    return output, count


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

        files = sorted([f for f in os.listdir(folder_path)
                        if f.lower().endswith(('.png', '.jpg'))])

        for f in files:
            full_path = os.path.join(folder_path, f)
            image_list.append((date, f, full_path))

    return image_list


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

        result, count = detect_porhyrin(img)

        # 좌우 비교
        combined = np.hstack((img, result))

        # 크기 줄이기 (너무 클 경우)
        combined = cv2.resize(combined, (1400, 700))

        # -----------------------------
        # 텍스트 표시
        # -----------------------------
        cv2.putText(combined, name, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(combined, f"{idx+1} / {total}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(combined, f"Date: {date}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(combined, f"Detected: {count}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

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
    base_path = "../captures/cam4_660nm"
    viewer(base_path)