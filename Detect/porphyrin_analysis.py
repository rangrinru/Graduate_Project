import cv2
import numpy as np
import os

# -----------------------------
# 최신 이미지 찾기
# -----------------------------
def get_latest_image(base_path):
    folders = [f for f in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, f))]

    latest_folder = sorted(folders)[-1]
    latest_path = os.path.join(base_path, latest_folder)

    files = [f for f in os.listdir(latest_path)
             if f.lower().endswith(('.jpg', '.png'))]

    latest_file = sorted(files)[-1]
    image_path = os.path.join(latest_path, latest_file)

    print("사용 이미지:", image_path)
    return image_path


# -----------------------------
# 포르피린 검출 (강한 형광만)
# -----------------------------
def detect_porhyrin(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("이미지 로드 실패")
        return

    output = img.copy()

    # -----------------------------
    # 그레이 + 대비 강화
    # -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (5,5), 0)

    # -----------------------------
    # 상위 밝기만 추출 (핵심)
    # -----------------------------
    threshold_value = np.percentile(blur, 97)   # 👈 강한 것만

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

    print("검출 개수:", count)

    # -----------------------------
    # 결과 저장
    # -----------------------------
    cv2.imwrite("porphyrin_detect_result.jpg", output)

    # -----------------------------
    # 비교 화면 생성
    # -----------------------------
    h, w = img.shape[:2]
    output_resized = cv2.resize(output, (w, h))

    combined = np.hstack((img, output_resized))

    cv2.putText(combined, "Original", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(combined, "Detection", (w + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imwrite("compare_result.jpg", combined)

    # -----------------------------
    # 화면 출력
    # -----------------------------
    cv2.imshow("Porphyrin Compare", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    base_path = "../captures/cam4_660nm"

    img_path = get_latest_image(base_path)
    detect_porhyrin(img_path)