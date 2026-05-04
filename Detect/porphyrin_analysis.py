import cv2
import numpy as np

def analyze_porhyrin(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("이미지 로드 실패")
        return

    # -----------------------------
    # 전처리
    # -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # -----------------------------
    # 포르피린 (밝은 점) 추출
    # -----------------------------
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    # 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # -----------------------------
    # 전체 비율 계산
    # -----------------------------
    total_pixels = thresh.size
    bright_pixels = np.sum(thresh == 255)

    ratio = bright_pixels / total_pixels

    # -----------------------------
    # 점수화 (0~100)
    # -----------------------------
    score = int((1 - ratio) * 100)

    # -----------------------------
    # 영역 분할
    # -----------------------------
    h, w = thresh.shape

    regions = {
        "이마": thresh[0:h//3, :],
        "코": thresh[h//3:2*h//3, w//3:2*w//3],
        "왼쪽볼": thresh[h//3:2*h//3, 0:w//3],
        "오른쪽볼": thresh[h//3:2*h//3, 2*w//3:w],
        "턱": thresh[2*h//3:h, :]
    }

    region_result = {}

    for name, region in regions.items():
        r_total = region.size
        r_bright = np.sum(region == 255)
        r_ratio = r_bright / r_total

        if r_ratio > 0.05:
            level = "높음"
        elif r_ratio > 0.02:
            level = "보통"
        else:
            level = "낮음"

        region_result[name] = {
            "ratio": round(r_ratio, 4),
            "level": level
        }

    # -----------------------------
    # 결과 출력
    # -----------------------------
    print("\n===== 피부 분석 결과 =====")
    print(f"전체 포르피린 비율: {round(ratio,4)}")
    print(f"피부 점수: {score} / 100")

    print("\n[부위별 상태]")
    for k, v in region_result.items():
        print(f"{k}: {v['level']} ({v['ratio']})")

    # -----------------------------
    # 시각화 이미지 생성
    # -----------------------------
    result_img = img.copy()
    result_img[thresh == 255] = [0,0,255]  # 빨간색 표시

    cv2.imwrite("porphyrin_result.jpg", result_img)
    print("\n결과 이미지 저장: porphyrin_result.jpg")

    return {
        "score": score,
        "total_ratio": ratio,
        "regions": region_result
    }


# 테스트 실행
if __name__ == "__main__":
    analyze_porhyrin("capture.jpg")