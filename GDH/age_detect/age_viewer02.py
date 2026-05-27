import cv2
import numpy as np
import json
from pathlib import Path


# -----------------------------
# 경로
# -----------------------------
PROJECT_ROOT = Path(r"C:\Users\고대현\Desktop\Graduate_Project")
BASE_DIR = PROJECT_ROOT / "GDH" / "age_detect" / "info"


# -----------------------------
# 한글 경로 이미지 읽기
# -----------------------------
def imread_korean(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# -----------------------------
# 메타데이터 로드
# -----------------------------
def load_data():
    json_files = sorted(
        BASE_DIR.glob("*_uv_aging_metadata.json")
    )

    items = []

    for meta_path in json_files:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        input_images = data.get("input_images", {})

        # 새 구조: input_images["no_filter"]
        # 예전 구조도 혹시 남아 있으면 대응
        original_path = (
            input_images.get("no_filter")
            or input_images.get("cam2_no_filter")
        )

        if original_path is None:
            print("원본 이미지 경로 없음:", meta_path)
            continue

        items.append({
            "image": Path(original_path),
            "meta": data,
            "meta_path": meta_path
        })

    return items


# -----------------------------
# 긴 텍스트 줄바꿈
# -----------------------------
def draw_wrapped_text(
    img,
    text,
    x,
    y,
    max_chars,
    font_scale,
    color,
    thickness,
    line_gap
):
    words = text.split(" ")
    line = ""

    for word in words:
        test_line = line + word + " "

        if len(test_line) > max_chars:
            cv2.putText(
                img,
                line.strip(),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness
            )
            y += line_gap
            line = word + " "
        else:
            line = test_line

    if line.strip():
        cv2.putText(
            img,
            line.strip(),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness
        )
        y += line_gap

    return y


# -----------------------------
# 화면 생성
# -----------------------------
def create_screen(img, meta, index, total):
    result = meta["analysis_result"]
    solution = meta["solution"]
    coords = result["coordinates"]

    display = img.copy()

    # 검출 영역 표시
    for c in coords:
        x = int(c["x"])
        y = int(c["y"])
        w = int(c["width"])
        h = int(c["height"])

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        center_x = int(c.get("center_x", x + w / 2))
        center_y = int(c.get("center_y", y + h / 2))

        cv2.circle(
            display,
            (center_x, center_y),
            5,
            (0, 255, 255),
            -1
        )

    h, w = display.shape[:2]

    # 오른쪽 정보 패널
    panel_width = 700
    panel = np.zeros(
        (h, panel_width, 3),
        dtype=np.uint8
    )

    y = 50

    title_texts = [
        "UV Aging Viewer",
        "",
        f"Image : {index + 1}/{total}",
        f"Date : {meta.get('date_folder', 'Unknown')}",
        f"Capture : {meta.get('capture_key', 'Unknown')}",
        "",
        f"Detected Count : {result['aging_region_count']}",
        f"Detection Rate : {result['detection_rate_percent']:.2f}%",
        f"Grade : {result['grade']}",
        f"Coordinate Count : {len(coords)}",
        "",
        "Solution:"
    ]

    for text in title_texts:
        cv2.putText(
            panel,
            text,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )
        y += 42

    y = draw_wrapped_text(
        panel,
        solution["summary"],
        25,
        y,
        max_chars=36,
        font_scale=0.62,
        color=(255, 255, 255),
        thickness=2,
        line_gap=34
    )

    y += 20

    cv2.putText(
        panel,
        "Recommendation:",
        (25, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    y += 38

    for rec in solution["recommendation"]:
        y = draw_wrapped_text(
            panel,
            "- " + rec,
            25,
            y,
            max_chars=38,
            font_scale=0.58,
            color=(255, 255, 255),
            thickness=2,
            line_gap=32
        )

    cv2.putText(
        panel,
        "[A] Prev",
        (25, h - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        panel,
        "[D] Next",
        (210, h - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        panel,
        "[ESC] Exit",
        (410, h - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    final = np.hstack((display, panel))

    return final


# -----------------------------
# 실행
# -----------------------------
def main():
    items = load_data()

    if len(items) == 0:
        print("저장된 결과 없음:", BASE_DIR)
        return

    index = 0

    cv2.namedWindow(
        "UV Aging Viewer",
        cv2.WINDOW_NORMAL
    )

    while True:
        current = items[index]

        img = imread_korean(current["image"])

        if img is None:
            print("이미지 읽기 실패:", current["image"])
            index = (index + 1) % len(items)
            continue

        screen = create_screen(
            img,
            current["meta"],
            index,
            len(items)
        )

        screen = cv2.resize(
            screen,
            (2100, 1200)
        )

        cv2.imshow(
            "UV Aging Viewer",
            screen
        )

        key = cv2.waitKey(0) & 0xFF

        if key == ord("d"):
            index = (index + 1) % len(items)

        elif key == ord("a"):
            index = (index - 1) % len(items)

        elif key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()