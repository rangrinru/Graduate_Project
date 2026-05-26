import cv2
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime


# -----------------------------
# 포르피린 검출
# -----------------------------
def detect_porhyrin(img):

    if img is None:
        return None, 0

    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(
        enhanced,
        (5,5),
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

    kernel = np.ones((3,3), np.uint8)

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

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 10:
            continue

        (x,y), radius = cv2.minEnclosingCircle(cnt)

        if 3 < radius < 15:

            center = (int(x), int(y))

            cv2.circle(
                output,
                center,
                int(radius),
                (0,0,255),
                2
            )

            count += 1

    return output, count


# -----------------------------
# 메타데이터 저장
# -----------------------------
def save_metadata(date, name, path, count):

    image_path = Path(path)

    metadata = {

        "analyzed_at":
        datetime.now().isoformat(),

        "date_folder":
        date,

        "image_name":
        name,

        "image_path":
        str(image_path),

        "porphyrin_count":
        count
    }

    meta_path = image_path.parent / (
        image_path.stem
        + "_metadata.json"
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(
        "메타데이터 저장:",
        meta_path
    )


# -----------------------------
# 이미지 리스트
# -----------------------------
def load_images(base_path):

    image_list=[]

    date_folders=sorted(
        os.listdir(base_path)
    )

    for date in date_folders:

        folder_path=os.path.join(
            base_path,
            date
        )

        if not os.path.isdir(
            folder_path
        ):
            continue

        files=sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(
                ('.png','.jpg')
            )
        ])

        for f in files:

            if "metadata" in f:
                continue

            full_path=os.path.join(
                folder_path,
                f
            )

            image_list.append(
                (
                    date,
                    f,
                    full_path
                )
            )

    return image_list


# -----------------------------
# Viewer
# -----------------------------
def viewer(base_path):

    images=load_images(
        base_path
    )

    if len(images)==0:

        print(
            "이미지 없음"
        )

        return

    idx=0
    total=len(images)

    cv2.namedWindow(
        "Porphyrin Viewer",
        cv2.WINDOW_NORMAL
    )

    while True:

        date,name,path=images[idx]

        img=cv2.imread(path)

        if img is None:

            idx=(idx+1)%total
            continue

        result,count=detect_porhyrin(
            img
        )

        save_metadata(
            date,
            name,
            path,
            count
        )

        combined=np.hstack(
            (
                img,
                result
            )
        )

        combined=cv2.resize(
            combined,
            (
                1400,
                700
            )
        )

        cv2.putText(
            combined,
            name,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            combined,
            f"{idx+1}/{total}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            combined,
            f"Date:{date}",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

        cv2.putText(
            combined,
            f"Detected:{count}",
            (20,160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,0,255),
            3
        )

        cv2.imshow(
            "Porphyrin Viewer",
            combined
        )

        key=cv2.waitKey(0)&0xFF

        if key==ord('d'):
            idx=(idx+1)%total

        elif key==ord('a'):
            idx=(idx-1)%total

        elif key==27:
            break

    cv2.destroyAllWindows()


# -----------------------------
# 실행
# -----------------------------
if __name__=="__main__":

    base_path="../captures/cam4_660nm"

    viewer(base_path)