import cv2
import numpy as np
import json
from pathlib import Path


# -----------------------------
# 경로
# -----------------------------
BASE_DIR=Path(
r"C:\Users\고대현\Desktop\Graduate_Project\GDH\age_detect\info"
)


# -----------------------------
# 한글 경로 이미지 읽기
# -----------------------------
def imread_korean(path):

    data=np.fromfile(
        str(path),
        dtype=np.uint8
    )

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )


# -----------------------------
# 메타데이터 로드
# -----------------------------
def load_data():

    json_files=sorted(
        BASE_DIR.glob(
            "*_metadata.json"
        )
    )

    items=[]

    for meta in json_files:

        with open(
            meta,
            "r",
            encoding="utf-8"
        ) as f:

            data=json.load(f)

        original=Path(
            data[
            "input_images"
            ][
            "cam2_no_filter"
            ]
        )

        items.append({

            "image":original,
            "meta":data

        })

    return items


# -----------------------------
# 표시
# -----------------------------
def create_screen(
    img,
    meta,
    index,
    total
):

    result=meta[
        "analysis_result"
    ]

    solution=meta[
        "solution"
    ]

    coords=result[
        "coordinates"
    ]

    display=img.copy()

    # 검출영역 표시
    for c in coords:

        x=c["x"]
        y=c["y"]
        w=c["width"]
        h=c["height"]

        cv2.rectangle(
            display,
            (x,y),
            (x+w,y+h),
            (0,0,255),
            3
        )

        cv2.circle(
            display,
            (
            c["center_x"],
            c["center_y"]
            ),
            5,
            (0,255,255),
            -1
        )


    h,w=display.shape[:2]

    # 오른쪽 정보 패널
    panel=np.zeros(
        (
        h,
        600,
        3
        ),
        dtype=np.uint8
    )

    y=50

    texts=[

    f"Image : {index+1}/{total}",
    "",
    f"Detected Count : {result['aging_region_count']}",
    f"Detection Rate : {result['detection_rate_percent']:.2f}%",
    f"Grade : {result['grade']}",
    f"Coordinate Count : {len(coords)}",
    "",
    "Solution:",
    solution["summary"]

    ]

    for text in texts:

        cv2.putText(
            panel,
            text,
            (20,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        y+=45


    for rec in solution[
        "recommendation"
    ]:

        cv2.putText(
            panel,
            "- "+rec,
            (20,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        y+=40


    cv2.putText(
        panel,
        "[A] Prev",
        (20,h-80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        panel,
        "[D] Next",
        (180,h-80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        panel,
        "[ESC] Exit",
        (350,h-80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    final=np.hstack(
        (
        display,
        panel
        )
    )

    return final


# -----------------------------
# 실행
# -----------------------------
def main():

    items=load_data()

    if len(items)==0:

        print(
        "저장된 결과 없음"
        )

        return


    index=0

    cv2.namedWindow(
        "UV Aging Viewer",
        cv2.WINDOW_NORMAL
    )


    while True:

        current=items[
            index
        ]

        img=imread_korean(
            current[
            "image"
            ]
        )

        if img is None:

            print(
            "이미지 읽기 실패"
            )

            break


        screen=create_screen(
            img,
            current[
            "meta"
            ],
            index,
            len(items)
        )


        # 기존보다 3배 이상 크게
        screen=cv2.resize(
            screen,
            (
            2100,
            1200
            )
        )


        cv2.imshow(
            "UV Aging Viewer",
            screen
        )


        key=cv2.waitKey(
            0
        )&0xFF


        if key==ord(
        'd'
        ):

            index=(
            index+1
            )%len(items)


        elif key==ord(
        'a'
        ):

            index=(
            index-1
            )%len(items)


        elif key==27:

            break


    cv2.destroyAllWindows()


if __name__=="__main__":

    main()