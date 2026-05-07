from __future__ import annotations

import cv2
import numpy as np

CAM_INDEX = 0
WINDOW_NAME = "Face ROI + Eye State + Eye Motion (OpenCV)"
ROI_EXPAND_X = 0.12
ROI_EXPAND_Y = 0.18
USE_ELLIPSE_MASK = False

# 얼굴 / 눈 cascade
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (80, 80)

EYE_SCALE_FACTOR = 1.1
EYE_MIN_NEIGHBORS = 8
EYE_MIN_SIZE = (20, 20)

# 눈 감김 판정
# eye ROI 높이가 너무 작거나 동공 검출 실패가 반복되면 closed 쪽으로 판단
MIN_EYE_OPEN_HEIGHT = 12
DARK_PIXEL_RATIO_TH = 0.015

# 깜빡임 카운트
MIN_CLOSED_FRAMES_FOR_BLINK = 2

# 시선 방향 임계값
GAZE_LEFT_TH = 0.38
GAZE_RIGHT_TH = 0.62
GAZE_UP_TH = 0.38
GAZE_DOWN_TH = 0.62


STATE = {
    "blink_count": 0,
    "closed_frame_count": 0,
    "prev_eye_closed": False,
    "eye_state": "Unknown",
    "gaze_direction": "Unknown",
}


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def expand_bbox(x: int, y: int, w_box: int, h_box: int, frame_w: int, frame_h: int):
    ex = int(w_box * ROI_EXPAND_X)
    ey = int(h_box * ROI_EXPAND_Y)

    x1 = clamp(x - ex, 0, frame_w - 1)
    y1 = clamp(y - ey, 0, frame_h - 1)
    x2 = clamp(x + w_box + ex, 0, frame_w - 1)
    y2 = clamp(y + h_box + ey, 0, frame_h - 1)

    return x1, y1, x2, y2


def make_face_roi_mask(frame: np.ndarray, bbox):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if bbox is None:
        return mask

    x1, y1, x2, y2 = bbox

    if USE_ELLIPSE_MASK:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ax = max((x2 - x1) // 2, 1)
        ay = max((y2 - y1) // 2, 1)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    else:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


def build_cascades():
    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"

    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)

    if face_cascade.empty():
        raise RuntimeError(f"얼굴 cascade 로드 실패: {face_path}")
    if eye_cascade.empty():
        raise RuntimeError(f"눈 cascade 로드 실패: {eye_path}")

    return face_cascade, eye_cascade


def detect_face_bbox(frame: np.ndarray, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )

    if len(faces) == 0:
        return None

    x, y, w_box, h_box = max(faces, key=lambda f: f[2] * f[3])

    frame_h, frame_w = frame.shape[:2]
    return expand_bbox(x, y, w_box, h_box, frame_w, frame_h)


def select_two_eyes(face_gray: np.ndarray, eye_cascade):
    """
    얼굴 ROI 안에서 눈 후보 2개 선택
    - 얼굴 상단 절반 정도만 탐색
    - 가장 그럴듯한 2개를 x축 기준으로 정렬
    """
    h, w = face_gray.shape[:2]
    upper_face = face_gray[: max(h // 2 + h // 8, 1), :]

    eyes = eye_cascade.detectMultiScale(
        upper_face,
        scaleFactor=EYE_SCALE_FACTOR,
        minNeighbors=EYE_MIN_NEIGHBORS,
        minSize=EYE_MIN_SIZE
    )

    if len(eyes) == 0:
        return []

    # 너무 아래 있거나 너무 큰 박스 제거
    candidates = []
    for (x, y, ew, eh) in eyes:
        if y > h * 0.55:
            continue
        area = ew * eh
        candidates.append((x, y, ew, eh, area))

    if not candidates:
        return []

    # x축으로 많이 겹치는 중복 줄이기
    candidates.sort(key=lambda e: e[4], reverse=True)
    filtered = []
    for cand in candidates:
        x, y, ew, eh, _ = cand
        keep = True
        for fx, fy, few, feh, _ in filtered:
            if abs(x - fx) < min(ew, few) * 0.5 and abs(y - fy) < min(eh, feh) * 0.5:
                keep = False
                break
        if keep:
            filtered.append(cand)

    if len(filtered) >= 2:
        filtered = sorted(filtered[:2], key=lambda e: e[0])
    else:
        filtered = sorted(filtered, key=lambda e: e[0])

    return [(x, y, ew, eh) for (x, y, ew, eh, _) in filtered]


def detect_pupil_center(eye_bgr: np.ndarray):
    """
    OpenCV만 사용한 간단한 동공 중심 추정.
    정확한 시선추적이 아니라 대략적인 좌/우/상/하 판정용.
    """
    if eye_bgr is None or eye_bgr.size == 0:
        return None, None

    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return None, None

    # 어두운 영역을 동공 후보로 가정
    thresh_val = max(10, int(np.percentile(gray, 20)))
    dark_mask = (gray <= thresh_val).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    dark_ratio = float(np.count_nonzero(dark_mask)) / float(h * w)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, dark_ratio

    # 중앙에 가깝고 면적이 너무 작지 않은 contour 선택
    best = None
    best_score = -1e9
    cx_ref, cy_ref = w / 2.0, h / 2.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist = ((cx - cx_ref) ** 2 + (cy - cy_ref) ** 2) ** 0.5

        score = area - 0.8 * dist
        if score > best_score:
            best_score = score
            best = (int(cx), int(cy), cnt)

    if best is None:
        return None, dark_ratio

    return (best[0], best[1]), dark_ratio


def estimate_eye_state_and_motion(frame: np.ndarray, face_bbox, eye_cascade):
    if face_bbox is None:
        STATE["eye_state"] = "Unknown"
        STATE["gaze_direction"] = "Unknown"
        STATE["closed_frame_count"] = 0
        STATE["prev_eye_closed"] = False
        return None

    x1, y1, x2, y2 = face_bbox
    face_bgr = frame[y1:y2, x1:x2]
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    eyes = select_two_eyes(face_gray, eye_cascade)

    if len(eyes) < 2:
        # 눈을 못 찾았다고 무조건 감김은 아니지만, 보수적으로 Unknown 처리
        STATE["eye_state"] = "Unknown"
        STATE["gaze_direction"] = "Unknown"
        return {
            "eye_state": "Unknown",
            "gaze_direction": "Unknown",
            "blink_count": STATE["blink_count"],
            "eyes": [],
            "face_bbox": face_bbox,
        }

    eye_infos = []
    open_flags = []

    for (ex, ey, ew, eh) in eyes:
        eye_roi = face_bgr[ey:ey + eh, ex:ex + ew]
        pupil_center, dark_ratio = detect_pupil_center(eye_roi)

        eye_open = True
        if eh < MIN_EYE_OPEN_HEIGHT:
            eye_open = False
        if pupil_center is None and dark_ratio is not None and dark_ratio < DARK_PIXEL_RATIO_TH:
            eye_open = False

        ratio_x = None
        ratio_y = None
        pupil_global = None

        if pupil_center is not None:
            px, py = pupil_center
            ratio_x = px / max(ew, 1)
            ratio_y = py / max(eh, 1)
            pupil_global = (x1 + ex + px, y1 + ey + py)

        eye_infos.append({
            "bbox_face_local": (ex, ey, ew, eh),
            "bbox_global": (x1 + ex, y1 + ey, ew, eh),
            "pupil_center_local": pupil_center,
            "pupil_center_global": pupil_global,
            "ratio_x": ratio_x,
            "ratio_y": ratio_y,
            "open": eye_open,
            "dark_ratio": dark_ratio,
        })
        open_flags.append(eye_open)

    eyes_closed = not any(open_flags)
    eye_state = "Closed" if eyes_closed else "Open"

    # 깜빡임 카운트
    if eyes_closed:
        STATE["closed_frame_count"] += 1
    else:
        if STATE["prev_eye_closed"] and STATE["closed_frame_count"] >= MIN_CLOSED_FRAMES_FOR_BLINK:
            STATE["blink_count"] += 1
        STATE["closed_frame_count"] = 0

    STATE["prev_eye_closed"] = eyes_closed
    STATE["eye_state"] = eye_state

    # 시선 방향 추정
    valid_ratios_x = [e["ratio_x"] for e in eye_infos if e["ratio_x"] is not None]
    valid_ratios_y = [e["ratio_y"] for e in eye_infos if e["ratio_y"] is not None]

    if eyes_closed:
        gaze_direction = "Eyes Closed"
    elif not valid_ratios_x or not valid_ratios_y:
        gaze_direction = "Unknown"
    else:
        rx = float(np.mean(valid_ratios_x))
        ry = float(np.mean(valid_ratios_y))

        horizontal = "Center"
        vertical = "Center"

        if rx < GAZE_LEFT_TH:
            horizontal = "Left"
        elif rx > GAZE_RIGHT_TH:
            horizontal = "Right"

        if ry < GAZE_UP_TH:
            vertical = "Up"
        elif ry > GAZE_DOWN_TH:
            vertical = "Down"

        if horizontal == "Center" and vertical == "Center":
            gaze_direction = "Center"
        elif vertical == "Center":
            gaze_direction = horizontal
        elif horizontal == "Center":
            gaze_direction = vertical
        else:
            gaze_direction = f"{vertical}-{horizontal}"

    STATE["gaze_direction"] = gaze_direction

    return {
        "eye_state": eye_state,
        "gaze_direction": gaze_direction,
        "blink_count": STATE["blink_count"],
        "eyes": eye_infos,
        "face_bbox": face_bbox,
    }


def draw_eye_info(display: np.ndarray, eye_result):
    if eye_result is None:
        cv2.putText(
            display, "Eye State: Unknown", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        cv2.putText(
            display, "Gaze: Unknown", (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
        )
        return

    for eye in eye_result["eyes"]:
        gx, gy, gw, gh = eye["bbox_global"]
        color = (255, 255, 0) if eye["open"] else (0, 0, 255)
        cv2.rectangle(display, (gx, gy), (gx + gw, gy + gh), color, 2)

        if eye["pupil_center_global"] is not None:
            cv2.circle(display, eye["pupil_center_global"], 3, (0, 0, 255), -1)

    cv2.putText(
        display,
        f"Eye State: {eye_result['eye_state']}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Gaze: {eye_result['gaze_direction']}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Blink Count: {eye_result['blink_count']}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


def main():
    face_cascade, eye_cascade = build_cascades()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패")
                break

            bbox = detect_face_bbox(frame, face_cascade)
            display = frame.copy()

            face_mask = make_face_roi_mask(frame, bbox)
            masked = cv2.bitwise_and(frame, frame, mask=face_mask)

            if bbox is not None:
                x1, y1, x2, y2 = bbox

                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    "Face ROI",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    display,
                    "No Face Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

            eye_result = estimate_eye_state_and_motion(frame, bbox, eye_cascade)
            draw_eye_info(display, eye_result)

            combined = np.hstack((display, masked))
            combined = cv2.resize(combined, (1400, 600))

            cv2.putText(
                combined,
                "Original + Face Box + Eye State",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                combined,
                "Face ROI Masked",
                (720, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
