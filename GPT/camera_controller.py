from __future__ import annotations

"""
camera_controller.py
카메라/자동촬영/릴레이/백색 LED 제어 담당 스켈레톤
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 실제 장비에서 사용할 때 주석 해제
# from gpiozero import LED
# from picamera2 import Picamera2


@dataclass
class CaptureResult:
    success: bool
    session_id: str = ""
    capture_dir: str = ""
    cam2_path: str = ""
    cam3_path: str = ""
    cam4_path: str = ""
    metadata_path: str = ""
    reason: str = ""


@dataclass
class DetectionState:
    face_ok: bool = False
    eyes_closed: bool = False
    center_ok: bool = False
    size_ok: bool = False
    yaw_ok: bool = False
    pitch_ok: bool = False
    roll_ok: bool = False
    status_text: str = "대기 중"


@dataclass
class ControllerConfig:
    save_root: str = str(Path.home() / "Graduate_Project" / "captures")
    relay_pin: int = 17
    white_led_pin: int = 22
    relay_active_high: bool = False
    white_led_active_high: bool = True
    exposure_ms: int = 8
    gain: float = 1.0
    eyes_closed_hold_frames: int = 8
    capture_cooldown_sec: float = 2.0
    center_tolerance_ratio: float = 0.18
    min_face_ratio: float = 0.18
    max_face_ratio: float = 0.72

def find_haarcascade_file(filename: str) -> str:
    candidates = []

    # OpenCV에 cv2.data가 있는 경우
    if hasattr(cv2, "data"):
        candidates.append(Path(cv2.data.haarcascades) / filename)

    # 라즈베리파이/리눅스에서 자주 쓰이는 경로들
    candidates.extend([
        Path("/usr/share/opencv4/haarcascades") / filename,
        Path("/usr/share/opencv/haarcascades") / filename,
        Path("/usr/local/share/opencv4/haarcascades") / filename,
        Path("/usr/local/share/opencv/haarcascades") / filename,
    ])

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"Haar cascade 파일을 찾을 수 없습니다: {filename}\\n"
        f"확인한 경로: {[str(p) for p in candidates]}"
    )

class AutoCaptureController:
    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()

        self._lock = threading.Lock()
        self._running = False
        self._armed = False
        self._cooldown_until = 0.0
        self._eyes_closed_counter = 0

        self._latest_frame = None
        self._display_frame = None
        self._last_capture_result: Optional[CaptureResult] = None
        self._last_detection = DetectionState()
        self._worker = None

        self.picam = None
        self.relay = None
        self.white_led = None

        # 스켈레톤: Haar Cascade 사용
        face_cascade_path = find_haarcascade_file("haarcascade_frontalface_default.xml")
        eye_cascade_path = find_haarcascade_file("haarcascade_eye.xml")

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError(f"얼굴 Cascade 로드 실패: {face_cascade_path}")

        if self.eye_cascade.empty():
            raise RuntimeError(f"눈 Cascade 로드 실패: {eye_cascade_path}")

    # -------------------------
    # GUI가 호출하는 공개 메서드
    # -------------------------
    def start(self) -> None:
        if self._running:
            return
        self._setup_hardware()
        self._running = True
        self._worker = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._white_led_off()
        self._relay_off()
        self._teardown_hardware()

    def request_auto_capture(self) -> None:
        with self._lock:
            self._armed = True
            self._eyes_closed_counter = 0
            self._last_detection.status_text = "자동 촬영 대기 중"
        self._white_led_on()

    def cancel_auto_capture(self) -> None:
        with self._lock:
            self._armed = False
            self._eyes_closed_counter = 0
            self._last_detection.status_text = "촬영 취소"
        self._white_led_off()

    def get_latest_frame(self):
        with self._lock:
            return None if self._display_frame is None else self._display_frame.copy()

    def get_status_text(self) -> str:
        with self._lock:
            return self._last_detection.status_text

    def get_detection_state(self) -> DetectionState:
        with self._lock:
            return DetectionState(**self._last_detection.__dict__)

    def get_last_capture_result(self) -> Optional[CaptureResult]:
        with self._lock:
            return self._last_capture_result

    # -------------------------
    # 실제 하드웨어 초기화 위치
    # -------------------------
    def _setup_hardware(self) -> None:
        # 예시:
        # self.relay = LED(self.config.relay_pin, active_high=self.config.relay_active_high, initial_value=False)
        # self.white_led = LED(self.config.white_led_pin, active_high=self.config.white_led_active_high, initial_value=False)
        # self.picam = Picamera2(0)
        pass

    def _teardown_hardware(self) -> None:
        if hasattr(self, "_demo_cap"):
            self._demo_cap.release()

    # -------------------------
    # 메인 루프
    # -------------------------
    def _capture_loop(self) -> None:
        while self._running:
            frame = self._read_preview_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            detection = self._detect_face_and_eyes(frame)
            annotated = self._draw_overlay(frame, detection)

            trigger_capture = False
            now = time.monotonic()

            with self._lock:
                self._latest_frame = frame
                self._display_frame = annotated
                self._last_detection = detection

                if self._armed and now >= self._cooldown_until:
                    if self._conditions_satisfied(detection):
                        self._eyes_closed_counter += 1
                        self._last_detection.status_text = (
                            f"조건 만족 중... {self._eyes_closed_counter}/{self.config.eyes_closed_hold_frames}"
                        )
                    else:
                        self._eyes_closed_counter = 0

                    if self._eyes_closed_counter >= self.config.eyes_closed_hold_frames:
                        trigger_capture = True

            if trigger_capture:
                self._white_led_off()
                result = self._capture_sequence()
                with self._lock:
                    self._last_capture_result = result
                    self._armed = False
                    self._eyes_closed_counter = 0
                    self._cooldown_until = time.monotonic() + self.config.capture_cooldown_sec
                    self._last_detection.status_text = "촬영 완료" if result.success else f"촬영 실패: {result.reason}"

            time.sleep(0.03)

    # -------------------------
    # 프리뷰 프레임 획득
    # -------------------------
    def _read_preview_frame(self):
        # 실제 rpicam_03.py 구조에서는 Picamera2 프레임을 읽고 Cam2를 잘라서 반환하면 됨
        if not hasattr(self, "_demo_cap"):
            self._demo_cap = cv2.VideoCapture(0)

        ok, frame = self._demo_cap.read()
        if not ok:
            return None
        return frame

    # -------------------------
    # 얼굴/눈 감김 판정
    # -------------------------
    def _detect_face_and_eyes(self, frame) -> DetectionState:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        state = DetectionState(status_text="얼굴을 화면 중앙에 맞춰주세요")

        if len(faces) == 0:
            state.status_text = "얼굴이 감지되지 않았습니다"
            return state

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        state.face_ok = True

        fh, fw = frame.shape[:2]
        cx = x + w / 2
        cy = y + h / 2
        frame_cx = fw / 2
        frame_cy = fh / 2

        tol_x = fw * self.config.center_tolerance_ratio
        tol_y = fh * self.config.center_tolerance_ratio
        state.center_ok = abs(cx - frame_cx) <= tol_x and abs(cy - frame_cy) <= tol_y

        face_ratio = max(w / fw, h / fh)
        state.size_ok = self.config.min_face_ratio <= face_ratio <= self.config.max_face_ratio

        # 스켈레톤 단계: 각도 판정은 True
        state.yaw_ok = True
        state.pitch_ok = True
        state.roll_ok = True

        roi_gray = gray[y:y + h, x:x + w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=6)
        state.eyes_closed = len(eyes) == 0

        if not state.center_ok:
            state.status_text = "얼굴을 중앙으로 맞춰주세요"
        elif not state.size_ok:
            state.status_text = "얼굴 크기를 맞춰주세요"
        elif not state.eyes_closed:
            state.status_text = "눈을 감아주세요"
        else:
            state.status_text = "촬영 조건 만족"

        return state

    def _conditions_satisfied(self, det: DetectionState) -> bool:
        return (
            det.face_ok
            and det.center_ok
            and det.size_ok
            and det.yaw_ok
            and det.pitch_ok
            and det.roll_ok
            and det.eyes_closed
        )

    def _draw_overlay(self, frame, det: DetectionState):
        out = frame.copy()
        cv2.putText(out, det.status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        mode = "AUTO-ARMED" if self._armed else "IDLE"
        cv2.putText(out, mode, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out

    # -------------------------
    # 촬영 시퀀스
    # -------------------------
    def _capture_sequence(self) -> CaptureResult:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_dir = Path(self.config.save_root) / session_id
        capture_dir.mkdir(parents=True, exist_ok=True)

        self._relay_on()
        time.sleep(0.3)

        try:
            frame = self._read_preview_frame()
            if frame is None:
                return CaptureResult(success=False, reason="프레임 획득 실패")

            cam2_path = str(capture_dir / "cam2.png")
            cam3_path = str(capture_dir / "cam3.png")
            cam4_path = str(capture_dir / "cam4.png")
            metadata_path = str(capture_dir / "metadata.json")

            # 스켈레톤: 동일 프레임 임시 저장
            cv2.imwrite(cam2_path, frame)
            cv2.imwrite(cam3_path, frame)
            cv2.imwrite(cam4_path, frame)

            import json
            metadata = {
                "session_id": session_id,
                "capture_dir": str(capture_dir),
                "created_at": datetime.now().isoformat(),
                "note": "replace with real cam2/cam3/cam4 capture logic"
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

            return CaptureResult(
                success=True,
                session_id=session_id,
                capture_dir=str(capture_dir),
                cam2_path=cam2_path,
                cam3_path=cam3_path,
                cam4_path=cam4_path,
                metadata_path=metadata_path,
            )
        finally:
            self._relay_off()

    # -------------------------
    # GPIO 제어
    # -------------------------
    def _relay_on(self) -> None:
        # if self.relay: self.relay.on()
        pass

    def _relay_off(self) -> None:
        # if self.relay: self.relay.off()
        pass

    def _white_led_on(self) -> None:
        # if self.white_led: self.white_led.on()
        pass

    def _white_led_off(self) -> None:
        # if self.white_led: self.white_led.off()
        pass
