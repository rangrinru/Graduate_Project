
from __future__ import annotations

import cv2
import json
import math
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from time import sleep, monotonic
from collections import deque
from typing import Optional, Dict, Any

from gpiozero import LED
from picamera2 import Picamera2

try:
    import mediapipe as mp
except ImportError:
    mp = None


logger = logging.getLogger("capture_service")


@dataclass
class CapturePaths:
    session_dir: str
    capture_id: str
    cam2_path: str
    cam3_path: str
    cam4_path: str
    session_metadata_path: str


class CaptureService:
    CAPTURE_WIDTH = 5120
    CAPTURE_HEIGHT = 800
    SINGLE_WIDTH = CAPTURE_WIDTH // 4

    MIN_EXPOSURE_MS = 1
    MAX_EXPOSURE_MS = 100
    DEFAULT_EXPOSURE_MS = 8
    CURRENT_GAIN = 1.0
    SAVE_AS_PNG = True

    RELAY_WARMUP_SEC = 0.3
    WHITE_LED_OFF_BEFORE_CAPTURE_SEC = 0.15

    FACE_CENTER_TOL_X = 0.18
    FACE_CENTER_TOL_Y = 0.20
    FACE_MIN_AREA_RATIO = 0.08
    FACE_MAX_AREA_RATIO = 0.70

    MAX_ABS_ROLL_DEG = 8.0
    MAX_ABS_YAW_SCORE = 0.12
    MAX_ABS_PITCH_SCORE = 0.12

    DEFAULT_EYE_AR_THRESHOLD = 0.18
    MIN_DYNAMIC_EYE_AR_THRESHOLD = 0.14
    MAX_DYNAMIC_EYE_AR_THRESHOLD = 0.26
    EYE_THRESHOLD_RATIO = 0.72

    STABLE_FACE_HOLD_FRAMES = 6
    OPEN_EYE_BASELINE_SAMPLES = 15
    EYES_CLOSED_HOLD_FRAMES = 4
    CAPTURE_COOLDOWN_SEC = 1.5

    CAMERA_INFO = {
        "cam2": {"label": "CAM 2 - NO FILTER", "filter": "no_filter", "display_name": "No_Filter", "x_start": SINGLE_WIDTH * 1, "x_end": SINGLE_WIDTH * 2, "sequence_order": 1},
        "cam3": {"label": "CAM 3 - 405nm FILTER", "filter": "405nm_filter", "display_name": "405nm_Filter", "x_start": SINGLE_WIDTH * 2, "x_end": SINGLE_WIDTH * 3, "sequence_order": 2},
        "cam4": {"label": "CAM 4 - 660nm FILTER", "filter": "660nm_filter", "display_name": "660nm_Filter", "x_start": SINGLE_WIDTH * 3, "x_end": SINGLE_WIDTH * 4, "sequence_order": 3},
    }

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    NOSE_TIP_IDX = 1
    LEFT_EYE_OUTER_IDX = 33
    RIGHT_EYE_OUTER_IDX = 263
    LEFT_FACE_IDX = 234
    RIGHT_FACE_IDX = 454
    UPPER_FACE_IDX = 10
    LOWER_FACE_IDX = 152
    MOUTH_LEFT_IDX = 61
    MOUTH_RIGHT_IDX = 291

    def __init__(
        self,
        save_root: Path | str,
        relay_pin: int = 17,
        relay_active_high: bool = False,
        white_led_pin: int = 22,
        white_led_active_high: bool = True,
    ) -> None:
        self.save_root = Path(save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)

        self.camera_lock = threading.RLock()
        self.picam2: Optional[Picamera2] = None
        self.preview_config = None
        self.still_config = None
        self.camera_ready = False

        self.relay = LED(relay_pin, active_high=relay_active_high, initial_value=False)
        self.white_led = LED(white_led_pin, active_high=white_led_active_high, initial_value=False)

        self.exposure_ms = self.DEFAULT_EXPOSURE_MS
        self.gain = self.CURRENT_GAIN

        self.state: Dict[str, Any] = {
            "status": "대기 중",
            "armed": False,
            "white_led_is_on": False,
            "relay_is_on": False,
            "stable_face_count": 0,
            "eyes_closed_count": 0,
            "dynamic_eye_threshold": self.DEFAULT_EYE_AR_THRESHOLD,
            "last_capture_monotonic": 0.0,
            "last_detection": None,
        }
        self.open_eye_ear_history = deque(maxlen=self.OPEN_EYE_BASELINE_SAMPLES)

        if mp is not None:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.face_mesh = None

    # ---------- lifecycle ----------
    def init_camera(self) -> None:
        with self.camera_lock:
            if self.camera_ready:
                return

            camera_list = Picamera2.global_camera_info()
            logger.info("Detected cameras: %s", camera_list)
            if len(camera_list) == 0:
                raise RuntimeError("카메라가 감지되지 않았습니다.")

            self.picam2 = Picamera2(0)
            self.preview_config = self.picam2.create_preview_configuration(
                main={"size": (self.CAPTURE_WIDTH, self.CAPTURE_HEIGHT), "format": "XRGB8888"}
            )
            self.still_config = self.picam2.create_still_configuration(
                main={"size": (self.CAPTURE_WIDTH, self.CAPTURE_HEIGHT), "format": "RGB888"}
            )
            self.picam2.configure(self.preview_config)
            self.picam2.options["quality"] = 100
            self.picam2.options["compress_level"] = 0
            self.picam2.start()
            self.set_manual_controls(self.exposure_ms, self.gain)
            self.relay_off()
            self.white_led_off()
            self.camera_ready = True

    def close(self) -> None:
        with self.camera_lock:
            self.white_led_off()
            self.relay_off()
            if self.picam2 is not None:
                try:
                    self.picam2.stop()
                except Exception:
                    pass
                self.picam2 = None
            self.camera_ready = False
            if self.face_mesh is not None:
                try:
                    self.face_mesh.close()
                except Exception:
                    pass
                self.face_mesh = None

    # ---------- gpio ----------
    def relay_on(self) -> None:
        if not self.state["relay_is_on"]:
            self.relay.on()
            self.state["relay_is_on"] = True

    def relay_off(self) -> None:
        if self.state["relay_is_on"]:
            self.relay.off()
            self.state["relay_is_on"] = False

    def white_led_on(self) -> None:
        if not self.state["white_led_is_on"]:
            self.white_led.on()
            self.state["white_led_is_on"] = True

    def white_led_off(self) -> None:
        if self.state["white_led_is_on"]:
            self.white_led.off()
            self.state["white_led_is_on"] = False

    # ---------- camera helpers ----------
    def set_manual_controls(self, exposure_ms: int, gain: float) -> None:
        if self.picam2 is None:
            return
        exposure_ms = max(self.MIN_EXPOSURE_MS, min(self.MAX_EXPOSURE_MS, int(exposure_ms)))
        self.exposure_ms = exposure_ms
        self.gain = float(gain)
        self.picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": exposure_ms * 1000,
            "AnalogueGain": gain,
        })

    def safe_capture_array(self):
        if self.picam2 is None:
            return None
        try:
            return self.picam2.capture_array()
        except Exception as exc:
            logger.exception("프레임 획득 실패: %s", exc)
            return None

    def read_full_frame_bgr(self):
        frame = self.safe_capture_array()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def get_stream_jpeg(self, cam_key: str = "cam4", jpeg_quality: int = 85):
        with self.camera_lock:
            if not self.camera_ready:
                return None
            full_frame = self.read_full_frame_bgr()
            if full_frame is None:
                return None
            crop = self.extract_cam_frame(full_frame, cam_key).copy()
            ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                return None
            return buffer.tobytes()

    def extract_cam_frame(self, full_frame_bgr, cam_key: str):
        info = self.CAMERA_INFO[cam_key]
        return full_frame_bgr[:, info["x_start"]:info["x_end"]]

    # ---------- detection ----------
    def _euclidean(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _eye_aspect_ratio(self, eye_pts):
        a = self._euclidean(eye_pts[1], eye_pts[5])
        b = self._euclidean(eye_pts[2], eye_pts[4])
        c = self._euclidean(eye_pts[0], eye_pts[3])
        return 1.0 if c == 0 else (a + b) / (2.0 * c)

    def update_dynamic_eye_threshold(self, avg_ear: float) -> None:
        self.open_eye_ear_history.append(avg_ear)
        if len(self.open_eye_ear_history) >= max(8, self.OPEN_EYE_BASELINE_SAMPLES // 2):
            baseline = sum(self.open_eye_ear_history) / len(self.open_eye_ear_history)
            dynamic = baseline * self.EYE_THRESHOLD_RATIO
            dynamic = max(self.MIN_DYNAMIC_EYE_AR_THRESHOLD, min(self.MAX_DYNAMIC_EYE_AR_THRESHOLD, dynamic))
            self.state["dynamic_eye_threshold"] = dynamic

    def analyze_face(self, cam2_bgr):
        h, w = cam2_bgr.shape[:2]
        result = {
            "face_found": False,
            "center_ok": False,
            "size_ok": False,
            "eyes_closed": False,
            "roll_ok": False,
            "yaw_ok": False,
            "pitch_ok": False,
            "angles_ok": False,
            "ear_left": 1.0,
            "ear_right": 1.0,
            "avg_ear": 1.0,
            "roll_deg": 0.0,
            "yaw_score": 0.0,
            "pitch_score": 0.0,
            "bbox": None,
            "guide_text": "얼굴을 화면에 맞춰주세요",
        }

        if self.face_mesh is None:
            result["guide_text"] = "mediapipe 미설치: 자동촬영 사용 불가"
            return result

        rgb = cv2.cvtColor(cam2_bgr, cv2.COLOR_BGR2RGB)
        mesh_result = self.face_mesh.process(rgb)
        if not mesh_result.multi_face_landmarks:
            result["guide_text"] = "얼굴이 감지되지 않습니다"
            return result

        pts = [(int(lm.x * w), int(lm.y * h)) for lm in mesh_result.multi_face_landmarks[0].landmark]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
        y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        area_ratio = (bw * bh) / float(w * h)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        norm_dx = abs(cx - (w / 2.0)) / w
        norm_dy = abs(cy - (h / 2.0)) / h

        left_eye_pts = [pts[i] for i in self.LEFT_EYE]
        right_eye_pts = [pts[i] for i in self.RIGHT_EYE]
        ear_left = self._eye_aspect_ratio(left_eye_pts)
        ear_right = self._eye_aspect_ratio(right_eye_pts)
        avg_ear = (ear_left + ear_right) / 2.0

        center_ok = (norm_dx <= self.FACE_CENTER_TOL_X) and (norm_dy <= self.FACE_CENTER_TOL_Y)
        size_ok = (self.FACE_MIN_AREA_RATIO <= area_ratio <= self.FACE_MAX_AREA_RATIO)

        nose = pts[self.NOSE_TIP_IDX]
        le = pts[self.LEFT_EYE_OUTER_IDX]
        re = pts[self.RIGHT_EYE_OUTER_IDX]
        lf = pts[self.LEFT_FACE_IDX]
        rf = pts[self.RIGHT_FACE_IDX]
        upper = pts[self.UPPER_FACE_IDX]
        lower = pts[self.LOWER_FACE_IDX]
        ml = pts[self.MOUTH_LEFT_IDX]
        mr = pts[self.MOUTH_RIGHT_IDX]

        eye_mid = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
        mouth_mid = ((ml[0] + mr[0]) / 2.0, (ml[1] + mr[1]) / 2.0)
        face_mid_x = (lf[0] + rf[0]) / 2.0
        face_width = max(1.0, float(rf[0] - lf[0]))
        face_height = max(1.0, float(lower[1] - upper[1]))

        roll_deg = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
        yaw_score = abs(nose[0] - face_mid_x) / face_width
        pitch_score = abs(nose[1] - ((eye_mid[1] + mouth_mid[1]) / 2.0)) / face_height

        roll_ok = abs(roll_deg) <= self.MAX_ABS_ROLL_DEG
        yaw_ok = yaw_score <= self.MAX_ABS_YAW_SCORE
        pitch_ok = pitch_score <= self.MAX_ABS_PITCH_SCORE
        angles_ok = roll_ok and yaw_ok and pitch_ok

        threshold = self.state["dynamic_eye_threshold"]
        eyes_closed = (ear_left < threshold) and (ear_right < threshold)

        result.update({
            "face_found": True,
            "center_ok": center_ok,
            "size_ok": size_ok,
            "eyes_closed": eyes_closed,
            "roll_ok": roll_ok,
            "yaw_ok": yaw_ok,
            "pitch_ok": pitch_ok,
            "angles_ok": angles_ok,
            "ear_left": ear_left,
            "ear_right": ear_right,
            "avg_ear": avg_ear,
            "roll_deg": roll_deg,
            "yaw_score": yaw_score,
            "pitch_score": pitch_score,
            "bbox": (x1, y1, x2, y2),
        })

        if not center_ok:
            result["guide_text"] = "얼굴을 화면 중앙에 맞춰주세요"
        elif not size_ok:
            result["guide_text"] = "얼굴 거리를 조정해주세요"
        elif not angles_ok:
            result["guide_text"] = "얼굴 각도를 정면으로 맞춰주세요"
        elif not eyes_closed:
            result["guide_text"] = "눈을 감아주세요"
        else:
            result["guide_text"] = "조건 충족"

        self.state["last_detection"] = result
        return result

    def get_status(self) -> Dict[str, Any]:
        det = self.state.get("last_detection")
        return {
            "ok": True,
            "camera_ready": self.camera_ready,
            "status": self.state["status"],
            "armed": self.state["armed"],
            "analysis_in_progress": False,
            "white_led_on": self.state["white_led_is_on"],
            "relay_on": self.state["relay_is_on"],
            "dynamic_eye_threshold": self.state["dynamic_eye_threshold"],
            "stable_face_count": self.state["stable_face_count"],
            "eyes_closed_count": self.state["eyes_closed_count"],
            "last_detection": det,
        }

    def _handle_auto_state(self, detection: Dict[str, Any]) -> bool:
        now = monotonic()
        if now - self.state["last_capture_monotonic"] < self.CAPTURE_COOLDOWN_SEC:
            self.state["status"] = "촬영 후 안정화 대기 중"
            self.state["stable_face_count"] = 0
            self.state["eyes_closed_count"] = 0
            return False

        stable_face_ok = (
            detection["face_found"]
            and detection["center_ok"]
            and detection["size_ok"]
            and detection["angles_ok"]
        )

        if stable_face_ok:
            self.state["stable_face_count"] += 1
        else:
            self.state["stable_face_count"] = 0
            self.state["eyes_closed_count"] = 0
            self.state["status"] = detection["guide_text"]
            return False

        if self.state["stable_face_count"] >= self.STABLE_FACE_HOLD_FRAMES and not detection["eyes_closed"]:
            self.update_dynamic_eye_threshold(detection["avg_ear"])
            self.state["status"] = f"눈을 감아주세요 (기준 EAR 수집 {len(self.open_eye_ear_history)}/{self.OPEN_EYE_BASELINE_SAMPLES})"
            self.state["eyes_closed_count"] = 0
            return False

        if self.state["stable_face_count"] < self.STABLE_FACE_HOLD_FRAMES:
            self.state["status"] = f"얼굴 고정 중 {self.state['stable_face_count']}/{self.STABLE_FACE_HOLD_FRAMES}"
            self.state["eyes_closed_count"] = 0
            return False

        if detection["eyes_closed"]:
            self.state["eyes_closed_count"] += 1
            self.state["status"] = f"눈감음 확인 중 {self.state['eyes_closed_count']}/{self.EYES_CLOSED_HOLD_FRAMES}"
        else:
            self.state["eyes_closed_count"] = 0
            self.state["status"] = "얼굴을 유지한 채 눈을 감아주세요"

        return self.state["eyes_closed_count"] >= self.EYES_CLOSED_HOLD_FRAMES

    # ---------- saving ----------
    def save_image(self, path: Path, img_bgr):
        ok = False
        if path.suffix.lower() == ".png":
            ok = cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            ok = cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not ok:
            raise RuntimeError(f"이미지 저장 실패: {path}")

    def _capture_high_quality_full_frame(self, exposure_ms: int, gain: float):
        try:
            self.picam2.stop()
            self.picam2.configure(self.still_config)
            self.picam2.start()
            self.set_manual_controls(exposure_ms, gain)
            still_frame = self.safe_capture_array()
            if still_frame is None:
                raise RuntimeError("고화질 프레임 획득 실패")
            return cv2.cvtColor(still_frame, cv2.COLOR_RGB2BGR)
        finally:
            self.picam2.stop()
            self.picam2.configure(self.preview_config)
            self.picam2.start()
            self.set_manual_controls(exposure_ms, gain)

    def _make_session_dir(self, profile_root: Path):
        capture_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        session_dir = profile_root / capture_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return capture_id, session_dir

    def _save_one_camera_image(
        self,
        session_dir: Path,
        cam_key: str,
        frame_bgr,
        timestamp: datetime,
        exposure_ms: int,
        gain: float,
        ext: str,
        profile_name: str,
        folder_id: str,
    ):
        info = self.CAMERA_INFO[cam_key]
        image_path = session_dir / f"{cam_key}.{ext}"
        meta_path = session_dir / f"{cam_key}_metadata.json"
        self.save_image(image_path, frame_bgr)
        metadata = {
            "captured_at": timestamp.isoformat(),
            "profile_name": profile_name,
            "profile_folder_id": folder_id,
            "camera_name": cam_key,
            "camera_label": info["label"],
            "filter_type": info["filter"],
            "display_name": info["display_name"],
            "sequence_order": info["sequence_order"],
            "file_format": ext,
            "camera_control": {
                "AeEnable": False,
                "ExposureTime_ms": exposure_ms,
                "ExposureTime_us": exposure_ms * 1000,
                "AnalogueGain": gain,
            },
            "saved_file": str(image_path),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return str(image_path), str(meta_path)

    def _save_session_metadata(
        self,
        session_dir: Path,
        timestamp: datetime,
        exposure_ms: int,
        gain: float,
        profile_name: str,
        folder_id: str,
        detection_snapshot: Optional[Dict[str, Any]],
        ext: str,
    ) -> Path:
        session_meta_path = session_dir / "session_metadata.json"
        data = {
            "captured_at": timestamp.isoformat(),
            "profile_name": profile_name,
            "profile_folder_id": folder_id,
            "exposure_ms": exposure_ms,
            "gain": gain,
            "dynamic_eye_threshold": self.state["dynamic_eye_threshold"],
            "open_eye_baseline_samples": list(self.open_eye_ear_history),
            "trigger_detection": detection_snapshot,
            "files": {
                key: str(session_dir / f"{key}.{ext}") for key in ["cam2", "cam3", "cam4"]
            },
        }
        with open(session_meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return session_meta_path

    def _capture_and_save_session(
        self,
        profile_root: Path,
        profile_name: str,
        folder_id: str,
        exposure_ms: int,
        gain: float,
        detection_snapshot: Optional[Dict[str, Any]],
    ) -> CapturePaths:
        ext = "png" if self.SAVE_AS_PNG else "jpg"
        self.relay_on()
        sleep(self.RELAY_WARMUP_SEC)
        capture_id, session_dir = self._make_session_dir(profile_root)
        try:
            full_frame_bgr = self._capture_high_quality_full_frame(exposure_ms, gain)
            timestamp = datetime.now()
            saved = {}
            for cam_key in ["cam2", "cam3", "cam4"]:
                target_frame = self.extract_cam_frame(full_frame_bgr, cam_key).copy()
                image_path, _ = self._save_one_camera_image(
                    session_dir, cam_key, target_frame, timestamp, exposure_ms, gain, ext, profile_name, folder_id
                )
                saved[cam_key] = image_path
            session_meta = self._save_session_metadata(
                session_dir, timestamp, exposure_ms, gain, profile_name, folder_id, detection_snapshot, ext
            )
            self.state["last_capture_monotonic"] = monotonic()
            self.state["status"] = "촬영 완료"
            return CapturePaths(
                session_dir=str(session_dir),
                capture_id=capture_id,
                cam2_path=saved["cam2"],
                cam3_path=saved["cam3"],
                cam4_path=saved["cam4"],
                session_metadata_path=str(session_meta),
            )
        finally:
            self.relay_off()

    # ---------- public capture ----------
    def manual_capture(self, profile_root: Path, profile_name: str, folder_id: str, exposure_ms: Optional[int] = None):
        with self.camera_lock:
            if not self.camera_ready:
                raise RuntimeError("카메라가 준비되지 않았습니다.")
            exposure_ms = exposure_ms or self.exposure_ms
            full_frame = self.read_full_frame_bgr()
            detection_snapshot = None
            if full_frame is not None:
                cam2 = self.extract_cam_frame(full_frame, "cam2").copy()
                det = self.analyze_face(cam2)
                if det is not None:
                    detection_snapshot = {
                        "bbox": list(det["bbox"]) if det["bbox"] is not None else None,
                        "roll_deg": det["roll_deg"],
                        "yaw_score": det["yaw_score"],
                        "pitch_score": det["pitch_score"],
                        "dynamic_eye_threshold": self.state["dynamic_eye_threshold"],
                    }
            paths = self._capture_and_save_session(profile_root, profile_name, folder_id, exposure_ms, self.gain, detection_snapshot)
            return {
                "ok": True,
                "mode": "manual",
                "captureId": paths.capture_id,
                "sessionDir": paths.session_dir,
                "files": {
                    "cam2": paths.cam2_path,
                    "cam3": paths.cam3_path,
                    "cam4": paths.cam4_path,
                },
                "metadataPath": paths.session_metadata_path,
            }

    def auto_capture(self, profile_root: Path, profile_name: str, folder_id: str, exposure_ms: Optional[int] = None, timeout_sec: float = 30.0):
        if self.face_mesh is None:
            raise RuntimeError("mediapipe가 설치되지 않아 자동촬영을 사용할 수 없습니다.")
        exposure_ms = exposure_ms or self.exposure_ms
        start = monotonic()
        self.state["armed"] = True
        self.state["stable_face_count"] = 0
        self.state["eyes_closed_count"] = 0
        self.state["status"] = "얼굴 위치/각도를 맞추고 눈을 감아주세요"
        self.white_led_on()
        try:
            while monotonic() - start <= timeout_sec:
                with self.camera_lock:
                    if not self.camera_ready:
                        raise RuntimeError("카메라가 준비되지 않았습니다.")
                    frame = self.read_full_frame_bgr()
                    if frame is None:
                        self.state["status"] = "프레임 획득 실패"
                        sleep(0.08)
                        continue

                    cam2 = self.extract_cam_frame(frame, "cam2").copy()
                    detection = self.analyze_face(cam2)
                    should_capture = self._handle_auto_state(detection)

                    if should_capture:
                        self.state["status"] = "조건 충족: 백색 LED OFF 후 촬영"
                        self.white_led_off()
                        sleep(self.WHITE_LED_OFF_BEFORE_CAPTURE_SEC)
                        detection_snapshot = {
                            "bbox": list(detection["bbox"]) if detection["bbox"] is not None else None,
                            "roll_deg": detection["roll_deg"],
                            "yaw_score": detection["yaw_score"],
                            "pitch_score": detection["pitch_score"],
                            "dynamic_eye_threshold": self.state["dynamic_eye_threshold"],
                        }
                        paths = self._capture_and_save_session(
                            profile_root, profile_name, folder_id, exposure_ms, self.gain, detection_snapshot
                        )
                        self.state["armed"] = False
                        self.state["stable_face_count"] = 0
                        self.state["eyes_closed_count"] = 0
                        return {
                            "ok": True,
                            "mode": "auto",
                            "captureId": paths.capture_id,
                            "sessionDir": paths.session_dir,
                            "files": {
                                "cam2": paths.cam2_path,
                                "cam3": paths.cam3_path,
                                "cam4": paths.cam4_path,
                            },
                            "metadataPath": paths.session_metadata_path,
                        }
                sleep(0.08)

            raise TimeoutError("자동 촬영 시간이 초과되었습니다.")
        finally:
            self.state["armed"] = False
            self.white_led_off()
