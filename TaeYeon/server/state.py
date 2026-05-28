import threading
from config import DEFAULT_EYE_AR_THRESHOLD

raw_stream_process = None
raw_stream_thread = None
raw_stream_stop_event = threading.Event()
raw_stream_lock = threading.Lock()
latest_preview_gray8 = None
latest_preview_frame_time = 0.0
raw_fifo_fd = None

cam4_jpeg_thread = None
cam4_jpeg_stop_event = threading.Event()
cam4_jpeg_condition = threading.Condition()
latest_cam4_jpeg = None
latest_cam4_jpeg_time = 0.0
latest_cam4_jpeg_seq = 0

arducam_media_device = None
uc788_media_configured = False

camera_lock = threading.Lock()
camera_ready = False
picam2 = None
preview_config = None
still_config = None

auto_capture_thread = None
auto_state_lock = threading.Lock()


def make_default_auto_checks():
    return {
        "face_found": False,
        "center_ok": False,
        "size_ok": False,
        "angle_ok": False,
        "eyes_closed": False,
        "stable_ok": False,
    }


AUTO_STATE = {
    "running": False,
    "captured": False,
    "profile_id": None,
    "capture_id": None,
    "status": "자동 촬영 대기 중",
    "error": None,
    "checks": make_default_auto_checks(),
    "stable_face_count": 0,
    "eyes_closed_count": 0,
    "dynamic_eye_threshold": DEFAULT_EYE_AR_THRESHOLD,
    "last_update": None,
}

AUTO_EYE_STATE = {
    "blink_count": 0,
    "closed_frame_count": 0,
    "prev_eye_closed": False,
    "eyes_closed_started_at": None,
    "eye_state": "Unknown",
    "gaze_direction": "Unknown",
}
