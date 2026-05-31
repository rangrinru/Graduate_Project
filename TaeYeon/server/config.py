from pathlib import Path

# Capture image size.
CAPTURE_WIDTH = 5120
CAPTURE_HEIGHT = 800
SINGLE_WIDTH = CAPTURE_WIDTH // 4

SAVE_ROOT = Path.home() / "Graduate_Project" / "TaeYeon" / "captures"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
PROFILES_FILE = SAVE_ROOT / "profiles.json"

CURRENT_GAIN = 1.0
SAVE_AS_PNG = True
INITIAL_EXPOSURE_MS = 1
REFERENCE_EXPOSURE_MS = 1
FLUORESCENCE_EXPOSURE_MS = 100
PREVIEW_EXPOSURE_MS = 100
PICAMERA_PREVIEW_WARMUP_SEC = 0.5
PICAMERA_CONTROL_SETTLE_SEC = 0.2
PREVIEW_CAM_KEY = "cam4"

CAPTURE_EXPOSURE_MS_BY_CAMERA = {
    "cam2": REFERENCE_EXPOSURE_MS,
    "cam3": REFERENCE_EXPOSURE_MS,
    "cam4": FLUORESCENCE_EXPOSURE_MS,
}

STREAM_FPS = 15
STREAM_WIDTH = 720
STREAM_HEIGHT = 1152
STREAM_JPEG_QUALITY = 70
PREVIEW_MIRROR_HORIZONTAL = True

RELAY_PIN = 17
RELAY_ACTIVE_HIGH = False
RELAY_WARMUP_SEC = 2.4

RGB1_RED_PIN = 27
RGB1_GREEN_PIN = 22
RGB1_BLUE_PIN = 23
RGB2_RED_PIN = 5
RGB2_GREEN_PIN = 6
RGB2_BLUE_PIN = 13
RGB_LED_ACTIVE_HIGH = True
WHITE_LED_COLOR = (1, 1, 1)
WHITE_LED_OFF_BEFORE_CAPTURE_SEC = 0.15
WHITE_LED_WARMUP_SEC = 0.25
WHITE_REFERENCE_EXPOSURE_MS = REFERENCE_EXPOSURE_MS
WHITE_CAM4_REFERENCE_NAME = "cam4_white"

DEFAULT_EYE_AR_THRESHOLD = 0.20
EYES_CLOSED_DELAY_SEC = 2.0
AUTO_DETECTION_CAM_KEY = "cam2"
AUTO_DETECTION_PROCESS_WIDTH = 960
AUTO_CAPTURE_INTERVAL_SEC = 0.12

RAW_WIDTH = CAPTURE_WIDTH
RAW_HEIGHT = CAPTURE_HEIGHT
RAW_VIDEO_DEVICE = "/dev/video0"
RAW_FRAME_PATH = Path("/dev/shm/uc788_raw_frame.bin")
RAW_EXPECTED_BYTES = RAW_WIDTH * RAW_HEIGHT * 5 // 4
RAW_FIFO_PATH = Path("/dev/shm/uc788_y10p_fifo")
RAW_NORMALIZE_LOW_PERCENTILE = 1
RAW_NORMALIZE_HIGH_PERCENTILE = 99
UC788_TRIGGER_MODE = 0
UC788_EXPOSURE = 800
UC788_REFERENCE_EXPOSURE = 0
UC788_FLUORESCENCE_EXPOSURE = 800
UC788_PREVIEW_EXPOSURE = 800
UC788_ANALOGUE_GAIN = 100

CAMERA_INFO = {
    "cam2": {
        "label": "CAM 2 - NO FILTER",
        "folder": "cam2_no_filter",
        "filter": "no_filter",
        "display_name": "No_Filter",
        "x_start": SINGLE_WIDTH * 1,
        "x_end": SINGLE_WIDTH * 2,
        "sequence_order": 1,
    },
    "cam3": {
        "label": "CAM 3 - 405nm FILTER",
        "folder": "cam3_405nm",
        "filter": "405nm_filter",
        "display_name": "405nm_Filter",
        "x_start": SINGLE_WIDTH * 2,
        "x_end": SINGLE_WIDTH * 3,
        "sequence_order": 2,
    },
    "cam4": {
        "label": "CAM 4 - 660nm FILTER",
        "folder": "cam4_660nm",
        "filter": "660nm_filter",
        "display_name": "660nm_Filter",
        "x_start": SINGLE_WIDTH * 3,
        "x_end": SINGLE_WIDTH * 4,
        "sequence_order": 3,
    },
}
