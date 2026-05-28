import subprocess

import cv2
import numpy as np

from config import (
    RAW_EXPECTED_BYTES,
    RAW_HEIGHT,
    RAW_NORMALIZE_HIGH_PERCENTILE,
    RAW_NORMALIZE_LOW_PERCENTILE,
    RAW_WIDTH,
)


def run_command(command, check=True, capture_output=True):
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_output else subprocess.DEVNULL,
    )


def y10p_high8_to_gray8(raw_bytes):
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    if data.size != RAW_EXPECTED_BYTES:
        raise RuntimeError(f"raw 크기 오류: expected={RAW_EXPECTED_BYTES}, actual={data.size}")

    row_stride = RAW_WIDTH * 5 // 4
    packed = data.reshape(RAW_HEIGHT, row_stride)
    groups = packed.reshape(RAW_HEIGHT, RAW_WIDTH // 4, 5)
    gray8 = groups[:, :, :4].reshape(RAW_HEIGHT, RAW_WIDTH).copy()

    return cv2.convertScaleAbs(gray8, alpha=1.35, beta=0)


def unpack_y10p_to_gray8(raw_bytes):
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    if data.size != RAW_EXPECTED_BYTES:
        raise RuntimeError(f"raw 크기 오류: expected={RAW_EXPECTED_BYTES}, actual={data.size}")

    groups = data.reshape(-1, 5)
    p0 = (groups[:, 0].astype(np.uint16) << 2) | ((groups[:, 4] >> 0) & 0x03)
    p1 = (groups[:, 1].astype(np.uint16) << 2) | ((groups[:, 4] >> 2) & 0x03)
    p2 = (groups[:, 2].astype(np.uint16) << 2) | ((groups[:, 4] >> 4) & 0x03)
    p3 = (groups[:, 3].astype(np.uint16) << 2) | ((groups[:, 4] >> 6) & 0x03)

    img10 = np.empty(RAW_WIDTH * RAW_HEIGHT, dtype=np.uint16)
    img10[0::4] = p0
    img10[1::4] = p1
    img10[2::4] = p2
    img10[3::4] = p3
    img10 = img10.reshape(RAW_HEIGHT, RAW_WIDTH)

    low = np.percentile(img10, RAW_NORMALIZE_LOW_PERCENTILE)
    high = np.percentile(img10, RAW_NORMALIZE_HIGH_PERCENTILE)
    if high <= low:
        return (img10 >> 2).astype(np.uint8)

    normalized = np.clip((img10.astype(np.float32) - low) * 255.0 / (high - low), 0, 255)
    return normalized.astype(np.uint8)
