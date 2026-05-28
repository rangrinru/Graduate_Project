import subprocess
import numpy as np


def run_command(command, check=True, capture_output=True):
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_output else subprocess.DEVNULL,
    )


def y10p_high8_to_gray8(raw_bytes):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    groups = arr.reshape(-1, 5)
    pixels = np.empty(groups.shape[0] * 4, dtype=np.uint8)
    pixels[0::4] = groups[:, 0]
    pixels[1::4] = groups[:, 1]
    pixels[2::4] = groups[:, 2]
    pixels[3::4] = groups[:, 3]
    return pixels


def unpack_y10p_to_gray8(raw_bytes):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    groups = arr.reshape(-1, 5).astype(np.uint16)
    pixels10 = np.empty(groups.shape[0] * 4, dtype=np.uint16)
    pixels10[0::4] = (groups[:, 0] << 2) | (groups[:, 4] & 0b00000011)
    pixels10[1::4] = (groups[:, 1] << 2) | ((groups[:, 4] >> 2) & 0b00000011)
    pixels10[2::4] = (groups[:, 2] << 2) | ((groups[:, 4] >> 4) & 0b00000011)
    pixels10[3::4] = (groups[:, 3] << 2) | ((groups[:, 4] >> 6) & 0b00000011)
    return (pixels10 >> 2).astype(np.uint8)
