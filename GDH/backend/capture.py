import subprocess

def run_capture():
    subprocess.run([
        "python",
        "../rpicam_03_white_led_auto_v5.py"
    ])