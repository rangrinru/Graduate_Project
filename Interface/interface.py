import tkinter as tk
from PIL import Image, ImageTk
import cv2

# -----------------------------
# 카메라 설정
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# 메인 윈도우
# -----------------------------
root = tk.Tk()
root.title("Skin Mirror Display")
root.attributes('-fullscreen', True)  # 전체화면

# -----------------------------
# 상태 변수
# -----------------------------
mode = "mirror"  # mirror / uv

# -----------------------------
# 카메라 프레임 표시 함수
# -----------------------------
def show_frame():
    global mode

    ret, frame = cap.read()
    if not ret:
        return

    # 좌우 반전 (거울 효과)
    frame = cv2.flip(frame, 1)

    if mode == "uv":
        # UV 느낌 효과 (대비 강화)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(frame)
        v = cv2.equalizeHist(v)
        frame = cv2.merge((h, s, v))
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    # tkinter 이미지 변환
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, show_frame)

# -----------------------------
# 버튼 기능
# -----------------------------
def set_mirror():
    global mode
    mode = "mirror"

def set_uv():
    global mode
    mode = "uv"

def capture_image():
    ret, frame = cap.read()
    if ret:
        filename = "capture.jpg"
        cv2.imwrite(filename, frame)
        print("촬영 완료:", filename)

def exit_app():
    cap.release()
    root.destroy()

# -----------------------------
# UI 구성
# -----------------------------
label = tk.Label(root)
label.pack()

btn_frame = tk.Frame(root, bg="black")
btn_frame.pack(side="bottom", fill="x")

btn_mirror = tk.Button(btn_frame, text="거울 모드", command=set_mirror, height=3)
btn_mirror.pack(side="left", expand=True, fill="x")

btn_uv = tk.Button(btn_frame, text="UV 모드", command=set_uv, height=3)
btn_uv.pack(side="left", expand=True, fill="x")

btn_capture = tk.Button(btn_frame, text="촬영", command=capture_image, height=3)
btn_capture.pack(side="left", expand=True, fill="x")

btn_exit = tk.Button(btn_frame, text="종료", command=exit_app, height=3)
btn_exit.pack(side="left", expand=True, fill="x")

# -----------------------------
# 실행
# -----------------------------
show_frame()
root.mainloop()