import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time

# -----------------------------
# 카메라
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# 윈도우
# -----------------------------
root = tk.Tk()
root.attributes('-fullscreen', True)
root.configure(bg='black')

mode = "mirror"  # mirror / capture
countdown = -1
countdown_start = 0

# -----------------------------
# 얼굴 검출
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -----------------------------
# 화면 표시
# -----------------------------
def update_frame():
    global mode, countdown, countdown_start

    if mode == "mirror":
        # 검정 화면
        canvas.create_rectangle(0, 0, 1920, 1080, fill="black")

    elif mode == "capture":
        ret, frame = cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # 얼굴 박스
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # 얼굴 중앙 체크
        if len(faces) > 0 and countdown == -1:
            countdown = 4
            countdown_start = time.time()

        # 카운트다운
        if countdown > 0:
            elapsed = time.time() - countdown_start
            if elapsed >= 1:
                countdown -= 1
                countdown_start = time.time()

        # 촬영
        if countdown == 0:
            cv2.imwrite("capture.jpg", frame)
            print("촬영 완료")
            countdown = -1
            mode = "mirror"

        # 화면에 숫자 표시
        if countdown > 0:
            cv2.putText(frame, str(countdown), (300,300),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 5)

        # tkinter 변환
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor='nw', image=imgtk)

    root.after(10, update_frame)

# -----------------------------
# 버튼 기능
# -----------------------------
def set_mirror():
    global mode
    mode = "mirror"

def set_capture():
    global mode
    mode = "capture"

# -----------------------------
# UI
# -----------------------------
canvas = tk.Canvas(root, width=1920, height=1080, bg="black")
canvas.pack()

btn_frame = tk.Frame(root, bg="black")
btn_frame.place(relx=0.5, rely=0.9, anchor='center')

btn_mirror = tk.Button(btn_frame, text="거울모드", command=set_mirror, width=15, height=2)
btn_mirror.pack(side="left", padx=20)

btn_capture = tk.Button(btn_frame, text="촬영모드", command=set_capture, width=15, height=2)
btn_capture.pack(side="left", padx=20)

# -----------------------------
# 실행
# -----------------------------
update_frame()
root.mainloop()