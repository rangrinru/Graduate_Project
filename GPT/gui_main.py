from __future__ import annotations

"""
gui_main.py
메인 GUI 스켈레톤
"""

import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import cv2
from PIL import Image, ImageTk

from camera_controller import AutoCaptureController, ControllerConfig
from analyzer import PorphyrinAnalyzer, AnalysisResult


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UVA 피부 분석 시스템")
        self.geometry("1200x800")

        self.controller = AutoCaptureController(
            ControllerConfig(
                save_root=str(Path.home() / "Graduate_Project" / "captures")
            )
        )
        self.analyzer = PorphyrinAnalyzer()

        self.preview_image_ref = None
        self.analysis_image_ref = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.controller.start()
        self.after(30, self.update_preview)

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Button(top, text="자동 촬영 시작", command=self.on_capture_clicked).pack(side="left", padx=5)
        ttk.Button(top, text="촬영 취소", command=self.on_cancel_clicked).pack(side="left", padx=5)
        ttk.Button(top, text="최신 촬영 분석", command=self.on_analyze_clicked).pack(side="left", padx=5)

        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(top, textvariable=self.status_var).pack(side="left", padx=20)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.LabelFrame(body, text="실시간 프리뷰")
        left.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        right = ttk.LabelFrame(body, text="분석 결과")
        right.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.preview_label = ttk.Label(left)
        self.preview_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.analysis_label = ttk.Label(right)
        self.analysis_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.result_text = tk.Text(right, height=12)
        self.result_text.pack(fill="x", padx=10, pady=10)

    def on_capture_clicked(self):
        self.controller.request_auto_capture()
        self.status_var.set("자동 촬영 요청됨")

    def on_cancel_clicked(self):
        self.controller.cancel_auto_capture()
        self.status_var.set("촬영 취소됨")

    def on_analyze_clicked(self):
        threading.Thread(target=self._run_latest_analysis, daemon=True).start()

    def _run_latest_analysis(self):
        self.status_var.set("분석 중...")
        result = self.analyzer.analyze_latest_capture(self.controller.config.save_root)
        self.after(0, lambda: self.show_analysis_result(result))

    def show_analysis_result(self, result: AnalysisResult):
        self.result_text.delete("1.0", tk.END)

        if not result.success:
            self.result_text.insert(tk.END, f"분석 실패: {result.reason}\n")
            self.status_var.set("분석 실패")
            return

        self.result_text.insert(tk.END, f"capture_dir: {result.capture_dir}\n")
        self.result_text.insert(tk.END, f"porphyrin_count: {result.porphyrin_count}\n")
        self.result_text.insert(tk.END, f"porphyrin_area: {result.porphyrin_area}\n")
        self.result_text.insert(tk.END, f"mean_intensity: {result.mean_intensity:.2f}\n")
        self.result_text.insert(tk.END, f"mask_path: {result.mask_path}\n")
        self.result_text.insert(tk.END, f"overlay_path: {result.overlay_path}\n")
        self.status_var.set("분석 완료")

        if result.overlay_path:
            img = cv2.imread(result.overlay_path)
            if img is not None:
                self._show_cv_image_on_label(img, self.analysis_label, preview=False)

    def update_preview(self):
        frame = self.controller.get_latest_frame()
        self.status_var.set(self.controller.get_status_text())

        if frame is not None:
            self._show_cv_image_on_label(frame, self.preview_label, preview=True)

        self.after(30, self.update_preview)

    def _show_cv_image_on_label(self, frame, label: ttk.Label, preview: bool):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        target_w = 520 if preview else 420
        ratio = target_w / image.width
        target_h = int(image.height * ratio)
        image = image.resize((target_w, target_h))

        tk_image = ImageTk.PhotoImage(image=image)
        label.configure(image=tk_image)

        if preview:
            self.preview_image_ref = tk_image
        else:
            self.analysis_image_ref = tk_image

    def on_close(self):
        try:
            self.controller.stop()
        finally:
            self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
