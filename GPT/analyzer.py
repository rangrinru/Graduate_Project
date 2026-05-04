from __future__ import annotations

"""
analyzer.py
촬영된 이미지 분석 스켈레톤
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class AnalysisResult:
    success: bool
    capture_dir: str = ""
    porphyrin_count: int = 0
    porphyrin_area: int = 0
    mean_intensity: float = 0.0
    mask_path: str = ""
    overlay_path: str = ""
    report_path: str = ""
    reason: str = ""


class PorphyrinAnalyzer:
    def __init__(self, result_root: Optional[str] = None):
        self.result_root = Path(result_root) if result_root else None

    def analyze_capture_set(self, capture_dir: str) -> AnalysisResult:
        capture_path = Path(capture_dir)

        cam2_path = capture_path / "cam2.png"
        cam3_path = capture_path / "cam3.png"
        cam4_path = capture_path / "cam4.png"

        if not cam4_path.exists():
            return AnalysisResult(
                success=False,
                capture_dir=str(capture_path),
                reason="cam4 이미지가 없습니다."
            )

        cam2 = self._read_gray(cam2_path)
        cam3 = self._read_gray(cam3_path)
        cam4 = self._read_gray(cam4_path)

        if cam4 is None:
            return AnalysisResult(
                success=False,
                capture_dir=str(capture_path),
                reason="cam4 이미지 로드 실패"
            )

        if cam2 is None:
            cam2 = np.zeros_like(cam4)
        if cam3 is None:
            cam3 = np.zeros_like(cam4)

        # 간단한 포르피린 후보 점수
        score = (
            0.65 * cam4.astype(np.float32)
            + 0.20 * np.maximum(cam4.astype(np.float32) - cam3.astype(np.float32), 0)
            + 0.15 * np.maximum(cam4.astype(np.float32) - cam2.astype(np.float32), 0)
        )

        score_norm = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(score_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        min_area = 20
        cleaned = np.zeros_like(mask)
        porphyrin_count = 0
        porphyrin_area = 0

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == label_idx] = 255
                porphyrin_count += 1
                porphyrin_area += int(area)

        mean_intensity = float(score_norm[cleaned > 0].mean()) if np.any(cleaned > 0) else 0.0

        cam4_bgr = cv2.imread(str(cam4_path), cv2.IMREAD_COLOR)
        if cam4_bgr is None:
            cam4_bgr = cv2.cvtColor(cam4, cv2.COLOR_GRAY2BGR)

        overlay = cam4_bgr.copy()
        overlay[cleaned > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(cam4_bgr, 0.7, overlay, 0.3, 0)

        out_dir = self._resolve_output_dir(capture_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        mask_path = str(out_dir / "porphyrin_mask.png")
        overlay_path = str(out_dir / "porphyrin_overlay.png")
        report_path = str(out_dir / "report.json")

        cv2.imwrite(mask_path, cleaned)
        cv2.imwrite(overlay_path, overlay)

        result = AnalysisResult(
            success=True,
            capture_dir=str(capture_path),
            porphyrin_count=porphyrin_count,
            porphyrin_area=porphyrin_area,
            mean_intensity=mean_intensity,
            mask_path=mask_path,
            overlay_path=overlay_path,
            report_path=report_path,
        )

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=4)

        return result

    def analyze_latest_capture(self, save_root: str) -> AnalysisResult:
        root = Path(save_root)
        if not root.exists():
            return AnalysisResult(success=False, reason="촬영 폴더가 없습니다.")

        folders = [p for p in root.iterdir() if p.is_dir()]
        if not folders:
            return AnalysisResult(success=False, reason="분석할 촬영 폴더가 없습니다.")

        latest = max(folders, key=lambda p: p.stat().st_mtime)
        return self.analyze_capture_set(str(latest))

    def _resolve_output_dir(self, capture_path: Path) -> Path:
        if self.result_root is not None:
            return self.result_root / capture_path.name
        return capture_path / "analysis"

    @staticmethod
    def _read_gray(path: Path):
        if not path.exists():
            return None
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
