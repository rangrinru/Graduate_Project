from __future__ import annotations

"""
skin_scoring.py

역할
- porphyrin_analysis 결과를 입력으로 받아 피부 상태 점수를 계산
- 지금은 규칙 기반(rule-based) 뼈대 코드
- 추후 임상 기준/실험 기준에 맞게 가중치와 구간을 조정하면 됨
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SkinScoreBreakdown:
    score_0_to_100: float
    grade: str
    burden_level: str
    count_component: float
    area_component: float
    intensity_component: float
    regional_component: float
    note: str


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_linear(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    return clamp((value - min_value) / (max_value - min_value), 0.0, 1.0)


def compute_regional_component(regional_distribution: Dict[str, Any]) -> float:
    """
    얼굴 부위별 분포 지도가 있을 때 사용하는 보조 점수.
    예:
    {
        "forehead": {"count": 3, "area": 120},
        "left_cheek": {"count": 10, "area": 420},
        ...
    }
    """
    if not regional_distribution:
        return 0.0

    max_region_area = 0.0
    total_area = 0.0

    for value in regional_distribution.values():
        area = float(value.get("area", 0.0))
        total_area += area
        max_region_area = max(max_region_area, area)

    if total_area <= 0:
        return 0.0

    concentration_ratio = max_region_area / total_area
    return clamp(concentration_ratio, 0.0, 1.0)


def classify_grade(score: float) -> tuple[str, str]:
    if score < 20:
        return "A", "매우 양호"
    if score < 40:
        return "B", "양호"
    if score < 60:
        return "C", "보통"
    if score < 80:
        return "D", "주의"
    return "E", "높음"


def compute_skin_score(
    porphyrin_result: Dict[str, Any],
    session_dir: Optional[str | Path] = None,
) -> SkinScoreBreakdown:
    """
    porphyrin_result 예시:
    {
        "porphyrin_count": 12,
        "porphyrin_area": 3580,
        "mean_intensity": 143.2,
        "regional_distribution": {...}
    }
    """

    count = float(porphyrin_result.get("porphyrin_count", 0))
    area = float(porphyrin_result.get("porphyrin_area", 0))
    intensity = float(porphyrin_result.get("mean_intensity", 0.0))
    regional_distribution = porphyrin_result.get("regional_distribution", {}) or {}

    # TODO: 실제 실험 데이터 기준으로 재조정 필요
    count_norm = normalize_linear(count, 0, 80)
    area_norm = normalize_linear(area, 0, 15000)
    intensity_norm = normalize_linear(intensity, 0, 255)
    regional_norm = compute_regional_component(regional_distribution)

    score = (
        0.35 * count_norm
        + 0.30 * area_norm
        + 0.25 * intensity_norm
        + 0.10 * regional_norm
    ) * 100.0

    score = round(clamp(score, 0.0, 100.0), 2)
    grade, burden_level = classify_grade(score)

    note = (
        "현재 점수는 규칙 기반 초기 버전입니다. "
        "실험 데이터가 쌓이면 정규화 범위와 가중치를 다시 보정하세요."
    )

    return SkinScoreBreakdown(
        score_0_to_100=score,
        grade=grade,
        burden_level=burden_level,
        count_component=round(count_norm * 100.0, 2),
        area_component=round(area_norm * 100.0, 2),
        intensity_component=round(intensity_norm * 100.0, 2),
        regional_component=round(regional_norm * 100.0, 2),
        note=note,
    )
