from __future__ import annotations

"""
skin_scoring.py

porphyrin_viewer.py의 검출 로직을 기준으로 사용하는 피부 상태 점수 계산 뼈대 코드

핵심 가정
- porphyrin_viewer.py는 percentile threshold(97), contour area, radius 조건으로
  "특정 밝기 이상의 포르피린 점 개수"를 카운트한다.
- 이 skin_scoring.py는 그 결과를 받아
  1) 개수
  2) 면적
  3) 강도
  4) 부위 편중도
  를 바탕으로 0~100 점수를 계산한다.

주의
- 현재는 규칙 기반(rule-based) 뼈대 코드
- 정규화 범위와 가중치는 반드시 실험 데이터로 재보정해야 함
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SkinScoreBreakdown:
    score_0_to_100: float
    grade: str
    burden_level: str

    # 원본 입력값
    porphyrin_count: float
    porphyrin_area: float
    mean_intensity: float
    regional_bias: float

    # 정규화된 성분 점수
    count_component: float
    area_component: float
    intensity_component: float
    regional_component: float

    # 참고 정보
    threshold_source: str
    note: str


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_linear(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    return clamp((value - min_value) / (max_value - min_value), 0.0, 1.0)


def compute_regional_bias(regional_distribution: Dict[str, Any]) -> float:
    """
    regional_distribution 예시:
    {
        "forehead": {"count": 3, "area": 120},
        "left_cheek": {"count": 10, "area": 420},
        "right_cheek": {"count": 2, "area": 80},
        "chin": {"count": 1, "area": 30}
    }

    편중도가 클수록 1에 가까워짐.
    여기서는 가장 면적이 큰 부위가 전체 면적에서 차지하는 비율을 사용.
    """
    if not regional_distribution:
        return 0.0

    total_area = 0.0
    max_region_area = 0.0

    for value in regional_distribution.values():
        area = float(value.get("area", 0.0))
        total_area += area
        max_region_area = max(max_region_area, area)

    if total_area <= 0:
        return 0.0

    return clamp(max_region_area / total_area, 0.0, 1.0)


def classify_grade(score: float) -> tuple[str, str]:
    """
    점수가 높을수록 포르피린 burden이 높다고 가정
    """
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
    session_dir: Optional[str] = None,
) -> SkinScoreBreakdown:
    """
    porphyrin_result 입력 형식 예시:
    {
        "porphyrin_count": 12,
        "porphyrin_area": 3580,
        "mean_intensity": 143.2,
        "regional_distribution": {...},
        "threshold_source": "viewer_percentile97_area10_radius3to15"
    }

    porphyrin_viewer.py 기반으로 count를 가장 직접 지표로 사용하고,
    area / intensity / regional bias를 함께 점수화한다.
    """

    count = float(porphyrin_result.get("porphyrin_count", 0.0))
    area = float(porphyrin_result.get("porphyrin_area", 0.0))
    intensity = float(porphyrin_result.get("mean_intensity", 0.0))
    regional_distribution = porphyrin_result.get("regional_distribution", {}) or {}
    threshold_source = porphyrin_result.get(
        "threshold_source",
        "viewer_percentile97_area10_radius3to15"
    )

    regional_bias = compute_regional_bias(regional_distribution)

    # TODO: 반드시 실험 데이터 기준으로 재보정
    count_norm = normalize_linear(count, 0, 80)
    area_norm = normalize_linear(area, 0, 15000)
    intensity_norm = normalize_linear(intensity, 0, 255)
    regional_norm = clamp(regional_bias, 0.0, 1.0)

    # viewer 기반 count를 가장 중요하게 둠
    score = (
        0.40 * count_norm
        + 0.25 * area_norm
        + 0.20 * intensity_norm
        + 0.15 * regional_norm
    ) * 100.0

    score = round(clamp(score, 0.0, 100.0), 2)
    grade, burden_level = classify_grade(score)

    note = (
        "porphyrin_viewer.py의 percentile threshold + contour count 로직을 기준으로 "
        "개수 비중을 가장 높게 둔 초기 점수화 버전입니다. "
        "실험 데이터가 쌓이면 area/intensity/regional weight와 정규화 범위를 재보정하세요."
    )

    return SkinScoreBreakdown(
        score_0_to_100=score,
        grade=grade,
        burden_level=burden_level,
        porphyrin_count=count,
        porphyrin_area=area,
        mean_intensity=intensity,
        regional_bias=round(regional_bias, 4),
        count_component=round(count_norm * 100.0, 2),
        area_component=round(area_norm * 100.0, 2),
        intensity_component=round(intensity_norm * 100.0, 2),
        regional_component=round(regional_norm * 100.0, 2),
        threshold_source=threshold_source,
        note=note,
    )


def compute_skin_score_from_metrics(
    porphyrin_count: float,
    porphyrin_area: float,
    mean_intensity: float,
    regional_distribution: Optional[Dict[str, Any]] = None,
    threshold_source: str = "viewer_percentile97_area10_radius3to15",
) -> SkinScoreBreakdown:
    """
    porphyrin_analysis.py가 아직 dict 구조를 완전히 안 맞췄을 때,
    값만 직접 넣어서 계산할 수 있게 만든 편의 함수
    """
    payload = {
        "porphyrin_count": porphyrin_count,
        "porphyrin_area": porphyrin_area,
        "mean_intensity": mean_intensity,
        "regional_distribution": regional_distribution or {},
        "threshold_source": threshold_source,
    }
    return compute_skin_score(payload)


if __name__ == "__main__":
    demo = {
        "porphyrin_count": 18,
        "porphyrin_area": 4200,
        "mean_intensity": 158.4,
        "regional_distribution": {
            "forehead": {"count": 3, "area": 500},
            "left_cheek": {"count": 9, "area": 2400},
            "right_cheek": {"count": 4, "area": 900},
            "chin": {"count": 2, "area": 400},
        },
        "threshold_source": "viewer_percentile97_area10_radius3to15",
    }

    result = compute_skin_score(demo)
    print(result)
