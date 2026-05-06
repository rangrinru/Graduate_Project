from __future__ import annotations

"""
acne_risk_model.py

역할
- 현재 세션의 포르피린 분석 결과 + 피부 상태 점수 + (선택적으로) 과거 세션 경향을 입력받아
  여드름 발생 위험을 예측하는 뼈대 코드
- 지금은 규칙 기반 기본 구현
- 추후 데이터셋이 쌓이면 sklearn / xgboost / torch 모델로 교체 가능
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AcneRiskResult:
    risk_label: str
    risk_score_0_to_1: float
    confidence_0_to_1: float
    contributing_factors: List[str]
    note: str


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_recent_pipeline_reports(session_dir: Path, lookback: int = 5) -> List[Dict[str, Any]]:
    """
    같은 상위 sessions 폴더 내 최근 pipeline_report.json들을 읽는다.
    현재 세션 제외 또는 포함 여부는 상황에 따라 조정 가능.
    """
    sessions_root = session_dir.parent
    if not sessions_root.exists():
        return []

    reports: List[Dict[str, Any]] = []
    session_dirs = sorted([p for p in sessions_root.iterdir() if p.is_dir()], reverse=True)

    for sdir in session_dirs[:lookback]:
        report_path = sdir / "analysis" / "pipeline_report.json"
        if report_path.exists():
            try:
                reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            except Exception:
                continue

    return reports


def compute_time_trend_bonus(recent_reports: List[Dict[str, Any]]) -> tuple[float, List[str]]:
    """
    최근 보고서가 있으면 score 증가 추세를 반영하는 보너스 항목.
    반환:
    - trend_bonus: 0.0 ~ 0.2 정도
    - factors: 설명 문자열 리스트
    """
    factors: List[str] = []

    if len(recent_reports) < 2:
        return 0.0, factors

    try:
        scores = []
        for item in recent_reports:
            skin_score = item.get("skin_score", {})
            scores.append(float(skin_score.get("score_0_to_100", 0.0)))

        if len(scores) < 2:
            return 0.0, factors

        latest = scores[0]
        oldest = scores[-1]

        if latest - oldest >= 15:
            factors.append("최근 세션 대비 피부 상태 점수가 악화되는 추세")
            return 0.15, factors
        if latest - oldest >= 8:
            factors.append("최근 세션 대비 포르피린 burden이 다소 증가하는 추세")
            return 0.08, factors

    except Exception:
        pass

    return 0.0, factors


def predict_acne_risk(
    porphyrin_result: Dict[str, Any],
    skin_score: Any,
    session_dir: Optional[str | Path] = None,
) -> AcneRiskResult:
    """
    입력:
    - porphyrin_result: 포르피린 분석 결과 dict
    - skin_score: SkinScoreBreakdown dataclass 또는 dict
    - session_dir: 세션 폴더 경로
    """

    if hasattr(skin_score, "__dict__"):
        score_dict = skin_score.__dict__
    else:
        score_dict = dict(skin_score)

    score_100 = float(score_dict.get("score_0_to_100", 0.0))
    porphyrin_count = float(porphyrin_result.get("porphyrin_count", 0.0))
    porphyrin_area = float(porphyrin_result.get("porphyrin_area", 0.0))
    mean_intensity = float(porphyrin_result.get("mean_intensity", 0.0))

    contributing_factors: List[str] = []

    base_risk = (
        0.45 * clamp(score_100 / 100.0, 0.0, 1.0)
        + 0.20 * clamp(porphyrin_count / 80.0, 0.0, 1.0)
        + 0.20 * clamp(porphyrin_area / 15000.0, 0.0, 1.0)
        + 0.15 * clamp(mean_intensity / 255.0, 0.0, 1.0)
    )

    if score_100 >= 60:
        contributing_factors.append("피부 상태 점수가 높은 부담 영역에 해당")
    if porphyrin_count >= 20:
        contributing_factors.append("포르피린 점 개수가 상대적으로 많음")
    if porphyrin_area >= 4000:
        contributing_factors.append("포르피린 총 면적이 큼")
    if mean_intensity >= 150:
        contributing_factors.append("포르피린 평균 강도가 높음")

    trend_bonus = 0.0
    if session_dir is not None:
        recent_reports = load_recent_pipeline_reports(Path(session_dir))
        trend_bonus, trend_factors = compute_time_trend_bonus(recent_reports)
        contributing_factors.extend(trend_factors)

    risk_score = clamp(base_risk + trend_bonus, 0.0, 1.0)

    if risk_score < 0.25:
        label = "낮음"
    elif risk_score < 0.50:
        label = "보통"
    elif risk_score < 0.75:
        label = "높음"
    else:
        label = "매우 높음"

    confidence = 0.55
    if session_dir is not None and trend_bonus > 0:
        confidence = 0.65

    note = (
        "현재 결과는 규칙 기반 초기 예측입니다. "
        "라벨된 임상/실험 데이터가 확보되면 ML 모델로 교체하세요."
    )

    return AcneRiskResult(
        risk_label=label,
        risk_score_0_to_1=round(risk_score, 4),
        confidence_0_to_1=round(confidence, 4),
        contributing_factors=contributing_factors,
        note=note,
    )
