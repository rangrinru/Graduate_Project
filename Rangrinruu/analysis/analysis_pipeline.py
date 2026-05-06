from __future__ import annotations

"""
analysis_pipeline.py

역할
- 촬영 세션 폴더(session_dir)를 입력으로 받음
- porphyrin_analysis.py의 1차 분석 결과를 불러옴
- skin_scoring.py로 피부 상태 점수 계산
- acne_risk_model.py로 여드름 발생 위험 예측
- 최종 결과를 pipeline_report.json으로 저장

주의
- 현재는 "뼈대 코드"이므로 규칙 기반 기본값과 TODO가 포함되어 있음
- porphyrin_analysis.py 안의 실제 분석 함수 이름이 프로젝트마다 다를 수 있어
  아래 resolve_analysis_function()이 여러 이름을 자동으로 탐색함
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict

from skin_scoring import SkinScoreBreakdown, compute_skin_score
from acne_risk_model import AcneRiskResult, predict_acne_risk

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    success: bool
    session_dir: str
    porphyrin_analysis: Dict[str, Any] = field(default_factory=dict)
    skin_score: Dict[str, Any] = field(default_factory=dict)
    acne_risk: Dict[str, Any] = field(default_factory=dict)
    pipeline_report_path: str = ""
    error: str = ""


def resolve_analysis_function() -> Callable[[str | Path], Dict[str, Any]]:
    """
    porphyrin_analysis.py 안의 실제 함수 이름을 유연하게 찾는다.
    우선순위:
    1) analyze_session_dir
    2) analyze_porphyrin_session
    3) run_analysis_for_session
    """
    import porphyrin_analysis as pa

    candidate_names = [
        "analyze_session_dir",
        "analyze_porphyrin_session",
        "run_analysis_for_session",
    ]

    for name in candidate_names:
        fn = getattr(pa, name, None)
        if callable(fn):
            logger.info("Using porphyrin analysis entrypoint: %s", name)
            return fn

    raise AttributeError(
        "porphyrin_analysis.py 안에서 사용할 분석 함수를 찾지 못했습니다. "
        "analyze_session_dir(session_dir) 형태의 함수를 하나 만들어주세요."
    )


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_porphyrin_result(raw: Any, session_dir: Path) -> Dict[str, Any]:
    """
    porphyrin_analysis.py가 반환하는 형식이 조금 달라도
    공통 키로 맞춰주는 함수
    """
    if raw is None:
        return {}

    if isinstance(raw, dict):
        result = dict(raw)
    else:
        try:
            result = asdict(raw)
        except Exception:
            result = raw.__dict__.copy() if hasattr(raw, "__dict__") else {}

    analysis_dir = session_dir / "analysis"
    report_json = analysis_dir / "report.json"

    if not result and report_json.exists():
        result = load_json_if_exists(report_json)

    result.setdefault("session_dir", str(session_dir))
    result.setdefault("porphyrin_count", 0)
    result.setdefault("porphyrin_area", 0)
    result.setdefault("mean_intensity", 0.0)
    result.setdefault("overlay_path", str(analysis_dir / "porphyrin_overlay.png"))
    result.setdefault("mask_path", str(analysis_dir / "porphyrin_mask.png"))
    result.setdefault("report_path", str(report_json))
    result.setdefault("regional_distribution", {})
    return result


def run_pipeline(session_dir: str | Path, save_report: bool = True) -> PipelineResult:
    session_path = Path(session_dir)

    if not session_path.exists():
        return PipelineResult(
            success=False,
            session_dir=str(session_path),
            error="session_dir가 존재하지 않습니다.",
        )

    try:
        analyze_fn = resolve_analysis_function()
        raw_porphyrin = analyze_fn(session_path)
        porphyrin_result = normalize_porphyrin_result(raw_porphyrin, session_path)

        skin_score_result: SkinScoreBreakdown = compute_skin_score(
            porphyrin_result=porphyrin_result,
            session_dir=session_path,
        )

        acne_risk_result: AcneRiskResult = predict_acne_risk(
            porphyrin_result=porphyrin_result,
            skin_score=skin_score_result,
            session_dir=session_path,
        )

        pipeline_result = PipelineResult(
            success=True,
            session_dir=str(session_path),
            porphyrin_analysis=porphyrin_result,
            skin_score=asdict(skin_score_result),
            acne_risk=asdict(acne_risk_result),
        )

        if save_report:
            analysis_dir = session_path / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            report_path = analysis_dir / "pipeline_report.json"
            report_path.write_text(
                json.dumps(asdict(pipeline_result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            pipeline_result.pipeline_report_path = str(report_path)

        return pipeline_result

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return PipelineResult(
            success=False,
            session_dir=str(session_path),
            error=str(exc),
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python3 analysis_pipeline.py <session_dir>")
        raise SystemExit(1)

    result = run_pipeline(sys.argv[1], save_report=True)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
