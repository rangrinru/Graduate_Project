from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any


# =========================================================
# 여드름 발생 위험도 예측 기준값
# 실제 촬영 데이터가 쌓이면 이 기준값은 조정하는 것이 좋음
# =========================================================
BAD_COUNT = 120
BAD_AREA_RATIO = 0.015

AVG_BRIGHTNESS_LOW = 120
AVG_BRIGHTNESS_HIGH = 230

MAX_BRIGHTNESS_LOW = 160
MAX_BRIGHTNESS_HIGH = 255

T_ZONE_SAFE = 0.45
T_ZONE_BAD = 0.75

CONCENTRATION_SAFE = 0.35
CONCENTRATION_BAD = 0.65

ASYMMETRY_SAFE = 0.15
ASYMMETRY_BAD = 0.45

# 시간 변화 반영값
COUNT_INCREASE_BAD = 30
SCORE_DROP_BAD = -15


# =========================================================
# 경로 자동 탐색
# =========================================================
def find_default_base_path() -> Path:
    candidates = [
        Path.home() / "Graduate_Project" / "captures" / "sessions" / "cam4_660nm",
        Path.home() / "Graduate_Project" / "captures" / "cam4_660nm",
        Path.cwd() / "captures" / "sessions" / "cam4_660nm",
        Path.cwd() / "captures" / "cam4_660nm",
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\captures\sessions\cam4_660nm"),
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\captures\cam4_660nm"),
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def find_latest_metadata(base_path: Path) -> Path | None:
    base_path = Path(base_path)

    if not base_path.exists():
        print("기준 폴더가 없습니다:", base_path)
        return None

    metadata_files = list(base_path.glob("**/metadata.json"))
    metadata_files.extend(list(base_path.glob("**/*metadata*.json")))

    metadata_files = sorted(
        set(metadata_files),
        key=lambda p: p.stat().st_mtime if p.exists() else 0
    )

    if not metadata_files:
        print("metadata.json 파일을 찾지 못했습니다:", base_path)
        return None

    return metadata_files[-1]


def find_default_trend_summary(base_path: Path) -> Path | None:
    candidates = [
        Path(base_path) / "trend_analysis" / "trend_summary.json",
        Path(base_path).parent / "trend_analysis" / "trend_summary.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    found = list(Path(base_path).glob("**/trend_summary.json"))
    if found:
        return sorted(found, key=lambda p: p.stat().st_mtime)[-1]

    return None


# =========================================================
# JSON 유틸
# =========================================================
def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("JSON 읽기 실패:", path, e)
        return None


def save_json(path: Path, data: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_text_report(path: Path, result: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("[여드름 발생 위험도 예측 결과]")
    lines.append("")
    lines.append(f"분석 시각: {result.get('analyzed_at')}")
    lines.append(f"사용 metadata: {result.get('metadata_file')}")
    lines.append("")
    lines.append(f"위험도 점수: {result['acne_risk']['risk_score']} / 100")
    lines.append(f"위험도 등급: {result['acne_risk']['risk_level']}")
    lines.append(f"상태 요약: {result['acne_risk']['summary']}")
    lines.append("")
    lines.append("[주요 지표]")
    metrics = result.get("input_metrics", {})
    lines.append(f"- 포르피린 개수: {metrics.get('porphyrin_count')}")
    lines.append(f"- 포르피린 면적 비율: {metrics.get('porphyrin_area_ratio')}")
    lines.append(f"- 평균 밝기: {metrics.get('average_brightness')}")
    lines.append(f"- 최대 밝기: {metrics.get('max_brightness')}")
    lines.append(f"- T존 비율: {metrics.get('t_zone_ratio')}")
    lines.append(f"- 부위 집중도: {metrics.get('concentration_ratio')}")
    lines.append(f"- 좌우 볼 비대칭도: {metrics.get('cheek_asymmetry')}")
    lines.append("")
    lines.append("[위험 구성 요소]")
    for key, value in result.get("risk_components", {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("[권장 사항]")
    for rec in result.get("recommendations", []):
        lines.append(f"- {rec}")

    path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# 숫자 유틸
# =========================================================
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def clamp_float(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def normalize_linear(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp_float((value - low) / (high - low))


def percent_or_zero(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


# =========================================================
# metadata에서 지표 추출
# =========================================================
def extract_analysis_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    analysis_result = metadata.get("analysis_result", {})
    if not isinstance(analysis_result, dict):
        analysis_result = {}

    distribution = analysis_result.get("distribution", {})
    if not isinstance(distribution, dict):
        distribution = {}

    distribution = {
        "left_cheek": safe_int(distribution.get("left_cheek", 0)),
        "right_cheek": safe_int(distribution.get("right_cheek", 0)),
        "forehead": safe_int(distribution.get("forehead", 0)),
        "nose": safe_int(distribution.get("nose", 0)),
        "chin": safe_int(distribution.get("chin", 0)),
    }

    porphyrin_count = safe_int(analysis_result.get("porphyrin_count", 0))
    average_brightness = safe_float(analysis_result.get("average_brightness", 0.0))
    max_brightness = safe_float(analysis_result.get("max_brightness", 0.0))

    area_ratio = safe_float(analysis_result.get("porphyrin_area_ratio", 0.0))
    total_area_px = safe_float(analysis_result.get("porphyrin_total_area_px", 0.0))

    skin_scoring = analysis_result.get("skin_scoring", {})
    if not isinstance(skin_scoring, dict):
        skin_scoring = {}

    skin_score = safe_int(skin_scoring.get("skin_score", 0), default=0)
    skin_grade = str(skin_scoring.get("grade", ""))

    distribution_metrics = skin_scoring.get("distribution_metrics", {})
    if not isinstance(distribution_metrics, dict):
        distribution_metrics = {}

    concentration_ratio = safe_float(distribution_metrics.get("concentration_ratio", 0.0))

    distribution_total = sum(distribution.values())

    if concentration_ratio == 0.0 and distribution_total > 0:
        concentration_ratio = max(distribution.values()) / distribution_total

    t_zone_count = distribution["forehead"] + distribution["nose"] + distribution["chin"]
    t_zone_ratio = percent_or_zero(t_zone_count, distribution_total)

    cheek_total = distribution["left_cheek"] + distribution["right_cheek"]
    cheek_asymmetry = 0.0
    if cheek_total > 0:
        cheek_asymmetry = abs(distribution["left_cheek"] - distribution["right_cheek"]) / cheek_total

    if distribution_total > 0:
        main_region = max(distribution, key=distribution.get)
        main_region_count = distribution[main_region]
    else:
        main_region = "unknown"
        main_region_count = 0

    return {
        "porphyrin_count": porphyrin_count,
        "porphyrin_total_area_px": total_area_px,
        "porphyrin_area_ratio": area_ratio,
        "average_brightness": average_brightness,
        "max_brightness": max_brightness,
        "distribution": distribution,
        "distribution_total": distribution_total,
        "main_region": main_region,
        "main_region_count": main_region_count,
        "concentration_ratio": concentration_ratio,
        "t_zone_count": t_zone_count,
        "t_zone_ratio": t_zone_ratio,
        "cheek_asymmetry": cheek_asymmetry,
        "skin_score": skin_score,
        "skin_grade": skin_grade,
    }


def extract_trend_metrics(trend_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not trend_summary:
        return {
            "has_trend": False,
            "latest_status": "unknown",
            "skin_score_delta": 0.0,
            "porphyrin_count_delta": 0,
            "porphyrin_count_change_percent": None,
        }

    latest_vs_previous = trend_summary.get("latest_vs_previous", {})
    if not isinstance(latest_vs_previous, dict) or not latest_vs_previous.get("available"):
        return {
            "has_trend": False,
            "latest_status": "unknown",
            "skin_score_delta": 0.0,
            "porphyrin_count_delta": 0,
            "porphyrin_count_change_percent": None,
        }

    return {
        "has_trend": True,
        "latest_status": str(latest_vs_previous.get("status", "unknown")),
        "skin_score_delta": safe_float(latest_vs_previous.get("skin_score_delta", 0.0)),
        "porphyrin_count_delta": safe_int(latest_vs_previous.get("porphyrin_count_delta", 0)),
        "porphyrin_count_change_percent": latest_vs_previous.get("porphyrin_count_change_percent"),
    }


# =========================================================
# 여드름 위험도 계산
# =========================================================
def get_risk_level(score: int) -> str:
    if score < 30:
        return "낮음"
    if score < 55:
        return "보통"
    if score < 75:
        return "높음"
    return "매우 높음"


def get_risk_summary(level: str) -> str:
    if level == "낮음":
        return "현재 기준에서는 여드름 발생 위험 신호가 낮습니다."
    if level == "보통":
        return "일부 포르피린 지표가 관찰되어 주기적 관찰이 필요합니다."
    if level == "높음":
        return "포르피린 지표가 높아 여드름 발생 가능성이 높은 패턴입니다."
    return "포르피린 개수, 강도 또는 분포 집중도가 높아 집중 관리가 필요한 상태입니다."


def calculate_acne_risk(metrics: dict[str, Any], trend: dict[str, Any] | None = None) -> dict[str, Any]:
    if trend is None:
        trend = extract_trend_metrics(None)

    porphyrin_count = safe_int(metrics.get("porphyrin_count", 0))
    area_ratio = safe_float(metrics.get("porphyrin_area_ratio", 0.0))
    average_brightness = safe_float(metrics.get("average_brightness", 0.0))
    max_brightness = safe_float(metrics.get("max_brightness", 0.0))
    t_zone_ratio = safe_float(metrics.get("t_zone_ratio", 0.0))
    concentration_ratio = safe_float(metrics.get("concentration_ratio", 0.0))
    cheek_asymmetry = safe_float(metrics.get("cheek_asymmetry", 0.0))

    count_risk = clamp_float(porphyrin_count / BAD_COUNT)

    # 기존 porphyrin_analysis.py는 area_ratio가 없을 수 있음.
    # area_ratio가 없으면 면적 위험도는 0으로 두고 나머지 지표 중심으로 판단.
    area_risk = clamp_float(area_ratio / BAD_AREA_RATIO) if area_ratio > 0 else 0.0

    avg_intensity_risk = normalize_linear(average_brightness, AVG_BRIGHTNESS_LOW, AVG_BRIGHTNESS_HIGH)
    max_intensity_risk = normalize_linear(max_brightness, MAX_BRIGHTNESS_LOW, MAX_BRIGHTNESS_HIGH)
    t_zone_risk = normalize_linear(t_zone_ratio, T_ZONE_SAFE, T_ZONE_BAD)
    concentration_risk = normalize_linear(concentration_ratio, CONCENTRATION_SAFE, CONCENTRATION_BAD)
    asymmetry_risk = normalize_linear(cheek_asymmetry, ASYMMETRY_SAFE, ASYMMETRY_BAD)

    count_delta = safe_int(trend.get("porphyrin_count_delta", 0))
    score_delta = safe_float(trend.get("skin_score_delta", 0.0))

    count_increase_risk = clamp_float(count_delta / COUNT_INCREASE_BAD) if count_delta > 0 else 0.0
    score_drop_risk = clamp_float(abs(score_delta) / abs(SCORE_DROP_BAD)) if score_delta < 0 else 0.0

    trend_risk = max(count_increase_risk, score_drop_risk)

    # 면적 지표가 있는 경우와 없는 경우 가중치를 다르게 적용
    if area_ratio > 0:
        weighted_risk = (
            0.30 * count_risk +
            0.18 * area_risk +
            0.18 * avg_intensity_risk +
            0.10 * max_intensity_risk +
            0.10 * t_zone_risk +
            0.07 * concentration_risk +
            0.03 * asymmetry_risk +
            0.04 * trend_risk
        )
    else:
        weighted_risk = (
            0.38 * count_risk +
            0.22 * avg_intensity_risk +
            0.12 * max_intensity_risk +
            0.13 * t_zone_risk +
            0.08 * concentration_risk +
            0.03 * asymmetry_risk +
            0.04 * trend_risk
        )

    risk_score = int(round(100 * clamp_float(weighted_risk)))
    risk_level = get_risk_level(risk_score)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "summary": get_risk_summary(risk_level),
        "risk_components": {
            "count_risk": round(count_risk, 4),
            "area_risk": round(area_risk, 4),
            "avg_intensity_risk": round(avg_intensity_risk, 4),
            "max_intensity_risk": round(max_intensity_risk, 4),
            "t_zone_risk": round(t_zone_risk, 4),
            "concentration_risk": round(concentration_risk, 4),
            "asymmetry_risk": round(asymmetry_risk, 4),
            "trend_risk": round(trend_risk, 4),
            "weighted_risk": round(weighted_risk, 4),
        }
    }


def make_recommendations(metrics: dict[str, Any], acne_risk: dict[str, Any], trend: dict[str, Any]) -> list[str]:
    recommendations = []

    region_name_ko = {
        "left_cheek": "왼쪽 볼",
        "right_cheek": "오른쪽 볼",
        "forehead": "이마",
        "nose": "코",
        "chin": "턱",
        "unknown": "알 수 없음",
    }

    level = acne_risk["risk_level"]
    risk_score = acne_risk["risk_score"]

    main_region = metrics.get("main_region", "unknown")
    t_zone_ratio = safe_float(metrics.get("t_zone_ratio", 0.0))
    concentration_ratio = safe_float(metrics.get("concentration_ratio", 0.0))
    cheek_asymmetry = safe_float(metrics.get("cheek_asymmetry", 0.0))
    average_brightness = safe_float(metrics.get("average_brightness", 0.0))
    porphyrin_count = safe_int(metrics.get("porphyrin_count", 0))

    if porphyrin_count == 0:
        recommendations.append("검출된 포르피린이 없습니다. 단, 노출 부족 또는 필터/조명 조건 문제로 0개가 나왔는지 확인하세요.")
        return recommendations

    recommendations.append(f"가장 많이 검출된 부위는 {region_name_ko.get(main_region, main_region)}입니다.")

    if level in ["높음", "매우 높음"]:
        recommendations.append("여드름 위험도가 높게 계산되었습니다. 동일한 조명·거리·노출 조건에서 재촬영하여 결과를 확인하세요.")

    if t_zone_ratio >= 0.75:
        recommendations.append("이마·코·턱으로 이어지는 T존 비율이 높습니다. 피지 분비와 모공 관리 지표로 관찰하는 것이 좋습니다.")
    elif t_zone_ratio >= 0.45:
        recommendations.append("T존에 포르피린이 어느 정도 분포합니다. 시간 변화 분석에서 T존 변화량을 같이 확인하세요.")

    if concentration_ratio >= 0.65:
        recommendations.append("특정 부위 집중도가 높습니다. 해당 부위의 포르피린 히트맵을 확인하세요.")

    if cheek_asymmetry >= 0.45:
        recommendations.append("좌우 볼의 포르피린 분포 차이가 큽니다. 촬영 각도 차이인지 실제 피부 상태 차이인지 재촬영으로 확인하세요.")

    if average_brightness >= 200:
        recommendations.append("평균 형광 강도가 높습니다. 밝게 반응하는 포르피린 영역이 많을 수 있습니다.")

    if trend.get("has_trend"):
        count_delta = safe_int(trend.get("porphyrin_count_delta", 0))
        score_delta = safe_float(trend.get("skin_score_delta", 0.0))

        if count_delta > 0:
            recommendations.append(f"직전 촬영 대비 포르피린 개수가 {count_delta:+d}개 증가했습니다.")
        elif count_delta < 0:
            recommendations.append(f"직전 촬영 대비 포르피린 개수가 {count_delta:+d}개 감소했습니다.")

        if score_delta < 0:
            recommendations.append(f"직전 촬영 대비 피부 점수가 {score_delta:+.1f}점 감소했습니다.")
        elif score_delta > 0:
            recommendations.append(f"직전 촬영 대비 피부 점수가 {score_delta:+.1f}점 증가했습니다.")

    if risk_score < 30:
        recommendations.append("현재 기준에서는 위험도가 낮습니다. 주기적으로 같은 조건에서 촬영하여 기준 데이터를 쌓으세요.")

    recommendations.append("이 결과는 포르피린 영상 기반 규칙 점수이며 의료 진단이 아닙니다.")

    return recommendations


# =========================================================
# 실행 함수
# =========================================================
def predict_acne_risk(
    metadata_path: Path,
    trend_summary_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any] | None:
    metadata_path = Path(metadata_path)

    metadata = load_json(metadata_path)
    if metadata is None:
        return None

    trend_summary = None
    if trend_summary_path is not None and Path(trend_summary_path).exists():
        trend_summary = load_json(Path(trend_summary_path))

    metrics = extract_analysis_metrics(metadata)
    trend = extract_trend_metrics(trend_summary)

    acne_risk = calculate_acne_risk(metrics, trend)
    recommendations = make_recommendations(metrics, acne_risk, trend)

    if output_dir is None:
        output_dir = metadata_path.parent
    else:
        output_dir = Path(output_dir)

    result = {
        "analyzed_at": datetime.now().isoformat(),
        "analysis_type": "acne_risk_prediction_rule_based",
        "metadata_file": str(metadata_path),
        "trend_summary_file": str(trend_summary_path) if trend_summary_path else None,
        "input_metrics": {
            "porphyrin_count": metrics["porphyrin_count"],
            "porphyrin_total_area_px": metrics["porphyrin_total_area_px"],
            "porphyrin_area_ratio": metrics["porphyrin_area_ratio"],
            "average_brightness": metrics["average_brightness"],
            "max_brightness": metrics["max_brightness"],
            "distribution": metrics["distribution"],
            "main_region": metrics["main_region"],
            "main_region_count": metrics["main_region_count"],
            "t_zone_count": metrics["t_zone_count"],
            "t_zone_ratio": round(metrics["t_zone_ratio"], 4),
            "concentration_ratio": round(metrics["concentration_ratio"], 4),
            "cheek_asymmetry": round(metrics["cheek_asymmetry"], 4),
            "skin_score": metrics["skin_score"],
            "skin_grade": metrics["skin_grade"],
            "trend": trend,
        },
        "acne_risk": acne_risk,
        "risk_components": acne_risk["risk_components"],
        "recommendations": recommendations,
        "note": "포르피린 형광 기반 규칙 예측입니다. 의료 진단 목적이 아니라 졸업작품의 정량 지표로 사용하세요."
    }

    json_path = output_dir / "acne_risk_result.json"
    txt_path = output_dir / "acne_risk_report.txt"

    save_json(json_path, result)
    save_text_report(txt_path, result)

    result["output_files"] = {
        "json": str(json_path),
        "text_report": str(txt_path),
    }

    # output_files까지 포함해서 다시 저장
    save_json(json_path, result)

    print_acne_risk_result(result)

    return result


def print_acne_risk_result(result: dict[str, Any]):
    print("\n여드름 발생 위험도 예측 완료")
    print("위험도 점수:", result["acne_risk"]["risk_score"], "/ 100")
    print("위험도 등급:", result["acne_risk"]["risk_level"])
    print("요약:", result["acne_risk"]["summary"])

    print("\n주요 지표")
    metrics = result["input_metrics"]
    print("- 포르피린 개수:", metrics["porphyrin_count"])
    print("- 평균 밝기:", metrics["average_brightness"])
    print("- 최대 밝기:", metrics["max_brightness"])
    print("- 주요 부위:", metrics["main_region"])
    print("- T존 비율:", metrics["t_zone_ratio"])
    print("- 부위 집중도:", metrics["concentration_ratio"])
    print("- 좌우 볼 비대칭도:", metrics["cheek_asymmetry"])

    print("\n권장/해석")
    for rec in result["recommendations"]:
        print("-", rec)

    print("\n저장 파일")
    for _, path in result["output_files"].items():
        print("-", path)


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="분석할 metadata.json 경로"
    )
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="cam4_660nm 기준 폴더. metadata를 자동으로 찾을 때 사용"
    )
    parser.add_argument(
        "--trend-summary",
        type=str,
        default=None,
        help="trend_summary.json 경로. 없으면 자동 탐색"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="여드름 위험도 결과 저장 폴더"
    )

    args = parser.parse_args()

    if args.metadata:
        metadata = Path(args.metadata).expanduser()
        base_path = metadata.parents[1] if len(metadata.parents) > 1 else metadata.parent
    else:
        base_path = Path(args.base).expanduser() if args.base else find_default_base_path()
        metadata = find_latest_metadata(base_path)

    if metadata is None:
        raise RuntimeError("분석할 metadata.json을 찾지 못했습니다.")

    if args.trend_summary:
        trend_summary = Path(args.trend_summary).expanduser()
    else:
        trend_summary = find_default_trend_summary(base_path)

    output_dir = Path(args.out).expanduser() if args.out else metadata.parent

    predict_acne_risk(
        metadata_path=metadata,
        trend_summary_path=trend_summary,
        output_dir=output_dir
    )
