from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Any


# =========================================================
# 시간 변화 분석 기본 설정
# =========================================================
DEFAULT_OUTPUT_FOLDER_NAME = "trend_analysis"

# 점수 fallback용 기준값
# porphyrin_analysis_with_score.py와 같은 기준으로 맞춤
BAD_COUNT = 120
BAD_AREA_RATIO = 0.015
AVG_BRIGHTNESS_LOW = 120
AVG_BRIGHTNESS_HIGH = 230
MAX_BRIGHTNESS_LOW = 160
MAX_BRIGHTNESS_HIGH = 255
CONCENTRATION_SAFE = 0.35
CONCENTRATION_BAD = 0.65


# =========================================================
# 경로 탐색
# =========================================================
def find_default_base_path() -> Path:
    """
    cam4_660nm 기준 폴더 자동 탐색.
    라즈베리파이와 Windows 개발 환경을 모두 고려함.
    """
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


def find_metadata_files(base_path: Path) -> list[Path]:
    """
    base_path 아래에 있는 metadata.json 파일을 모두 찾음.

    예상 구조 예시:
    captures/sessions/cam4_660nm/2026-04-07/analysis/metadata.json

    또는 추후 개선 구조:
    captures/sessions/cam4_660nm/2026-04-07/analysis/20260407_224057_metadata.json
    """
    base_path = Path(base_path)

    if not base_path.exists():
        print("기준 폴더가 없습니다:", base_path)
        return []

    metadata_files = sorted(base_path.glob("**/metadata.json"))

    # 혹시 metadata_*.json 형태로 저장하는 경우까지 대응
    metadata_files.extend(sorted(base_path.glob("**/*metadata*.json")))

    # 중복 제거
    unique_files = []
    seen = set()

    for file in metadata_files:
        key = str(file.resolve())
        if key not in seen and file.is_file():
            unique_files.append(file)
            seen.add(key)

    return sorted(unique_files)


# =========================================================
# 파싱 유틸
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


def parse_datetime(value: Any, fallback_path: Path | None = None) -> datetime:
    """
    metadata의 captured_at을 datetime으로 변환.
    실패하면 파일 수정 시간을 사용.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            pass

    if fallback_path is not None and fallback_path.exists():
        return datetime.fromtimestamp(fallback_path.stat().st_mtime)

    return datetime.now()


def clamp_float(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def normalize_linear(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp_float((value - low) / (high - low))


def get_grade(score: int) -> str:
    if score >= 85:
        return "양호"
    if score >= 70:
        return "보통"
    if score >= 55:
        return "주의"
    return "위험"


def fallback_skin_score(
    porphyrin_count: int,
    area_ratio: float,
    avg_brightness: float,
    max_brightness: float,
    distribution: dict[str, int],
) -> dict[str, Any]:
    """
    기존 metadata에 skin_scoring이 없을 때를 위한 fallback 계산.
    porphyrin_analysis_with_score.py의 점수 기준과 동일한 방향.
    """
    total_distribution_count = sum(distribution.values()) if distribution else 0

    if total_distribution_count > 0:
        concentration_ratio = max(distribution.values()) / total_distribution_count
    else:
        concentration_ratio = 0.0

    count_risk = clamp_float(porphyrin_count / BAD_COUNT)
    area_risk = clamp_float(area_ratio / BAD_AREA_RATIO)
    avg_intensity_risk = normalize_linear(avg_brightness, AVG_BRIGHTNESS_LOW, AVG_BRIGHTNESS_HIGH)
    max_intensity_risk = normalize_linear(max_brightness, MAX_BRIGHTNESS_LOW, MAX_BRIGHTNESS_HIGH)
    concentration_risk = normalize_linear(concentration_ratio, CONCENTRATION_SAFE, CONCENTRATION_BAD)

    weighted_risk = (
        0.35 * count_risk +
        0.25 * area_risk +
        0.20 * avg_intensity_risk +
        0.10 * max_intensity_risk +
        0.10 * concentration_risk
    )

    score = int(round(100 * (1.0 - weighted_risk)))
    score = max(0, min(100, score))

    if distribution:
        main_region = max(distribution, key=distribution.get)
        main_region_count = distribution[main_region]
    else:
        main_region = "unknown"
        main_region_count = 0

    return {
        "skin_score": score,
        "grade": get_grade(score),
        "risk_components": {
            "count_risk": round(count_risk, 4),
            "area_risk": round(area_risk, 4),
            "avg_intensity_risk": round(avg_intensity_risk, 4),
            "max_intensity_risk": round(max_intensity_risk, 4),
            "concentration_risk": round(concentration_risk, 4),
            "weighted_risk": round(weighted_risk, 4),
        },
        "distribution_metrics": {
            "main_region": main_region,
            "main_region_count": main_region_count,
            "concentration_ratio": round(concentration_ratio, 4),
        },
        "scoring_note": "metadata에 skin_scoring이 없어 시간 변화 분석 코드에서 fallback으로 계산한 점수입니다.",
    }


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("[스킵] metadata 읽기 실패:", path, e)
        return None


def record_from_metadata(metadata_path: Path) -> dict[str, Any] | None:
    """
    porphyrin_analysis.py 또는 porphyrin_analysis_with_score.py의 metadata.json을
    시간 변화 분석용 record 한 줄로 변환.
    """
    meta = load_json(metadata_path)
    if meta is None:
        return None

    analysis_result = meta.get("analysis_result", {})

    captured_at = parse_datetime(
        meta.get("captured_at"),
        fallback_path=metadata_path
    )

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
    avg_brightness = safe_float(analysis_result.get("average_brightness", 0.0))
    max_brightness = safe_float(analysis_result.get("max_brightness", 0.0))

    total_area_px = safe_float(
        analysis_result.get("porphyrin_total_area_px", 0.0)
    )

    area_ratio = safe_float(
        analysis_result.get("porphyrin_area_ratio", 0.0)
    )

    skin_scoring = analysis_result.get("skin_scoring", None)

    if not isinstance(skin_scoring, dict):
        skin_scoring = fallback_skin_score(
            porphyrin_count=porphyrin_count,
            area_ratio=area_ratio,
            avg_brightness=avg_brightness,
            max_brightness=max_brightness,
            distribution=distribution,
        )

    skin_score = safe_int(skin_scoring.get("skin_score", 0))
    grade = str(skin_scoring.get("grade", get_grade(skin_score)))

    distribution_metrics = skin_scoring.get("distribution_metrics", {})
    if not isinstance(distribution_metrics, dict):
        distribution_metrics = {}

    main_region = str(distribution_metrics.get("main_region", "unknown"))
    concentration_ratio = safe_float(distribution_metrics.get("concentration_ratio", 0.0))

    saved_file = meta.get("saved_file", "")
    analysis_files = meta.get("analysis_files", {})
    compare_image = ""
    distribution_map = ""

    if isinstance(analysis_files, dict):
        compare_image = str(analysis_files.get("compare_image", ""))
        distribution_map = str(analysis_files.get("distribution_map", ""))

    return {
        "captured_at": captured_at.isoformat(),
        "date": captured_at.strftime("%Y-%m-%d"),
        "time": captured_at.strftime("%H:%M:%S"),
        "timestamp": captured_at.timestamp(),

        "porphyrin_count": porphyrin_count,
        "porphyrin_total_area_px": round(total_area_px, 2),
        "porphyrin_area_ratio": round(area_ratio, 8),
        "average_brightness": round(avg_brightness, 4),
        "max_brightness": round(max_brightness, 4),

        "skin_score": skin_score,
        "grade": grade,

        "main_region": main_region,
        "concentration_ratio": round(concentration_ratio, 4),

        "left_cheek": distribution["left_cheek"],
        "right_cheek": distribution["right_cheek"],
        "forehead": distribution["forehead"],
        "nose": distribution["nose"],
        "chin": distribution["chin"],

        "saved_file": str(saved_file),
        "metadata_file": str(metadata_path),
        "compare_image": compare_image,
        "distribution_map": distribution_map,
    }


# =========================================================
# 변화량 계산
# =========================================================
def percent_change(new_value: float, old_value: float) -> float | None:
    if old_value == 0:
        return None
    return ((new_value - old_value) / old_value) * 100.0


def trend_direction_score(score_delta: float) -> str:
    if score_delta >= 5:
        return "개선"
    if score_delta <= -5:
        return "악화"
    return "유지"


def trend_direction_count(count_delta: int) -> str:
    if count_delta <= -10:
        return "개선"
    if count_delta >= 10:
        return "악화"
    return "유지"


def make_latest_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) < 2:
        return {
            "available": False,
            "message": "비교할 기록이 2개 미만입니다."
        }

    prev = records[-2]
    latest = records[-1]

    score_delta = safe_float(latest["skin_score"]) - safe_float(prev["skin_score"])
    count_delta = safe_int(latest["porphyrin_count"]) - safe_int(prev["porphyrin_count"])
    area_delta = safe_float(latest["porphyrin_area_ratio"]) - safe_float(prev["porphyrin_area_ratio"])
    avg_brightness_delta = safe_float(latest["average_brightness"]) - safe_float(prev["average_brightness"])

    count_change_pct = percent_change(
        safe_float(latest["porphyrin_count"]),
        safe_float(prev["porphyrin_count"])
    )

    area_change_pct = percent_change(
        safe_float(latest["porphyrin_area_ratio"]),
        safe_float(prev["porphyrin_area_ratio"])
    )

    score_status = trend_direction_score(score_delta)
    count_status = trend_direction_count(count_delta)

    if score_status == "개선" and count_status != "악화":
        total_status = "개선"
    elif score_status == "악화" or count_status == "악화":
        total_status = "악화"
    else:
        total_status = "유지"

    return {
        "available": True,
        "previous_date": prev["captured_at"],
        "latest_date": latest["captured_at"],
        "status": total_status,
        "skin_score_delta": round(score_delta, 2),
        "porphyrin_count_delta": count_delta,
        "porphyrin_count_change_percent": None if count_change_pct is None else round(count_change_pct, 2),
        "porphyrin_area_ratio_delta": round(area_delta, 8),
        "porphyrin_area_ratio_change_percent": None if area_change_pct is None else round(area_change_pct, 2),
        "average_brightness_delta": round(avg_brightness_delta, 2),
        "previous_main_region": prev["main_region"],
        "latest_main_region": latest["main_region"],
    }


def make_first_latest_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) < 2:
        return {
            "available": False,
            "message": "비교할 기록이 2개 미만입니다."
        }

    first = records[0]
    latest = records[-1]

    score_delta = safe_float(latest["skin_score"]) - safe_float(first["skin_score"])
    count_delta = safe_int(latest["porphyrin_count"]) - safe_int(first["porphyrin_count"])

    count_change_pct = percent_change(
        safe_float(latest["porphyrin_count"]),
        safe_float(first["porphyrin_count"])
    )

    return {
        "available": True,
        "first_date": first["captured_at"],
        "latest_date": latest["captured_at"],
        "skin_score_delta": round(score_delta, 2),
        "porphyrin_count_delta": count_delta,
        "porphyrin_count_change_percent": None if count_change_pct is None else round(count_change_pct, 2),
        "status": trend_direction_score(score_delta),
    }


def make_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "record_count": 0,
            "message": "분석 기록이 없습니다."
        }

    scores = [safe_float(r["skin_score"]) for r in records]
    counts = [safe_float(r["porphyrin_count"]) for r in records]
    avg_brightness_values = [safe_float(r["average_brightness"]) for r in records]

    latest = records[-1]

    region_totals = {
        "left_cheek": sum(safe_int(r["left_cheek"]) for r in records),
        "right_cheek": sum(safe_int(r["right_cheek"]) for r in records),
        "forehead": sum(safe_int(r["forehead"]) for r in records),
        "nose": sum(safe_int(r["nose"]) for r in records),
        "chin": sum(safe_int(r["chin"]) for r in records),
    }

    most_frequent_region = max(region_totals, key=region_totals.get)

    latest_comparison = make_latest_comparison(records)
    first_latest_comparison = make_first_latest_comparison(records)

    recommendations = make_trend_recommendations(
        records=records,
        latest_comparison=latest_comparison,
        first_latest_comparison=first_latest_comparison,
        most_frequent_region=most_frequent_region
    )

    return {
        "record_count": len(records),
        "start_date": records[0]["captured_at"],
        "end_date": records[-1]["captured_at"],
        "latest": {
            "captured_at": latest["captured_at"],
            "skin_score": latest["skin_score"],
            "grade": latest["grade"],
            "porphyrin_count": latest["porphyrin_count"],
            "average_brightness": latest["average_brightness"],
            "main_region": latest["main_region"],
            "concentration_ratio": latest["concentration_ratio"],
        },
        "statistics": {
            "skin_score": {
                "min": round(min(scores), 2),
                "max": round(max(scores), 2),
                "mean": round(sum(scores) / len(scores), 2),
            },
            "porphyrin_count": {
                "min": round(min(counts), 2),
                "max": round(max(counts), 2),
                "mean": round(sum(counts) / len(counts), 2),
            },
            "average_brightness": {
                "min": round(min(avg_brightness_values), 2),
                "max": round(max(avg_brightness_values), 2),
                "mean": round(sum(avg_brightness_values) / len(avg_brightness_values), 2),
            },
        },
        "region_totals": region_totals,
        "most_frequent_region": most_frequent_region,
        "latest_vs_previous": latest_comparison,
        "first_vs_latest": first_latest_comparison,
        "recommendations": recommendations,
        "note": "시간 변화 분석은 동일 촬영 조건, 동일 노출, 동일 거리에서 반복 촬영할수록 신뢰도가 높아집니다."
    }


def make_trend_recommendations(
    records: list[dict[str, Any]],
    latest_comparison: dict[str, Any],
    first_latest_comparison: dict[str, Any],
    most_frequent_region: str
) -> list[str]:
    recommendations = []

    region_name_ko = {
        "left_cheek": "왼쪽 볼",
        "right_cheek": "오른쪽 볼",
        "forehead": "이마",
        "nose": "코",
        "chin": "턱",
        "unknown": "알 수 없음"
    }

    if len(records) < 2:
        recommendations.append("시간 변화 분석을 하려면 최소 2회 이상의 촬영/분석 기록이 필요합니다.")
        return recommendations

    if latest_comparison.get("available"):
        status = latest_comparison.get("status")
        score_delta = safe_float(latest_comparison.get("skin_score_delta", 0))
        count_delta = safe_int(latest_comparison.get("porphyrin_count_delta", 0))

        if status == "개선":
            recommendations.append(f"최근 촬영 대비 피부 점수가 {score_delta:+.1f}점 변화하여 개선 경향입니다.")
        elif status == "악화":
            recommendations.append(f"최근 촬영 대비 피부 점수가 {score_delta:+.1f}점 변화했고 포르피린 변화량은 {count_delta:+d}개입니다. 동일 조건 재촬영 후 확인이 필요합니다.")
        else:
            recommendations.append("최근 촬영 대비 큰 변화는 없습니다.")

    if first_latest_comparison.get("available"):
        long_score_delta = safe_float(first_latest_comparison.get("skin_score_delta", 0))
        long_count_delta = safe_int(first_latest_comparison.get("porphyrin_count_delta", 0))

        if abs(long_score_delta) >= 5:
            recommendations.append(f"전체 기간 기준 피부 점수 변화는 {long_score_delta:+.1f}점입니다.")
        if abs(long_count_delta) >= 10:
            recommendations.append(f"전체 기간 기준 포르피린 개수 변화는 {long_count_delta:+d}개입니다.")

    recommendations.append(f"누적 기준 포르피린이 가장 많이 나타난 부위는 {region_name_ko.get(most_frequent_region, most_frequent_region)}입니다.")

    latest = records[-1]
    if safe_float(latest.get("concentration_ratio", 0)) >= 0.65:
        recommendations.append("최근 결과에서 특정 부위 집중도가 높습니다. 부위별 분포 이미지를 함께 확인하세요.")

    return recommendations


# =========================================================
# 저장
# =========================================================
def write_records_csv(records: list[dict[str, Any]], csv_path: Path):
    if not records:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "captured_at",
        "date",
        "time",
        "porphyrin_count",
        "porphyrin_total_area_px",
        "porphyrin_area_ratio",
        "average_brightness",
        "max_brightness",
        "skin_score",
        "grade",
        "main_region",
        "concentration_ratio",
        "left_cheek",
        "right_cheek",
        "forehead",
        "nose",
        "chin",
        "saved_file",
        "metadata_file",
        "compare_image",
        "distribution_map",
    ]

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def write_summary_json(summary: dict[str, Any], json_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)


def make_optional_plots(records: list[dict[str, Any]], out_dir: Path):
    """
    matplotlib이 설치되어 있으면 추세 그래프 저장.
    설치되어 있지 않으면 그래프만 건너뜀.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[그래프 스킵] matplotlib이 설치되어 있지 않습니다.")
        print("설치하려면: python -m pip install matplotlib")
        return []

    if len(records) < 2:
        print("[그래프 스킵] 기록이 2개 미만입니다.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)

    x_labels = [r["date"] + "\n" + r["time"] for r in records]
    x = list(range(len(records)))

    generated = []

    def save_line_plot(values, title, ylabel, filename):
        plt.figure(figsize=(10, 5))
        plt.plot(x, values, marker="o")
        plt.xticks(x, x_labels, rotation=45, ha="right")
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel("Capture Time")
        plt.tight_layout()

        path = out_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        generated.append(str(path))

    save_line_plot(
        [safe_float(r["skin_score"]) for r in records],
        "Skin Score Trend",
        "Score",
        "skin_score_trend.png"
    )

    save_line_plot(
        [safe_float(r["porphyrin_count"]) for r in records],
        "Porphyrin Count Trend",
        "Count",
        "porphyrin_count_trend.png"
    )

    save_line_plot(
        [safe_float(r["average_brightness"]) for r in records],
        "Average Brightness Trend",
        "Brightness",
        "average_brightness_trend.png"
    )

    return generated


# =========================================================
# 메인 분석 함수
# =========================================================
def analyze_time_trend(
    base_path: Path,
    output_dir: Path | None = None,
    make_plots: bool = True
) -> dict[str, Any]:
    base_path = Path(base_path)

    if output_dir is None:
        output_dir = base_path / DEFAULT_OUTPUT_FOLDER_NAME
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_files = find_metadata_files(base_path)

    print("기준 폴더:", base_path)
    print("발견한 metadata 파일 수:", len(metadata_files))

    records = []

    for metadata_path in metadata_files:
        record = record_from_metadata(metadata_path)
        if record is not None:
            records.append(record)

    # 같은 metadata가 중복 검색될 수 있으므로 metadata_file 기준 중복 제거
    unique = {}
    for record in records:
        unique[record["metadata_file"]] = record

    records = list(unique.values())
    records.sort(key=lambda r: safe_float(r["timestamp"]))

    # timestamp는 내부 정렬용이라 저장 전 제거
    for record in records:
        record.pop("timestamp", None)

    summary = make_summary(records)

    csv_path = output_dir / "trend_records.csv"
    summary_path = output_dir / "trend_summary.json"

    write_records_csv(records, csv_path)
    write_summary_json(summary, summary_path)

    generated_plots = []
    if make_plots:
        generated_plots = make_optional_plots(records, output_dir)

    summary["output_files"] = {
        "records_csv": str(csv_path),
        "summary_json": str(summary_path),
        "plots": generated_plots,
    }

    # 그래프 목록까지 포함해서 summary 재저장
    write_summary_json(summary, summary_path)

    print("\n시간 변화 분석 완료")
    print("CSV:", csv_path)
    print("요약 JSON:", summary_path)

    if generated_plots:
        print("그래프:")
        for path in generated_plots:
            print("-", path)

    print("\n요약:")
    print_summary_to_console(summary)

    return {
        "records": records,
        "summary": summary,
    }


def print_summary_to_console(summary: dict[str, Any]):
    if summary.get("record_count", 0) == 0:
        print(summary.get("message", "기록 없음"))
        return

    latest = summary["latest"]
    print(f"- 기록 수: {summary['record_count']}")
    print(f"- 기간: {summary['start_date']} ~ {summary['end_date']}")
    print(f"- 최근 피부 점수: {latest['skin_score']}점 ({latest['grade']})")
    print(f"- 최근 포르피린 개수: {latest['porphyrin_count']}개")
    print(f"- 최근 주요 부위: {latest['main_region']}")

    latest_vs_previous = summary.get("latest_vs_previous", {})
    if latest_vs_previous.get("available"):
        print(f"- 직전 대비 상태: {latest_vs_previous.get('status')}")
        print(f"- 직전 대비 점수 변화: {latest_vs_previous.get('skin_score_delta'):+.1f}점")
        print(f"- 직전 대비 포르피린 개수 변화: {latest_vs_previous.get('porphyrin_count_delta'):+d}개")

    print("- 권장/해석:")
    for rec in summary.get("recommendations", []):
        print("  ·", rec)


# =========================================================
# 실행
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="cam4_660nm 기준 폴더. 예: ~/Graduate_Project/captures/sessions/cam4_660nm"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="시간 변화 분석 결과 저장 폴더"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="matplotlib 그래프 생성을 하지 않음"
    )

    args = parser.parse_args()

    base = Path(args.base).expanduser() if args.base else find_default_base_path()
    out = Path(args.out).expanduser() if args.out else None

    analyze_time_trend(
        base_path=base,
        output_dir=out,
        make_plots=not args.no_plots
    )
