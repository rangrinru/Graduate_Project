from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any


# =========================================================
# 포르피린 정도별 기준
# =========================================================
# 이 기준은 졸업작품용 규칙 기반 기준입니다.
# 실제 촬영 조건, 노출, 필터, 얼굴 거리, 임계값에 따라 조정하세요.

COUNT_LEVELS = [
    {
        "min": 0,
        "max": 0,
        "level": "없음",
        "severity_score": 0,
        "summary": "해당 부위에서 포르피린이 거의 검출되지 않았습니다.",
        "tone": "good",
    },
    {
        "min": 1,
        "max": 10,
        "level": "매우 양호",
        "severity_score": 5,
        "summary": "해당 부위의 포르피린 검출량이 매우 적습니다.",
        "tone": "good",
    },
    {
        "min": 11,
        "max": 20,
        "level": "양호",
        "severity_score": 18,
        "summary": "해당 부위에 포르피린이 소량 검출되었습니다.",
        "tone": "normal",
    },
    {
        "min": 21,
        "max": 40,
        "level": "보통",
        "severity_score": 35,
        "summary": "해당 부위에 포르피린이 보통 수준으로 검출되었습니다.",
        "tone": "caution",
    },
    {
        "min": 41,
        "max": 70,
        "level": "주의",
        "severity_score": 60,
        "summary": "해당 부위에 포르피린이 다소 많이 집중되어 있습니다.",
        "tone": "warning",
    },
    {
        "min": 71,
        "max": 100,
        "level": "높음",
        "severity_score": 78,
        "summary": "해당 부위의 포르피린 검출량이 높은 편입니다.",
        "tone": "danger",
    },
    {
        "min": 101,
        "max": 10**9,
        "level": "매우 높음",
        "severity_score": 92,
        "summary": "해당 부위의 포르피린 검출량이 매우 높습니다.",
        "tone": "critical",
    },
]


REGION_KO = {
    "forehead": "이마",
    "nose": "코",
    "chin": "턱",
    "left_cheek": "왼쪽 볼",
    "right_cheek": "오른쪽 볼",
    "unknown": "알 수 없음",
}


# =========================================================
# 부위별 관리 솔루션
# =========================================================
# 각 부위의 포르피린 개수 단계에 따라 기본 관리 + 부위별 관리 문구를 조합합니다.

REGION_BASE_SOLUTIONS = {
    "forehead": {
        "cause": [
            "이마는 피지, 땀, 앞머리, 헤어 제품의 영향을 받기 쉬운 부위입니다.",
            "운동 후 땀을 오래 방치하거나 헤어 왁스·오일이 닿으면 포르피린 신호가 증가할 수 있습니다."
        ],
        "low": [
            "앞머리와 헤어 제품이 이마에 많이 닿지 않게 관리하세요.",
            "운동 후에는 이마의 땀과 피지를 가볍게 제거하세요."
        ],
        "medium": [
            "이마 라인과 헤어라인 주변을 세안할 때 충분히 헹구세요.",
            "헤어 제품 사용량을 줄이거나, 얼굴에 닿지 않도록 사용 위치를 조절하세요.",
            "땀이 많은 날에는 세안 전까지 손수건이나 티슈로 가볍게 눌러 관리하세요."
        ],
        "high": [
            "헤어라인 주변 포르피린 집중이 반복되면 헤어 제품 사용을 줄여보세요.",
            "이마 부위에 유분감이 많은 제품을 겹쳐 바르는 습관을 점검하세요.",
            "반복적으로 높은 수치가 나오면 동일 조건에서 재촬영해 증가 추세를 확인하세요."
        ]
    },
    "nose": {
        "cause": [
            "코는 피지 분비가 많아 포르피린이 가장 자주 집중되는 부위입니다.",
            "블랙헤드, 피지, 모공 주변 잔여물이 많으면 코 주변 신호가 높게 나타날 수 있습니다."
        ],
        "low": [
            "코 주변은 세안 시 가볍게 원을 그리듯 부드럽게 닦아주세요.",
            "무리한 코팩이나 강한 압출보다 꾸준한 세안 관리가 좋습니다."
        ],
        "medium": [
            "저녁 세안 시 코 옆, 콧망울, 코 아래 부분을 조금 더 꼼꼼히 관리하세요.",
            "유분감이 많은 선크림이나 베이스 제품이 코 주변에 남지 않도록 충분히 헹구세요.",
            "피지 제거 제품을 사용할 경우 자극이 적은 제품부터 시도하세요."
        ],
        "high": [
            "코 부위 포르피린이 높게 반복되면 피지 관리 루틴을 강화하세요.",
            "강한 스크럽, 잦은 코팩, 손으로 짜는 행동은 피하세요.",
            "코 주변 붉은기나 염증성 여드름이 함께 있으면 피부과 상담을 고려하세요."
        ]
    },
    "chin": {
        "cause": [
            "턱은 마스크, 손 접촉, 면도 자극, 침구류 접촉의 영향을 받기 쉬운 부위입니다.",
            "턱 주변 포르피린은 생활 습관과 반복 접촉에 의해 집중될 수 있습니다."
        ],
        "low": [
            "턱을 손으로 괴거나 만지는 습관을 줄이세요.",
            "마스크 착용 후에는 턱 주변을 깨끗하게 관리하세요."
        ],
        "medium": [
            "마스크를 오래 착용한 날에는 턱과 입 주변 세안을 더 신경 쓰세요.",
            "면도 후 자극이 남지 않게 보습을 충분히 해주세요.",
            "턱 주변에 유분감이 많은 제품이 쌓이지 않도록 관리하세요."
        ],
        "high": [
            "턱 부위 포르피린이 높다면 마스크 교체 주기와 손 접촉 습관을 점검하세요.",
            "면도 도구를 청결하게 유지하고, 면도 방향과 압력을 줄이세요.",
            "턱 주변 염증성 트러블이 반복되면 피부과 상담을 고려하세요."
        ]
    },
    "left_cheek": {
        "cause": [
            "왼쪽 볼은 베개, 휴대폰, 손 접촉, 마스크 마찰의 영향을 받을 수 있습니다.",
            "한쪽 볼에만 집중되면 실제 피부 차이뿐 아니라 촬영 각도 차이도 확인해야 합니다."
        ],
        "low": [
            "베개 커버와 수건을 주기적으로 교체하세요.",
            "휴대폰 화면이 볼에 닿는 습관을 줄이세요."
        ],
        "medium": [
            "왼쪽 볼에 닿는 침구류, 휴대폰, 손 접촉 습관을 점검하세요.",
            "마스크가 한쪽 볼에 더 강하게 닿지 않는지 확인하세요.",
            "볼 부위는 과한 문지름보다 순한 세안과 보습을 유지하세요."
        ],
        "high": [
            "왼쪽 볼 포르피린 집중이 반복되면 베개 커버 교체 주기를 줄이세요.",
            "휴대폰 통화 습관이나 손으로 얼굴을 만지는 습관을 적극적으로 줄이세요.",
            "좌우 차이가 지속되면 같은 위치·각도로 재촬영해 실제 차이인지 확인하세요."
        ]
    },
    "right_cheek": {
        "cause": [
            "오른쪽 볼은 베개, 휴대폰, 손 접촉, 마스크 마찰의 영향을 받을 수 있습니다.",
            "한쪽 볼에만 집중되면 실제 피부 차이뿐 아니라 촬영 각도 차이도 확인해야 합니다."
        ],
        "low": [
            "베개 커버와 수건을 주기적으로 교체하세요.",
            "휴대폰 화면이 볼에 닿는 습관을 줄이세요."
        ],
        "medium": [
            "오른쪽 볼에 닿는 침구류, 휴대폰, 손 접촉 습관을 점검하세요.",
            "마스크가 한쪽 볼에 더 강하게 닿지 않는지 확인하세요.",
            "볼 부위는 과한 문지름보다 순한 세안과 보습을 유지하세요."
        ],
        "high": [
            "오른쪽 볼 포르피린 집중이 반복되면 베개 커버 교체 주기를 줄이세요.",
            "휴대폰 통화 습관이나 손으로 얼굴을 만지는 습관을 적극적으로 줄이세요.",
            "좌우 차이가 지속되면 같은 위치·각도로 재촬영해 실제 차이인지 확인하세요."
        ]
    },
}


OVERALL_GUIDE = {
    "good": [
        "현재 상태는 양호한 편입니다. 기존 세안 루틴을 유지하세요.",
        "과도한 세안보다 아침·저녁 규칙적인 세안과 보습을 유지하는 것이 좋습니다."
    ],
    "normal": [
        "큰 문제는 아니지만 세안 후 잔여물이 남지 않도록 충분히 헹구세요.",
        "운동이나 외출 후 땀과 피지가 오래 남지 않게 관리하세요."
    ],
    "caution": [
        "피지 관리와 생활 습관 점검이 필요합니다.",
        "베개 커버, 마스크, 휴대폰 등 얼굴에 닿는 물건의 청결을 확인하세요."
    ],
    "warning": [
        "부위별 포르피린 집중 관리가 필요합니다.",
        "동일 조건으로 3~7일 간격 재촬영하여 증가/감소 추세를 확인하세요."
    ],
    "danger": [
        "포르피린 신호가 높은 편이므로 세안, 화장품, 수면 습관을 적극적으로 조정하세요.",
        "염증성 여드름이나 붉은기가 동반되면 피부과 상담을 고려하세요."
    ],
    "critical": [
        "먼저 같은 조건에서 재촬영해 과노출이나 반사광으로 인한 과검출이 아닌지 확인하세요.",
        "반복적으로 매우 높은 수치가 나오면 전문가 상담을 권장합니다."
    ],
}


# =========================================================
# 유틸
# =========================================================
def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def find_count_level(count: int) -> dict[str, Any]:
    for item in COUNT_LEVELS:
        if item["min"] <= count <= item["max"]:
            return item
    return COUNT_LEVELS[-1]


def level_to_solution_bucket(level: str) -> str:
    if level in ["없음", "매우 양호"]:
        return "low"
    if level in ["양호", "보통"]:
        return "medium"
    return "high"


def remove_duplicates(items: list[str]) -> list[str]:
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def get_region_order() -> list[str]:
    return ["forehead", "nose", "chin", "left_cheek", "right_cheek"]


def get_main_region(distribution: dict[str, int]) -> tuple[str, int]:
    if not distribution:
        return "unknown", 0

    main_region = max(distribution, key=distribution.get)
    return main_region, safe_int(distribution.get(main_region, 0))


def calculate_t_zone_count(distribution: dict[str, int]) -> int:
    return (
        safe_int(distribution.get("forehead", 0))
        + safe_int(distribution.get("nose", 0))
        + safe_int(distribution.get("chin", 0))
    )


def calculate_t_zone_ratio(distribution: dict[str, int]) -> float:
    total = sum(safe_int(v) for v in distribution.values())
    if total <= 0:
        return 0.0
    return calculate_t_zone_count(distribution) / total


def calculate_cheek_asymmetry(distribution: dict[str, int]) -> float:
    left = safe_int(distribution.get("left_cheek", 0))
    right = safe_int(distribution.get("right_cheek", 0))
    total = left + right
    if total <= 0:
        return 0.0
    return abs(left - right) / total


def normalize_distribution(distribution: Any) -> dict[str, int]:
    if not isinstance(distribution, dict):
        distribution = {}

    return {
        "forehead": safe_int(distribution.get("forehead", 0)),
        "nose": safe_int(distribution.get("nose", 0)),
        "chin": safe_int(distribution.get("chin", 0)),
        "left_cheek": safe_int(distribution.get("left_cheek", 0)),
        "right_cheek": safe_int(distribution.get("right_cheek", 0)),
    }


def extract_metrics_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    analysis_result = metadata.get("analysis_result", {})
    if not isinstance(analysis_result, dict):
        analysis_result = {}

    distribution = normalize_distribution(analysis_result.get("distribution", {}))

    porphyrin_count = safe_int(analysis_result.get("porphyrin_count", sum(distribution.values())))
    avg_brightness = safe_float(analysis_result.get("average_brightness", 0.0))
    max_brightness = safe_float(analysis_result.get("max_brightness", 0.0))

    main_region, main_region_count = get_main_region(distribution)

    return {
        "porphyrin_count": porphyrin_count,
        "average_brightness": avg_brightness,
        "max_brightness": max_brightness,
        "distribution": distribution,
        "main_region": main_region,
        "main_region_count": main_region_count,
        "t_zone_count": calculate_t_zone_count(distribution),
        "t_zone_ratio": calculate_t_zone_ratio(distribution),
        "cheek_asymmetry": calculate_cheek_asymmetry(distribution),
    }


# =========================================================
# 부위별 솔루션 생성
# =========================================================
def build_region_solution(region_key: str, count: int, total_count: int) -> dict[str, Any]:
    region_name = REGION_KO.get(region_key, region_key)
    level_info = find_count_level(count)
    bucket = level_to_solution_bucket(level_info["level"])
    region_rule = REGION_BASE_SOLUTIONS.get(region_key, {})

    ratio = 0.0
    if total_count > 0:
        ratio = count / total_count

    base_steps = []
    base_steps.extend(region_rule.get("cause", []))
    base_steps.extend(region_rule.get(bucket, []))

    if count == 0:
        base_steps = [
            f"{region_name} 부위는 현재 포르피린이 거의 검출되지 않았습니다.",
            "현재 관리 습관을 유지하고, 다른 집중 부위를 우선 확인하세요."
        ]

    if ratio >= 0.5 and total_count >= 10:
        base_steps.append(f"전체 포르피린 중 {region_name} 비율이 높습니다. 이 부위를 우선 관리 부위로 설정하세요.")

    return {
        "region": region_key,
        "region_ko": region_name,
        "count": count,
        "ratio": round(ratio, 4),
        "level": level_info["level"],
        "severity_score": level_info["severity_score"],
        "summary": f"{region_name}: {count}개 → {level_info['summary']}",
        "solution": remove_duplicates(base_steps)
    }


def build_overall_solution(metrics: dict[str, Any], region_solutions: list[dict[str, Any]]) -> dict[str, Any]:
    total_count = safe_int(metrics.get("porphyrin_count", 0))
    avg_brightness = safe_float(metrics.get("average_brightness", 0.0))
    max_brightness = safe_float(metrics.get("max_brightness", 0.0))
    distribution = metrics.get("distribution", {})
    main_region = str(metrics.get("main_region", "unknown"))
    main_region_count = safe_int(metrics.get("main_region_count", 0))
    t_zone_ratio = safe_float(metrics.get("t_zone_ratio", 0.0))
    cheek_asymmetry = safe_float(metrics.get("cheek_asymmetry", 0.0))

    overall_level = find_count_level(total_count)
    tone = overall_level["tone"]

    if tone in ["good"]:
        guide_key = "good"
    elif tone in ["normal"]:
        guide_key = "normal"
    elif tone in ["caution"]:
        guide_key = "caution"
    elif tone in ["warning"]:
        guide_key = "warning"
    elif tone in ["danger"]:
        guide_key = "danger"
    else:
        guide_key = "critical"

    warnings = []

    if total_count == 0:
        warnings.append("전체 포르피린이 0개라면 실제로 깨끗한 상태일 수 있지만, 노출 부족이나 660nm 영상이 너무 어두운 경우도 확인하세요.")

    if avg_brightness >= 200:
        warnings.append("평균 형광 밝기가 높습니다. 밝게 반응하는 영역이 많거나 노출 조건의 영향을 받았을 수 있습니다.")

    if max_brightness >= 240:
        warnings.append("최대 밝기가 매우 높습니다. 반사광 또는 과노출로 인한 과검출 가능성을 확인하세요.")

    if t_zone_ratio >= 0.65 and total_count >= 10:
        warnings.append("T존 비율이 높습니다. 이마·코·턱 부위 피지 관리를 우선하세요.")

    if cheek_asymmetry >= 0.45:
        warnings.append("좌우 볼 차이가 큽니다. 생활 습관 차이인지 촬영 각도 차이인지 재촬영으로 확인하세요.")

    priority_regions = sorted(
        [r for r in region_solutions if r["count"] > 0],
        key=lambda r: (r["count"], r["severity_score"]),
        reverse=True
    )

    top_regions = priority_regions[:3]

    return {
        "total_count": total_count,
        "level": overall_level["level"],
        "severity_score": overall_level["severity_score"],
        "summary": overall_level["summary"],
        "main_region": main_region,
        "main_region_ko": REGION_KO.get(main_region, main_region),
        "main_region_count": main_region_count,
        "t_zone_ratio": round(t_zone_ratio, 4),
        "cheek_asymmetry": round(cheek_asymmetry, 4),
        "overall_guide": OVERALL_GUIDE[guide_key],
        "priority_regions": [
            {
                "region": r["region"],
                "region_ko": r["region_ko"],
                "count": r["count"],
                "level": r["level"],
                "summary": r["summary"]
            }
            for r in top_regions
        ],
        "warnings": warnings,
        "note": "이 결과는 포르피린 형광 기반 관리 가이드이며 의료 진단이 아닙니다."
    }


def generate_region_based_skin_care_solution_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    distribution = normalize_distribution(metrics.get("distribution", {}))
    total_count = safe_int(metrics.get("porphyrin_count", sum(distribution.values())))

    if total_count == 0 and sum(distribution.values()) > 0:
        total_count = sum(distribution.values())

    region_solutions = []

    for region_key in get_region_order():
        count = safe_int(distribution.get(region_key, 0))
        region_solutions.append(
            build_region_solution(
                region_key=region_key,
                count=count,
                total_count=total_count
            )
        )

    overall_solution = build_overall_solution(metrics, region_solutions)

    return {
        "generated_at": datetime.now().isoformat(),
        "solution_type": "region_based_porphyrin_skin_care_solution",
        "overall_solution": overall_solution,
        "region_solutions": {
            item["region"]: item for item in region_solutions
        },
        "region_solutions_list": region_solutions,
        "ui_summary": build_ui_summary(overall_solution, region_solutions),
    }


def generate_region_based_skin_care_solution_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    metrics = extract_metrics_from_metadata(metadata)
    return generate_region_based_skin_care_solution_from_metrics(metrics)


def build_ui_summary(overall_solution: dict[str, Any], region_solutions: list[dict[str, Any]]) -> dict[str, Any]:
    cards = []

    for item in region_solutions:
        cards.append({
            "title": f"{item['region_ko']} : {item['count']}개",
            "level": item["level"],
            "severity_score": item["severity_score"],
            "summary": item["summary"],
            "solution_preview": item["solution"][:2],
        })

    return {
        "headline": f"전체 포르피린 {overall_solution['total_count']}개 · 주요 집중 부위: {overall_solution['main_region_ko']}",
        "overall_level": overall_solution["level"],
        "priority_regions": overall_solution["priority_regions"],
        "region_cards": cards,
    }


def append_region_solution_to_metadata_file(metadata_path: Path) -> dict[str, Any]:
    metadata_path = Path(metadata_path)

    metadata = load_json(metadata_path)
    solution = generate_region_based_skin_care_solution_from_metadata(metadata)

    if "analysis_result" not in metadata or not isinstance(metadata["analysis_result"], dict):
        metadata["analysis_result"] = {}

    metadata["analysis_result"]["region_based_skin_care_solution"] = solution

    solution_path = metadata_path.parent / "region_skin_care_solution.json"
    save_json(solution_path, solution)

    metadata.setdefault("analysis_files", {})
    metadata["analysis_files"]["region_skin_care_solution"] = str(solution_path)

    save_json(metadata_path, metadata)

    return {
        "metadata": metadata,
        "solution": solution,
        "solution_path": str(solution_path)
    }


def print_region_solution(solution: dict[str, Any]):
    overall = solution["overall_solution"]
    region_list = solution["region_solutions_list"]

    print("\n[부위별 포르피린 피부관리 솔루션]")
    print(f"전체 포르피린: {overall['total_count']}개")
    print(f"전체 단계: {overall['level']}")
    print(f"주요 집중 부위: {overall['main_region_ko']} ({overall['main_region_count']}개)")

    print("\n전체 관리 안내:")
    for guide in overall["overall_guide"]:
        print(f"- {guide}")

    if overall["warnings"]:
        print("\n주의:")
        for warning in overall["warnings"]:
            print(f"- {warning}")

    print("\n부위별 관리 솔루션:")
    for item in region_list:
        print(f"\n[{item['region_ko']}: {item['count']}개 / {item['level']}]")
        print(f"- {item['summary']}")
        for step in item["solution"]:
            print(f"- {step}")

    print("\n우선 관리 부위:")
    if overall["priority_regions"]:
        for region in overall["priority_regions"]:
            print(f"- {region['region_ko']}: {region['count']}개 / {region['level']}")
    else:
        print("- 우선 관리가 필요한 집중 부위가 뚜렷하지 않습니다.")


# =========================================================
# CLI
# =========================================================
def find_latest_metadata(base_path: Path) -> Path | None:
    base_path = Path(base_path)

    if not base_path.exists():
        return None

    candidates = list(base_path.glob("**/metadata.json"))
    candidates.extend(list(base_path.glob("**/*metadata*.json")))
    candidates = [p for p in candidates if p.is_file()]

    if not candidates:
        return None

    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def find_default_base_path() -> Path:
    candidates = [
        Path.home() / "Graduate_Project" / "TaeYeon" / "captures",
        Path.home() / "Graduate_Project" / "captures" / "sessions" / "cam4_660nm",
        Path.home() / "Graduate_Project" / "captures" / "cam4_660nm",
        Path.cwd() / "captures",
        Path.cwd(),
    ]

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default=None, help="포르피린 분석 metadata.json 경로")
    parser.add_argument("--base", type=str, default=None, help="metadata.json을 찾을 기준 폴더")
    args = parser.parse_args()

    if args.metadata:
        target_metadata = Path(args.metadata).expanduser()
    else:
        base = Path(args.base).expanduser() if args.base else find_default_base_path()
        target_metadata = find_latest_metadata(base)

    if target_metadata is None:
        raise RuntimeError("metadata.json을 찾지 못했습니다.")

    result = append_region_solution_to_metadata_file(target_metadata)
    print_region_solution(result["solution"])
    print("\n저장 완료:")
    print("-", result["solution_path"])
    print("-", target_metadata)
