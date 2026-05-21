from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any


# =========================================================
# 포르피린 정도별 피부관리 솔루션 기준
# =========================================================
# count 기준은 졸업작품용 규칙 기반 기준입니다.
# 실제 촬영 조건, 노출, 필터, 얼굴 거리, 분석 임계값에 따라 조정하세요.

SOLUTION_LEVELS = [
    {
        "min": 0,
        "max": 10,
        "level": "매우 양호",
        "severity_score": 5,
        "summary": "포르피린 검출량이 매우 적어 현재 관리 상태가 좋은 편입니다.",
        "message": "좋습니다. 현재 세안과 생활 습관을 잘 유지하고 있는 것으로 보입니다.",
        "care": [
            "현재 세안 루틴을 유지하세요.",
            "과도한 세안보다 아침·저녁 규칙적인 세안을 유지하세요.",
            "자외선 차단제와 충분한 수분 공급을 꾸준히 유지하세요.",
            "같은 조건에서 주기적으로 촬영하여 변화만 확인하세요."
        ]
    },
    {
        "min": 11,
        "max": 20,
        "level": "양호",
        "severity_score": 18,
        "summary": "포르피린이 소량 검출되었습니다. 큰 문제는 아니지만 세안 습관 점검이 필요합니다.",
        "message": "대체로 양호하지만, 피지와 노폐물이 쌓이기 쉬운 부위는 조금 더 신경 쓰면 좋습니다.",
        "care": [
            "저녁 세안을 조금 더 꼼꼼히 하세요.",
            "운동이나 외출 후에는 얼굴에 땀과 피지가 오래 남지 않게 관리하세요.",
            "코 주변과 턱 주변을 부드럽게 세안하세요.",
            "피부를 문지르는 강한 세안은 피하세요."
        ]
    },
    {
        "min": 21,
        "max": 40,
        "level": "보통",
        "severity_score": 35,
        "summary": "포르피린이 보통 수준으로 검출되었습니다. 피지 관리와 생활 습관 개선이 권장됩니다.",
        "message": "피지 분비가 많은 부위에 포르피린이 쌓이기 시작한 상태로 볼 수 있습니다.",
        "care": [
            "세안 후 잔여물이 남지 않도록 충분히 헹구세요.",
            "유분감이 많은 화장품 사용량을 줄이고, 논코메도제닉 제품을 고려하세요.",
            "베개 커버와 마스크처럼 얼굴에 닿는 물건을 자주 교체하세요.",
            "야식, 수면 부족, 스트레스 등 피지 분비에 영향을 줄 수 있는 생활 요인을 점검하세요."
        ]
    },
    {
        "min": 41,
        "max": 70,
        "level": "주의",
        "severity_score": 60,
        "summary": "포르피린이 다소 많이 검출되었습니다. 부위별 집중 관리가 필요합니다.",
        "message": "여드름 유발 가능성과 관련된 형광 반응이 비교적 뚜렷하게 나타난 상태입니다.",
        "care": [
            "T존과 볼 주변의 피지 관리를 강화하세요.",
            "세안은 하루 2회 정도로 유지하고, 과도한 스크럽은 피하세요.",
            "사용 중인 화장품이나 선크림이 모공을 막는지 확인하세요.",
            "같은 조건으로 3~7일 간격 재촬영하여 증가/감소 추세를 확인하세요.",
            "염증성 여드름이 동반되면 피부과 상담을 고려하세요."
        ]
    },
    {
        "min": 71,
        "max": 100,
        "level": "높음",
        "severity_score": 78,
        "summary": "포르피린 검출량이 높은 편입니다. 관리 루틴을 적극적으로 조정해야 합니다.",
        "message": "포르피린 신호가 넓거나 강하게 나타나 피부 트러블 위험 신호로 볼 수 있습니다.",
        "care": [
            "저녁 세안 루틴을 반드시 점검하세요.",
            "유분이 많은 제품, 두꺼운 메이크업, 장시간 마스크 착용을 줄이세요.",
            "피지가 많은 부위는 순한 클렌저로 꼼꼼히 관리하세요.",
            "수면 부족과 스트레스 관리가 필요합니다.",
            "반복적으로 높은 수치가 나오면 피부과 상담을 권장합니다."
        ]
    },
    {
        "min": 101,
        "max": 10**9,
        "level": "매우 높음",
        "severity_score": 92,
        "summary": "포르피린 검출량이 매우 높습니다. 촬영 조건 재확인과 집중 관리가 필요합니다.",
        "message": "포르피린 신호가 매우 많이 검출되었습니다. 단, 노출 과다나 반사광 때문에 과검출되었는지도 확인해야 합니다.",
        "care": [
            "먼저 같은 조건에서 재촬영하여 결과가 반복되는지 확인하세요.",
            "세안, 수면, 화장품, 마스크 착용 습관을 전반적으로 점검하세요.",
            "피지 분비가 많은 부위는 자극 없이 꾸준히 관리하세요.",
            "강한 스크럽이나 압출은 피하세요.",
            "염증성 여드름, 통증, 붉은기, 악화가 동반되면 피부과 상담을 권장합니다."
        ]
    },
]


REGION_KO = {
    "left_cheek": "왼쪽 볼",
    "right_cheek": "오른쪽 볼",
    "forehead": "이마",
    "nose": "코",
    "chin": "턱",
    "unknown": "알 수 없음",
}


REGION_CARE = {
    "forehead": [
        "이마 부위는 앞머리, 헤어 제품, 땀의 영향을 받을 수 있습니다.",
        "운동 후 땀을 오래 방치하지 말고, 헤어 왁스·오일이 이마에 닿지 않게 관리하세요."
    ],
    "nose": [
        "코 부위는 피지 분비가 많아 포르피린이 집중되기 쉽습니다.",
        "코 주변은 자극적인 코팩보다 부드러운 세안과 꾸준한 피지 관리가 좋습니다."
    ],
    "chin": [
        "턱 부위는 마스크, 손으로 만지는 습관, 면도 자극의 영향을 받을 수 있습니다.",
        "턱을 손으로 만지는 습관을 줄이고, 마스크와 면도 도구를 청결하게 유지하세요."
    ],
    "left_cheek": [
        "왼쪽 볼에 집중되어 있다면 베개, 휴대폰, 손 접촉 습관을 확인하세요.",
        "좌우 차이가 반복되면 생활 습관이나 촬영 각도 차이도 함께 확인하세요."
    ],
    "right_cheek": [
        "오른쪽 볼에 집중되어 있다면 베개, 휴대폰, 손 접촉 습관을 확인하세요.",
        "좌우 차이가 반복되면 생활 습관이나 촬영 각도 차이도 함께 확인하세요."
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


def find_solution_level(porphyrin_count: int) -> dict[str, Any]:
    for item in SOLUTION_LEVELS:
        if item["min"] <= porphyrin_count <= item["max"]:
            return item

    return SOLUTION_LEVELS[-1]


def get_main_region(distribution: dict[str, int]) -> tuple[str, int]:
    if not distribution:
        return "unknown", 0

    main_region = max(distribution, key=distribution.get)
    return main_region, safe_int(distribution.get(main_region, 0))


def calculate_t_zone_ratio(distribution: dict[str, int]) -> float:
    total = sum(safe_int(v) for v in distribution.values())

    if total <= 0:
        return 0.0

    t_zone = (
        safe_int(distribution.get("forehead", 0))
        + safe_int(distribution.get("nose", 0))
        + safe_int(distribution.get("chin", 0))
    )

    return t_zone / total


def calculate_cheek_asymmetry(distribution: dict[str, int]) -> float:
    left = safe_int(distribution.get("left_cheek", 0))
    right = safe_int(distribution.get("right_cheek", 0))
    total = left + right

    if total <= 0:
        return 0.0

    return abs(left - right) / total


def extract_metrics_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
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
        "t_zone_ratio": calculate_t_zone_ratio(distribution),
        "cheek_asymmetry": calculate_cheek_asymmetry(distribution),
    }


# =========================================================
# 피부관리 솔루션 생성
# =========================================================
def generate_skin_care_solution_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    porphyrin_count = safe_int(metrics.get("porphyrin_count", 0))
    avg_brightness = safe_float(metrics.get("average_brightness", 0.0))
    max_brightness = safe_float(metrics.get("max_brightness", 0.0))
    distribution = metrics.get("distribution", {})

    if not isinstance(distribution, dict):
        distribution = {}

    main_region = str(metrics.get("main_region", "unknown"))
    main_region_count = safe_int(metrics.get("main_region_count", 0))
    t_zone_ratio = safe_float(metrics.get("t_zone_ratio", 0.0))
    cheek_asymmetry = safe_float(metrics.get("cheek_asymmetry", 0.0))

    base = find_solution_level(porphyrin_count)

    warnings = []
    focus_care = []

    if porphyrin_count == 0:
        warnings.append("포르피린이 0개로 검출된 경우 실제로 깨끗한 상태일 수도 있지만, 노출 부족 또는 660nm 영상이 너무 어두운 경우도 확인해야 합니다.")

    if avg_brightness >= 200:
        warnings.append("평균 형광 밝기가 높습니다. 밝게 반응하는 영역이 많을 수 있습니다.")
        focus_care.append("밝게 반응하는 부위를 중심으로 세안 잔여물, 피지, 화장품 잔여 가능성을 확인하세요.")

    if max_brightness >= 240:
        warnings.append("최대 밝기가 매우 높습니다. 반사광 또는 과노출로 인한 과검출 가능성도 확인하세요.")

    if t_zone_ratio >= 0.65:
        focus_care.append("T존 비율이 높습니다. 이마·코·턱 부위의 피지 관리와 세안 습관을 우선 점검하세요.")

    if cheek_asymmetry >= 0.45:
        focus_care.append("좌우 볼 차이가 큽니다. 한쪽으로 자는 습관, 휴대폰 접촉, 손으로 얼굴을 만지는 습관을 확인하세요.")

    if main_region in REGION_CARE and main_region_count > 0:
        focus_care.extend(REGION_CARE[main_region])

    care_steps = []
    for item in base["care"] + focus_care:
        if item not in care_steps:
            care_steps.append(item)

    warnings_unique = []
    for item in warnings:
        if item not in warnings_unique:
            warnings_unique.append(item)

    result = {
        "generated_at": datetime.now().isoformat(),
        "solution_type": "porphyrin_based_skin_care_solution",
        "porphyrin_level": {
            "count": porphyrin_count,
            "range": f"{base['min']}~{base['max'] if base['max'] < 10**9 else '이상'}",
            "level": base["level"],
            "severity_score": base["severity_score"],
            "summary": base["summary"],
            "message": base["message"]
        },
        "key_metrics": {
            "porphyrin_count": porphyrin_count,
            "average_brightness": avg_brightness,
            "max_brightness": max_brightness,
            "main_region": main_region,
            "main_region_ko": REGION_KO.get(main_region, main_region),
            "main_region_count": main_region_count,
            "t_zone_ratio": round(t_zone_ratio, 4),
            "cheek_asymmetry": round(cheek_asymmetry, 4),
            "distribution": distribution
        },
        "skin_care_solution": {
            "main_message": base["message"],
            "recommended_steps": care_steps,
            "warnings": warnings_unique,
            "follow_up": [
                "동일한 거리, 조명, 노출 조건에서 반복 촬영해야 변화 비교가 정확합니다.",
                "이 결과는 포르피린 형광 기반 관리 가이드이며 의료 진단이 아닙니다."
            ]
        }
    }

    return result


def generate_skin_care_solution_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    metrics = extract_metrics_from_metadata(metadata)
    return generate_skin_care_solution_from_metrics(metrics)


def append_solution_to_metadata_file(metadata_path: Path) -> dict[str, Any]:
    metadata_path = Path(metadata_path)

    metadata = load_json(metadata_path)
    solution = generate_skin_care_solution_from_metadata(metadata)

    if "analysis_result" not in metadata or not isinstance(metadata["analysis_result"], dict):
        metadata["analysis_result"] = {}

    metadata["analysis_result"]["skin_care_solution"] = solution

    solution_path = metadata_path.parent / "skin_care_solution.json"
    save_json(solution_path, solution)

    metadata.setdefault("analysis_files", {})
    metadata["analysis_files"]["skin_care_solution"] = str(solution_path)

    save_json(metadata_path, metadata)

    return {
        "metadata": metadata,
        "solution": solution,
        "solution_path": str(solution_path)
    }


def print_solution(solution: dict[str, Any]):
    level = solution["porphyrin_level"]
    metrics = solution["key_metrics"]
    care = solution["skin_care_solution"]

    print("\n[피부관리 솔루션]")
    print(f"포르피린 개수: {level['count']}개")
    print(f"단계: {level['level']} ({level['range']})")
    print(f"요약: {level['summary']}")
    print(f"주요 부위: {metrics['main_region_ko']}")

    print("\n관리 안내:")
    print(f"- {care['main_message']}")

    print("\n추천 관리:")
    for step in care["recommended_steps"]:
        print(f"- {step}")

    if care["warnings"]:
        print("\n주의:")
        for warning in care["warnings"]:
            print(f"- {warning}")

    print("\n추적:")
    for item in care["follow_up"]:
        print(f"- {item}")


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

    result = append_solution_to_metadata_file(target_metadata)
    print_solution(result["solution"])
    print("\n저장 완료:")
    print("-", result["solution_path"])
    print("-", target_metadata)
