import type { PorphyrinResult } from "./types";

export const PORPHYRIN_REGION_LABELS: Record<string, string> = {
  forehead: "이마",
  chin: "턱",
  nose: "코",
  right_cheek: "오른쪽 볼",
  left_cheek: "왼쪽 볼",
  philtrum: "인중",
};

export const PORPHYRIN_REGION_ORDER = [
  "forehead",
  "chin",
  "nose",
  "right_cheek",
  "left_cheek",
  "philtrum",
];

const PORPHYRIN_RISK_SCORE_MEAN = 50.0;
const PORPHYRIN_RISK_SCORE_STD = 25.0;
const PORPHYRIN_RISK_Z_RANGE = 2.0;
const PORPHYRIN_RISK_COUNT_MEAN = 578.8636363636364;
const PORPHYRIN_RISK_COUNT_STD = 360.73123909567926;
const PORPHYRIN_RISK_MEAN_BRIGHTNESS_MEAN = 60.91818181818182;
const PORPHYRIN_RISK_MEAN_BRIGHTNESS_STD = 11.652024262826076;
const PORPHYRIN_RISK_TOP5_MAX_MEAN = 95.04545454545455;
const PORPHYRIN_RISK_TOP5_MAX_STD = 26.06892485269016;
const PORPHYRIN_RISK_COUNT_MAX_SCORE = 60.0;
const PORPHYRIN_RISK_MEAN_BRIGHTNESS_MAX_SCORE = 30.0;
const PORPHYRIN_RISK_TOP5_MAX_SCORE = 10.0;
const PORPHYRIN_RISK_SCORE_BOUNDARIES = {
  a_max: 20.0,
  b_max: 40.0,
  c_max: 60.0,
  d_max: 80.0,
};

const PORPHYRIN_GRADE_LABELS: Record<string, string> = {
  A: "매우 양호",
  B: "양호",
  C: "보통",
  D: "주의",
  E: "관리 필요",
};

const PORPHYRIN_GRADE_LEVELS: Record<string, number> = {
  A: 1,
  B: 2,
  C: 3,
  D: 4,
  E: 5,
};

export function normalizePercentagesTo100(values: Record<string, number>) {
  const rawValues = PORPHYRIN_REGION_ORDER.map((key) => ({
    key,
    value: Math.max(0, Number(values[key] || 0)),
  }));
  const total = rawValues.reduce((sum, item) => sum + item.value, 0);

  if (total <= 0) {
    return Object.fromEntries(PORPHYRIN_REGION_ORDER.map((key) => [key, 0]));
  }

  const normalized = rawValues.map((item) => {
    const raw = (item.value / total) * 100;
    const rounded = Math.round(raw);
    return {
      ...item,
      raw,
      rounded,
      remainder: Math.abs(raw - rounded),
    };
  });

  let diff = 100 - normalized.reduce((sum, item) => sum + item.rounded, 0);
  const ordered = [...normalized].sort((a, b) => b.remainder - a.remainder);

  for (let idx = 0; diff !== 0 && ordered.length > 0; idx += 1) {
    const target = ordered[idx % ordered.length];
    if (diff > 0) {
      target.rounded += 1;
      diff -= 1;
    } else if (target.rounded > 0) {
      target.rounded -= 1;
      diff += 1;
    }
  }

  return Object.fromEntries(normalized.map((item) => [item.key, item.rounded]));
}

function isPorphyrinGrade(value: string) {
  return Object.prototype.hasOwnProperty.call(PORPHYRIN_GRADE_LABELS, value);
}

function clampNumber(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function getMetricRiskScore(value: number, mean: number, std: number, maxScore: number) {
  if (std <= 0) {
    return maxScore / 2;
  }

  const zScore = (Number(value || 0) - mean) / std;
  const normalized = (zScore + PORPHYRIN_RISK_Z_RANGE) / (PORPHYRIN_RISK_Z_RANGE * 2);
  return clampNumber(normalized * maxScore, 0, maxScore);
}

function getPorphyrinRiskScore(
  porphyrinCount: number,
  meanBrightness: number,
  top5MaxBrightness: number
) {
  return (
    getMetricRiskScore(
      porphyrinCount,
      PORPHYRIN_RISK_COUNT_MEAN,
      PORPHYRIN_RISK_COUNT_STD,
      PORPHYRIN_RISK_COUNT_MAX_SCORE
    )
    + getMetricRiskScore(
      meanBrightness,
      PORPHYRIN_RISK_MEAN_BRIGHTNESS_MEAN,
      PORPHYRIN_RISK_MEAN_BRIGHTNESS_STD,
      PORPHYRIN_RISK_MEAN_BRIGHTNESS_MAX_SCORE
    )
    + getMetricRiskScore(
      top5MaxBrightness,
      PORPHYRIN_RISK_TOP5_MAX_MEAN,
      PORPHYRIN_RISK_TOP5_MAX_STD,
      PORPHYRIN_RISK_TOP5_MAX_SCORE
    )
  );
}

function getPorphyrinRiskGrade(riskScore: number) {
  if (riskScore <= PORPHYRIN_RISK_SCORE_BOUNDARIES.a_max) return "A";
  if (riskScore <= PORPHYRIN_RISK_SCORE_BOUNDARIES.b_max) return "B";
  if (riskScore <= PORPHYRIN_RISK_SCORE_BOUNDARIES.c_max) return "C";
  if (riskScore <= PORPHYRIN_RISK_SCORE_BOUNDARIES.d_max) return "D";
  return "E";
}

function getPorphyrinRiskComment(grade: string) {
  if (grade === "A") return "위험점수가 낮은 구간입니다.";
  if (grade === "B") return "위험점수가 비교적 낮은 구간입니다.";
  if (grade === "C") return "위험점수가 보통 구간입니다.";
  if (grade === "D") return "위험점수가 높은 구간입니다.";
  return "위험점수가 매우 높은 구간입니다.";
}

export function getPorphyrinRiskReview(result: PorphyrinResult) {
  const riskScore = Number(result.risk_score || 0);
  const zScore =
    Number(result.risk_z_score || 0)
    || ((riskScore - PORPHYRIN_RISK_SCORE_MEAN) / PORPHYRIN_RISK_SCORE_STD);
  const grade = isPorphyrinGrade(result.grade)
    ? result.grade
    : getPorphyrinRiskGrade(riskScore);

  return {
    level: PORPHYRIN_GRADE_LEVELS[grade] || 0,
    grade,
    status: PORPHYRIN_GRADE_LABELS[grade] || "분석 필요",
    comment: getPorphyrinRiskComment(grade),
    riskScore,
    zScore,
  };
}

export function mapPorphyrinResult(data: Record<string, unknown>): PorphyrinResult {
  const meanBrightness = Number(data.porphyrin_mean_brightness || 0);
  const top5MaxBrightness = Number(data.porphyrin_top5_max_brightness || 0);
  const porphyrinCount = Number(data.porphyrin_count || 0);
  const skinScore =
    typeof data.skin_score === "object" && data.skin_score !== null
      ? data.skin_score as Partial<PorphyrinResult["skin_score"]>
      : {};
  const fallbackRiskScore = getPorphyrinRiskScore(
    porphyrinCount,
    meanBrightness,
    top5MaxBrightness
  );
  const apiRiskScore = Number(data.risk_score);
  const usesCurrentRiskBasis = skinScore.basis === "standardized_metric_risk_score";
  const apiRiskScoreMean = Number(data.risk_score_mean);
  const apiRiskScoreStd = Number(data.risk_score_std);
  const usesCurrentRiskScale =
    usesCurrentRiskBasis
    && Math.abs(apiRiskScoreMean - PORPHYRIN_RISK_SCORE_MEAN) < 0.01
    && Math.abs(apiRiskScoreStd - PORPHYRIN_RISK_SCORE_STD) < 0.01;
  const riskScore = usesCurrentRiskBasis && Number.isFinite(apiRiskScore) && apiRiskScore > 0
    ? apiRiskScore
    : fallbackRiskScore;
  const apiRiskZScore = Number(data.risk_z_score);
  const riskZScore = usesCurrentRiskScale && Number.isFinite(apiRiskZScore)
    ? apiRiskZScore
    : (riskScore - PORPHYRIN_RISK_SCORE_MEAN) / PORPHYRIN_RISK_SCORE_STD;
  const apiGrade = typeof data.grade === "string" ? data.grade : "";
  const grade = usesCurrentRiskScale && isPorphyrinGrade(apiGrade)
    ? apiGrade
    : getPorphyrinRiskGrade(riskScore);

  return {
    porphyrin_count: porphyrinCount,
    porphyrin_area: Number(data.porphyrin_area || 0),
    detection_rate_percent: Number(data.detection_rate_percent || 0),
    porphyrin_mean_brightness: meanBrightness,
    porphyrin_top5_max_brightness: top5MaxBrightness,
    risk_score: riskScore,
    risk_z_score: riskZScore,
    risk_score_mean: PORPHYRIN_RISK_SCORE_MEAN,
    risk_score_std: PORPHYRIN_RISK_SCORE_STD,
    risk_score_boundaries: {
      a_max: PORPHYRIN_RISK_SCORE_BOUNDARIES.a_max,
      b_max: PORPHYRIN_RISK_SCORE_BOUNDARIES.b_max,
      c_max: PORPHYRIN_RISK_SCORE_BOUNDARIES.c_max,
      d_max: PORPHYRIN_RISK_SCORE_BOUNDARIES.d_max,
    },
    grade,
    skin_score: {
      ...skinScore,
      score: riskScore,
      grade,
      label: PORPHYRIN_GRADE_LABELS[grade] || "분석 필요",
      level: PORPHYRIN_GRADE_LEVELS[grade] || 0,
      basis: "standardized_metric_risk_score",
      porphyrin_count: porphyrinCount || Number(skinScore.porphyrin_count || 0),
      reference_bad_count: Number(skinScore.reference_bad_count || 0),
      risk_score: riskScore,
      risk_z_score: riskZScore,
      risk_score_mean: PORPHYRIN_RISK_SCORE_MEAN,
      risk_score_std: PORPHYRIN_RISK_SCORE_STD,
      risk_score_boundaries: {
        a_max: PORPHYRIN_RISK_SCORE_BOUNDARIES.a_max,
        b_max: PORPHYRIN_RISK_SCORE_BOUNDARIES.b_max,
        c_max: PORPHYRIN_RISK_SCORE_BOUNDARIES.c_max,
        d_max: PORPHYRIN_RISK_SCORE_BOUNDARIES.d_max,
      },
    },
    region_analysis:
      typeof data.region_analysis === "object" && data.region_analysis !== null
        ? data.region_analysis as Record<string, number>
        : {},
    threshold_percentile: Number(data.threshold_percentile || 0),
    threshold_value: Number(data.threshold_value || 0),
    min_area: Number(data.min_area || 0),
    max_area: Number(data.max_area || 0),
    heatmap_url: typeof data.heatmap_url === "string" ? data.heatmap_url : "",
    uv_heatmap_url: typeof data.uv_heatmap_url === "string" ? data.uv_heatmap_url : null,
    white_overlay_url: typeof data.white_overlay_url === "string" ? data.white_overlay_url : null,
  };
}
