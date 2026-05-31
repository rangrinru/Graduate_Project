export type Screen =
  | "profiles"
  | "camera"
  | "history"
  | "historyDetail";

export type Profile = {
  id: number;
  name: string;
  folderId: string;
  createdAt: string;
};

export type HistoryItem = {
  captureId: string;
  capturedAt: string;
  displayTime: string;
  profileId: string;
  profileName: string;
};

export type HistoryImageItem = {
  camera: string;
  display_name: string;
  filter_type: string;
  exists: boolean;
  image_url: string | null;
  metadata?: Record<string, unknown>;
};

export type HistoryDetail = {
  captureId: string;
  capturedAt: string;
  displayTime: string;
  profileId: string;
  profileName: string;
  images: {
    no_filter?: HistoryImageItem;
    "405nm_filter"?: HistoryImageItem;
    "660nm_filter"?: HistoryImageItem;
  };
};

export type Toast = {
  message: string;
  type: "success" | "error" | "info";
};

export type PorphyrinResult = {
  porphyrin_count: number;
  porphyrin_area: number;
  detection_rate_percent: number;
  grade: string;
  skin_score: {
    score: number;
    grade: string;
    label: string;
    basis: string;
    porphyrin_count: number;
    reference_bad_count: number;
  };
  region_analysis: Record<string, number>;
  threshold_percentile: number;
  threshold_value: number;
  min_area: number;
  max_area: number;
  heatmap_url: string;
  white_overlay_url?: string | null;
};

export type FocusCareArea = {
  id: number;
  region: string;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: {
    x: number;
    y: number;
  };
  area: number;
  mean_intensity: number;
  risk_score: number;
};

export type TroubleRiskResult = {
  risk_area: number;
  risk_rate_percent: number;
  risk_grade: string;
  region_analysis: Record<string, number>;
  focus_areas: FocusCareArea[];
  top_region: string | null;
  threshold_value: number;
  risk_heatmap_url: string;
  focus_overlay_url: string;
  risk_mask_url: string;
};

export type AnalysisImageMode =
  | "source"
  | "porphyrin_heatmap"
  | "trouble_risk_heatmap"
  | "focus_care_overlay";

export type SkinAgingResult = {
  freckle_count: number;
  freckle_area: number;
  freckle_area_rate_percent: number;
  predicted_skin_age: number;
  skin_age_offset: number;
  skin_age_level: string;
  grade: string;
  label: string;
  basis: string;
  base_age: number;
  threshold_value: number;
  min_area: number;
  max_area: number;
  face_detection_method: string;
  result_url: string;
  mask_url: string;
};

export type AutoCaptureChecks = {
  face_found: boolean;
  center_ok: boolean;
  size_ok: boolean;
  angle_ok: boolean;
  eyes_closed: boolean;
  stable_ok: boolean;
};

export type AutoCaptureStatus = {
  running: boolean;
  captured: boolean;
  profile_id: string | null;
  capture_id: string | null;
  status: string;
  error: string | null;
  checks: AutoCaptureChecks;
  stable_face_count: number;
  eyes_closed_count: number;
  dynamic_eye_threshold: number;
  white_led_is_on: boolean;
  last_update: string | null;
};

export type KeyboardMode = "ko" | "en" | "num";

export type HangulBuffer = {
  cho: string | null;
  jung: string | null;
  jong: string | null;
};
