import type { AutoCaptureChecks } from "./types";

export const EMPTY_AUTO_CHECKS: AutoCaptureChecks = {
  face_found: false,
  center_ok: false,
  size_ok: false,
  angle_ok: false,
  eyes_closed: false,
  stable_ok: false,
};

export const AUTO_CHECK_LABELS: Array<{ key: keyof AutoCaptureChecks; label: string }> = [
  { key: "face_found", label: "얼굴 인식" },
  { key: "center_ok", label: "얼굴 중앙 정렬" },
  { key: "size_ok", label: "얼굴 크기" },
  { key: "angle_ok", label: "얼굴 각도" },
  { key: "eyes_closed", label: "눈 감음" },
  { key: "stable_ok", label: "안정 유지" },
];

export const REGION_LABELS: Record<string, string> = {
  forehead: "이마",
  nose: "코",
  philtrum: "인중",
  chin: "턱",
  right_cheek: "오른쪽 볼",
  left_cheek: "왼쪽 볼",
};
