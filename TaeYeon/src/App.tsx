import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import type {
  AutoCaptureStatus,
  HistoryDetail,
  HistoryItem,
  KeyboardMode,
  PorphyrinCompareItem,
  PorphyrinResult,
  Profile,
  Screen,
  Toast,
  HangulBuffer,
} from "./types";
import { AUTO_CHECK_LABELS, EMPTY_AUTO_CHECKS } from "./constants";
import {
  COMPOUND_JONG_MAP,
  COMPOUND_JUNG_MAP,
  DOUBLE_CHO_MAP,
  EMPTY_HANGUL_BUFFER,
  ENGLISH_KEY_ROWS,
  JONG_LIST,
  KOREAN_CONSONANTS,
  KOREAN_KEY_ROWS,
  KOREAN_VOWELS,
  NUMBER_KEY_ROWS,
  SPLIT_JONG_MAP,
  composeHangul,
  composeHangulWithoutJong,
} from "./hangul";
import { API_BASE } from "./api";

const PROFILE_FETCH_TIMEOUT_MS = 4000;

type HistoryFilter = "no_filter" | "405nm_filter" | "660nm_filter";
type DetailImageView = "source" | "porphyrin_heatmap";

const PORPHYRIN_REGION_LABELS: Record<string, string> = {
  forehead: "이마",
  chin: "턱",
  nose: "코",
  right_cheek: "오른쪽 볼",
  left_cheek: "왼쪽 볼",
  philtrum: "인중",
};

const PORPHYRIN_REGION_ORDER = [
  "forehead",
  "chin",
  "nose",
  "right_cheek",
  "left_cheek",
  "philtrum",
];

function normalizePercentagesTo100(values: Record<string, number>) {
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

function getPorphyrinBrightnessReview(meanBrightness: number) {
  if (meanBrightness > 70) {
    return {
      level: 5,
      status: "집중 관리 필요",
      comment: "피부 상태 개선을 위한 관리가 권장됩니다.",
    };
  }

  if (meanBrightness >= 60) {
    return {
      level: 4,
      status: "주의",
      comment: "피지 및 모공 관리가 필요합니다.",
    };
  }

  if (meanBrightness >= 50) {
    return {
      level: 3,
      status: "보통",
      comment: "피부 관리가 필요한 상태입니다.",
    };
  }

  if (meanBrightness >= 40) {
    return {
      level: 2,
      status: "양호",
      comment: "전반적으로 건강한 피부 상태입니다.",
    };
  }

  return {
    level: 1,
    status: "매우 양호",
    comment: "깨끗한 피부 상태로 보입니다.",
  };
}

function mapAutoCaptureStatus(
  data: Record<string, unknown>,
  fallbackStatus: string
): AutoCaptureStatus {
  return {
    running: Boolean(data.running),
    captured: Boolean(data.captured),
    profile_id: typeof data.profile_id === "string" ? data.profile_id : null,
    capture_id: typeof data.capture_id === "string" ? data.capture_id : null,
    status: typeof data.status === "string" ? data.status : fallbackStatus,
    error: typeof data.error === "string" ? data.error : null,
    checks:
      typeof data.checks === "object" && data.checks !== null
        ? { ...EMPTY_AUTO_CHECKS, ...data.checks }
        : EMPTY_AUTO_CHECKS,
    stable_face_count: Number(data.stable_face_count || 0),
    eyes_closed_count: Number(data.eyes_closed_count || 0),
    dynamic_eye_threshold: Number(data.dynamic_eye_threshold || 0),
    white_led_is_on: Boolean(data.white_led_is_on),
    last_update: typeof data.last_update === "string" ? data.last_update : null,
  };
}

function mapPorphyrinResult(data: Record<string, unknown>): PorphyrinResult {
  return {
    porphyrin_count: Number(data.porphyrin_count || 0),
    porphyrin_area: Number(data.porphyrin_area || 0),
    detection_rate_percent: Number(data.detection_rate_percent || 0),
    porphyrin_mean_brightness: Number(data.porphyrin_mean_brightness || 0),
    porphyrin_top5_max_brightness: Number(data.porphyrin_top5_max_brightness || 0),
    grade: typeof data.grade === "string" ? data.grade : "-",
    skin_score:
      typeof data.skin_score === "object" && data.skin_score !== null
        ? data.skin_score as PorphyrinResult["skin_score"]
        : {
            score: 0,
            grade: "-",
            label: "분석 필요",
            basis: "porphyrin_count",
            porphyrin_count: Number(data.porphyrin_count || 0),
            reference_bad_count: 80,
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

function App() {
  const [screen, setScreen] = useState<Screen>("profiles");

  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<Profile | null>(null);

  const [showGuide, setShowGuide] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);

  const [profileInputText, setProfileInputText] = useState("");
  const [hangulBuffer, setHangulBuffer] = useState<HangulBuffer>(EMPTY_HANGUL_BUFFER);
  const [showVirtualKeyboard, setShowVirtualKeyboard] = useState(false);
  const [keyboardMode, setKeyboardMode] = useState<KeyboardMode>("ko");

  const [isLoadingProfiles, setIsLoadingProfiles] = useState(true);
  const [isCreatingProfile, setIsCreatingProfile] = useState(false);
  const [isDeletingProfile, setIsDeletingProfile] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);

  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isHistoryCompareMode, setIsHistoryCompareMode] = useState(false);
  const [selectedCompareIds, setSelectedCompareIds] = useState<string[]>([]);
  const [isLoadingCompare, setIsLoadingCompare] = useState(false);
  const [compareResults, setCompareResults] = useState<PorphyrinCompareItem[]>([]);

  const [selectedHistory, setSelectedHistory] = useState<HistoryDetail | null>(null);
  const [isLoadingHistoryDetail, setIsLoadingHistoryDetail] = useState(false);

  const [selectedFilter, setSelectedFilter] = useState<HistoryFilter>("no_filter");
  const [detailImageView, setDetailImageView] = useState<DetailImageView>("source");

  const [toast, setToast] = useState<Toast | null>(null);
  const toastTimerRef = useRef<number | null>(null);

  const [deleteTarget, setDeleteTarget] = useState<Profile | null>(null);
  const [historyDeleteTarget, setHistoryDeleteTarget] = useState<HistoryItem | null>(null);
  const [isDeletingHistory, setIsDeletingHistory] = useState<string | null>(null);

  const [isAnalyzingPorphyrin, setIsAnalyzingPorphyrin] = useState(false);
  const [porphyrinResult, setPorphyrinResult] = useState<PorphyrinResult | null>(null);
  const historyScrollRef = useRef<HTMLDivElement | null>(null);
  const compareScrollRef = useRef<HTMLDivElement | null>(null);
  const historyScrollTopRef = useRef(0);

  const [whiteLedOn, setWhiteLedOn] = useState(false);
  const [isChangingWhiteLed, setIsChangingWhiteLed] = useState(false);
  const [isStartingAutoCapture, setIsStartingAutoCapture] = useState(false);
  const [autoStatus, setAutoStatus] = useState<AutoCaptureStatus | null>(null);

  const currentImage = useMemo(() => {
    if (!selectedHistory) return null;
    return selectedHistory.images[selectedFilter] || null;
  }, [selectedHistory, selectedFilter]);

  const isViewingPorphyrinHeatmap =
    detailImageView === "porphyrin_heatmap" && Boolean(porphyrinResult?.heatmap_url);

  const porphyrinRegionPercentages = useMemo(() => {
    if (!porphyrinResult?.region_analysis) {
      return normalizePercentagesTo100({});
    }

    return normalizePercentagesTo100(porphyrinResult.region_analysis);
  }, [porphyrinResult]);

  const porphyrinBrightnessReview = useMemo(() => {
    if (!porphyrinResult) return null;
    return getPorphyrinBrightnessReview(porphyrinResult.porphyrin_mean_brightness);
  }, [porphyrinResult]);

  const compareChartMax = useMemo(() => {
    return {
      count: Math.max(1, ...compareResults.map((item) => item.porphyrin_count)),
      rate: Math.max(1, ...compareResults.map((item) => item.detection_rate_percent)),
    };
  }, [compareResults]);

  const selectedImageLabel = isViewingPorphyrinHeatmap
    ? "Porphyrin_Heatmap"
    : currentImage?.display_name || "-";

  const selectedHistoryIndex = useMemo(() => {
    if (!selectedHistory) return -1;
    return historyItems.findIndex((item) => item.captureId === selectedHistory.captureId);
  }, [historyItems, selectedHistory]);

  const newerHistoryItem =
    selectedHistoryIndex > 0 ? historyItems[selectedHistoryIndex - 1] : null;
  const olderHistoryItem =
    selectedHistoryIndex >= 0 && selectedHistoryIndex < historyItems.length - 1
      ? historyItems[selectedHistoryIndex + 1]
      : null;

  const profileNameInputValue = profileInputText + composeHangul(hangulBuffer);

  const autoChecks = autoStatus?.checks ?? EMPTY_AUTO_CHECKS;
  const isAutoRunning = autoStatus?.running ?? false;
  const showAutoPanel = Boolean(autoStatus && (autoStatus.running || autoStatus.error));

  const showToast = useMemo(() => (
    message: string,
    type: "success" | "error" | "info" = "info"
  ) => {
    setToast({ message, type });

    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
    }

    toastTimerRef.current = window.setTimeout(() => {
      setToast(null);
      toastTimerRef.current = null;
    }, 2500);
  }, []);

  const selectHistoryFilter = (filter: HistoryFilter) => {
    setSelectedFilter(filter);
    setDetailImageView("source");
  };

  useEffect(() => {
    return () => {
      if (toastTimerRef.current !== null) {
        window.clearTimeout(toastTimerRef.current);
      }
    };
  }, []);

  const fetchProfiles = useMemo(() => async () => {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => {
      controller.abort();
    }, PROFILE_FETCH_TIMEOUT_MS);

    try {
      setIsLoadingProfiles(true);

      const res = await fetch(`${API_BASE}/profiles`, {
        signal: controller.signal,
      });
      const data = await res.json();

      if (!data.ok) {
        console.error(data);
        showToast(data.error || "프로필 목록 불러오기 실패", "error");
        return;
      }

      setProfiles(data.profiles || []);
    } catch (error) {
      console.error(error);
      showToast("프로필 목록 불러오기 실패: 서버 연결을 확인하세요.", "error");
    } finally {
      window.clearTimeout(timeoutId);
      setIsLoadingProfiles(false);
    }
  }, [showToast]);

  useEffect(() => {
    fetchProfiles();
  }, [fetchProfiles]);

  useEffect(() => {
    if (!autoStatus?.captured || !autoStatus.capture_id) {
      return;
    }

    showToast("자동 촬영이 완료되었습니다.", "success");
  }, [autoStatus?.captured, autoStatus?.capture_id, showToast]);

  useEffect(() => {
    if (screen !== "history") return;

    window.requestAnimationFrame(() => {
      if (historyScrollRef.current) {
        historyScrollRef.current.scrollTop = historyScrollTopRef.current;
      }
    });
  }, [screen]);

  const createProfile = async () => {
    const trimmed = profileNameInputValue.trim();

    if (!trimmed) {
      showToast("프로필 이름을 입력하세요.", "error");
      return;
    }

    try {
      setIsCreatingProfile(true);

      const res = await fetch(`${API_BASE}/profiles`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name: trimmed }),
      });

      const data = await res.json();

      if (!data.ok) {
        showToast(data.error || "프로필 생성 실패", "error");
        return;
      }

      setProfiles((prev) => [...prev, data.profile]);
      setProfileInputText("");
      setHangulBuffer(EMPTY_HANGUL_BUFFER);
      setShowVirtualKeyboard(false);
      setKeyboardMode("ko");
      setShowCreateModal(false);
      showToast("프로필이 생성되었습니다.", "success");
    } catch (error) {
      console.error(error);
      showToast("프로필 생성 실패", "error");
    } finally {
      setIsCreatingProfile(false);
    }
  };

  const commitHangulBuffer = () => {
    const composed = composeHangul(hangulBuffer);

    if (composed) {
      setProfileInputText((prev) => prev + composed);
    }

    setHangulBuffer(EMPTY_HANGUL_BUFFER);
  };

  const resetProfileInput = () => {
    setProfileInputText("");
    setHangulBuffer(EMPTY_HANGUL_BUFFER);
    setShowVirtualKeyboard(false);
    setKeyboardMode("ko");
  };

  const openCreateProfileModal = () => {
    resetProfileInput();
    setShowCreateModal(true);
  };

  const handleKoreanConsonant = (key: string) => {
    const prev = hangulBuffer;

    if (!prev.cho && !prev.jung && !prev.jong) {
      setHangulBuffer({
        cho: key,
        jung: null,
        jong: null,
      });
      return;
    }

    if (prev.cho && !prev.jung) {
      const doubleCho = DOUBLE_CHO_MAP[`${prev.cho}${key}`];

      if (doubleCho) {
        setHangulBuffer({
          cho: doubleCho,
          jung: null,
          jong: null,
        });
        return;
      }

      setProfileInputText((text) => text + composeHangul(prev));
      setHangulBuffer({
        cho: key,
        jung: null,
        jong: null,
      });
      return;
    }

    if (prev.cho && prev.jung && !prev.jong) {
      if (JONG_LIST.includes(key)) {
        setHangulBuffer({
          ...prev,
          jong: key,
        });
        return;
      }

      setProfileInputText((text) => text + composeHangul(prev));
      setHangulBuffer({
        cho: key,
        jung: null,
        jong: null,
      });
      return;
    }

    if (prev.cho && prev.jung && prev.jong) {
      const compoundJong = COMPOUND_JONG_MAP[`${prev.jong}${key}`];

      if (compoundJong) {
        setHangulBuffer({
          ...prev,
          jong: compoundJong,
        });
        return;
      }

      setProfileInputText((text) => text + composeHangul(prev));
      setHangulBuffer({
        cho: key,
        jung: null,
        jong: null,
      });
      return;
    }
  };

  const handleKoreanVowel = (key: string) => {
    const prev = hangulBuffer;

    if (!prev.cho && !prev.jung && !prev.jong) {
      setHangulBuffer({
        cho: "ㅇ",
        jung: key,
        jong: null,
      });
      return;
    }

    if (prev.cho && !prev.jung) {
      setHangulBuffer({
        ...prev,
        jung: key,
      });
      return;
    }

    if (prev.cho && prev.jung && !prev.jong) {
      const compoundJung = COMPOUND_JUNG_MAP[`${prev.jung}${key}`];

      if (compoundJung) {
        setHangulBuffer({
          ...prev,
          jung: compoundJung,
        });
        return;
      }

      setProfileInputText((text) => text + composeHangul(prev));
      setHangulBuffer({
        cho: "ㅇ",
        jung: key,
        jong: null,
      });
      return;
    }

    if (prev.cho && prev.jung && prev.jong) {
      const splitJong = SPLIT_JONG_MAP[prev.jong];

      if (splitJong) {
        const [remainJong, nextCho] = splitJong;

        setProfileInputText(
          (text) =>
            text +
            composeHangul({
              cho: prev.cho,
              jung: prev.jung,
              jong: remainJong,
            })
        );

        setHangulBuffer({
          cho: nextCho,
          jung: key,
          jong: null,
        });
        return;
      }

      setProfileInputText((text) => text + composeHangulWithoutJong(prev));
      setHangulBuffer({
        cho: prev.jong,
        jung: key,
        jong: null,
      });
      return;
    }
  };

  const handleVirtualKey = (key: string) => {
    if (KOREAN_CONSONANTS.has(key)) {
      handleKoreanConsonant(key);
      return;
    }

    if (KOREAN_VOWELS.has(key)) {
      handleKoreanVowel(key);
      return;
    }

    const composed = composeHangul(hangulBuffer);

    if (composed) {
      setProfileInputText((prev) => prev + composed + key);
    } else {
      setProfileInputText((prev) => prev + key);
    }

    setHangulBuffer(EMPTY_HANGUL_BUFFER);
  };

  const handleVirtualBackspace = () => {
    const prev = hangulBuffer;

    if (prev.jong) {
      const splitJong = SPLIT_JONG_MAP[prev.jong];

      if (splitJong) {
        setHangulBuffer({
          ...prev,
          jong: splitJong[0],
        });
        return;
      }

      setHangulBuffer({
        ...prev,
        jong: null,
      });
      return;
    }

    if (prev.jung) {
      setHangulBuffer({
        ...prev,
        jung: null,
      });
      return;
    }

    if (prev.cho) {
      setHangulBuffer(EMPTY_HANGUL_BUFFER);
      return;
    }

    setProfileInputText((prevText) => prevText.slice(0, -1));
  };

  const handleVirtualSpace = () => {
    const composed = composeHangul(hangulBuffer);

    if (composed) {
      setProfileInputText((prev) => prev + composed + " ");
    } else {
      setProfileInputText((prev) => prev + " ");
    }

    setHangulBuffer(EMPTY_HANGUL_BUFFER);
  };

  const handleVirtualDone = () => {
    const composed = composeHangul(hangulBuffer);

    if (composed) {
      setProfileInputText((prev) => prev + composed);
    }

    setHangulBuffer(EMPTY_HANGUL_BUFFER);
    setShowVirtualKeyboard(false);
  };

  const requestDeleteProfile = (profile: Profile) => {
    setDeleteTarget(profile);
  };

  const deleteProfile = async () => {
    if (!deleteTarget) return;

    const profile = deleteTarget;

    try {
      setIsDeletingProfile(profile.folderId);

      const encodedId = encodeURIComponent(profile.folderId);
      const res = await fetch(`${API_BASE}/profiles/${encodedId}`, {
        method: "DELETE",
      });

      const data = await res.json();

      if (!data.ok) {
        showToast(data.error || "프로필 삭제 실패", "error");
        return;
      }

      setProfiles((prev) => prev.filter((p) => p.folderId !== profile.folderId));

      if (selectedProfile?.folderId === profile.folderId) {
        setSelectedProfile(null);
        setScreen("profiles");
      }

      setDeleteTarget(null);
      showToast("프로필이 삭제되었습니다.", "success");
    } catch (error) {
      console.error(error);
      showToast("프로필 삭제 실패", "error");
    } finally {
      setIsDeletingProfile(null);
    }
  };

  const selectProfile = (profile: Profile) => {
    setSelectedProfile(profile);
    setSelectedHistory(null);
    setHistoryItems([]);
    setIsHistoryCompareMode(false);
    setSelectedCompareIds([]);
    setCompareResults([]);
    setPorphyrinResult(null);
    setDetailImageView("source");
    setAutoStatus(null);
    setScreen("camera");
  };

  const goToProfiles = () => {
    setScreen("profiles");
    setSelectedProfile(null);
    setSelectedHistory(null);
    setHistoryItems([]);
    setIsHistoryCompareMode(false);
    setSelectedCompareIds([]);
    setCompareResults([]);
    setPorphyrinResult(null);
    setDetailImageView("source");
    setAutoStatus(null);
  };

  const capturePhoto = async () => {
    if (!selectedProfile) {
      showToast("프로필을 먼저 선택하세요.", "error");
      return;
    }

    try {
      setIsCapturing(true);
      showToast("촬영을 시작합니다.", "info");

      const res = await fetch(`${API_BASE}/capture-all`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          profileId: selectedProfile.folderId,
        }),
      });

      const data = await res.json();

      if (!data.ok) {
        showToast(data.error || "촬영 실패", "error");
        console.error(data);
        return;
      }

      showToast(`${selectedProfile.name} 프로필에 저장 완료`, "success");
      console.log(data);
    } catch (error) {
      console.error(error);
      showToast("촬영 실패", "error");
    } finally {
      setIsCapturing(false);
    }
  };

  const fetchHistory = async () => {
    if (!selectedProfile) {
      showToast("프로필을 먼저 선택하세요.", "error");
      return;
    }

    try {
      setIsLoadingHistory(true);

      const encodedId = encodeURIComponent(selectedProfile.folderId);
      const res = await fetch(`${API_BASE}/profiles/${encodedId}/history`);
      const data = await res.json();

      if (!data.ok) {
        showToast(data.error || "이전 기록 불러오기 실패", "error");
        return;
      }

      setHistoryItems(data.history || []);
      setScreen("history");
    } catch (error) {
      console.error(error);
      showToast("이전 기록 불러오기 실패", "error");
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const fetchPorphyrinReport = async (
    encodedProfileId: string,
    encodedCaptureId: string
  ): Promise<PorphyrinResult | null> => {
    const res = await fetch(
      `${API_BASE}/profiles/${encodedProfileId}/history/${encodedCaptureId}/analysis/porphyrin-report`
    );
    const data = await res.json();

    if (!data.ok) {
      return null;
    }

    return mapPorphyrinResult(data);
  };

  const openHistoryDetail = async (captureId: string, resetFilter = true) => {
    if (!selectedProfile) {
      showToast("프로필을 먼저 선택하세요.", "error");
      return;
    }

    try {
      setIsLoadingHistoryDetail(true);

      const encodedProfileId = encodeURIComponent(selectedProfile.folderId);
      const encodedCaptureId = encodeURIComponent(captureId);

      const res = await fetch(
        `${API_BASE}/profiles/${encodedProfileId}/history/${encodedCaptureId}`
      );
      const data = await res.json();

      if (!data.ok) {
        showToast(data.error || "기록 상세 불러오기 실패", "error");
        return;
      }

      setSelectedHistory(data);
      if (resetFilter) {
        setSelectedFilter("no_filter");
      }
      setDetailImageView("source");
      const savedPorphyrinResult = await fetchPorphyrinReport(
        encodedProfileId,
        encodedCaptureId
      );
      setPorphyrinResult(savedPorphyrinResult);
      setScreen("historyDetail");
    } catch (error) {
      console.error(error);
      showToast("기록 상세 불러오기 실패", "error");
    } finally {
      setIsLoadingHistoryDetail(false);
    }
  };

  const requestDeleteHistory = (item: HistoryItem) => {
    setHistoryDeleteTarget(item);
  };

  const deleteHistory = async () => {
    if (!selectedProfile || !historyDeleteTarget) return;

    const target = historyDeleteTarget;

    try {
      setIsDeletingHistory(target.captureId);

      const encodedProfileId = encodeURIComponent(selectedProfile.folderId);
      const encodedCaptureId = encodeURIComponent(target.captureId);

      const res = await fetch(
        `${API_BASE}/profiles/${encodedProfileId}/history/${encodedCaptureId}`,
        {
          method: "DELETE",
        }
      );

      const data = await res.json();

      if (!data.ok) {
        showToast(data.error || "기록 삭제 실패", "error");
        return;
      }

      setHistoryItems((prev) =>
        prev.filter((item) => item.captureId !== target.captureId)
      );
      setSelectedCompareIds((prev) =>
        prev.filter((captureId) => captureId !== target.captureId)
      );
      setCompareResults((prev) =>
        prev.filter((item) => item.captureId !== target.captureId)
      );

      if (selectedHistory?.captureId === target.captureId) {
        setSelectedHistory(null);
        setPorphyrinResult(null);
        setDetailImageView("source");
        setScreen("history");
      }

      setHistoryDeleteTarget(null);
      showToast("촬영 기록이 삭제되었습니다.", "success");
    } catch (error) {
      console.error(error);
      showToast("기록 삭제 실패", "error");
    } finally {
      setIsDeletingHistory(null);
    }
  };

  const startHistoryCompareMode = () => {
    setIsHistoryCompareMode(true);
    setSelectedCompareIds([]);
    setCompareResults([]);
  };

  const cancelHistoryCompareMode = () => {
    setIsHistoryCompareMode(false);
    setSelectedCompareIds([]);
  };

  const toggleCompareSelection = (captureId: string) => {
    setSelectedCompareIds((prev) => {
      if (prev.includes(captureId)) {
        return prev.filter((id) => id !== captureId);
      }

      if (prev.length >= 5) {
        showToast("비교는 최대 5개까지 선택할 수 있습니다.", "error");
        return prev;
      }

      return [...prev, captureId];
    });
  };

  const openHistoryCompare = async () => {
    if (!selectedProfile) {
      showToast("프로필을 먼저 선택하세요.", "error");
      return;
    }

    if (selectedCompareIds.length < 2) {
      showToast("비교할 기록을 2개 이상 선택하세요.", "error");
      return;
    }

    try {
      setIsLoadingCompare(true);

      const encodedProfileId = encodeURIComponent(selectedProfile.folderId);
      const reports: PorphyrinCompareItem[] = [];
      let analyzedMissingResult = false;

      for (const captureId of selectedCompareIds) {
        const encodedCaptureId = encodeURIComponent(captureId);
        const historyItem = historyItems.find((history) => history.captureId === captureId);
        const reportUrl =
          `${API_BASE}/profiles/${encodedProfileId}/history/${encodedCaptureId}/analysis/porphyrin-report`;

        let res = await fetch(reportUrl);
        let data = await res.json();

        if (!data.ok) {
          analyzedMissingResult = true;
          showToast(
            `${historyItem?.displayTime || "선택한 기록"} 포르피린 분석 중...`,
            "info"
          );

          res = await fetch(
            `${API_BASE}/profiles/${encodedProfileId}/history/${encodedCaptureId}/analyze-porphyrin`,
            {
              method: "POST",
            }
          );
          data = await res.json();

          if (!data.ok) {
            throw new Error(
              `${historyItem?.displayTime || captureId}: ${data.error || "포르피린 분석 실패"}`
            );
          }
        }

        reports.push({
          ...mapPorphyrinResult(data),
          captureId,
          displayTime: historyItem?.displayTime || captureId,
        });
      }

      setCompareResults(reports);
      setScreen("historyCompare");
      showToast(
        analyzedMissingResult
          ? "분석 후 비교 결과를 불러왔습니다."
          : "비교 결과를 불러왔습니다.",
        "success"
      );
    } catch (error) {
      console.error(error);
      showToast(
        error instanceof Error
          ? error.message
          : "비교 결과를 불러오지 못했습니다.",
        "error"
      );
    } finally {
      setIsLoadingCompare(false);
    }
  };

  const analyzePorphyrin = async () => {
    if (!selectedProfile || !selectedHistory) {
      showToast("분석할 촬영 기록이 없습니다.", "error");
      return;
    }

    try {
      setIsAnalyzingPorphyrin(true);
      showToast("포르피린 분석을 시작합니다.", "info");

      const encodedProfileId = encodeURIComponent(selectedProfile.folderId);
      const encodedCaptureId = encodeURIComponent(selectedHistory.captureId);

      const res = await fetch(
        `${API_BASE}/profiles/${encodedProfileId}/history/${encodedCaptureId}/analyze-porphyrin`,
        {
          method: "POST",
        }
      );

      const data = await res.json();

      if (!data.ok) {
        showToast(data.error || "포르피린 분석 실패", "error");
        return;
      }

      setPorphyrinResult(mapPorphyrinResult(data));
      setSelectedFilter("660nm_filter");
      setDetailImageView("porphyrin_heatmap");
      showToast("포르피린 분석 완료", "success");
    } catch (error) {
      console.error(error);
      showToast("포르피린 분석 실패", "error");
    } finally {
      setIsAnalyzingPorphyrin(false);
    }
  };

  const fetchWhiteLedStatus = useMemo(() => async () => {
    try {
      const res = await fetch(`${API_BASE}/white-led/status`);
      const data = await res.json();

      if (data.ok) {
        setWhiteLedOn(Boolean(data.white_led_is_on));
      }
    } catch (error) {
      console.error(error);
    }
  }, []);

  const toggleWhiteLed = async () => {
    try {
      setIsChangingWhiteLed(true);

      const nextOn = !whiteLedOn;
      const res = await fetch(`${API_BASE}/white-led/${nextOn ? "on" : "off"}`, {
        method: "POST",
      });
      const data = await res.json();

      if (!data.ok) {
        showToast(data.error || "백색 조명 제어 실패", "error");
        return;
      }

      setWhiteLedOn(Boolean(data.white_led_is_on));
      showToast(data.white_led_is_on ? "백색 조명 켜짐" : "백색 조명 꺼짐", "success");
    } catch (error) {
      console.error(error);
      showToast("백색 조명 제어 실패", "error");
    } finally {
      setIsChangingWhiteLed(false);
    }
  };

  const fetchAutoCaptureStatus = useMemo(() => async (showError = false) => {
    try {
      const res = await fetch(`${API_BASE}/auto-capture/status`);
      const data = await res.json();

      if (!data.ok) {
        if (showError) {
          showToast(data.error || "자동 촬영 상태 확인 실패", "error");
        }
        return;
      }

      setAutoStatus(mapAutoCaptureStatus(data, "자동 촬영 대기 중"));

      if (typeof data.white_led_is_on === "boolean") {
        setWhiteLedOn(data.white_led_is_on);
      }
    } catch (error) {
      console.error(error);
      if (showError) {
        showToast("자동 촬영 상태 확인 실패", "error");
      }
    }
  }, [showToast]);

  useEffect(() => {
    if (screen !== "camera") {
      return;
    }

    fetchWhiteLedStatus();
    fetchAutoCaptureStatus(false);

    const intervalId = window.setInterval(() => {
      fetchAutoCaptureStatus(false);
    }, 500);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [fetchAutoCaptureStatus, fetchWhiteLedStatus, screen]);

  const startAutoCapture = async () => {
    if (!selectedProfile) {
      showToast("프로필을 먼저 선택하세요.", "error");
      return;
    }

    try {
      setIsStartingAutoCapture(true);
      setAutoStatus(null);
      showToast("자동 촬영 조건 확인을 시작합니다.", "info");

      const res = await fetch(`${API_BASE}/auto-capture/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          profileId: selectedProfile.folderId,
        }),
      });
      const data = await res.json();

      if (!data.ok) {
        showToast(data.error || "자동 촬영 시작 실패", "error");
        return;
      }

      setAutoStatus(mapAutoCaptureStatus(data, "자동 촬영 조건 확인 중"));

      if (typeof data.white_led_is_on === "boolean") {
        setWhiteLedOn(data.white_led_is_on);
      }
    } catch (error) {
      console.error(error);
      showToast("자동 촬영 시작 실패", "error");
    } finally {
      setIsStartingAutoCapture(false);
    }
  };

  const cancelAutoCapture = async () => {
    try {
      const res = await fetch(`${API_BASE}/auto-capture/cancel`, {
        method: "POST",
      });
      const data = await res.json();

      if (!data.ok) {
        showToast(data.error || "자동 촬영 취소 실패", "error");
        return;
      }

      await fetchAutoCaptureStatus(false);
      showToast("자동 촬영을 취소했습니다.", "info");
    } catch (error) {
      console.error(error);
      showToast("자동 촬영 취소 실패", "error");
    }
  };

  const openHistory = () => {
    historyScrollTopRef.current = 0;
    fetchHistory();
  };

  const backToCamera = () => {
    setScreen("camera");
  };

  const backToHistory = () => {
    setScreen("history");
  };

  const openHistoryCard = (captureId: string) => {
    historyScrollTopRef.current = historyScrollRef.current?.scrollTop ?? 0;
    openHistoryDetail(captureId);
  };

  const moveHistoryDetail = (item: HistoryItem | null) => {
    if (!item || isLoadingHistoryDetail) return;
    openHistoryDetail(item.captureId, false);
  };

  const scrollHistoryList = (direction: "up" | "down") => {
    const target = historyScrollRef.current;
    if (!target) return;

    const distance = Math.max(260, Math.floor(target.clientHeight * 0.72));
    target.scrollBy({
      top: direction === "up" ? -distance : distance,
      behavior: "smooth",
    });
  };

  const scrollCompareView = (direction: "up" | "down") => {
    const target = compareScrollRef.current;
    if (!target) return;

    const distance = Math.max(260, Math.floor(target.clientHeight * 0.72));
    target.scrollBy({
      top: direction === "up" ? -distance : distance,
      behavior: "smooth",
    });
  };

  const getImageSrc = (imageUrl: string | null | undefined) => {
    if (!imageUrl) return "";

    if (imageUrl.startsWith("http://") || imageUrl.startsWith("https://")) {
      return imageUrl;
    }

    return `${API_BASE}${imageUrl}`;
  };

  return (
    <>

      <div className="app-bg">
        <div className="mirror-frame">
          {toast && (
            <div className={`toast ${toast.type}`}>
              {toast.message}
            </div>
          )}

          {screen === "profiles" && (
            <>
              <div className="header">
                <div>
                  <h1 className="header-title">프로필 선택</h1>
                  <div className="header-subtitle">
                    사용할 프로필을 선택하거나 새로 추가해 주세요.
                    <br />
                    프로필별로 촬영 기록이 구분 저장됩니다.
                  </div>
                </div>

                <div className="time-badge">
                  {new Intl.DateTimeFormat("ko-KR", {
                    hour: "2-digit",
                    minute: "2-digit",
                    hour12: false,
                  }).format(new Date())}
                </div>
              </div>

              <div className="profiles-container">
                {isLoadingProfiles ? (
                  <div className="loading-box">프로필 불러오는 중...</div>
                ) : (
                  <div className="profiles-grid">
                    {profiles.map((profile) => (
                      <div key={profile.id} className="profile-card">
                        <div
                          className="profile-click-area"
                          onClick={() => selectProfile(profile)}
                        >
                          <div className="profile-icon">👤</div>
                          <div className="profile-name">{profile.name}</div>
                          <div className="profile-date">
                            생성일 {profile.createdAt}
                          </div>
                          <div className="profile-select-tag">선택하기</div>
                        </div>

                        <div className="profile-card-actions">
                          <button
                            className="delete-btn"
                            onClick={(e) => {
                              e.stopPropagation();
                              requestDeleteProfile(profile);
                            }}
                            disabled={isDeletingProfile === profile.folderId}
                          >
                            {isDeletingProfile === profile.folderId
                              ? "삭제 중..."
                              : "삭제"}
                          </button>
                        </div>
                      </div>
                    ))}

                    <div
                      className="add-card"
                      onClick={openCreateProfileModal}
                    >
                      <div className="add-icon">＋</div>
                      <div style={{ fontSize: "20px", fontWeight: 700 }}>
                        프로필 추가
                      </div>
                      <div
                        style={{
                          marginTop: "10px",
                          fontSize: "14px",
                          lineHeight: 1.6,
                          color: "rgba(255,255,255,0.6)",
                        }}
                      >
                        새 사용자 프로필을 만들어
                        <br />
                        촬영 기록을 구분합니다.
                      </div>
                    </div>
                  </div>
                )}

                {!isLoadingProfiles && profiles.length === 0 && (
                  <div className="empty-box">
                    아직 프로필이 없습니다.
                    <br />
                    오른쪽 카드의 프로필 추가를 눌러 시작하세요.
                    <br />
                    <button className="retry-btn" onClick={fetchProfiles}>
                      다시 불러오기
                    </button>
                  </div>
                )}
              </div>
            </>
          )}

          {screen === "camera" && (
            <div className="camera-screen">
              <img
                className="camera-live"
                src={`${API_BASE}/stream-cam4`}
                alt="CAM 4 live stream"
              />

              <div className="camera-overlay"></div>

              <div className="selected-profile">
                <small>선택된 프로필</small>
                <strong>{selectedProfile?.name}</strong>
              </div>

              <div className="controls">
                <button
                  className="control-btn"
                  onClick={() => setShowGuide((prev) => !prev)}
                  disabled={isCapturing || isAutoRunning}
                >
                  {showGuide ? "실루엣 끄기" : "실루엣 켜기"}
                </button>

                <button
                  className={`control-btn white-led-control ${whiteLedOn ? "on" : ""}`}
                  onClick={toggleWhiteLed}
                  disabled={isCapturing || isChangingWhiteLed}
                >
                  {whiteLedOn ? "백색 조명 끄기" : "백색 조명 켜기"}
                </button>

                <button
                  className="control-btn auto-capture-control"
                  onClick={startAutoCapture}
                  disabled={isCapturing || isAutoRunning || isStartingAutoCapture}
                >
                  {isStartingAutoCapture ? "자동 촬영 준비 중..." : "자동 얼굴 촬영"}
                </button>

                {isAutoRunning && (
                  <button
                    className="control-btn auto-cancel-control"
                    onClick={cancelAutoCapture}
                  >
                    자동 촬영 취소
                  </button>
                )}

                <button
                  className="control-btn"
                  onClick={openHistory}
                  disabled={isCapturing || isAutoRunning}
                >
                  이전 기록 확인
                </button>
              </div>

              {showAutoPanel && (
                <div className={`auto-check-panel ${autoStatus?.captured ? "done" : ""}`}>
                  <div className="auto-check-header">
                    <div>
                      <div className="auto-check-title">자동 촬영 조건 확인</div>
                      <div className="auto-check-message">
                        {autoStatus?.error || autoStatus?.status || "자동 촬영을 시작해 주세요."}
                      </div>
                    </div>

                    <div className={`auto-check-state ${autoStatus?.captured ? "done" : isAutoRunning ? "running" : "idle"}`}>
                      {autoStatus?.captured ? "완료" : isAutoRunning ? "확인 중" : "대기"}
                    </div>
                  </div>

                  <div className="auto-check-list">
                    {AUTO_CHECK_LABELS.map((item) => {
                      const ok = Boolean(autoChecks[item.key]);

                      return (
                        <div
                          key={item.key}
                          className={`auto-check-item ${ok ? "ok" : "fail"}`}
                        >
                          <span className="auto-check-icon">
                            {ok ? "✓" : "!"}
                          </span>
                          <span>{item.label}</span>
                        </div>
                      );
                    })}
                  </div>

                  {autoStatus?.capture_id && (
                    <div className="auto-capture-id">
                      저장된 captureId: {autoStatus.capture_id}
                    </div>
                  )}
                </div>
              )}

              {showGuide && (
                <div className="guide-wrap">
                  <div className="guide-face" aria-label="얼굴 위치 기준선">
                    <div className="guide-eye-line"></div>
                    <div className="guide-nose-line"></div>
                    <div className="guide-mouth-line"></div>
                  </div>
                </div>
              )}

              <button
                className="back-btn"
                onClick={goToProfiles}
                disabled={isCapturing || isAutoRunning}
              >
                프로필로 돌아가기
              </button>

              {(isCapturing || isAutoRunning) && (
                <div className="capture-status">
                  {isCapturing ? "촬영 중..." : autoStatus?.status || "자동 촬영 조건 확인 중..."}
                </div>
              )}

              <div className="capture-area">
                <button
                  className="capture-btn"
                  onClick={capturePhoto}
                  disabled={isCapturing || isAutoRunning}
                >
                  <div className="capture-inner"></div>
                </button>
              </div>
            </div>
          )}

          {screen === "history" && (
            <>
              <div className="header">
                <div>
                  <h1 className="header-title">이전 기록</h1>
                  <div className="header-subtitle">
                    {selectedProfile?.name} 프로필의 촬영 날짜와 시간을 확인합니다.
                    <br />
                    원하는 기록을 누르면 필터별 사진을 볼 수 있습니다.
                  </div>
                </div>

                <div className="time-badge">
                  {new Intl.DateTimeFormat("ko-KR", {
                    hour: "2-digit",
                    minute: "2-digit",
                    hour12: false,
                  }).format(new Date())}
                </div>
              </div>

              <div className="history-scroll-controls" aria-label="이전 기록 스크롤">
                <button
                  className="history-scroll-btn"
                  type="button"
                  onClick={() => scrollHistoryList("up")}
                  aria-label="이전 기록 위로 스크롤"
                >
                  ▲
                </button>
                <button
                  className="history-scroll-btn"
                  type="button"
                  onClick={() => scrollHistoryList("down")}
                  aria-label="이전 기록 아래로 스크롤"
                >
                  ▼
                </button>
              </div>

              <div className="history-container" ref={historyScrollRef}>
                <div className="history-action-row">
                  <button
                    className="mini-back-btn history-camera-back-btn"
                    type="button"
                    onClick={backToCamera}
                  >
                    카메라로 돌아가기
                  </button>

                  {historyItems.length > 0 && (
                    <div className="history-compare-actions">
                      {isHistoryCompareMode ? (
                        <>
                          <button
                            className="compare-secondary-btn"
                            type="button"
                            onClick={cancelHistoryCompareMode}
                            disabled={isLoadingCompare}
                          >
                            취소
                          </button>
                          <button
                            className="compare-primary-btn"
                            type="button"
                            onClick={openHistoryCompare}
                            disabled={isLoadingCompare || selectedCompareIds.length < 2}
                          >
                            {isLoadingCompare
                              ? "불러오는 중..."
                              : `비교 보기 ${selectedCompareIds.length}/5`}
                          </button>
                        </>
                      ) : (
                        <button
                          className="compare-primary-btn"
                          type="button"
                          onClick={startHistoryCompareMode}
                        >
                          비교하기
                        </button>
                      )}
                    </div>
                  )}
                </div>

                {isLoadingHistory ? (
                  <div className="loading-box">이전 기록 불러오는 중...</div>
                ) : historyItems.length === 0 ? (
                  <div className="empty-box">
                    아직 저장된 촬영 기록이 없습니다.
                    <br />
                    먼저 촬영을 진행해 주세요.
                  </div>
                ) : (
                  <>
                    {historyItems.map((item) => (
                      <div
                        key={item.captureId}
                        className={`history-card ${isHistoryCompareMode ? "compare-mode" : ""} ${
                          selectedCompareIds.includes(item.captureId) ? "selected" : ""
                        }`}
                        onClick={() => {
                          if (isHistoryCompareMode) {
                            toggleCompareSelection(item.captureId);
                            return;
                          }

                          openHistoryCard(item.captureId);
                        }}
                      >
                        <div className="history-card-header">
                          {isHistoryCompareMode && (
                            <label className="history-compare-check">
                              <input
                                type="checkbox"
                                checked={selectedCompareIds.includes(item.captureId)}
                                onChange={() => toggleCompareSelection(item.captureId)}
                                onClick={(e) => e.stopPropagation()}
                              />
                              <span></span>
                            </label>
                          )}

                          <div>
                            <div className="history-card-time">{item.displayTime}</div>
                            <div className="history-card-sub">
                              프로필: {item.profileName}
                            </div>
                          </div>

                          {!isHistoryCompareMode && (
                            <button
                              className="history-delete-btn"
                              onClick={(e) => {
                                e.stopPropagation();
                                requestDeleteHistory(item);
                              }}
                              disabled={isDeletingHistory === item.captureId}
                            >
                              {isDeletingHistory === item.captureId ? "삭제 중..." : "삭제"}
                            </button>
                          )}
                        </div>

                        <div className="history-tag">
                          {isHistoryCompareMode ? "비교 선택" : "기록 열기"}
                        </div>
                      </div>
                    ))}
                  </>
                )}
              </div>
            </>
          )}

          {screen === "historyCompare" && (
            <>
              <div className="header">
                <div>
                  <h1 className="header-title">포르피린 비교</h1>
                  <div className="header-subtitle">
                    선택한 촬영 기록의 포르피린 수치와 부위별 분포를 비교합니다.
                  </div>
                </div>

                <div className="time-badge">
                  {new Intl.DateTimeFormat("ko-KR", {
                    hour: "2-digit",
                    minute: "2-digit",
                    hour12: false,
                  }).format(new Date())}
                </div>
              </div>

              <div className="history-scroll-controls" aria-label="비교 결과 스크롤">
                <button
                  className="history-scroll-btn"
                  type="button"
                  onClick={() => scrollCompareView("up")}
                  aria-label="비교 결과 위로 스크롤"
                >
                  ▲
                </button>
                <button
                  className="history-scroll-btn"
                  type="button"
                  onClick={() => scrollCompareView("down")}
                  aria-label="비교 결과 아래로 스크롤"
                >
                  ▼
                </button>
              </div>

              <div className="history-compare-container" ref={compareScrollRef}>
                <button className="mini-back-btn" type="button" onClick={backToHistory}>
                  기록 목록으로 돌아가기
                </button>

                {compareResults.length === 0 ? (
                  <div className="empty-box">
                    비교할 포르피린 분석 결과가 없습니다.
                    <br />
                    기록 목록에서 비교할 기록을 다시 선택해 주세요.
                  </div>
                ) : (
                  <>
                    <div className="compare-summary-grid">
                      {compareResults.map((item) => (
                        <div className="compare-summary-card" key={item.captureId}>
                          <div className="compare-summary-date">{item.displayTime}</div>
                          <div className="compare-summary-values">
                            <span>{item.porphyrin_count.toLocaleString()}개</span>
                            <span>{item.detection_rate_percent.toFixed(2)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="compare-chart-section">
                      <div className="compare-section-title">포르피린 디텍션 수</div>
                      <div className="compare-bars">
                        {compareResults.map((item) => (
                          <div className="compare-bar-row" key={`count-${item.captureId}`}>
                            <div className="compare-bar-label">{item.displayTime}</div>
                            <div className="compare-bar-track">
                              <div
                                className="compare-bar-fill count"
                                style={{
                                  width: `${Math.max(
                                    4,
                                    (item.porphyrin_count / compareChartMax.count) * 100
                                  )}%`,
                                }}
                              ></div>
                            </div>
                            <div className="compare-bar-value">
                              {item.porphyrin_count.toLocaleString()}개
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="compare-chart-section">
                      <div className="compare-section-title">포르피린 분포 비율</div>
                      <div className="compare-bars">
                        {compareResults.map((item) => (
                          <div className="compare-bar-row" key={`rate-${item.captureId}`}>
                            <div className="compare-bar-label">{item.displayTime}</div>
                            <div className="compare-bar-track">
                              <div
                                className="compare-bar-fill rate"
                                style={{
                                  width: `${Math.max(
                                    4,
                                    (item.detection_rate_percent / compareChartMax.rate) * 100
                                  )}%`,
                                }}
                              ></div>
                            </div>
                            <div className="compare-bar-value">
                              {item.detection_rate_percent.toFixed(2)}%
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="compare-chart-section">
                      <div className="compare-section-title">부위별 분포 비교</div>
                      <div className="compare-region-grid">
                        {PORPHYRIN_REGION_ORDER.map((regionKey) => (
                          <div className="compare-region-card" key={regionKey}>
                            <div className="compare-region-name">
                              {PORPHYRIN_REGION_LABELS[regionKey]}
                            </div>

                            {compareResults.map((item) => {
                              const percentages = normalizePercentagesTo100(item.region_analysis);
                              const value = percentages[regionKey] || 0;

                              return (
                                <div
                                  className="compare-region-row"
                                  key={`${regionKey}-${item.captureId}`}
                                >
                                  <span>{item.displayTime}</span>
                                  <div className="compare-region-track">
                                    <div
                                      className="compare-region-fill"
                                      style={{ width: `${Math.max(4, value)}%` }}
                                    ></div>
                                  </div>
                                  <strong>{value}%</strong>
                                </div>
                              );
                            })}
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </>
          )}

          {screen === "historyDetail" && (
            <>
              <div className="header">
                <div>
                  <h1 className="header-title">촬영 기록 상세</h1>
                  <div className="header-subtitle">
                    필터를 선택하면 해당 촬영 이미지를 확인할 수 있습니다.
                  </div>
                </div>

                <div className="time-badge">
                  {new Intl.DateTimeFormat("ko-KR", {
                    hour: "2-digit",
                    minute: "2-digit",
                    hour12: false,
                  }).format(new Date())}
                </div>
              </div>

              <div className="history-detail-container">
                <button className="mini-back-btn" onClick={backToHistory}>
                  기록 목록으로 돌아가기
                </button>

                {isLoadingHistoryDetail ? (
                  <div className="loading-box">기록 상세 불러오는 중...</div>
                ) : (
                  <>
                    <div className="history-detail-top">
                      <div className="history-detail-nav">
                        <button
                          className="history-nav-btn"
                          type="button"
                          onClick={() => moveHistoryDetail(olderHistoryItem)}
                          disabled={!olderHistoryItem || isLoadingHistoryDetail}
                          aria-label="이전 촬영 기록 보기"
                        >
                          ‹
                        </button>

                        <h2 className="history-detail-title">
                          {selectedHistory?.displayTime || "-"}
                        </h2>

                        <button
                          className="history-nav-btn"
                          type="button"
                          onClick={() => moveHistoryDetail(newerHistoryItem)}
                          disabled={!newerHistoryItem || isLoadingHistoryDetail}
                          aria-label="다음 촬영 기록 보기"
                        >
                          ›
                        </button>
                      </div>

                      <div className="history-detail-sub">
                        프로필: {selectedHistory?.profileName || "-"}
                        <br />
                        선택한 이미지: {selectedImageLabel}
                      </div>

                      <div className="filter-row">
                        <button
                          className={`filter-chip ${
                            selectedFilter === "no_filter" && detailImageView === "source" ? "active" : ""
                          }`}
                          onClick={() => selectHistoryFilter("no_filter")}
                        >
                          No_Filter
                        </button>

                        <button
                          className={`filter-chip ${
                            selectedFilter === "405nm_filter" && detailImageView === "source" ? "active" : ""
                          }`}
                          onClick={() => selectHistoryFilter("405nm_filter")}
                        >
                          405nm_Filter
                        </button>

                        <button
                          className={`filter-chip ${
                            selectedFilter === "660nm_filter" && detailImageView === "source" ? "active" : ""
                          }`}
                          onClick={() => selectHistoryFilter("660nm_filter")}
                        >
                          660nm_Filter
                        </button>

                        {porphyrinResult?.heatmap_url && (
                          <button
                            className={`filter-chip porphyrin-chip ${
                              isViewingPorphyrinHeatmap ? "active" : ""
                            }`}
                            onClick={() => {
                              setSelectedFilter("660nm_filter");
                              setDetailImageView("porphyrin_heatmap");
                            }}
                          >
                            Porphyrin
                          </button>
                        )}
                      </div>
                    </div>

                    <div className="image-viewer">
                      {isViewingPorphyrinHeatmap && porphyrinResult?.heatmap_url ? (
                        porphyrinResult.white_overlay_url && porphyrinResult.uv_heatmap_url ? (
                          <div className="porphyrin-comparison-view">
                            <div className="porphyrin-comparison-item">
                              <div className="porphyrin-comparison-label">White_LED_Match</div>
                              <img
                                className="history-image comparison-image"
                                src={getImageSrc(porphyrinResult.white_overlay_url)}
                                alt="Porphyrin matched on white LED face"
                              />
                            </div>
                            <div className="porphyrin-comparison-item">
                              <div className="porphyrin-comparison-label">UV_Heatmap</div>
                              <img
                                className="history-image comparison-image"
                                src={getImageSrc(porphyrinResult.uv_heatmap_url)}
                                alt="Porphyrin UV heatmap"
                              />
                            </div>
                          </div>
                        ) : (
                          <img
                            className="history-image"
                            src={getImageSrc(porphyrinResult.heatmap_url)}
                            alt="Porphyrin heatmap"
                          />
                        )
                      ) : currentImage?.exists && currentImage?.image_url ? (
                        <img
                          className="history-image"
                          src={getImageSrc(currentImage.image_url)}
                          alt={currentImage.display_name}
                        />
                      ) : (
                        <div className="history-empty-image">
                          선택한 필터의 이미지가 없습니다.
                        </div>
                      )}

                      {isViewingPorphyrinHeatmap && porphyrinBrightnessReview && (
                        <div className="porphyrin-review-box">
                          <div className="porphyrin-review-meta">
                            {porphyrinBrightnessReview.level}단계 · {porphyrinBrightnessReview.status}
                          </div>
                          <div className="porphyrin-review-comment">
                            {porphyrinBrightnessReview.comment}
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="analysis-panel">
                      <div className="analysis-title">포르피린 분석</div>
                      <div className="analysis-description">
                        660nm 필터 이미지에서 강한 형광 후보 영역을 검출합니다.
                        분석 결과는 원본과 검출 결과를 나란히 보여줍니다.
                      </div>

                      {!porphyrinResult && (
                        <button
                          className="analysis-btn"
                          onClick={analyzePorphyrin}
                          disabled={isAnalyzingPorphyrin || !selectedHistory}
                        >
                          {isAnalyzingPorphyrin ? "포르피린 분석 중..." : "포르피린 분석하기"}
                        </button>
                      )}

                      {porphyrinResult && (
                        <>
                          <div className="analysis-stat-grid">
                            <div className="analysis-stat-card analysis-stat-primary">
                              <div className="analysis-stat-label">포르피린 디텍션 수</div>
                              <div className="analysis-stat-value">
                                {porphyrinResult.porphyrin_count.toLocaleString()}개
                              </div>
                            </div>
                            <div className="analysis-stat-card analysis-stat-primary">
                              <div className="analysis-stat-label">포르피린 분포 비율</div>
                              <div className="analysis-stat-value">
                                {porphyrinResult.detection_rate_percent.toFixed(2)}%
                              </div>
                            </div>
                            <div className="analysis-stat-card analysis-stat-primary">
                              <div className="analysis-stat-label">포르피린 최대 밝기</div>
                              <div className="analysis-stat-value">
                                {porphyrinResult.porphyrin_top5_max_brightness.toFixed(0)}
                              </div>
                            </div>
                            <div className="analysis-stat-card analysis-stat-primary">
                              <div className="analysis-stat-label">포르피린 평균 밝기</div>
                              <div className="analysis-stat-value">
                                {porphyrinResult.porphyrin_mean_brightness.toFixed(1)}
                              </div>
                            </div>
                          </div>

                          <div className="analysis-region-title">포르피린 분포도</div>
                          <div className="analysis-region-grid">
                            {PORPHYRIN_REGION_ORDER.map((regionKey) => (
                              <div className="analysis-region-item" key={regionKey}>
                                <span>{PORPHYRIN_REGION_LABELS[regionKey]}</span>
                                <strong>{porphyrinRegionPercentages[regionKey]}%</strong>
                              </div>
                            ))}
                          </div>
                        </>
                      )}

                    </div>
                  </>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {showCreateModal && (
        <div className="modal-overlay">
          <div className="modal-box">
            <h2 className="modal-title">프로필 만들기</h2>

            <label className="input-label">프로필 이름</label>
            <input
              className="text-input"
              type="text"
              placeholder="이름을 입력하세요"
              value={profileNameInputValue}
              readOnly
              onFocus={() => setShowVirtualKeyboard(true)}
              onClick={() => setShowVirtualKeyboard(true)}
              disabled={isCreatingProfile}
            />

            {showVirtualKeyboard && (
              <div className="virtual-keyboard">
                <div className="keyboard-mode-row">
                  <button
                    className={`keyboard-mode-btn ${keyboardMode === "ko" ? "active" : ""}`}
                    onClick={() => setKeyboardMode("ko")}
                    disabled={isCreatingProfile}
                  >
                    한글
                  </button>

                  <button
                    className={`keyboard-mode-btn ${keyboardMode === "en" ? "active" : ""}`}
                    onClick={() => {
                      commitHangulBuffer();
                      setKeyboardMode("en");
                    }}
                    disabled={isCreatingProfile}
                  >
                    영어
                  </button>

                  <button
                    className={`keyboard-mode-btn ${keyboardMode === "num" ? "active" : ""}`}
                    onClick={() => {
                      commitHangulBuffer();
                      setKeyboardMode("num");
                    }}
                    disabled={isCreatingProfile}
                  >
                    숫자
                  </button>
                </div>

                {keyboardMode === "ko" &&
                  KOREAN_KEY_ROWS.map((row, rowIndex) => (
                    <div className="keyboard-row" key={`ko-${rowIndex}`}>
                      {row.map((key) => (
                        <button
                          className="keyboard-key"
                          key={key}
                          onClick={() => handleVirtualKey(key)}
                          disabled={isCreatingProfile}
                        >
                          {key}
                        </button>
                      ))}
                    </div>
                  ))}

                {keyboardMode === "en" &&
                  ENGLISH_KEY_ROWS.map((row, rowIndex) => (
                    <div className="keyboard-row" key={`en-${rowIndex}`}>
                      {row.map((key) => (
                        <button
                          className="keyboard-key"
                          key={key}
                          onClick={() => handleVirtualKey(key)}
                          disabled={isCreatingProfile}
                        >
                          {key}
                        </button>
                      ))}
                    </div>
                  ))}

                {keyboardMode === "num" &&
                  NUMBER_KEY_ROWS.map((row, rowIndex) => (
                    <div className="keyboard-row" key={`num-${rowIndex}`}>
                      {row.map((key) => (
                        <button
                          className="keyboard-key"
                          key={key}
                          onClick={() => handleVirtualKey(key)}
                          disabled={isCreatingProfile}
                        >
                          {key}
                        </button>
                      ))}
                    </div>
                  ))}

                <div className="keyboard-control-row">
                  <button
                    className="keyboard-control-key"
                    onClick={handleVirtualBackspace}
                    disabled={isCreatingProfile}
                  >
                    지우기
                  </button>

                  <button
                    className="keyboard-control-key space"
                    onClick={handleVirtualSpace}
                    disabled={isCreatingProfile}
                  >
                    띄어쓰기
                  </button>

                  <button
                    className="keyboard-control-key done"
                    onClick={handleVirtualDone}
                    disabled={isCreatingProfile}
                  >
                    완료
                  </button>
                </div>
              </div>
            )}

            <div className="modal-actions">
              <button
                className="modal-btn cancel"
                onClick={() => {
                  if (isCreatingProfile) return;
                  setShowCreateModal(false);
                  resetProfileInput();
                }}
                disabled={isCreatingProfile}
              >
                취소
              </button>

              <button
                className="modal-btn create"
                onClick={createProfile}
                disabled={isCreatingProfile}
              >
                {isCreatingProfile ? "생성 중..." : "생성하기"}
              </button>
            </div>
          </div>
        </div>
      )}

      {deleteTarget && (
        <div className="modal-overlay">
          <div className="modal-box">
            <h2 className="modal-title">프로필 삭제</h2>

            <p className="modal-text">
              <strong>{deleteTarget.name}</strong> 프로필을 삭제할까요?
            </p>

            <div className="modal-warning">
              프로필 폴더와 촬영 데이터가 모두 삭제됩니다.
              <br />
              이 작업은 되돌릴 수 없습니다.
            </div>

            <div className="modal-actions">
              <button
                className="modal-btn cancel"
                onClick={() => {
                  if (isDeletingProfile) return;
                  setDeleteTarget(null);
                }}
                disabled={isDeletingProfile !== null}
              >
                취소
              </button>

              <button
                className="modal-btn danger"
                onClick={deleteProfile}
                disabled={isDeletingProfile !== null}
              >
                {isDeletingProfile ? "삭제 중..." : "삭제하기"}
              </button>
            </div>
          </div>
        </div>
      )}

      {historyDeleteTarget && (
        <div className="modal-overlay">
          <div className="modal-box">
            <h2 className="modal-title">촬영 기록 삭제</h2>

            <p className="modal-text">
              아래 촬영 기록을 삭제할까요?
              <br />
              <strong>{historyDeleteTarget.displayTime}</strong>
            </p>

            <div className="modal-warning">
              이 기록에 저장된 파일이 모두 삭제됩니다.
              <br />
              이 작업은 되돌릴 수 없습니다.
            </div>

            <div className="modal-actions">
              <button
                className="modal-btn cancel"
                onClick={() => {
                  if (isDeletingHistory) return;
                  setHistoryDeleteTarget(null);
                }}
                disabled={isDeletingHistory !== null}
              >
                취소
              </button>

              <button
                className="modal-btn danger"
                onClick={deleteHistory}
                disabled={isDeletingHistory !== null}
              >
                {isDeletingHistory ? "삭제 중..." : "삭제하기"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default App;

