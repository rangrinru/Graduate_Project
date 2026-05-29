import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import type {
  AnalysisImageMode,
  AutoCaptureStatus,
  HistoryDetail,
  HistoryItem,
  KeyboardMode,
  PorphyrinResult,
  Profile,
  Screen,
  Toast,
  HangulBuffer,
  TroubleRiskResult,
} from "./types";
import { AUTO_CHECK_LABELS, EMPTY_AUTO_CHECKS, REGION_LABELS } from "./constants";
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

  const [selectedHistory, setSelectedHistory] = useState<HistoryDetail | null>(null);
  const [isLoadingHistoryDetail, setIsLoadingHistoryDetail] = useState(false);

  const [selectedFilter, setSelectedFilter] = useState<
    "no_filter" | "405nm_filter" | "660nm_filter"
  >("no_filter");

  const [toast, setToast] = useState<Toast | null>(null);
  const toastTimerRef = useRef<number | null>(null);

  const [deleteTarget, setDeleteTarget] = useState<Profile | null>(null);
  const [historyDeleteTarget, setHistoryDeleteTarget] = useState<HistoryItem | null>(null);
  const [isDeletingHistory, setIsDeletingHistory] = useState<string | null>(null);

  const [isAnalyzingPorphyrin, setIsAnalyzingPorphyrin] = useState(false);
  const [porphyrinResult, setPorphyrinResult] = useState<PorphyrinResult | null>(null);
  const [isAnalyzingTroubleRisk, setIsAnalyzingTroubleRisk] = useState(false);
  const [troubleRiskResult, setTroubleRiskResult] = useState<TroubleRiskResult | null>(null);
  const [analysisImageMode, setAnalysisImageMode] = useState<AnalysisImageMode>("source");
  const [showAnalysisResultModal, setShowAnalysisResultModal] = useState(false);
  const historyScrollRef = useRef<HTMLDivElement | null>(null);

  const [whiteLedOn, setWhiteLedOn] = useState(false);
  const [isChangingWhiteLed, setIsChangingWhiteLed] = useState(false);
  const [isStartingAutoCapture, setIsStartingAutoCapture] = useState(false);
  const [autoStatus, setAutoStatus] = useState<AutoCaptureStatus | null>(null);

  const currentImage = useMemo(() => {
    if (!selectedHistory) return null;
    return selectedHistory.images[selectedFilter] || null;
  }, [selectedHistory, selectedFilter]);

  const profileNameInputValue = profileInputText + composeHangul(hangulBuffer);

  const autoChecks = autoStatus?.checks ?? EMPTY_AUTO_CHECKS;
  const isAutoRunning = autoStatus?.running ?? false;
  const showAutoPanel = Boolean(autoStatus && (autoStatus.running || autoStatus.error));

  const showToast = (
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
  };

  useEffect(() => {
    return () => {
      if (toastTimerRef.current !== null) {
        window.clearTimeout(toastTimerRef.current);
      }
    };
  }, []);

  const fetchProfiles = async () => {
    try {
      setIsLoadingProfiles(true);

      const res = await fetch(`${API_BASE}/profiles`);
      const data = await res.json();

      if (!data.ok) {
        console.error(data);
        showToast(data.error || "프로필 목록 불러오기 실패", "error");
        return;
      }

      setProfiles(data.profiles || []);
    } catch (error) {
      console.error(error);
      showToast("프로필 목록 불러오기 실패", "error");
    } finally {
      setIsLoadingProfiles(false);
    }
  };

  useEffect(() => {
    fetchProfiles();
  }, []);

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
  }, [screen]);

  useEffect(() => {
    if (!autoStatus?.captured || !autoStatus.capture_id) {
      return;
    }

    showToast("자동 촬영이 완료되었습니다.", "success");
  }, [autoStatus?.captured, autoStatus?.capture_id]);

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
    setPorphyrinResult(null);
    setTroubleRiskResult(null);
    setAnalysisImageMode("source");
    setShowAnalysisResultModal(false);
    setAutoStatus(null);
    setScreen("camera");
  };

  const goToProfiles = () => {
    setScreen("profiles");
    setSelectedProfile(null);
    setSelectedHistory(null);
    setHistoryItems([]);
    setPorphyrinResult(null);
    setTroubleRiskResult(null);
    setAnalysisImageMode("source");
    setShowAnalysisResultModal(false);
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

  const openHistoryDetail = async (captureId: string) => {
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
      setSelectedFilter("no_filter");
      setPorphyrinResult(null);
      setTroubleRiskResult(null);
      setAnalysisImageMode("source");
      setShowAnalysisResultModal(false);
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

      if (selectedHistory?.captureId === target.captureId) {
        setSelectedHistory(null);
        setPorphyrinResult(null);
        setTroubleRiskResult(null);
        setAnalysisImageMode("source");
        setShowAnalysisResultModal(false);
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

      setPorphyrinResult({
        porphyrin_area: data.porphyrin_area,
        detection_rate_percent: data.detection_rate_percent,
        grade: data.grade,
        region_analysis: data.region_analysis || {},
        threshold_percentile: data.threshold_percentile,
        threshold_value: data.threshold_value,
        min_area: data.min_area,
        max_area: data.max_area,
        heatmap_url: data.heatmap_url,
      });
      setShowAnalysisResultModal(false);

      setSelectedFilter("660nm_filter");
      setAnalysisImageMode("porphyrin_heatmap");
      showToast("포르피린 분석 완료", "success");
    } catch (error) {
      console.error(error);
      showToast("포르피린 분석 실패", "error");
    } finally {
      setIsAnalyzingPorphyrin(false);
    }
  };

  const analyzeTroubleRisk = async () => {
    if (!selectedProfile || !selectedHistory) {
      showToast("분석할 촬영 기록이 없습니다.", "error");
      return;
    }

    try {
      setIsAnalyzingTroubleRisk(true);
      showToast("트러블 위험 분석을 시작합니다.", "info");

      const encodedProfileId = encodeURIComponent(selectedProfile.folderId);
      const encodedCaptureId = encodeURIComponent(selectedHistory.captureId);

      const res = await fetch(
        `${API_BASE}/profiles/${encodedProfileId}/history/${encodedCaptureId}/analyze-trouble-risk`,
        {
          method: "POST",
        }
      );

      const data = await res.json();

      if (!data.ok) {
        showToast(data.error || "트러블 위험 분석 실패", "error");
        return;
      }

      setTroubleRiskResult({
        risk_area: data.risk_area,
        risk_rate_percent: data.risk_rate_percent,
        risk_grade: data.risk_grade,
        region_analysis: data.region_analysis || {},
        focus_areas: data.focus_areas || [],
        top_region: data.top_region ?? null,
        threshold_value: data.threshold_value,
        risk_heatmap_url: data.risk_heatmap_url,
        focus_overlay_url: data.focus_overlay_url,
        risk_mask_url: data.risk_mask_url,
      });

      setSelectedFilter("660nm_filter");
      setAnalysisImageMode("trouble_risk_heatmap");
      showToast("트러블 위험 분석 완료", "success");
    } catch (error) {
      console.error(error);
      showToast("트러블 위험 분석 실패", "error");
    } finally {
      setIsAnalyzingTroubleRisk(false);
    }
  };

  const fetchWhiteLedStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/white-led/status`);
      const data = await res.json();

      if (data.ok) {
        setWhiteLedOn(Boolean(data.white_led_is_on));
      }
    } catch (error) {
      console.error(error);
    }
  };

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

  const fetchAutoCaptureStatus = async (showError = false) => {
    try {
      const res = await fetch(`${API_BASE}/auto-capture/status`);
      const data = await res.json();

      if (!data.ok) {
        if (showError) {
          showToast(data.error || "자동 촬영 상태 확인 실패", "error");
        }
        return;
      }

      setAutoStatus({
        running: Boolean(data.running),
        captured: Boolean(data.captured),
        profile_id: data.profile_id ?? null,
        capture_id: data.capture_id ?? null,
        status: data.status || "자동 촬영 대기 중",
        error: data.error ?? null,
        checks: data.checks || EMPTY_AUTO_CHECKS,
        stable_face_count: Number(data.stable_face_count || 0),
        eyes_closed_count: Number(data.eyes_closed_count || 0),
        dynamic_eye_threshold: Number(data.dynamic_eye_threshold || 0),
        white_led_is_on: Boolean(data.white_led_is_on),
        last_update: data.last_update ?? null,
      });

      if (typeof data.white_led_is_on === "boolean") {
        setWhiteLedOn(data.white_led_is_on);
      }
    } catch (error) {
      console.error(error);
      if (showError) {
        showToast("자동 촬영 상태 확인 실패", "error");
      }
    }
  };

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

      setAutoStatus({
        running: Boolean(data.running),
        captured: Boolean(data.captured),
        profile_id: data.profile_id ?? null,
        capture_id: data.capture_id ?? null,
        status: data.status || "자동 촬영 조건 확인 중",
        error: data.error ?? null,
        checks: data.checks || EMPTY_AUTO_CHECKS,
        stable_face_count: Number(data.stable_face_count || 0),
        eyes_closed_count: Number(data.eyes_closed_count || 0),
        dynamic_eye_threshold: Number(data.dynamic_eye_threshold || 0),
        white_led_is_on: Boolean(data.white_led_is_on),
        last_update: data.last_update ?? null,
      });

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
    fetchHistory();
  };

  const backToCamera = () => {
    setScreen("camera");
  };

  const backToHistory = () => {
    setScreen("history");
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

  const getImageSrc = (imageUrl: string | null | undefined) => {
    if (!imageUrl) return "";

    if (imageUrl.startsWith("http://") || imageUrl.startsWith("https://")) {
      return imageUrl;
    }

    return `${API_BASE}${imageUrl}`;
  };

  const activeAnalysisImageUrl =
    analysisImageMode === "porphyrin_heatmap"
      ? porphyrinResult?.heatmap_url
      : analysisImageMode === "trouble_risk_heatmap"
        ? troubleRiskResult?.risk_heatmap_url
        : analysisImageMode === "focus_care_overlay"
          ? troubleRiskResult?.focus_overlay_url
          : null;

  const activeAnalysisImageAlt =
    analysisImageMode === "porphyrin_heatmap"
      ? "포르피린 히트맵"
      : analysisImageMode === "trouble_risk_heatmap"
        ? "트러블 위험 예측 히트맵"
        : analysisImageMode === "focus_care_overlay"
          ? "집중 케어 영역 표시"
          : currentImage?.display_name || "촬영 이미지";

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
                <button className="mini-back-btn" onClick={backToCamera}>
                  카메라로 돌아가기
                </button>

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
                        className="history-card"
                        onClick={() => openHistoryDetail(item.captureId)}
                      >
                        <div className="history-card-header">
                          <div>
                            <div className="history-card-time">{item.displayTime}</div>
                            <div className="history-card-sub">
                              captureId: {item.captureId}
                            </div>
                          </div>

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
                        </div>

                        <div className="history-tag">기록 열기</div>
                      </div>
                    ))}
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
                      <h2 className="history-detail-title">
                        {selectedHistory?.displayTime || "-"}
                      </h2>

                      <div className="history-detail-sub">
                        프로필: {selectedHistory?.profileName || "-"}
                        <br />
                        선택한 필터: {currentImage?.display_name || "-"}
                      </div>

                      <div className="filter-row">
                        <button
                          className={`filter-chip ${
                            selectedFilter === "no_filter" ? "active" : ""
                          }`}
                          onClick={() => {
                            setSelectedFilter("no_filter");
                            setAnalysisImageMode("source");
                          }}
                        >
                          No_Filter
                        </button>

                        <button
                          className={`filter-chip ${
                            selectedFilter === "405nm_filter" ? "active" : ""
                          }`}
                          onClick={() => {
                            setSelectedFilter("405nm_filter");
                            setAnalysisImageMode("source");
                          }}
                        >
                          405nm_Filter
                        </button>

                        <button
                          className={`filter-chip ${
                            selectedFilter === "660nm_filter" ? "active" : ""
                          }`}
                          onClick={() => {
                            setSelectedFilter("660nm_filter");
                            setAnalysisImageMode("source");
                          }}
                        >
                          660nm_Filter
                        </button>

                        {porphyrinResult?.heatmap_url && (
                          <button
                            className={`filter-chip ${
                              analysisImageMode === "porphyrin_heatmap" ? "active" : ""
                            }`}
                            onClick={() => setAnalysisImageMode("porphyrin_heatmap")}
                          >
                            포르피린 히트맵
                          </button>
                        )}

                        {troubleRiskResult?.risk_heatmap_url && (
                          <button
                            className={`filter-chip ${
                              analysisImageMode === "trouble_risk_heatmap" ? "active" : ""
                            }`}
                            onClick={() => setAnalysisImageMode("trouble_risk_heatmap")}
                          >
                            위험 히트맵
                          </button>
                        )}

                        {troubleRiskResult?.focus_overlay_url && (
                          <button
                            className={`filter-chip ${
                              analysisImageMode === "focus_care_overlay" ? "active" : ""
                            }`}
                            onClick={() => setAnalysisImageMode("focus_care_overlay")}
                          >
                            집중 케어
                          </button>
                        )}
                      </div>
                    </div>

                    <div className="image-viewer">
                      {activeAnalysisImageUrl ? (
                        <img
                          className="history-image"
                          src={getImageSrc(activeAnalysisImageUrl)}
                          alt={activeAnalysisImageAlt}
                        />
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
                    </div>

                    <div className="analysis-panel">
                      <div className="analysis-title">포르피린 분석</div>
                      <div className="analysis-description">
                        660nm 필터 이미지에서 강한 형광 후보 영역을 검출합니다.
                        분석 결과는 원본과 검출 결과를 나란히 보여줍니다.
                      </div>

                      <button
                        className="analysis-btn"
                        onClick={analyzePorphyrin}
                        disabled={isAnalyzingPorphyrin || !selectedHistory}
                      >
                        {isAnalyzingPorphyrin ? "포르피린 분석 중..." : "포르피린 분석하기"}
                      </button>

                      <button
                        className="analysis-btn trouble-risk-btn"
                        onClick={analyzeTroubleRisk}
                        disabled={isAnalyzingTroubleRisk || !selectedHistory}
                      >
                        {isAnalyzingTroubleRisk ? "트러블 위험 분석 중..." : "트러블 위험 분석하기"}
                      </button>

                      {false && porphyrinResult && (
                        <div className="analysis-result-compact">
                          <div className="analysis-compact-text">
                            분석 완료
                          </div>

                          <button
                            className="analysis-view-btn"
                            onClick={() => setShowAnalysisResultModal(true)}
                          >
                            분석 결과 크게 보기
                          </button>
                        </div>
                      )}

                      {porphyrinResult && (
                        <div className="analysis-result-box">
                          <div className="analysis-result-grid">
                            <div className="analysis-stat">
                              <div className="analysis-stat-label">얼굴 영역 대비 포르피린</div>
                              <div className="analysis-stat-value">
                                {porphyrinResult.detection_rate_percent.toFixed(2)}%
                              </div>
                            </div>

                            <div className="analysis-stat">
                              <div className="analysis-stat-label">등급</div>
                              <div className="analysis-stat-value">
                                {porphyrinResult.grade}
                              </div>
                            </div>
                          </div>

                          <div className="analysis-region-grid">
                            {Object.entries(porphyrinResult.region_analysis).map(
                              ([key, value]) => (
                                <div className="analysis-region-item" key={key}>
                                  <span>{REGION_LABELS[key] || key}</span>
                                  <strong>{value}%</strong>
                                </div>
                              )
                            )}
                          </div>
                        </div>
                      )}

                      {troubleRiskResult && (
                        <div className="analysis-result-box trouble-risk-result">
                          <div className="analysis-result-grid">
                            <div className="analysis-stat">
                              <div className="analysis-stat-label">트러블 위험 면적</div>
                              <div className="analysis-stat-value">
                                {troubleRiskResult.risk_rate_percent.toFixed(2)}%
                              </div>
                            </div>

                            <div className="analysis-stat">
                              <div className="analysis-stat-label">위험 등급</div>
                              <div className="analysis-stat-value">
                                {troubleRiskResult.risk_grade}
                              </div>
                            </div>

                            <div className="analysis-stat">
                              <div className="analysis-stat-label">집중 케어</div>
                              <div className="analysis-stat-value">
                                {troubleRiskResult.focus_areas.length}
                              </div>
                            </div>
                          </div>

                          <div className="analysis-view-row">
                            <button
                              className="analysis-view-btn"
                              onClick={() => setAnalysisImageMode("trouble_risk_heatmap")}
                            >
                              위험 히트맵 보기
                            </button>

                            <button
                              className="analysis-view-btn"
                              onClick={() => setAnalysisImageMode("focus_care_overlay")}
                            >
                              집중 케어 보기
                            </button>
                          </div>

                          <div className="analysis-region-grid">
                            {Object.entries(troubleRiskResult.region_analysis).map(
                              ([key, value]) => (
                                <div className="analysis-region-item" key={key}>
                                  <span>{REGION_LABELS[key] || key}</span>
                                  <strong>{value}%</strong>
                                </div>
                              )
                            )}
                          </div>

                          {troubleRiskResult.focus_areas.length > 0 && (
                            <div className="focus-care-list">
                              {troubleRiskResult.focus_areas.slice(0, 4).map((area) => (
                                <div className="focus-care-item" key={area.id}>
                                  <span>
                                    #{area.id} {REGION_LABELS[area.region] || area.region}
                                  </span>
                                  <strong>{Math.round(area.risk_score)}</strong>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {showAnalysisResultModal && porphyrinResult && (
        <div className="analysis-full-overlay">
          <div className="analysis-full-header">
            <div>
              <div className="analysis-full-title">포르피린 분석 결과</div>
              <div className="analysis-full-subtitle">
                원본 이미지와 검출 결과를 한 화면에서 확인합니다.
              </div>
            </div>

            <button
              className="analysis-full-close"
              onClick={() => setShowAnalysisResultModal(false)}
            >
              닫기
            </button>
          </div>

          <div className="analysis-full-stats">
            <div className="analysis-full-stat">
              <div className="analysis-full-stat-label">검출 개수</div>
              <div className="analysis-full-stat-value">
                {porphyrinResult.detection_rate_percent.toFixed(2)}%
              </div>
            </div>

            <div className="analysis-full-stat">
              <div className="analysis-full-stat-label">검출 면적</div>
              <div className="analysis-full-stat-value">
                {porphyrinResult.porphyrin_area.toFixed(1)}
              </div>
            </div>

            <div className="analysis-full-stat">
              <div className="analysis-full-stat-label">임계값</div>
              <div className="analysis-full-stat-value">
                {porphyrinResult.threshold_value.toFixed(1)}
              </div>
            </div>
          </div>

          <div className="analysis-full-image-wrap">
            <img
              className="analysis-full-image"
              src={getImageSrc(porphyrinResult.heatmap_url)}
              alt="포르피린 분석 결과"
            />
          </div>
        </div>
      )}

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

