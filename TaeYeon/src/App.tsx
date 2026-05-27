import { useEffect, useMemo, useRef, useState } from "react";

type Screen =
  | "profiles"
  | "camera"
  | "history"
  | "historyDetail";

type Profile = {
  id: number;
  name: string;
  folderId: string;
  createdAt: string;
};

type HistoryItem = {
  captureId: string;
  capturedAt: string;
  displayTime: string;
  profileId: string;
  profileName: string;
};

type HistoryImageItem = {
  camera: string;
  display_name: string;
  filter_type: string;
  exists: boolean;
  image_url: string | null;
  metadata?: Record<string, unknown>;
};

type HistoryDetail = {
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

type Toast = {
  message: string;
  type: "success" | "error" | "info";
};

type PorphyrinResult = {
  porphyrin_area: number;
  detection_rate_percent: number;
  grade: string;
  region_analysis: Record<string, number>;
  threshold_percentile: number;
  threshold_value: number;
  min_area: number;
  max_area: number;
  heatmap_url: string;
};

type AutoCaptureChecks = {
  face_found: boolean;
  center_ok: boolean;
  size_ok: boolean;
  angle_ok: boolean;
  eyes_closed: boolean;
  stable_ok: boolean;
};

type AutoCaptureStatus = {
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

const EMPTY_AUTO_CHECKS: AutoCaptureChecks = {
  face_found: false,
  center_ok: false,
  size_ok: false,
  angle_ok: false,
  eyes_closed: false,
  stable_ok: false,
};

const AUTO_CHECK_LABELS: Array<{ key: keyof AutoCaptureChecks; label: string }> = [
  { key: "face_found", label: "얼굴 인식" },
  { key: "center_ok", label: "얼굴 중앙 정렬" },
  { key: "size_ok", label: "얼굴 크기" },
  { key: "angle_ok", label: "얼굴 각도" },
  { key: "eyes_closed", label: "눈 감음" },
  { key: "stable_ok", label: "안정 유지" },
];

const REGION_LABELS: Record<string, string> = {
  Upper: "상단",
  Middle: "중앙",
  Lower: "하단",
  Left: "왼쪽",
  Center: "가운데",
  Right: "오른쪽",
};

type KeyboardMode = "ko" | "en" | "num";

type HangulBuffer = {
  cho: string | null;
  jung: string | null;
  jong: string | null;
};

const EMPTY_HANGUL_BUFFER: HangulBuffer = {
  cho: null,
  jung: null,
  jong: null,
};

const CHO_LIST = [
  "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
  "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
];

const JUNG_LIST = [
  "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
  "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
];

const JONG_LIST = [
  "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
  "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
  "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
];

const KOREAN_CONSONANTS = new Set([
  "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
  "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]);

const KOREAN_VOWELS = new Set([
  "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅛ",
  "ㅜ", "ㅠ", "ㅡ", "ㅣ",
]);

const DOUBLE_CHO_MAP: Record<string, string> = {
  "ㄱㄱ": "ㄲ",
  "ㄷㄷ": "ㄸ",
  "ㅂㅂ": "ㅃ",
  "ㅅㅅ": "ㅆ",
  "ㅈㅈ": "ㅉ",
};

const COMPOUND_JUNG_MAP: Record<string, string> = {
  "ㅗㅏ": "ㅘ",
  "ㅗㅐ": "ㅙ",
  "ㅗㅣ": "ㅚ",
  "ㅜㅓ": "ㅝ",
  "ㅜㅔ": "ㅞ",
  "ㅜㅣ": "ㅟ",
  "ㅡㅣ": "ㅢ",
};

const COMPOUND_JONG_MAP: Record<string, string> = {
  "ㄱㅅ": "ㄳ",
  "ㄴㅈ": "ㄵ",
  "ㄴㅎ": "ㄶ",
  "ㄹㄱ": "ㄺ",
  "ㄹㅁ": "ㄻ",
  "ㄹㅂ": "ㄼ",
  "ㄹㅅ": "ㄽ",
  "ㄹㅌ": "ㄾ",
  "ㄹㅍ": "ㄿ",
  "ㄹㅎ": "ㅀ",
  "ㅂㅅ": "ㅄ",
};

const SPLIT_JONG_MAP: Record<string, [string, string]> = {
  "ㄳ": ["ㄱ", "ㅅ"],
  "ㄵ": ["ㄴ", "ㅈ"],
  "ㄶ": ["ㄴ", "ㅎ"],
  "ㄺ": ["ㄹ", "ㄱ"],
  "ㄻ": ["ㄹ", "ㅁ"],
  "ㄼ": ["ㄹ", "ㅂ"],
  "ㄽ": ["ㄹ", "ㅅ"],
  "ㄾ": ["ㄹ", "ㅌ"],
  "ㄿ": ["ㄹ", "ㅍ"],
  "ㅀ": ["ㄹ", "ㅎ"],
  "ㅄ": ["ㅂ", "ㅅ"],
};

const KOREAN_KEY_ROWS = [
  ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅎ"],
  ["ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ"],
  ["ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"],
  ["ㅐ", "ㅒ", "ㅔ", "ㅖ"],
];

const ENGLISH_KEY_ROWS = [
  ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
  ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
  ["z", "x", "c", "v", "b", "n", "m"],
];

const NUMBER_KEY_ROWS = [
  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
  ["-", "_", ".", "(", ")", "/", "@"],
];

const composeHangul = (buffer: HangulBuffer) => {
  if (!buffer.cho && !buffer.jung && !buffer.jong) {
    return "";
  }

  if (buffer.cho && !buffer.jung) {
    return buffer.cho;
  }

  if (!buffer.cho && buffer.jung) {
    return buffer.jung;
  }

  if (!buffer.cho || !buffer.jung) {
    return "";
  }

  const choIndex = CHO_LIST.indexOf(buffer.cho);
  const jungIndex = JUNG_LIST.indexOf(buffer.jung);
  const jongIndex = buffer.jong ? JONG_LIST.indexOf(buffer.jong) : 0;

  if (choIndex < 0 || jungIndex < 0 || jongIndex < 0) {
    return `${buffer.cho}${buffer.jung}${buffer.jong || ""}`;
  }

  const unicode = 0xac00 + choIndex * 588 + jungIndex * 28 + jongIndex;

  return String.fromCharCode(unicode);
};

const composeHangulWithoutJong = (buffer: HangulBuffer) => {
  return composeHangul({
    cho: buffer.cho,
    jung: buffer.jung,
    jong: null,
  });
};

const API_BASE = "http://192.168.137.47:8000";

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
  const [showAnalysisResultModal, setShowAnalysisResultModal] = useState(false);

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
      showToast("포르피린 분석 완료", "success");
    } catch (error) {
      console.error(error);
      showToast("포르피린 분석 실패", "error");
    } finally {
      setIsAnalyzingPorphyrin(false);
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

  const getImageSrc = (imageUrl: string | null | undefined) => {
    if (!imageUrl) return "";

    if (imageUrl.startsWith("http://") || imageUrl.startsWith("https://")) {
      return imageUrl;
    }

    return `${API_BASE}${imageUrl}`;
  };

  return (
    <>
      <style>{`
        * {
          box-sizing: border-box;
        }

        html, body, #root {
          margin: 0;
          width: 100%;
          height: 100%;
          font-family: Arial, Helvetica, sans-serif;
          background: #0b1220;
          overflow: hidden;
        }

        button, input {
          font-family: inherit;
        }

        .app-bg {
          width: 100vw;
          height: 100vh;
          min-height: 100vh;
          background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
          display: flex;
          justify-content: stretch;
          align-items: stretch;
          padding: 0;
          overflow: hidden;
        }

        .mirror-frame {
          width: 100vw;
          max-width: none;
          height: 100vh;
          min-height: 0;
          background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
          border: none;
          border-radius: 0;
          overflow: hidden;
          position: relative;
          box-shadow: none;
        }

        .toast {
          position: absolute;
          left: 50%;
          top: 24px;
          transform: translateX(-50%);
          z-index: 300;
          min-width: 260px;
          max-width: 86%;
          padding: 14px 18px;
          border-radius: 18px;
          color: white;
          font-size: 14px;
          font-weight: 700;
          text-align: center;
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255,255,255,0.14);
          box-shadow: 0 12px 32px rgba(0,0,0,0.35);
          animation: toastFade 0.2s ease;
          line-height: 1.5;
        }

        .toast.success {
          background: rgba(34, 197, 94, 0.9);
        }

        .toast.error {
          background: rgba(239, 68, 68, 0.92);
        }

        .toast.info {
          background: rgba(14, 165, 233, 0.9);
        }

        @keyframes toastFade {
          from {
            opacity: 0;
            transform: translateX(-50%) translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
          }
        }

        .header {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          z-index: 20;
          padding: 20px;
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
        }

        .header-title {
          color: white;
          font-size: 28px;
          font-weight: 700;
          margin: 0;
        }

        .header-subtitle {
          color: rgba(255,255,255,0.65);
          font-size: 14px;
          margin-top: 8px;
          line-height: 1.6;
        }

        .time-badge {
          background: rgba(255,255,255,0.08);
          border: 1px solid rgba(255,255,255,0.12);
          color: white;
          padding: 10px 14px;
          border-radius: 16px;
          font-size: 14px;
          white-space: nowrap;
        }

        .profiles-container,
        .history-container {
          height: 100%;
          padding: 110px 20px 20px 20px;
          overflow-y: auto;
        }

        .history-detail-container {
          height: 100vh;
          padding: 86px 14px 10px 14px;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }

        .profiles-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
        }

        .profile-card,
        .add-card,
        .history-card {
          border-radius: 24px;
          transition: 0.2s ease;
        }

        .profile-card {
          min-height: 220px;
          padding: 20px;
          background: rgba(255,255,255,0.07);
          border: 1px solid rgba(255,255,255,0.1);
          color: white;
          position: relative;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
        }

        .history-card {
          padding: 18px;
          background: rgba(255,255,255,0.07);
          border: 1px solid rgba(255,255,255,0.1);
          color: white;
          cursor: pointer;
          margin-bottom: 14px;
        }

        .profile-click-area {
          cursor: pointer;
        }

        .profile-card:hover,
        .add-card:hover,
        .control-btn:hover,
        .capture-btn:hover,
        .modal-btn:hover,
        .back-btn:hover,
        .delete-btn:hover,
        .retry-btn:hover,
        .history-card:hover,
        .filter-chip:hover,
        .mini-back-btn:hover,
        .history-delete-btn:hover,
        .analysis-btn:hover {
          transform: scale(1.02);
        }

        .add-card {
          min-height: 220px;
          padding: 20px;
          background: rgba(255,255,255,0.04);
          border: 2px dashed rgba(255,255,255,0.2);
          color: white;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          text-align: center;
          cursor: pointer;
        }

        .profile-icon,
        .add-icon {
          width: 58px;
          height: 58px;
          border-radius: 16px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 28px;
          margin-bottom: 16px;
        }

        .profile-icon {
          background: rgba(34, 211, 238, 0.12);
        }

        .add-icon {
          background: rgba(255,255,255,0.08);
        }

        .profile-name {
          font-size: 20px;
          font-weight: 700;
          word-break: break-word;
        }

        .profile-date {
          margin-top: 8px;
          font-size: 14px;
          color: rgba(255,255,255,0.6);
        }

        .profile-select-tag,
        .history-tag {
          display: inline-block;
          margin-top: 16px;
          padding: 8px 12px;
          border-radius: 999px;
          background: rgba(34, 211, 238, 0.12);
          color: #a5f3fc;
          font-size: 12px;
          font-weight: 600;
        }

        .profile-card-actions {
          margin-top: 16px;
          display: flex;
          justify-content: flex-end;
        }

        .delete-btn,
        .history-delete-btn {
          background: rgba(239, 68, 68, 0.18);
          border: 1px solid rgba(255,255,255,0.12);
          color: white;
          border-radius: 14px;
          padding: 10px 14px;
          cursor: pointer;
          font-size: 13px;
          transition: 0.2s ease;
          white-space: nowrap;
        }

        .history-card-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 12px;
        }

        .history-delete-btn {
          padding: 9px 12px;
        }

        .history-delete-btn:hover,
        .delete-btn:hover {
          background: rgba(239, 68, 68, 0.28);
        }

        .empty-box,
        .loading-box {
          margin-top: 12px;
          width: 100%;
          padding: 22px;
          border-radius: 24px;
          color: white;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.1);
          text-align: center;
          line-height: 1.7;
        }

        .retry-btn {
          margin-top: 14px;
          border: none;
          border-radius: 14px;
          padding: 12px 16px;
          background: #22d3ee;
          color: #0f172a;
          font-weight: 700;
          cursor: pointer;
          transition: 0.2s ease;
        }

        .camera-screen {
          position: relative;
          width: 100%;
          height: 100%;
          background: #000;
        }

        .camera-live {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          object-fit: cover;
          background: #000;
        }

        .camera-overlay {
          position: absolute;
          inset: 0;
          background:
            linear-gradient(to bottom, rgba(0,0,0,0.18), rgba(0,0,0,0.05) 25%, rgba(0,0,0,0.18));
          pointer-events: none;
        }

        .selected-profile {
          position: absolute;
          left: 16px;
          top: 16px;
          z-index: 20;
          background: rgba(0,0,0,0.3);
          border: 1px solid rgba(255,255,255,0.12);
          color: white;
          border-radius: 18px;
          padding: 12px 14px;
          backdrop-filter: blur(8px);
          max-width: 60%;
        }

        .selected-profile small {
          display: block;
          color: rgba(255,255,255,0.6);
          margin-bottom: 6px;
          font-size: 12px;
        }

        .selected-profile strong {
          font-size: 15px;
          word-break: break-word;
        }

        .back-btn {
          position: absolute;
          left: 16px;
          bottom: 16px;
          z-index: 20;
          background: rgba(0,0,0,0.3);
          border: 1px solid rgba(255,255,255,0.12);
          color: white;
          border-radius: 16px;
          padding: 12px 16px;
          cursor: pointer;
          backdrop-filter: blur(8px);
          transition: 0.2s ease;
        }

        .controls {
          position: absolute;
          right: 16px;
          top: 16px;
          z-index: 80;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .control-btn,
        .mini-back-btn,
        .filter-chip {
          background: rgba(0,0,0,0.3);
          border: 1px solid rgba(255,255,255,0.12);
          color: white;
          border-radius: 16px;
          padding: 12px 16px;
          cursor: pointer;
          font-size: 14px;
          backdrop-filter: blur(8px);
          transition: 0.2s ease;
        }

        .white-led-control.on {
          background: rgba(255,255,255,0.92);
          color: #0f172a;
          border-color: rgba(255,255,255,0.95);
        }

        .auto-capture-control {
          background: rgba(34, 197, 94, 0.34);
          border-color: rgba(34, 197, 94, 0.55);
          color: #dcfce7;
          font-weight: 800;
        }

        .auto-cancel-control {
          background: rgba(239, 68, 68, 0.32);
          border-color: rgba(239, 68, 68, 0.58);
          color: #fee2e2;
          font-weight: 800;
        }

        .mini-back-btn {
          margin-top: 4px;
          margin-bottom: 6px;
          padding: 9px 13px;
          font-size: 12px;
          flex-shrink: 0;
        }

        .auto-check-panel {
          position: absolute;
          left: 18px;
          top: 102px;
          width: min(390px, calc(100vw - 190px));
          z-index: 30;
          padding: 16px;
          border-radius: 22px;
          background: rgba(0, 0, 0, 0.56);
          border: 1px solid rgba(255,255,255,0.16);
          color: white;
          backdrop-filter: blur(12px);
          box-shadow: 0 16px 40px rgba(0,0,0,0.28);
        }

        .auto-check-panel.done {
          border-color: rgba(34, 197, 94, 0.65);
        }

        .auto-check-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 12px;
          margin-bottom: 14px;
        }

        .auto-check-title {
          font-size: 20px;
          font-weight: 900;
          margin-bottom: 6px;
        }

        .auto-check-message {
          min-height: 22px;
          font-size: 15px;
          font-weight: 700;
          line-height: 1.4;
          color: rgba(255,255,255,0.9);
        }

        .auto-check-state {
          flex-shrink: 0;
          padding: 8px 12px;
          border-radius: 999px;
          font-size: 12px;
          font-weight: 900;
          border: 1px solid rgba(255,255,255,0.16);
        }

        .auto-check-state.running {
          background: rgba(59, 130, 246, 0.22);
          color: #bfdbfe;
        }

        .auto-check-state.done {
          background: rgba(34, 197, 94, 0.24);
          color: #bbf7d0;
        }

        .auto-check-state.idle {
          background: rgba(255,255,255,0.12);
          color: rgba(255,255,255,0.8);
        }

        .auto-check-list {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
        }

        .auto-check-item {
          display: flex;
          align-items: center;
          gap: 10px;
          min-height: 44px;
          padding: 0 12px;
          border-radius: 16px;
          font-size: 14px;
          font-weight: 900;
          transition: 0.18s ease;
        }

        .auto-check-item.fail {
          background: rgba(239, 68, 68, 0.25);
          border: 1px solid rgba(239, 68, 68, 0.72);
          color: #fecaca;
        }

        .auto-check-item.ok {
          background: rgba(34, 197, 94, 0.25);
          border: 1px solid rgba(34, 197, 94, 0.72);
          color: #bbf7d0;
        }

        .auto-check-icon {
          width: 24px;
          height: 24px;
          border-radius: 999px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 14px;
          font-weight: 900;
        }

        .auto-check-item.fail .auto-check-icon {
          background: #ef4444;
          color: white;
        }

        .auto-check-item.ok .auto-check-icon {
          background: #22c55e;
          color: white;
        }

        .auto-capture-id {
          margin-top: 12px;
          font-size: 12px;
          color: rgba(255,255,255,0.65);
          word-break: break-all;
        }

        .guide-wrap {
          position: absolute;
          inset: 0;
          pointer-events: none;
          z-index: 10;
        }

        .guide-face {
          position: absolute;
          left: 50%;
          top: clamp(120px, 14vh, 180px);
          width: clamp(280px, 58vw, 470px);
          height: clamp(420px, 50vh, 650px);
          transform: translateX(-50%);
          border: 4px dashed rgba(103,232,249,0.92);
          border-radius: 47% 47% 43% 43% / 34% 34% 57% 57%;
          background:
            radial-gradient(ellipse at 50% 30%, rgba(103,232,249,0.08), transparent 56%),
            rgba(255,255,255,0.018);
          box-shadow:
            0 0 0 999px rgba(0,0,0,0.08),
            0 0 34px rgba(34,211,238,0.28);
        }

        .guide-face::before {
          content: "";
          position: absolute;
          left: 8%;
          right: 8%;
          top: -2%;
          height: 31%;
          border-top: 4px solid rgba(103,232,249,0.64);
          border-radius: 50% 50% 34% 34% / 80% 80% 24% 24%;
        }

        .guide-face::after {
          content: "";
          position: absolute;
          left: 30%;
          right: 30%;
          bottom: -18%;
          height: 24%;
          border-left: 3px dashed rgba(103,232,249,0.56);
          border-right: 3px dashed rgba(103,232,249,0.56);
          border-bottom: 3px dashed rgba(103,232,249,0.46);
          border-radius: 0 0 42% 42%;
        }

        .guide-eye-line,
        .guide-nose-line,
        .guide-mouth-line {
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
          border-radius: 999px;
          background: rgba(165,243,252,0.72);
          box-shadow: 0 0 12px rgba(34,211,238,0.22);
        }

        .guide-eye-line {
          top: 47%;
          width: 48%;
          height: 3px;
        }

        .guide-nose-line {
          top: 51%;
          width: 3px;
          height: 16%;
        }

        .guide-mouth-line {
          top: 69%;
          width: 24%;
          height: 3px;
        }

        .capture-area {
          position: absolute;
          left: 50%;
          bottom: 26px;
          transform: translateX(-50%);
          z-index: 20;
        }

        .capture-btn {
          width: 96px;
          height: 96px;
          border-radius: 999px;
          border: 6px solid rgba(255,255,255,0.9);
          background: rgba(255,255,255,0.12);
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          backdrop-filter: blur(8px);
          transition: 0.2s ease;
        }

        .capture-btn:disabled,
        .control-btn:disabled,
        .back-btn:disabled,
        .modal-btn:disabled,
        .delete-btn:disabled,
        .filter-chip:disabled,
        .mini-back-btn:disabled,
        .history-delete-btn:disabled,
        .keyboard-key:disabled,
        .keyboard-mode-btn:disabled,
        .keyboard-control-key:disabled,
        .analysis-btn:disabled {
          opacity: 0.55;
          cursor: not-allowed;
          transform: none !important;
        }

        .capture-inner {
          width: 64px;
          height: 64px;
          border-radius: 999px;
          background: white;
        }

        .capture-status {
          position: absolute;
          left: 50%;
          bottom: 135px;
          transform: translateX(-50%);
          z-index: 20;
          background: rgba(0, 0, 0, 0.45);
          color: white;
          border: 1px solid rgba(255,255,255,0.12);
          padding: 10px 14px;
          border-radius: 16px;
          backdrop-filter: blur(8px);
          font-size: 14px;
        }

        .modal-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0,0,0,0.55);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
          z-index: 200;
        }

        .modal-box {
          width: 100%;
          max-width: 420px;
          background: #0f172a;
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 24px;
          padding: 24px;
          color: white;
          box-shadow: 0 20px 60px rgba(0,0,0,0.45);
        }

        .modal-title {
          margin: 0 0 18px 0;
          font-size: 22px;
          font-weight: 700;
        }

        .modal-text {
          color: rgba(255,255,255,0.72);
          font-size: 14px;
          line-height: 1.7;
          margin: 0;
        }

        .modal-warning {
          margin-top: 12px;
          padding: 12px 14px;
          border-radius: 16px;
          background: rgba(239, 68, 68, 0.13);
          border: 1px solid rgba(239, 68, 68, 0.25);
          color: rgba(255,255,255,0.86);
          font-size: 13px;
          line-height: 1.6;
        }

        .input-label {
          display: block;
          margin-bottom: 8px;
          color: rgba(255,255,255,0.75);
          font-size: 14px;
        }

        .text-input {
          width: 100%;
          padding: 14px 16px;
          border-radius: 16px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: white;
          font-size: 15px;
          outline: none;
        }

        .text-input::placeholder {
          color: rgba(255,255,255,0.35);
        }

        .virtual-keyboard {
          margin-top: 18px;
          padding: 14px;
          border-radius: 20px;
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.1);
        }

        .keyboard-mode-row {
          display: flex;
          gap: 8px;
          margin-bottom: 12px;
        }

        .keyboard-mode-btn {
          flex: 1;
          border: none;
          border-radius: 12px;
          padding: 10px 0;
          background: rgba(255,255,255,0.08);
          color: white;
          font-size: 13px;
          cursor: pointer;
        }

        .keyboard-mode-btn.active {
          background: #22d3ee;
          color: #0f172a;
          font-weight: 700;
        }

        .keyboard-row {
          display: flex;
          justify-content: center;
          gap: 5px;
          margin-bottom: 7px;
        }

        .keyboard-key {
          min-width: 30px;
          height: 38px;
          border: none;
          border-radius: 10px;
          background: rgba(255,255,255,0.12);
          color: white;
          font-size: 15px;
          font-weight: 700;
          cursor: pointer;
        }

        .keyboard-key:active,
        .keyboard-control-key:active,
        .keyboard-mode-btn:active {
          transform: scale(0.96);
        }

        .keyboard-control-row {
          display: flex;
          gap: 8px;
          margin-top: 10px;
        }

        .keyboard-control-key {
          border: none;
          border-radius: 12px;
          padding: 12px 10px;
          background: rgba(255,255,255,0.12);
          color: white;
          font-size: 13px;
          font-weight: 700;
          cursor: pointer;
        }

        .keyboard-control-key.space {
          flex: 1;
        }

        .keyboard-control-key.done {
          background: #22d3ee;
          color: #0f172a;
        }

        .modal-actions {
          margin-top: 20px;
          display: flex;
          justify-content: flex-end;
          gap: 10px;
        }

        .modal-btn {
          border: none;
          border-radius: 16px;
          padding: 12px 18px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 600;
          transition: 0.2s ease;
        }

        .modal-btn.cancel {
          background: rgba(255,255,255,0.08);
          color: white;
          border: 1px solid rgba(255,255,255,0.12);
        }

        .modal-btn.create {
          background: #22d3ee;
          color: #0f172a;
        }

        .modal-btn.danger {
          background: #ef4444;
          color: white;
        }

        .history-detail-top {
          display: flex;
          flex-direction: column;
          gap: 6px;
          align-items: center;
          text-align: center;
          flex-shrink: 0;
        }

        .history-detail-title {
          color: white;
          font-size: 20px;
          font-weight: 700;
          margin: 0;
        }

        .history-detail-sub {
          color: rgba(255,255,255,0.65);
          font-size: 12px;
          line-height: 1.4;
        }

        .filter-row {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
          margin-top: 6px;
          flex-shrink: 0;
        }

        .filter-chip.active {
          background: rgba(34, 211, 238, 0.16);
          color: #a5f3fc;
          border-color: rgba(34, 211, 238, 0.35);
        }

        .history-detail-container .filter-chip {
          border-radius: 14px;
          padding: 9px 12px;
          font-size: 12px;
        }

        .image-viewer {
          margin-top: 10px;
          border-radius: 20px;
          overflow: hidden;
          border: 1px solid rgba(255,255,255,0.1);
          background: rgba(255,255,255,0.04);
          flex: 1;
          min-height: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 8px;
        }

        .history-image {
          width: 100%;
          height: 100%;
          max-height: none;
          object-fit: contain;
          border-radius: 16px;
          background: #000;
        }

        .history-empty-image {
          color: rgba(255,255,255,0.7);
          text-align: center;
          line-height: 1.7;
          padding: 30px 16px;
        }

        .analysis-panel {
          margin-top: 8px;
          border-radius: 18px;
          border: 1px solid rgba(255,255,255,0.1);
          background: rgba(255,255,255,0.05);
          padding: 10px 12px;
          flex-shrink: 0;
        }

        .analysis-title {
          color: white;
          font-size: 15px;
          font-weight: 800;
          margin-bottom: 4px;
        }

        .analysis-description {
          display: none;
        }

        .analysis-btn {
          width: 100%;
          border: none;
          border-radius: 14px;
          padding: 12px 14px;
          background: #ef4444;
          color: white;
          font-size: 14px;
          font-weight: 800;
          cursor: pointer;
          transition: 0.2s ease;
        }

        .analysis-result-box {
          margin-top: 16px;
          border-radius: 18px;
          background: rgba(0,0,0,0.25);
          border: 1px solid rgba(255,255,255,0.08);
          padding: 14px;
        }

        .analysis-result-grid {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 10px;
          margin-bottom: 12px;
        }

        .analysis-stat {
          border-radius: 16px;
          background: rgba(255,255,255,0.07);
          padding: 12px;
          text-align: center;
        }

        .analysis-stat-label {
          color: rgba(255,255,255,0.6);
          font-size: 12px;
          margin-bottom: 6px;
        }

        .analysis-stat-value {
          color: white;
          font-size: 18px;
          font-weight: 800;
        }

        .analysis-region-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 8px;
          margin-top: 12px;
        }

        .analysis-region-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 8px;
          border-radius: 12px;
          background: rgba(255,255,255,0.06);
          color: rgba(255,255,255,0.72);
          font-size: 13px;
          padding: 10px 12px;
        }

        .analysis-region-item strong {
          color: white;
          font-size: 14px;
        }

        .analysis-image {
          width: 100%;
          max-height: 70vh;
          object-fit: contain;
          border-radius: 16px;
          background: #000;
        }

        .analysis-result-compact {
          margin-top: 14px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          padding: 14px;
          border-radius: 18px;
          background: rgba(0,0,0,0.28);
          border: 1px solid rgba(255,255,255,0.1);
        }

        .analysis-compact-text {
          color: white;
          font-size: 15px;
          font-weight: 800;
        }

        .analysis-view-btn {
          flex-shrink: 0;
          border: none;
          border-radius: 16px;
          padding: 12px 16px;
          background: #22d3ee;
          color: #0f172a;
          font-size: 14px;
          font-weight: 900;
          cursor: pointer;
        }

        .analysis-full-overlay {
          position: fixed;
          inset: 0;
          z-index: 260;
          background: rgba(0, 0, 0, 0.88);
          display: flex;
          flex-direction: column;
          padding: 18px;
          color: white;
        }

        .analysis-full-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 12px;
        }

        .analysis-full-title {
          font-size: 24px;
          font-weight: 900;
          margin-bottom: 6px;
        }

        .analysis-full-subtitle {
          color: rgba(255,255,255,0.68);
          font-size: 13px;
          line-height: 1.5;
        }

        .analysis-full-close {
          flex-shrink: 0;
          border: none;
          border-radius: 16px;
          padding: 13px 18px;
          background: rgba(255,255,255,0.14);
          color: white;
          font-size: 15px;
          font-weight: 900;
          cursor: pointer;
        }

        .analysis-full-stats {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 10px;
          margin-bottom: 12px;
        }

        .analysis-full-stat {
          border-radius: 18px;
          background: rgba(255,255,255,0.08);
          border: 1px solid rgba(255,255,255,0.1);
          padding: 12px;
          text-align: center;
        }

        .analysis-full-stat-label {
          color: rgba(255,255,255,0.62);
          font-size: 12px;
          margin-bottom: 6px;
        }

        .analysis-full-stat-value {
          color: white;
          font-size: 20px;
          font-weight: 900;
        }

        .analysis-full-image-wrap {
          flex: 1;
          min-height: 0;
          border-radius: 22px;
          background: #000;
          border: 1px solid rgba(255,255,255,0.12);
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
        }

        .analysis-full-image {
          width: 100%;
          height: 100%;
          object-fit: contain;
          background: #000;
        }

        .history-card-time {
          font-size: 18px;
          font-weight: 700;
        }

        .history-card-sub {
          margin-top: 8px;
          color: rgba(255,255,255,0.65);
          font-size: 14px;
          word-break: break-all;
        }

        @media (max-width: 560px) {
          .mirror-frame {
            width: 100vw;
            max-width: none;
            min-height: 0;
            height: 100vh;
            border-radius: 0;
          }

          .profiles-grid {
            grid-template-columns: 1fr;
          }

          .header-title {
            font-size: 24px;
          }

          .guide-face {
            top: clamp(105px, 13vh, 145px);
            width: clamp(250px, 58vw, 330px);
            height: clamp(380px, 50vh, 520px);
            border-width: 3px;
          }

          .selected-profile {
            max-width: 55%;
          }

          .filter-row {
            flex-direction: row;
          }

          .auto-check-panel {
            width: min(360px, calc(100vw - 180px));
          }
        }


        /* =========================
           15.6인치 세로 터치화면용 확대 설정
           버튼, 모달, 가상 키보드 터치 영역 확대
        ========================= */

        button {
          touch-action: manipulation;
          min-height: 56px;
        }

        .control-btn,
        .back-btn,
        .mini-back-btn,
        .filter-chip,
        .delete-btn,
        .history-delete-btn,
        .retry-btn,
        .analysis-btn {
          min-height: 66px;
          min-width: 154px;
          padding: 18px 22px;
          border-radius: 22px;
          font-size: 18px;
          font-weight: 800;
        }

        .controls {
          right: 22px;
          top: 22px;
          gap: 16px;
          z-index: 90;
        }

        .selected-profile {
          left: 22px;
          top: 22px;
          padding: 18px 20px;
          border-radius: 24px;
          max-width: 58%;
        }

        .selected-profile small {
          font-size: 15px;
        }

        .selected-profile strong {
          font-size: 22px;
        }

        .back-btn {
          left: 22px;
          bottom: 22px;
        }

        .capture-area {
          bottom: 34px;
        }

        .capture-btn {
          width: 140px;
          height: 140px;
          border-width: 8px;
        }

        .capture-inner {
          width: 92px;
          height: 92px;
        }

        .capture-status {
          bottom: 190px;
          padding: 16px 22px;
          border-radius: 22px;
          font-size: 18px;
          font-weight: 800;
        }

        .guide-face {
          top: clamp(135px, 14vh, 190px);
          width: clamp(330px, 58vw, 500px);
          height: clamp(500px, 51vh, 690px);
          border-width: 5px;
        }

        .auto-check-panel {
          left: 24px;
          top: 125px;
          width: min(520px, calc(100vw - 230px));
          padding: 22px;
          border-radius: 28px;
        }

        .auto-check-title {
          font-size: 25px;
        }

        .auto-check-message {
          font-size: 19px;
        }

        .auto-check-state {
          font-size: 15px;
          padding: 10px 14px;
        }

        .auto-check-list {
          gap: 12px;
        }

        .auto-check-item {
          min-height: 58px;
          font-size: 17px;
          border-radius: 20px;
        }

        .auto-check-icon {
          width: 32px;
          height: 32px;
          font-size: 18px;
        }

        .profiles-container,
        .history-container {
          padding: 130px 28px 28px 28px;
        }

        .profiles-grid {
          gap: 22px;
        }

        .profile-card,
        .add-card {
          min-height: 280px;
          padding: 28px;
          border-radius: 30px;
        }

        .profile-icon,
        .add-icon {
          width: 78px;
          height: 78px;
          border-radius: 22px;
          font-size: 38px;
        }

        .profile-name {
          font-size: 28px;
        }

        .profile-date {
          font-size: 18px;
        }

        .profile-select-tag,
        .history-tag {
          font-size: 16px;
          padding: 12px 18px;
        }

        .history-card {
          padding: 24px;
          border-radius: 28px;
          margin-bottom: 18px;
        }

        .history-detail-title {
          font-size: 26px;
        }

        .history-detail-sub {
          font-size: 16px;
        }

        .modal-box {
          max-width: 760px;
          padding: 34px;
          border-radius: 32px;
        }

        .modal-title {
          font-size: 30px;
          margin-bottom: 24px;
        }

        .modal-text {
          font-size: 20px;
        }

        .modal-warning {
          font-size: 18px;
          padding: 18px 20px;
          border-radius: 22px;
        }

        .input-label {
          font-size: 19px;
          margin-bottom: 12px;
        }

        .text-input {
          height: 72px;
          padding: 18px 22px;
          border-radius: 22px;
          font-size: 24px;
        }

        .modal-actions {
          gap: 16px;
          margin-top: 24px;
        }

        .modal-btn {
          min-width: 140px;
          min-height: 64px;
          padding: 18px 26px;
          border-radius: 22px;
          font-size: 20px;
          font-weight: 800;
        }

        .virtual-keyboard {
          margin-top: 22px;
          padding: 20px;
          border-radius: 26px;
        }

        .keyboard-mode-row {
          gap: 12px;
          margin-bottom: 18px;
        }

        .keyboard-mode-btn {
          min-height: 60px;
          border-radius: 18px;
          font-size: 20px;
          font-weight: 800;
        }

        .keyboard-row {
          gap: 8px;
          margin-bottom: 10px;
        }

        .keyboard-key {
          flex: 1;
          min-width: 0;
          height: 64px;
          border-radius: 18px;
          font-size: 24px;
          font-weight: 900;
        }

        .keyboard-control-row {
          gap: 12px;
          margin-top: 16px;
        }

        .keyboard-control-key {
          min-height: 66px;
          padding: 16px 18px;
          border-radius: 20px;
          font-size: 20px;
          font-weight: 900;
        }

        .keyboard-control-key.space {
          flex: 1.5;
        }

        .keyboard-control-key.done {
          min-width: 120px;
        }

        .toast {
          top: 28px;
          min-width: 360px;
          padding: 20px 26px;
          border-radius: 24px;
          font-size: 20px;
        }

        @media (max-width: 700px) {
          .modal-box {
            max-width: 96vw;
            padding: 24px;
          }

          .keyboard-key {
            height: 58px;
            font-size: 21px;
            border-radius: 16px;
          }

          .keyboard-row {
            gap: 6px;
          }

          .control-btn,
          .back-btn,
          .mini-back-btn,
          .filter-chip {
            min-width: 130px;
            font-size: 16px;
          }

          .capture-btn {
            width: 128px;
            height: 128px;
          }

          .capture-inner {
            width: 84px;
            height: 84px;
          }
        }

      `}</style>

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

              <div className="history-container">
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
                          onClick={() => setSelectedFilter("no_filter")}
                        >
                          No_Filter
                        </button>

                        <button
                          className={`filter-chip ${
                            selectedFilter === "405nm_filter" ? "active" : ""
                          }`}
                          onClick={() => setSelectedFilter("405nm_filter")}
                        >
                          405nm_Filter
                        </button>

                        <button
                          className={`filter-chip ${
                            selectedFilter === "660nm_filter" ? "active" : ""
                          }`}
                          onClick={() => setSelectedFilter("660nm_filter")}
                        >
                          660nm_Filter
                        </button>
                      </div>
                    </div>

                    <div className="image-viewer">
                      {selectedFilter === "660nm_filter" && porphyrinResult?.heatmap_url ? (
                        <img
                          className="history-image"
                          src={getImageSrc(porphyrinResult.heatmap_url)}
                          alt="Porphyrin heatmap"
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
                                  <strong>{Math.round(value)}%</strong>
                                </div>
                              )
                            )}
                          </div>
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
