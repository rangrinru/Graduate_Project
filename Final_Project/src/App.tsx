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

const API_BASE = "http://192.168.137.145:8000";

function App() {
  const [screen, setScreen] = useState<Screen>("profiles");

  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<Profile | null>(null);

  const [showGuide, setShowGuide] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newProfileName, setNewProfileName] = useState("");

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

  const currentImage = useMemo(() => {
    if (!selectedHistory) return null;
    return selectedHistory.images[selectedFilter] || null;
  }, [selectedHistory, selectedFilter]);

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

  const createProfile = async () => {
    const trimmed = newProfileName.trim();

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
      setNewProfileName("");
      setShowCreateModal(false);
      showToast("프로필이 생성되었습니다.", "success");
    } catch (error) {
      console.error(error);
      showToast("프로필 생성 실패", "error");
    } finally {
      setIsCreatingProfile(false);
    }
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
    setScreen("camera");
  };

  const goToProfiles = () => {
    setScreen("profiles");
    setSelectedProfile(null);
    setSelectedHistory(null);
    setHistoryItems([]);
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
        }

        button, input {
          font-family: inherit;
        }

        .app-bg {
          min-height: 100vh;
          width: 100%;
          background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
          display: flex;
          justify-content: center;
          align-items: center;
          padding: 16px;
        }

        .mirror-frame {
          width: 100%;
          max-width: 520px;
          height: 95vh;
          min-height: 760px;
          background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 32px;
          overflow: hidden;
          position: relative;
          box-shadow: 0 20px 60px rgba(0,0,0,0.4);
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
        .history-container,
        .history-detail-container {
          height: 100%;
          padding: 110px 20px 20px 20px;
          overflow-y: auto;
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
        .history-delete-btn:hover {
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
          top: 50%;
          left: 50%;
          width: 100vh;
          height: 100vw;
          object-fit: cover;
          background: #000;
          transform: translate(-50%, -50%) rotate(90deg);
          transform-origin: center center;
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
          z-index: 20;
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

        .mini-back-btn {
          margin-top: 14px;
        }

        .guide-wrap {
          position: absolute;
          inset: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          pointer-events: none;
          z-index: 10;
        }

        .guide-circle {
          width: 220px;
          height: 220px;
          border-radius: 50%;
          border: 3px dashed rgba(103,232,249,0.9);
          display: flex;
          align-items: center;
          justify-content: center;
          color: rgba(165,243,252,0.95);
          font-weight: 600;
          text-align: center;
          background: rgba(255,255,255,0.02);
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
        .history-delete-btn:disabled {
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
          background: rgba(0,0,0,0.45);
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
          gap: 14px;
        }

        .history-detail-title {
          color: white;
          font-size: 24px;
          font-weight: 700;
          margin: 0;
        }

        .history-detail-sub {
          color: rgba(255,255,255,0.65);
          font-size: 14px;
          line-height: 1.6;
        }

        .filter-row {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          margin-top: 8px;
        }

        .filter-chip.active {
          background: rgba(34, 211, 238, 0.16);
          color: #a5f3fc;
          border-color: rgba(34, 211, 238, 0.35);
        }

        .image-viewer {
          margin-top: 18px;
          border-radius: 24px;
          overflow: hidden;
          border: 1px solid rgba(255,255,255,0.1);
          background: rgba(255,255,255,0.04);
          min-height: 420px;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 12px;
        }

        .history-image {
          width: 100%;
          max-height: 70vh;
          object-fit: contain;
          border-radius: 18px;
          background: #000;
        }

        .history-empty-image {
          color: rgba(255,255,255,0.7);
          text-align: center;
          line-height: 1.7;
          padding: 30px 16px;
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
            max-width: 100%;
            min-height: 100vh;
            height: 100vh;
            border-radius: 0;
          }

          .profiles-grid {
            grid-template-columns: 1fr;
          }

          .header-title {
            font-size: 24px;
          }

          .guide-circle {
            width: 190px;
            height: 190px;
            font-size: 14px;
          }

          .selected-profile {
            max-width: 55%;
          }

          .filter-row {
            flex-direction: column;
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
                      onClick={() => setShowCreateModal(true)}
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
                  disabled={isCapturing}
                >
                  {showGuide ? "실루엣 끄기" : "실루엣 켜기"}
                </button>

                <button
                  className="control-btn"
                  onClick={openHistory}
                  disabled={isCapturing}
                >
                  이전 기록 확인
                </button>
              </div>

              {showGuide && (
                <div className="guide-wrap">
                  <div className="guide-circle">얼굴 위치 맞추기</div>
                </div>
              )}

              <button
                className="back-btn"
                onClick={goToProfiles}
                disabled={isCapturing}
              >
                프로필로 돌아가기
              </button>

              {isCapturing && (
                <div className="capture-status">촬영 중...</div>
              )}

              <div className="capture-area">
                <button
                  className="capture-btn"
                  onClick={capturePhoto}
                  disabled={isCapturing}
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
                      {currentImage?.exists && currentImage?.image_url ? (
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
              value={newProfileName}
              onChange={(e) => setNewProfileName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !isCreatingProfile) {
                  createProfile();
                }
              }}
              disabled={isCreatingProfile}
            />

            <div className="modal-actions">
              <button
                className="modal-btn cancel"
                onClick={() => {
                  if (isCreatingProfile) return;
                  setShowCreateModal(false);
                  setNewProfileName("");
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
              이 기록에 저장된 No_Filter, 405nm_Filter, 660nm_Filter 이미지와
              metadata.json 파일이 모두 삭제됩니다.
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
