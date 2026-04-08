import { useEffect, useState } from "react";

type Screen = "profiles" | "camera";

type Profile = {
  id: number;
  name: string;       // 화면 표시용 이름 (한글 가능)
  folderId: string;   // 실제 저장/삭제용 안전한 ID
  createdAt: string;
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

  // =========================
  // 백엔드에서 프로필 목록 불러오기
  // =========================
  const fetchProfiles = async () => {
    try {
      setIsLoadingProfiles(true);

      const res = await fetch(`${API_BASE}/profiles`);
      const data = await res.json();

      if (!data.ok) {
        console.error(data);
        alert(data.error || "프로필 목록 불러오기 실패");
        return;
      }

      setProfiles(data.profiles || []);
    } catch (error) {
      console.error(error);
      alert("프로필 목록 불러오기 실패");
    } finally {
      setIsLoadingProfiles(false);
    }
  };

  useEffect(() => {
    fetchProfiles();
  }, []);

  // =========================
  // 프로필 생성
  // =========================
  const createProfile = async () => {
    const trimmed = newProfileName.trim();

    if (!trimmed) {
      alert("프로필 이름을 입력하세요.");
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
        alert(data.error || "프로필 생성 실패");
        return;
      }

      setProfiles((prev) => [...prev, data.profile]);
      setNewProfileName("");
      setShowCreateModal(false);
    } catch (error) {
      console.error(error);
      alert("프로필 생성 실패");
    } finally {
      setIsCreatingProfile(false);
    }
  };

  // =========================
  // 프로필 삭제
  // 실제 삭제 기준은 folderId
  // =========================
  const deleteProfile = async (profile: Profile) => {
    const ok = window.confirm(
      `${profile.name} 프로필을 삭제할까요?\n프로필 폴더와 촬영 데이터가 모두 삭제됩니다.`
    );

    if (!ok) return;

    try {
      setIsDeletingProfile(profile.folderId);

      const encodedId = encodeURIComponent(profile.folderId);
      const res = await fetch(`${API_BASE}/profiles/${encodedId}`, {
        method: "DELETE",
      });

      const data = await res.json();

      if (!data.ok) {
        alert(data.error || "프로필 삭제 실패");
        return;
      }

      setProfiles((prev) => prev.filter((p) => p.folderId !== profile.folderId));

      if (selectedProfile?.folderId === profile.folderId) {
        setSelectedProfile(null);
        setScreen("profiles");
      }
    } catch (error) {
      console.error(error);
      alert("프로필 삭제 실패");
    } finally {
      setIsDeletingProfile(null);
    }
  };

  // =========================
  // 프로필 선택
  // =========================
  const selectProfile = (profile: Profile) => {
    setSelectedProfile(profile);
    setScreen("camera");
  };

  // =========================
  // 프로필 선택 화면으로 돌아가기
  // =========================
  const goToProfiles = () => {
    setScreen("profiles");
    setSelectedProfile(null);
  };

  // =========================
  // 촬영
  // 실제 촬영 기준은 profileId(folderId)
  // =========================
  const capturePhoto = async () => {
    if (!selectedProfile) {
      alert("프로필을 먼저 선택하세요.");
      return;
    }

    try {
      setIsCapturing(true);

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
        alert(data.error || "촬영 실패");
        console.error(data);
        return;
      }

      alert(`${selectedProfile.name} 프로필에 저장 완료`);
      console.log(data);
    } catch (error) {
      alert("촬영 실패");
      console.error(error);
    } finally {
      setIsCapturing(false);
    }
  };

  // =========================
  // 이전 기록 확인
  // =========================
  const openHistory = () => {
    if (!selectedProfile) {
      alert("프로필을 먼저 선택하세요.");
      return;
    }

    alert(
      `${selectedProfile.name} 프로필의 이전 기록 확인 화면은 다음 단계에서 연결하면 됩니다.`
    );
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

        .profiles-container {
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
        .add-card {
          border-radius: 24px;
          min-height: 220px;
          padding: 20px;
          transition: 0.2s ease;
        }

        .profile-card {
          background: rgba(255,255,255,0.07);
          border: 1px solid rgba(255,255,255,0.1);
          color: white;
          position: relative;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
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
        .retry-btn:hover {
          transform: scale(1.02);
        }

        .add-card {
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

        .profile-select-tag {
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

        .delete-btn {
          background: rgba(239, 68, 68, 0.18);
          border: 1px solid rgba(255,255,255,0.12);
          color: white;
          border-radius: 14px;
          padding: 10px 14px;
          cursor: pointer;
          font-size: 13px;
          transition: 0.2s ease;
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

        .control-btn {
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
        .delete-btn:disabled {
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
          z-index: 100;
        }

        .modal-box {
          width: 100%;
          max-width: 420px;
          background: #0f172a;
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 24px;
          padding: 24px;
          color: white;
        }

        .modal-title {
          margin: 0 0 18px 0;
          font-size: 22px;
          font-weight: 700;
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
        }
      `}</style>

      <div className="app-bg">
        <div className="mirror-frame">
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
                              deleteProfile(profile);
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
    </>
  );
}

export default App;