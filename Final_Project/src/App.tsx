import { useState } from "react";

type Screen = "profiles" | "camera";

type Profile = {
  id: number;
  name: string;
  createdAt: string;
};

function App() {
  const [screen, setScreen] = useState<Screen>("profiles");
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<Profile | null>(null);
  const [showGuide, setShowGuide] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newProfileName, setNewProfileName] = useState("");

  const createProfile = () => {
    const trimmed = newProfileName.trim();

    if (!trimmed) {
      alert("프로필 이름을 입력하세요.");
      return;
    }

    const today = new Date();
    const formattedDate = `${today.getFullYear()}.${String(
      today.getMonth() + 1
    ).padStart(2, "0")}.${String(today.getDate()).padStart(2, "0")}`;

    const newProfile: Profile = {
      id: Date.now(),
      name: trimmed,
      createdAt: formattedDate,
    };

    setProfiles((prev) => [...prev, newProfile]);
    setNewProfileName("");
    setShowCreateModal(false);
  };

  const selectProfile = (profile: Profile) => {
    setSelectedProfile(profile);
    setScreen("camera");
  };

  const goToProfiles = () => {
    setScreen("profiles");
    setSelectedProfile(null);
  };

  const capturePhoto = () => {
    alert("나중에 여기서 라즈베리파이 카메라 캡처 기능을 연결하면 됩니다.");
  };

  const openHistory = () => {
    alert("이전 기록 확인 화면은 다음 단계에서 연결하면 됩니다.");
  };

  return (
    <>
      <style>{`
        * {
          box-sizing: border-box;
        }

        body {
          margin: 0;
          font-family: Arial, Helvetica, sans-serif;
          background: #0b1220;
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
        }

        .time-badge {
          background: rgba(255,255,255,0.08);
          border: 1px solid rgba(255,255,255,0.12);
          color: white;
          padding: 10px 14px;
          border-radius: 16px;
          font-size: 14px;
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
          cursor: pointer;
          transition: 0.2s ease;
        }

        .profile-card {
          background: rgba(255,255,255,0.07);
          border: 1px solid rgba(255,255,255,0.1);
          color: white;
        }

        .profile-card:hover,
        .add-card:hover,
        .control-btn:hover,
        .capture-btn:hover,
        .modal-btn:hover,
        .back-btn:hover {
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

        .camera-screen {
          position: relative;
          width: 100%;
          height: 100%;
          background: radial-gradient(circle at center, rgba(255,255,255,0.06) 0%, rgba(0,0,0,0.12) 55%, rgba(0,0,0,0.28) 100%);
        }

        .camera-placeholder {
          position: absolute;
          inset: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 24px;
          color: white;
        }

        .camera-placeholder-box {
          background: rgba(0,0,0,0.28);
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 24px;
          padding: 24px;
          backdrop-filter: blur(8px);
        }

        .camera-title {
          font-size: 18px;
          font-weight: 700;
          margin-bottom: 10px;
        }

        .camera-desc {
          font-size: 14px;
          line-height: 1.6;
          color: rgba(255,255,255,0.7);
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
        }

        .selected-profile small {
          display: block;
          color: rgba(255,255,255,0.6);
          margin-bottom: 6px;
          font-size: 12px;
        }

        .selected-profile strong {
          font-size: 15px;
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

        .capture-inner {
          width: 64px;
          height: 64px;
          border-radius: 999px;
          background: white;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #111827;
          font-size: 28px;
          font-weight: bold;
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
                <div className="profiles-grid">
                  {profiles.map((profile) => (
                    <div
                      key={profile.id}
                      className="profile-card"
                      onClick={() => selectProfile(profile)}
                    >
                      <div className="profile-icon">👤</div>
                      <div className="profile-name">{profile.name}</div>
                      <div className="profile-date">
                        생성일 {profile.createdAt}
                      </div>
                      <div className="profile-select-tag">선택하기</div>
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
              </div>
            </>
          )}

          {screen === "camera" && (
            <div className="camera-screen">
              <div className="camera-placeholder">
                <div className="camera-placeholder-box">
                  <div className="camera-title">Raspberry Pi Camera Preview</div>
                  <div className="camera-desc">
                    실제 카메라 연결 전까지는 미리보기 영역입니다.
                    <br />
                    나중에 여기 부분에 라즈베리파이 카메라 스트림을 넣으면 됩니다.
                  </div>
                </div>
              </div>

              <div className="selected-profile">
                <small>선택된 프로필</small>
                <strong>{selectedProfile?.name}</strong>
              </div>

              <div className="controls">
                <button
                  className="control-btn"
                  onClick={() => setShowGuide((prev) => !prev)}
                >
                  {showGuide ? "실루엣 끄기" : "실루엣 켜기"}
                </button>
                <button className="control-btn" onClick={openHistory}>
                  이전 기록 확인
                </button>
              </div>

              {showGuide && (
                <div className="guide-wrap">
                  <div className="guide-circle">얼굴 위치 맞추기</div>
                </div>
              )}

              <button className="back-btn" onClick={goToProfiles}>
                프로필로 돌아가기
              </button>

              <div className="capture-area">
                <button className="capture-btn" onClick={capturePhoto}>
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
                if (e.key === "Enter") createProfile();
              }}
            />

            <div className="modal-actions">
              <button
                className="modal-btn cancel"
                onClick={() => {
                  setShowCreateModal(false);
                  setNewProfileName("");
                }}
              >
                취소
              </button>
              <button className="modal-btn create" onClick={createProfile}>
                생성하기
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default App;