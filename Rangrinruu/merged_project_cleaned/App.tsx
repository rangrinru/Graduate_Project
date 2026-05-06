
import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type Screen = 'profiles' | 'camera' | 'history' | 'historyDetail'

type Profile = {
  id: number
  name: string
  folderId: string
  createdAt: string
}

type HistoryItem = {
  captureId: string
  capturedAt: string
  displayTime: string
  profileId: string
  profileName: string
}

type HistoryImageItem = {
  camera: string
  display_name: string
  filter_type: string
  exists: boolean
  image_url: string | null
  metadata?: Record<string, unknown>
}

type AnalysisReport = {
  exists: boolean
  report: {
    porphyrin_count: number
    porphyrin_area: number
    mean_intensity: number
    face_bbox?: [number, number, number, number] | null
  } | null
  overlay_url: string | null
  mask_url: string | null
  roi_mask_url: string | null
}

type HistoryDetail = {
  captureId: string
  capturedAt: string
  displayTime: string
  profileId: string
  profileName: string
  sessionMetadata?: Record<string, unknown>
  images: {
    no_filter?: HistoryImageItem
    '405nm_filter'?: HistoryImageItem
    '660nm_filter'?: HistoryImageItem
  }
  analysis: AnalysisReport
}

type Toast = {
  type: 'success' | 'error' | 'info'
  message: string
}

type CaptureStatus = {
  ok: boolean
  status: string
  armed: boolean
  analysis_in_progress: boolean
  dynamic_eye_threshold?: number
  stable_face_count?: number
  eyes_closed_count?: number
  last_analysis_report?: {
    porphyrin_count: number
    porphyrin_area: number
    mean_intensity: number
  } | null
}

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

function App() {
  const [screen, setScreen] = useState<Screen>('profiles')

  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<Profile | null>(null)

  const [profileName, setProfileName] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showGuide, setShowGuide] = useState(true)

  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([])
  const [selectedHistory, setSelectedHistory] = useState<HistoryDetail | null>(null)

  const [selectedFilter, setSelectedFilter] = useState<'no_filter' | '405nm_filter' | '660nm_filter'>('660nm_filter')
  const [selectedAnalysisView, setSelectedAnalysisView] = useState<'overlay' | 'mask' | 'roi-mask'>('overlay')

  const [isLoadingProfiles, setIsLoadingProfiles] = useState(true)
  const [isSubmittingProfile, setIsSubmittingProfile] = useState(false)
  const [isDeletingProfile, setIsDeletingProfile] = useState<string | null>(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const [captureMode, setCaptureMode] = useState<'manual' | 'auto' | null>(null)
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)
  const [isLoadingDetail, setIsLoadingDetail] = useState(false)
  const [isDeletingHistory, setIsDeletingHistory] = useState<string | null>(null)

  const [status, setStatus] = useState<CaptureStatus | null>(null)
  const [toast, setToast] = useState<Toast | null>(null)

  const toastTimerRef = useRef<number | null>(null)

  const currentImage = useMemo(() => {
    if (!selectedHistory) return null
    return selectedHistory.images[selectedFilter] ?? null
  }, [selectedHistory, selectedFilter])

  const currentAnalysisImage = useMemo(() => {
    if (!selectedHistory?.analysis?.exists) return null
    if (selectedAnalysisView === 'mask') return selectedHistory.analysis.mask_url
    if (selectedAnalysisView === 'roi-mask') return selectedHistory.analysis.roi_mask_url
    return selectedHistory.analysis.overlay_url
  }, [selectedHistory, selectedAnalysisView])

  const showToast = (message: string, type: Toast['type']) => {
    setToast({ message, type })
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current)
    toastTimerRef.current = window.setTimeout(() => setToast(null), 2600)
  }

  useEffect(() => {
    return () => {
      if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current)
    }
  }, [])

  const getImageSrc = (url: string | null | undefined) => {
    if (!url) return ''
    return url.startsWith('http') ? url : `${API_BASE}${url}`
  }

  const fetchProfiles = async () => {
    try {
      setIsLoadingProfiles(true)
      const res = await fetch(`${API_BASE}/profiles`)
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || '프로필 불러오기 실패')
      setProfiles(data.profiles ?? [])
    } catch (error) {
      console.error(error)
      showToast('프로필 목록을 불러오지 못했습니다.', 'error')
    } finally {
      setIsLoadingProfiles(false)
    }
  }

  useEffect(() => {
    fetchProfiles()
  }, [])

  useEffect(() => {
    if (screen !== 'camera') return
    let alive = true
    const tick = async () => {
      try {
        const res = await fetch(`${API_BASE}/capture-status`)
        const data = await res.json()
        if (alive) setStatus(data)
      } catch {
        if (alive) setStatus(null)
      }
    }
    tick()
    const id = window.setInterval(tick, 600)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [screen])

  const submitCreateProfile = async () => {
    const name = profileName.trim()
    if (!name) {
      showToast('프로필 이름을 입력하세요.', 'error')
      return
    }
    try {
      setIsSubmittingProfile(true)
      const res = await fetch(`${API_BASE}/profiles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      })
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || '프로필 생성 실패')
      setProfiles((prev) => [...prev, data.profile])
      setShowCreateModal(false)
      setProfileName('')
      showToast('프로필이 생성되었습니다.', 'success')
    } catch (error) {
      console.error(error)
      showToast(error instanceof Error ? error.message : '프로필 생성 실패', 'error')
    } finally {
      setIsSubmittingProfile(false)
    }
  }

  const removeProfile = async (profile: Profile) => {
    try {
      setIsDeletingProfile(profile.folderId)
      const res = await fetch(`${API_BASE}/profiles/${encodeURIComponent(profile.folderId)}`, {
        method: 'DELETE',
      })
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || '프로필 삭제 실패')
      setProfiles((prev) => prev.filter((item) => item.folderId !== profile.folderId))
      if (selectedProfile?.folderId === profile.folderId) {
        setSelectedProfile(null)
        setScreen('profiles')
      }
      showToast('프로필이 삭제되었습니다.', 'success')
    } catch (error) {
      console.error(error)
      showToast(error instanceof Error ? error.message : '프로필 삭제 실패', 'error')
    } finally {
      setIsDeletingProfile(null)
    }
  }

  const selectProfile = (profile: Profile) => {
    setSelectedProfile(profile)
    setSelectedHistory(null)
    setHistoryItems([])
    setScreen('camera')
  }

  const capture = async (mode: 'manual' | 'auto') => {
    if (!selectedProfile) {
      showToast('프로필을 먼저 선택하세요.', 'error')
      return
    }
    try {
      setIsCapturing(true)
      setCaptureMode(mode)
      showToast(mode === 'manual' ? '수동 촬영을 시작합니다.' : '자동 촬영을 시작합니다.', 'info')
      const endpoint = mode === 'manual' ? '/capture-all' : '/capture-auto'
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profileId: selectedProfile.folderId }),
      })
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || '촬영 실패')
      showToast(`${mode === 'manual' ? '수동' : '자동'} 촬영과 분석이 완료되었습니다.`, 'success')
      await fetchHistory(false)
    } catch (error) {
      console.error(error)
      showToast(error instanceof Error ? error.message : '촬영 실패', 'error')
    } finally {
      setIsCapturing(false)
      setCaptureMode(null)
    }
  }

  const fetchHistory = async (moveScreen = true) => {
    if (!selectedProfile) {
      showToast('프로필을 먼저 선택하세요.', 'error')
      return
    }
    try {
      setIsLoadingHistory(true)
      const res = await fetch(`${API_BASE}/profiles/${encodeURIComponent(selectedProfile.folderId)}/history`)
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || '이전 기록 조회 실패')
      setHistoryItems(data.history ?? [])
      if (moveScreen) setScreen('history')
    } catch (error) {
      console.error(error)
      showToast(error instanceof Error ? error.message : '이전 기록 조회 실패', 'error')
    } finally {
      setIsLoadingHistory(false)
    }
  }

  const openHistoryDetail = async (captureId: string) => {
    if (!selectedProfile) return
    try {
      setIsLoadingDetail(true)
      const res = await fetch(`${API_BASE}/profiles/${encodeURIComponent(selectedProfile.folderId)}/history/${encodeURIComponent(captureId)}`)
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || '기록 상세 조회 실패')
      setSelectedHistory(data)
      setSelectedFilter('660nm_filter')
      setSelectedAnalysisView('overlay')
      setScreen('historyDetail')
    } catch (error) {
      console.error(error)
      showToast(error instanceof Error ? error.message : '기록 상세 조회 실패', 'error')
    } finally {
      setIsLoadingDetail(false)
    }
  }

  const deleteHistory = async (item: HistoryItem) => {
    if (!selectedProfile) return
    try {
      setIsDeletingHistory(item.captureId)
      const res = await fetch(
        `${API_BASE}/profiles/${encodeURIComponent(selectedProfile.folderId)}/history/${encodeURIComponent(item.captureId)}`,
        { method: 'DELETE' }
      )
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || '기록 삭제 실패')
      setHistoryItems((prev) => prev.filter((history) => history.captureId !== item.captureId))
      if (selectedHistory?.captureId === item.captureId) {
        setSelectedHistory(null)
        setScreen('history')
      }
      showToast('촬영 기록이 삭제되었습니다.', 'success')
    } catch (error) {
      console.error(error)
      showToast(error instanceof Error ? error.message : '기록 삭제 실패', 'error')
    } finally {
      setIsDeletingHistory(null)
    }
  }

  return (
    <div className="app-shell">
      {toast && <div className={`toast ${toast.type}`}>{toast.message}</div>}

      <div className="mirror-frame">
        <header className="top-bar">
          <div>
            <h1>UVA Skin Mirror</h1>
            <p>촬영 · 분석 · 기록 관리를 한 화면에서 진행합니다.</p>
          </div>
          <div className="clock-chip">
            {new Intl.DateTimeFormat('ko-KR', {
              hour: '2-digit',
              minute: '2-digit',
              hour12: false,
            }).format(new Date())}
          </div>
        </header>

        {screen === 'profiles' && (
          <section className="screen scrollable">
            <div className="section-head">
              <div>
                <h2>프로필 선택</h2>
                <p>사용자별로 촬영 기록과 분석 결과를 분리합니다.</p>
              </div>
              <button className="primary-btn" onClick={() => setShowCreateModal(true)}>
                프로필 추가
              </button>
            </div>

            {isLoadingProfiles ? (
              <div className="empty-box">프로필을 불러오는 중입니다...</div>
            ) : profiles.length === 0 ? (
              <div className="empty-box">
                아직 프로필이 없습니다.
                <br />
                상단의 프로필 추가 버튼을 눌러 시작하세요.
              </div>
            ) : (
              <div className="card-grid">
                {profiles.map((profile) => (
                  <article key={profile.id} className="profile-card">
                    <button className="card-main" onClick={() => selectProfile(profile)}>
                      <div className="profile-avatar">👤</div>
                      <div className="profile-name">{profile.name}</div>
                      <div className="profile-date">생성일 {profile.createdAt}</div>
                    </button>
                    <button
                      className="danger-btn"
                      onClick={() => removeProfile(profile)}
                      disabled={isDeletingProfile === profile.folderId}
                    >
                      {isDeletingProfile === profile.folderId ? '삭제 중...' : '삭제'}
                    </button>
                  </article>
                ))}
              </div>
            )}
          </section>
        )}

        {screen === 'camera' && (
          <section className="screen camera-screen">
            <img className="camera-live" src={`${API_BASE}/stream-cam4`} alt="camera live" />
            {showGuide && <div className="guide-circle">얼굴 위치 맞추기</div>}

            <div className="status-panel">
              <small>선택된 프로필</small>
              <strong>{selectedProfile?.name}</strong>
              <div className="status-line">{status?.status ?? '상태를 불러오는 중...'}</div>
              {status && (
                <div className="status-metrics">
                  <span>Stable {status.stable_face_count ?? 0}</span>
                  <span>Eyes {status.eyes_closed_count ?? 0}</span>
                  <span>TH {status.dynamic_eye_threshold?.toFixed(2) ?? '-'}</span>
                </div>
              )}
            </div>

            <div className="camera-actions right">
              <button className="glass-btn" onClick={() => setShowGuide((prev) => !prev)} disabled={isCapturing}>
                {showGuide ? '실루엣 끄기' : '실루엣 켜기'}
              </button>
              <button className="glass-btn" onClick={() => fetchHistory(true)} disabled={isCapturing}>
                이전 기록
              </button>
            </div>

            <div className="camera-actions left-bottom">
              <button className="glass-btn" onClick={() => setScreen('profiles')} disabled={isCapturing}>
                프로필로 돌아가기
              </button>
            </div>

            <div className="capture-dock">
              <button className="secondary-btn" onClick={() => capture('manual')} disabled={isCapturing}>
                {isCapturing && captureMode === 'manual' ? '촬영 중...' : '수동 촬영'}
              </button>
              <button className="capture-btn" onClick={() => capture('auto')} disabled={isCapturing}>
                <span />
              </button>
              <button className="secondary-btn" onClick={() => capture('auto')} disabled={isCapturing}>
                {isCapturing && captureMode === 'auto' ? '자동 촬영 중...' : '자동 촬영'}
              </button>
            </div>
          </section>
        )}

        {screen === 'history' && (
          <section className="screen scrollable">
            <div className="section-head">
              <div>
                <h2>이전 기록</h2>
                <p>{selectedProfile?.name} 프로필의 촬영 세션을 확인합니다.</p>
              </div>
              <button className="glass-btn dark" onClick={() => setScreen('camera')}>
                카메라로 돌아가기
              </button>
            </div>

            {isLoadingHistory ? (
              <div className="empty-box">기록을 불러오는 중입니다...</div>
            ) : historyItems.length === 0 ? (
              <div className="empty-box">아직 저장된 촬영 기록이 없습니다.</div>
            ) : (
              <div className="history-list">
                {historyItems.map((item) => (
                  <article key={item.captureId} className="history-card">
                    <button className="card-main left-align" onClick={() => openHistoryDetail(item.captureId)}>
                      <div className="history-time">{item.displayTime}</div>
                      <div className="history-id">captureId: {item.captureId}</div>
                    </button>
                    <button
                      className="danger-btn"
                      onClick={() => deleteHistory(item)}
                      disabled={isDeletingHistory === item.captureId}
                    >
                      {isDeletingHistory === item.captureId ? '삭제 중...' : '삭제'}
                    </button>
                  </article>
                ))}
              </div>
            )}
          </section>
        )}

        {screen === 'historyDetail' && (
          <section className="screen scrollable">
            <div className="section-head">
              <div>
                <h2>촬영 기록 상세</h2>
                <p>{selectedHistory?.displayTime}</p>
              </div>
              <button className="glass-btn dark" onClick={() => setScreen('history')}>
                목록으로 돌아가기
              </button>
            </div>

            {isLoadingDetail ? (
              <div className="empty-box">상세 정보를 불러오는 중입니다...</div>
            ) : selectedHistory ? (
              <div className="detail-layout">
                <div className="viewer-card">
                  <div className="chip-row">
                    <button className={`chip ${selectedFilter === 'no_filter' ? 'active' : ''}`} onClick={() => setSelectedFilter('no_filter')}>No_Filter</button>
                    <button className={`chip ${selectedFilter === '405nm_filter' ? 'active' : ''}`} onClick={() => setSelectedFilter('405nm_filter')}>405nm</button>
                    <button className={`chip ${selectedFilter === '660nm_filter' ? 'active' : ''}`} onClick={() => setSelectedFilter('660nm_filter')}>660nm</button>
                  </div>

                  {currentImage?.exists && currentImage.image_url ? (
                    <img className="detail-image" src={getImageSrc(currentImage.image_url)} alt={currentImage.display_name} />
                  ) : (
                    <div className="empty-box compact">선택한 필터 이미지가 없습니다.</div>
                  )}
                </div>

                <div className="analysis-panel">
                  <div className="panel-card">
                    <h3>분석 결과</h3>
                    {selectedHistory.analysis.exists && selectedHistory.analysis.report ? (
                      <>
                        <div className="analysis-stats">
                          <div><span>Porphyrin Count</span><strong>{selectedHistory.analysis.report.porphyrin_count}</strong></div>
                          <div><span>Porphyrin Area</span><strong>{selectedHistory.analysis.report.porphyrin_area}</strong></div>
                          <div><span>Mean Intensity</span><strong>{selectedHistory.analysis.report.mean_intensity.toFixed(1)}</strong></div>
                        </div>
                        <div className="chip-row">
                          <button className={`chip ${selectedAnalysisView === 'overlay' ? 'active' : ''}`} onClick={() => setSelectedAnalysisView('overlay')}>Overlay</button>
                          <button className={`chip ${selectedAnalysisView === 'mask' ? 'active' : ''}`} onClick={() => setSelectedAnalysisView('mask')}>Mask</button>
                          <button className={`chip ${selectedAnalysisView === 'roi-mask' ? 'active' : ''}`} onClick={() => setSelectedAnalysisView('roi-mask')}>ROI</button>
                        </div>
                        {currentAnalysisImage && (
                          <img className="analysis-image" src={getImageSrc(currentAnalysisImage)} alt="analysis result" />
                        )}
                      </>
                    ) : (
                      <div className="empty-box compact">분석 결과가 아직 없습니다.</div>
                    )}
                  </div>

                  <div className="panel-card">
                    <h3>세션 정보</h3>
                    <div className="meta-grid">
                      <div><span>프로필</span><strong>{selectedHistory.profileName}</strong></div>
                      <div><span>captureId</span><strong>{selectedHistory.captureId}</strong></div>
                      <div><span>촬영 시각</span><strong>{selectedHistory.capturedAt}</strong></div>
                      <div><span>현재 필터</span><strong>{currentImage?.display_name ?? '-'}</strong></div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="empty-box">불러올 상세 정보가 없습니다.</div>
            )}
          </section>
        )}
      </div>

      {showCreateModal && (
        <div className="modal-backdrop" onClick={() => !isSubmittingProfile && setShowCreateModal(false)}>
          <div className="modal-card" onClick={(event) => event.stopPropagation()}>
            <h3>새 프로필 생성</h3>
            <input
              value={profileName}
              onChange={(event) => setProfileName(event.target.value)}
              placeholder="프로필 이름을 입력하세요"
              className="text-input"
            />
            <div className="modal-actions">
              <button className="glass-btn dark" onClick={() => setShowCreateModal(false)} disabled={isSubmittingProfile}>
                취소
              </button>
              <button className="primary-btn" onClick={submitCreateProfile} disabled={isSubmittingProfile}>
                {isSubmittingProfile ? '생성 중...' : '생성'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
