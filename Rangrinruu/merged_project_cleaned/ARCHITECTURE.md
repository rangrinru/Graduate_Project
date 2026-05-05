# 역할 분리 정리

- `App.tsx` / `App.css` / `index.css` / `main.tsx`
  - 화면 표시 전용
  - 프로필 선택, 스트림 표시, 촬영 버튼, 기록/분석 결과 표시
  - 하드웨어 제어 없음

- `camera_server.py`
  - Flask API 라우팅과 프로필/기록 파일 관리 전용
  - 촬영은 `CaptureService` 호출
  - 분석은 `analyze_session_dir()` 호출
  - 촬영/분석 알고리즘 자체는 넣지 않음

- `rpicam_03_white_led_auto_v5.py`
  - 카메라, GPIO, 얼굴 정렬, 자동 촬영 전용
  - 저장 결과는 세션 폴더로 반환
  - 분석 로직 없음

- `porphyrin_analysis.py`
  - 세션 폴더를 입력받아 포르피린 분석만 수행
  - 세션 metadata를 읽어서 실제 파일 경로와 ROI를 결정
  - 카메라 제어/Flask/UI 로직 없음
