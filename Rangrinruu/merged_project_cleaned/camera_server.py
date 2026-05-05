from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from time import sleep

from flask import Flask, jsonify, Response, request, send_file
from flask_cors import CORS

from rpicam_03_white_led_auto_v5_vertical_save import CaptureService
from porphyrin_analysis import (
    analyze_session_dir,
    load_analysis_report,
    load_session_metadata,
    resolve_session_capture_file,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('camera_server')

app = Flask(__name__)
CORS(app)

SAVE_ROOT = Path.home() / 'Graduate_Project' / 'Final_Project' / 'captures'
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
PROFILES_FILE = SAVE_ROOT / 'profiles.json'
STREAM_FPS = 12

capture_service = CaptureService(save_root=SAVE_ROOT)
analysis_state = {'in_progress': False, 'last_report': None}
analysis_lock = threading.Lock()


def sanitize_profile_name(profile_name: str) -> str:
    cleaned = re.sub(r'\s+', ' ', profile_name.strip())
    if not cleaned:
        raise ValueError('유효한 프로필 이름이 아닙니다.')
    return cleaned


def make_folder_id() -> str:
    return f"profile_{int(datetime.now().timestamp() * 1000)}"


def load_profiles():
    if not PROFILES_FILE.exists():
        return []
    try:
        return json.loads(PROFILES_FILE.read_text(encoding='utf-8'))
    except Exception:
        return []


def save_profiles(profiles):
    PROFILES_FILE.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding='utf-8')


def ensure_profile_root(folder_id: str) -> Path:
    root = SAVE_ROOT / folder_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def find_profile_by_id(profile_id: str):
    for profile in load_profiles():
        if profile['folderId'] == profile_id:
            return profile
    return None


def create_profile(profile_name: str):
    display_name = sanitize_profile_name(profile_name)
    profiles = load_profiles()
    if any(p['name'] == display_name for p in profiles):
        raise ValueError('이미 존재하는 프로필입니다.')

    new_profile = {
        'id': int(datetime.now().timestamp() * 1000),
        'name': display_name,
        'folderId': make_folder_id(),
        'createdAt': datetime.now().strftime('%Y.%m.%d'),
    }
    profiles.append(new_profile)
    save_profiles(profiles)
    ensure_profile_root(new_profile['folderId'])
    return new_profile


def delete_profile(profile_id: str):
    profiles = load_profiles()
    target = next((p for p in profiles if p['folderId'] == profile_id), None)
    if target is None:
        raise ValueError('삭제할 프로필이 없습니다.')
    new_profiles = [p for p in profiles if p['folderId'] != profile_id]
    profile_root = SAVE_ROOT / target['folderId']
    if profile_root.exists():
        shutil.rmtree(profile_root)
    save_profiles(new_profiles)


def get_profile_root(profile_id: str) -> Path:
    profile = find_profile_by_id(profile_id)
    if profile is None:
        raise ValueError('존재하지 않는 프로필입니다.')
    root = SAVE_ROOT / profile['folderId']
    root.mkdir(parents=True, exist_ok=True)
    return root


def format_capture_id_to_text(capture_id: str) -> str:
    try:
        dt = datetime.strptime(capture_id, '%Y%m%d_%H%M%S_%f')
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return capture_id


def get_capture_history(profile_id: str):
    profile_root = get_profile_root(profile_id)
    profile = find_profile_by_id(profile_id)
    history = []
    for capture_dir in profile_root.iterdir():
        if not capture_dir.is_dir():
            continue
        capture_id = capture_dir.name
        meta = load_session_metadata(capture_dir)
        history.append({
            'captureId': capture_id,
            'capturedAt': meta.get('captured_at', capture_id),
            'displayTime': format_capture_id_to_text(capture_id),
            'profileId': profile['folderId'],
            'profileName': profile['name'],
        })
    history.sort(key=lambda x: x['captureId'], reverse=True)
    return history


def _build_image_item(profile_id: str, capture_id: str, cam_key: str, filter_type: str, display_name: str):
    session_dir = get_profile_root(profile_id) / capture_id
    try:
        image_path = resolve_session_capture_file(session_dir, cam_key)
    except Exception:
        image_path = None

    meta_path = session_dir / f'{cam_key}_metadata.json'
    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            metadata = {}

    return {
        'camera': cam_key,
        'display_name': display_name,
        'filter_type': filter_type,
        'exists': image_path is not None,
        'image_url': f'/profiles/{profile_id}/history/{capture_id}/image/{filter_type}' if image_path else None,
        'metadata': metadata,
    }


def get_capture_detail(profile_id: str, capture_id: str):
    session_dir = get_profile_root(profile_id) / capture_id
    profile = find_profile_by_id(profile_id)
    if not session_dir.exists():
        raise ValueError('해당 촬영 기록을 찾을 수 없습니다.')

    session_meta = load_session_metadata(session_dir)
    images = {
        'no_filter': _build_image_item(profile_id, capture_id, 'cam2', 'no_filter', 'No_Filter'),
        '405nm_filter': _build_image_item(profile_id, capture_id, 'cam3', '405nm_filter', '405nm_Filter'),
        '660nm_filter': _build_image_item(profile_id, capture_id, 'cam4', '660nm_filter', '660nm_Filter'),
    }
    if not any(item['exists'] for item in images.values()):
        raise ValueError('해당 촬영 기록의 이미지가 없습니다.')

    analysis = load_analysis_report(session_dir)
    analysis_payload = {
        'exists': False,
        'report': None,
        'overlay_url': None,
        'mask_url': None,
        'roi_mask_url': None,
    }
    if analysis:
        analysis_payload = {
            'exists': True,
            'report': analysis,
            'overlay_url': f'/profiles/{profile_id}/history/{capture_id}/analysis/overlay',
            'mask_url': f'/profiles/{profile_id}/history/{capture_id}/analysis/mask',
            'roi_mask_url': f'/profiles/{profile_id}/history/{capture_id}/analysis/roi-mask',
        }

    return {
        'captureId': capture_id,
        'capturedAt': session_meta.get('captured_at', capture_id),
        'displayTime': format_capture_id_to_text(capture_id),
        'profileId': profile['folderId'],
        'profileName': profile['name'],
        'sessionMetadata': session_meta,
        'images': images,
        'analysis': analysis_payload,
    }


def delete_capture_history(profile_id: str, capture_id: str):
    session_dir = get_profile_root(profile_id) / capture_id
    if not session_dir.exists():
        raise ValueError('삭제할 촬영 기록이 없습니다.')
    shutil.rmtree(session_dir)


def resolve_image_path(profile_id: str, capture_id: str, filter_type: str) -> Path:
    mapping = {'no_filter': 'cam2', '405nm_filter': 'cam3', '660nm_filter': 'cam4'}
    cam_key = mapping.get(filter_type)
    if cam_key is None:
        raise ValueError('유효하지 않은 필터 타입입니다.')
    session_dir = get_profile_root(profile_id) / capture_id
    return resolve_session_capture_file(session_dir, cam_key)


def resolve_analysis_asset(profile_id: str, capture_id: str, asset: str) -> Path:
    session_dir = get_profile_root(profile_id) / capture_id
    mapping = {
        'overlay': session_dir / 'analysis' / 'porphyrin_overlay.png',
        'mask': session_dir / 'analysis' / 'porphyrin_mask.png',
        'roi-mask': session_dir / 'analysis' / 'face_roi_mask.png',
        'report': session_dir / 'analysis' / 'report.json',
    }
    path = mapping.get(asset)
    if path is None or not path.exists():
        raise ValueError('분석 결과 파일이 존재하지 않습니다.')
    return path


def run_analysis_for_session(session_dir: str | Path):
    with analysis_lock:
        analysis_state['in_progress'] = True
        try:
            report = analyze_session_dir(session_dir)
            analysis_state['last_report'] = report
            return report
        finally:
            analysis_state['in_progress'] = False


def generate_stream(cam_key: str = 'cam4'):
    frame_delay = 1.0 / STREAM_FPS
    while True:
        try:
            jpg = capture_service.get_stream_jpeg(cam_key=cam_key)
            if jpg is None:
                sleep(frame_delay)
                continue
            yield (b'--frame
' b'Content-Type: image/jpeg

' + jpg + b'
')
            sleep(frame_delay)
        except Exception as exc:
            logger.exception('스트림 오류: %s', exc)
            sleep(0.3)


@app.route('/health')
def health():
    return jsonify({'ok': True, 'camera_ready': capture_service.camera_ready, 'analysis_in_progress': analysis_state['in_progress']})


@app.route('/capture-status')
def capture_status():
    payload = capture_service.get_status()
    payload['analysis_in_progress'] = analysis_state['in_progress']
    payload['last_analysis_report'] = analysis_state['last_report']
    return jsonify(payload)


@app.route('/profiles', methods=['GET'])
def get_profiles_api():
    return jsonify({'ok': True, 'profiles': load_profiles()})


@app.route('/profiles', methods=['POST'])
def create_profile_api():
    try:
        body = request.get_json(silent=True) or {}
        profile = create_profile(body.get('name', ''))
        return jsonify({'ok': True, 'profile': profile})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/profiles/<profile_id>', methods=['DELETE'])
def delete_profile_api(profile_id):
    try:
        delete_profile(profile_id)
        return jsonify({'ok': True})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/stream-cam4')
def stream_cam4():
    return Response(generate_stream('cam4'), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream/<cam_key>')
def stream_any_cam(cam_key):
    if cam_key not in {'cam2', 'cam3', 'cam4'}:
        return jsonify({'ok': False, 'error': '유효하지 않은 카메라 키입니다.'}), 400
    return Response(generate_stream(cam_key), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture-all', methods=['POST'])
def capture_all():
    try:
        body = request.get_json(silent=True) or {}
        profile_id = str(body.get('profileId', '')).strip()
        exposure_ms = int(body.get('exposureMs', capture_service.exposure_ms))
        if not profile_id:
            return jsonify({'ok': False, 'error': 'profileId가 필요합니다.'}), 400
        profile = find_profile_by_id(profile_id)
        if profile is None:
            return jsonify({'ok': False, 'error': '존재하지 않는 프로필입니다.'}), 400

        profile_root = ensure_profile_root(profile['folderId'])
        result = capture_service.manual_capture(profile_root, profile['name'], profile['folderId'], exposure_ms=exposure_ms)
        report = run_analysis_for_session(result['sessionDir'])
        return jsonify({'ok': True, 'mode': 'manual', 'profile_name': profile['name'], 'profile_id': profile['folderId'], **result, 'analysis': report})
    except Exception as exc:
        logger.exception('수동 촬영 오류: %s', exc)
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/capture-auto', methods=['POST'])
def capture_auto():
    try:
        body = request.get_json(silent=True) or {}
        profile_id = str(body.get('profileId', '')).strip()
        exposure_ms = int(body.get('exposureMs', capture_service.exposure_ms))
        timeout_sec = float(body.get('timeoutSec', 30))
        if not profile_id:
            return jsonify({'ok': False, 'error': 'profileId가 필요합니다.'}), 400
        profile = find_profile_by_id(profile_id)
        if profile is None:
            return jsonify({'ok': False, 'error': '존재하지 않는 프로필입니다.'}), 400

        profile_root = ensure_profile_root(profile['folderId'])
        result = capture_service.auto_capture(profile_root, profile['name'], profile['folderId'], exposure_ms=exposure_ms, timeout_sec=timeout_sec)
        report = run_analysis_for_session(result['sessionDir'])
        return jsonify({'ok': True, 'mode': 'auto', 'profile_name': profile['name'], 'profile_id': profile['folderId'], **result, 'analysis': report})
    except TimeoutError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 408
    except Exception as exc:
        logger.exception('자동 촬영 오류: %s', exc)
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/profiles/<profile_id>/history', methods=['GET'])
def history_api(profile_id):
    try:
        return jsonify({'ok': True, 'profileId': profile_id, 'history': get_capture_history(profile_id)})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/profiles/<profile_id>/history/<capture_id>', methods=['GET'])
def history_detail_api(profile_id, capture_id):
    try:
        return jsonify({'ok': True, **get_capture_detail(profile_id, capture_id)})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/profiles/<profile_id>/history/<capture_id>', methods=['DELETE'])
def delete_history_api(profile_id, capture_id):
    try:
        delete_capture_history(profile_id, capture_id)
        return jsonify({'ok': True})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/profiles/<profile_id>/history/<capture_id>/image/<filter_type>', methods=['GET'])
def history_image_api(profile_id, capture_id, filter_type):
    try:
        return send_file(resolve_image_path(profile_id, capture_id, filter_type))
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 404


@app.route('/profiles/<profile_id>/history/<capture_id>/analysis/<asset>', methods=['GET'])
def analysis_asset_api(profile_id, capture_id, asset):
    try:
        return send_file(resolve_analysis_asset(profile_id, capture_id, asset))
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 404


if __name__ == '__main__':
    capture_service.init_camera()
    capture_service.relay_off()
    app.run(host='0.0.0.0', port=8000, threaded=True)
