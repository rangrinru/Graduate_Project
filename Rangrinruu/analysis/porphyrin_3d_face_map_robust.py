from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("mediapipe가 필요합니다. 설치: python -m pip install mediapipe") from e

try:
    import plotly.graph_objects as go
except ImportError as e:
    raise ImportError("plotly가 필요합니다. 설치: python -m pip install plotly") from e


DEFAULT_OUTPUT_NAME = "porphyrin_3d_face_map.html"
DEFAULT_POINTS_JSON_NAME = "porphyrin_3d_face_points.json"
DEFAULT_SUMMARY_JSON_NAME = "porphyrin_3d_face_summary.json"
DEFAULT_DEBUG_FACE_NAME = "debug_face_mesh_input.jpg"

DENSITY_SIGMA = 35.0
POINT_Z_OFFSET = 5.0

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("JSON 읽기 실패:", path, e)
        return None


def save_json(path: Path, data: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_image_korean_path(image_path: Path):
    arr = np.fromfile(str(image_path), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def save_image_korean_path(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    encoded.tofile(str(path))


def find_default_base_path() -> Path:
    candidates = [
        Path.home() / "Graduate_Project" / "captures" / "sessions" / "cam4_660nm",
        Path.home() / "Graduate_Project" / "captures" / "cam4_660nm",
        Path.cwd() / "captures" / "sessions" / "cam4_660nm",
        Path.cwd() / "captures" / "cam4_660nm",
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\captures\sessions\cam4_660nm"),
        Path(r"C:\Users\kangm\PycharmProjects\PythonProject\Graduate_Project\captures\cam4_660nm"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def find_latest_metadata(base_path: Path) -> Path | None:
    base_path = Path(base_path)
    if not base_path.exists():
        print("기준 폴더가 없습니다:", base_path)
        return None

    candidates = list(base_path.glob("**/metadata.json"))
    candidates.extend(list(base_path.glob("**/*metadata*.json")))
    candidates = [p for p in candidates if p.is_file()]

    if not candidates:
        print("metadata.json 파일이 없습니다:", base_path)
        return None

    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def normalize_0_1(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    vmin = np.min(values)
    vmax = np.max(values)
    if float(vmax - vmin) < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def path_from_metadata_or_fallback(metadata: dict[str, Any], face_image_path: str | None) -> Path | None:
    if face_image_path:
        p = Path(face_image_path).expanduser()
        if p.exists():
            return p
        print("지정한 face image 경로가 없습니다:", p)

    saved_file = metadata.get("saved_file", None)
    if saved_file:
        p = Path(saved_file)
        if p.exists():
            return p
        print("metadata의 saved_file 경로가 존재하지 않습니다:", p)

    return None


def preprocess_face_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    흑백/저조도/저대비 얼굴 사진에서 FaceMesh 검출률을 높이기 위한 전처리.
    """
    if image_bgr is None:
        return image_bgr

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 너무 어둡거나 밝기 범위가 좁은 경우 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 밝기 범위 정규화
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 얼굴 검출기는 3채널 입력이 안정적
    enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return enhanced


def rotate_image(image_bgr: np.ndarray, angle_name: str) -> np.ndarray:
    if angle_name == "0":
        return image_bgr
    if angle_name == "90cw":
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    if angle_name == "90ccw":
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle_name == "180":
        return cv2.rotate(image_bgr, cv2.ROTATE_180)
    return image_bgr


def inverse_rotate_points(points_xyz: np.ndarray, angle_name: str, original_w: int, original_h: int) -> np.ndarray:
    """
    회전된 이미지에서 얻은 랜드마크 좌표를 원본 이미지 좌표계로 되돌림.
    z는 그대로 유지.
    """
    pts = points_xyz.copy()

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()

    if angle_name == "0":
        return pts

    if angle_name == "90cw":
        # 원본 (w,h) -> 회전 후 (h,w)
        # rotated_x = original_h - 1 - original_y
        # rotated_y = original_x
        orig_x = y
        orig_y = original_h - 1 - x
        pts[:, 0] = orig_x
        pts[:, 1] = orig_y
        return pts

    if angle_name == "90ccw":
        # rotated_x = original_y
        # rotated_y = original_w - 1 - original_x
        orig_x = original_w - 1 - y
        orig_y = x
        pts[:, 0] = orig_x
        pts[:, 1] = orig_y
        return pts

    if angle_name == "180":
        orig_x = original_w - 1 - x
        orig_y = original_h - 1 - y
        pts[:, 0] = orig_x
        pts[:, 1] = orig_y
        return pts

    return pts


def landmarks_to_xyz(face_landmarks, w: int, h: int) -> np.ndarray:
    points = []
    for lm in face_landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        z = -lm.z * w
        points.append([float(x), float(y), float(z)])
    return np.array(points, dtype=np.float32)


def run_facemesh_once(image_bgr: np.ndarray, min_conf: float):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_conf
    ) as face_mesh:
        return face_mesh.process(image_rgb)


def detect_face_mesh_3d_robust(image_bgr: np.ndarray, debug_dir: Path | None = None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    원본, 대비 보정본, 회전본을 순차적으로 시도해서 FaceMesh 검출률을 높임.
    """
    if image_bgr is None:
        return None, {"success": False, "reason": "image is None"}

    original_h, original_w = image_bgr.shape[:2]

    candidates = []
    for angle in ["0", "90cw", "90ccw", "180"]:
        rot = rotate_image(image_bgr, angle)
        candidates.append((f"original_{angle}", angle, rot))
        candidates.append((f"enhanced_{angle}", angle, preprocess_face_image(rot)))

    conf_list = [0.5, 0.35, 0.2]

    last_debug = {
        "success": False,
        "tried": [],
        "original_size": [original_w, original_h],
    }

    for name, angle, candidate in candidates:
        h, w = candidate.shape[:2]

        # 너무 큰 이미지는 검출 안정성을 위해 폭 1280 정도로 축소해서 시도
        scale = 1.0
        detect_img = candidate
        if w > 1280:
            scale = 1280.0 / w
            detect_img = cv2.resize(candidate, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)

        dh, dw = detect_img.shape[:2]

        for conf in conf_list:
            last_debug["tried"].append({"candidate": name, "confidence": conf, "size": [dw, dh]})
            result = run_facemesh_once(detect_img, conf)

            if result.multi_face_landmarks:
                mesh_scaled = landmarks_to_xyz(result.multi_face_landmarks[0], dw, dh)

                # 축소 검출 좌표를 candidate 원본 좌표계로 복원
                mesh_scaled[:, 0] /= scale
                mesh_scaled[:, 1] /= scale
                mesh_scaled[:, 2] /= scale

                # 회전 이미지 좌표계를 원본 이미지 좌표계로 복원
                mesh_original = inverse_rotate_points(mesh_scaled, angle, original_w, original_h)

                debug_info = {
                    "success": True,
                    "selected_candidate": name,
                    "selected_angle": angle,
                    "selected_confidence": conf,
                    "original_size": [original_w, original_h],
                    "detected_size": [dw, dh],
                    "scale": scale,
                }

                if debug_dir is not None:
                    dbg = candidate.copy()
                    for p in mesh_scaled.astype(int):
                        cv2.circle(dbg, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
                    save_image_korean_path(debug_dir / DEFAULT_DEBUG_FACE_NAME, dbg)

                return mesh_original, debug_info

    return None, last_debug


def compute_density_on_mesh(mesh_xy: np.ndarray, detected_points: list, sigma: float = DENSITY_SIGMA) -> np.ndarray:
    n_vertices = mesh_xy.shape[0]
    if not detected_points:
        return np.zeros(n_vertices, dtype=np.float32)

    mesh_xy = mesh_xy.astype(np.float32)
    pts = np.array(detected_points, dtype=np.float32)

    density = np.zeros(n_vertices, dtype=np.float32)
    sigma2 = sigma * sigma * 2.0

    for p in pts:
        d2 = np.sum((mesh_xy - p[:2]) ** 2, axis=1)
        density += np.exp(-d2 / sigma2)

    return density


def map_points_to_3d(mesh_xyz: np.ndarray, detected_points: list) -> np.ndarray:
    if not detected_points:
        return np.zeros((0, 3), dtype=np.float32)

    mesh_xy = mesh_xyz[:, :2]
    points_3d = []

    for p in detected_points:
        p2 = np.array(p[:2], dtype=np.float32)
        d2 = np.sum((mesh_xy - p2) ** 2, axis=1)
        idx = int(np.argmin(d2))

        x = float(p2[0])
        y = float(p2[1])
        z = float(mesh_xyz[idx, 2] + POINT_Z_OFFSET)
        points_3d.append([x, y, z])

    return np.array(points_3d, dtype=np.float32)


def build_plotly_figure(mesh_xyz: np.ndarray, density: np.ndarray, porphyrin_points_3d: np.ndarray):
    x = mesh_xyz[:, 0]
    y = -mesh_xyz[:, 1]
    z = mesh_xyz[:, 2]

    intensity = normalize_0_1(density)

    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=z,
        intensity=intensity,
        colorscale="Reds",
        opacity=0.92,
        colorbar=dict(title="Porphyrin Density"),
        flatshading=False,
        hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<br>density=%{intensity:.3f}<extra></extra>"
    ))

    if porphyrin_points_3d.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=porphyrin_points_3d[:, 0],
            y=-porphyrin_points_3d[:, 1],
            z=porphyrin_points_3d[:, 2],
            mode="markers",
            marker=dict(size=4, color="red", opacity=0.95, symbol="circle"),
            name="Porphyrin Points",
            hovertemplate="Porphyrin<br>x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>"
        ))

    fig.update_layout(
        title="3D Face Model with Porphyrin Distribution",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=0.0, y=-1.9, z=0.6))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.01, y=0.99)
    )
    return fig


def create_3d_porphyrin_face_map(metadata_path: Path, face_image_path: str | None = None, output_dir: Path | None = None):
    metadata_path = Path(metadata_path)

    metadata = load_json(metadata_path)
    if metadata is None:
        return None

    analysis_result = metadata.get("analysis_result", {})
    if not isinstance(analysis_result, dict):
        analysis_result = {}

    detected_points = analysis_result.get("detected_points", [])
    if not isinstance(detected_points, list):
        detected_points = []

    image_path = path_from_metadata_or_fallback(metadata, face_image_path)
    if image_path is None:
        raise RuntimeError("얼굴 이미지 경로를 찾지 못했습니다. --face-image 옵션으로 직접 지정하세요.")

    image = load_image_korean_path(image_path)
    if image is None:
        raise RuntimeError(f"이미지 로드 실패: {image_path}")

    if output_dir is None:
        output_dir = metadata_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("얼굴 이미지 로드 성공:", image_path)
    print("얼굴 이미지 크기:", image.shape)

    mesh_xyz, debug_info = detect_face_mesh_3d_robust(image, debug_dir=output_dir)

    debug_json_path = output_dir / "debug_face_mesh_detection.json"
    save_json(debug_json_path, debug_info)

    if mesh_xyz is None:
        raise RuntimeError(
            "얼굴 3D 랜드마크를 찾지 못했습니다. "
            f"debug 파일을 확인하세요: {debug_json_path}. "
            "정면 얼굴이 크고 밝게 보이는 일반광 사진을 사용하거나, 얼굴이 회전되어 있지 않은지 확인하세요."
        )

    mesh_xy = mesh_xyz[:, :2]
    density = compute_density_on_mesh(mesh_xy, detected_points, sigma=DENSITY_SIGMA)
    density_norm = normalize_0_1(density)
    porphyrin_points_3d = map_points_to_3d(mesh_xyz, detected_points)

    html_path = output_dir / DEFAULT_OUTPUT_NAME
    points_json_path = output_dir / DEFAULT_POINTS_JSON_NAME
    summary_json_path = output_dir / DEFAULT_SUMMARY_JSON_NAME

    fig = build_plotly_figure(mesh_xyz, density_norm, porphyrin_points_3d)
    fig.write_html(str(html_path), include_plotlyjs=True)

    points_payload = {
        "generated_at": datetime.now().isoformat(),
        "source_metadata": str(metadata_path),
        "source_face_image": str(image_path),
        "mesh_vertex_count": int(mesh_xyz.shape[0]),
        "porphyrin_point_count": int(len(detected_points)),
        "face_mesh_debug": debug_info,
        "mesh_points_xyz": mesh_xyz.round(4).tolist(),
        "porphyrin_points_2d": detected_points,
        "porphyrin_points_3d": porphyrin_points_3d.round(4).tolist(),
        "mesh_density": density_norm.round(6).tolist(),
    }
    save_json(points_json_path, points_payload)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "analysis_type": "3d_face_porphyrin_distribution",
        "metadata_file": str(metadata_path),
        "face_image_file": str(image_path),
        "output_files": {
            "html_3d_map": str(html_path),
            "points_json": str(points_json_path),
            "summary_json": str(summary_json_path),
            "debug_face_mesh_detection": str(debug_json_path),
            "debug_face_mesh_input": str(output_dir / DEFAULT_DEBUG_FACE_NAME),
        },
        "summary": {
            "mesh_vertex_count": int(mesh_xyz.shape[0]),
            "porphyrin_point_count": int(len(detected_points)),
            "max_density": float(np.max(density_norm)) if density_norm.size > 0 else 0.0,
            "mean_density": float(np.mean(density_norm)) if density_norm.size > 0 else 0.0,
        },
        "debug": debug_info,
        "note": (
            "전면 얼굴 사진 1장을 기반으로 MediaPipe FaceMesh가 추정한 시각화용 3D 메쉬입니다. "
            "cam4 포르피린 좌표와 face-image 좌표계가 일치할수록 정확합니다."
        )
    }
    save_json(summary_json_path, summary)

    print("\n3D 얼굴 모델 위 포르피린 분포 표시 완료")
    print("사용 이미지:", image_path)
    print("metadata:", metadata_path)
    print("FaceMesh 검출 방식:", debug_info.get("selected_candidate"))
    print("얼굴 메쉬 정점 수:", mesh_xyz.shape[0])
    print("포르피린 점 개수:", len(detected_points))
    print("HTML:", html_path)
    print("Debug:", debug_json_path)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default=None, help="porphyrin_analysis 결과 metadata.json 경로")
    parser.add_argument("--base", type=str, default=None, help="cam4_660nm 기준 폴더. metadata를 자동으로 찾을 때 사용")
    parser.add_argument("--face-image", type=str, default=None, help="3D 얼굴 메쉬를 만들 전면 얼굴 이미지 경로")
    parser.add_argument("--out", type=str, default=None, help="결과 저장 폴더")
    args = parser.parse_args()

    if args.metadata:
        metadata_path = Path(args.metadata).expanduser()
    else:
        base_path = Path(args.base).expanduser() if args.base else find_default_base_path()
        metadata_path = find_latest_metadata(base_path)

    if metadata_path is None:
        raise RuntimeError("metadata.json을 찾지 못했습니다.")

    out_dir = Path(args.out).expanduser() if args.out else None

    create_3d_porphyrin_face_map(
        metadata_path=metadata_path,
        face_image_path=args.face_image,
        output_dir=out_dir
    )
