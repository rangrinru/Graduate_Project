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
    raise ImportError(
        "mediapipe가 필요합니다. 설치: python -m pip install mediapipe"
    ) from e

try:
    import plotly.graph_objects as go
except ImportError as e:
    raise ImportError(
        "plotly가 필요합니다. 설치: python -m pip install plotly"
    ) from e


# =========================================================
# 기본 설정
# =========================================================
DEFAULT_OUTPUT_NAME = "porphyrin_3d_face_map.html"
DEFAULT_POINTS_JSON_NAME = "porphyrin_3d_face_points.json"
DEFAULT_SUMMARY_JSON_NAME = "porphyrin_3d_face_summary.json"

# 포르피린 분포를 얼굴 메쉬에 퍼뜨릴 때 사용하는 가우시안 sigma (픽셀 단위)
DENSITY_SIGMA = 35.0

# 3D 표시 시 포인트를 얼굴 표면보다 조금 띄워서 보여주기 위한 값
POINT_Z_OFFSET = 5.0

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh


# =========================================================
# 유틸
# =========================================================
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
    image_path = Path(image_path)
    arr = np.fromfile(str(image_path), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


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

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1]


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
    """
    우선순위:
    1) 사용자가 --face-image로 준 경로
    2) metadata의 saved_file
    """
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


# =========================================================
# MediaPipe로 3D 얼굴 랜드마크 추출
# =========================================================
def detect_face_mesh_3d(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    반환 shape: (N, 3)
    x, y는 픽셀 좌표
    z는 상대 깊이값(표시를 위해 스케일링한 값)
    """
    if image_bgr is None:
        return None

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        return None

    lms = result.multi_face_landmarks[0].landmark

    points = []
    for lm in lms:
        x = lm.x * w
        y = lm.y * h

        # MediaPipe z는 얼굴 크기에 대한 상대값에 가까움.
        # 보기 쉽게 픽셀 기준 스케일로 변환하고 부호를 반전해 앞쪽이 +가 되게 처리
        z = -lm.z * w

        points.append([float(x), float(y), float(z)])

    return np.array(points, dtype=np.float32)


# =========================================================
# 포르피린 점 -> 얼굴 메쉬 분포 생성
# =========================================================
def compute_density_on_mesh(
    mesh_xy: np.ndarray,
    detected_points: list[list[int]] | list[tuple[int, int]],
    sigma: float = DENSITY_SIGMA
) -> np.ndarray:
    """
    얼굴 메쉬 각 정점에서 포르피린 점들의 가우시안 누적값 계산.
    """
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


def map_points_to_3d(mesh_xyz: np.ndarray, detected_points: list[list[int]] | list[tuple[int, int]]) -> np.ndarray:
    """
    2D 검출 좌표를 가장 가까운 얼굴 메쉬 정점의 z로 올려 3D 포인트로 변환.
    """
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


# =========================================================
# Plotly 3D 시각화
# =========================================================
def build_plotly_figure(
    mesh_xyz: np.ndarray,
    density: np.ndarray,
    porphyrin_points_3d: np.ndarray,
    title: str = "3D Face Porphyrin Distribution"
):
    """
    Plotly Mesh3d를 사용한 인터랙티브 3D 얼굴 + 포르피린 분포 표시
    """
    x = mesh_xyz[:, 0]
    y = -mesh_xyz[:, 1]  # 화면에서 위쪽이 +로 보이도록 뒤집음
    z = mesh_xyz[:, 2]

    intensity = normalize_0_1(density)

    fig = go.Figure()

    # 얼굴 메쉬
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

    # 포르피린 점
    if porphyrin_points_3d.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=porphyrin_points_3d[:, 0],
            y=-porphyrin_points_3d[:, 1],
            z=porphyrin_points_3d[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color="red",
                opacity=0.95,
                symbol="circle"
            ),
            name="Porphyrin Points",
            hovertemplate="Porphyrin<br>x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(
                eye=dict(x=0.0, y=-1.9, z=0.6)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.01, y=0.99)
    )

    return fig


# =========================================================
# 메인 실행
# =========================================================
def create_3d_porphyrin_face_map(
    metadata_path: Path,
    face_image_path: str | None = None,
    output_dir: Path | None = None
) -> dict[str, Any] | None:
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
        raise RuntimeError(
            "얼굴 이미지 경로를 찾지 못했습니다. --face-image 옵션으로 전면 얼굴 사진 경로를 직접 지정하세요."
        )

    image = load_image_korean_path(image_path)
    if image is None:
        raise RuntimeError(f"이미지 로드 실패: {image_path}")

    mesh_xyz = detect_face_mesh_3d(image)
    if mesh_xyz is None:
        raise RuntimeError("얼굴 3D 랜드마크를 찾지 못했습니다. 정면 얼굴이 잘 보이는 사진을 사용하세요.")

    mesh_xy = mesh_xyz[:, :2]

    density = compute_density_on_mesh(mesh_xy, detected_points, sigma=DENSITY_SIGMA)
    density_norm = normalize_0_1(density)
    porphyrin_points_3d = map_points_to_3d(mesh_xyz, detected_points)

    # 출력 폴더
    if output_dir is None:
        output_dir = metadata_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / DEFAULT_OUTPUT_NAME
    points_json_path = output_dir / DEFAULT_POINTS_JSON_NAME
    summary_json_path = output_dir / DEFAULT_SUMMARY_JSON_NAME

    fig = build_plotly_figure(
        mesh_xyz=mesh_xyz,
        density=density_norm,
        porphyrin_points_3d=porphyrin_points_3d,
        title="3D Face Model with Porphyrin Distribution"
    )

    fig.write_html(str(html_path), include_plotlyjs=True)

    points_payload = {
        "generated_at": datetime.now().isoformat(),
        "source_metadata": str(metadata_path),
        "source_face_image": str(image_path),
        "mesh_vertex_count": int(mesh_xyz.shape[0]),
        "porphyrin_point_count": int(len(detected_points)),
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
        },
        "summary": {
            "mesh_vertex_count": int(mesh_xyz.shape[0]),
            "porphyrin_point_count": int(len(detected_points)),
            "max_density": float(np.max(density_norm)) if density_norm.size > 0 else 0.0,
            "mean_density": float(np.mean(density_norm)) if density_norm.size > 0 else 0.0,
        },
        "note": (
            "이 3D 모델은 전면 얼굴 사진 1장을 기반으로 MediaPipe FaceMesh가 추정한 3D 메쉬입니다. "
            "정밀 실측 3D가 아니라 시각화용 추정 3D입니다. "
            "cam4 포르피린 점과 전면 얼굴 이미지의 좌표계가 정확히 일치할수록 결과가 더 자연스럽습니다."
        )
    }
    save_json(summary_json_path, summary)
    summary["output_files"]["summary_json"] = str(summary_json_path)
    save_json(summary_json_path, summary)

    print("\n3D 얼굴 모델 위 포르피린 분포 표시 완료")
    print("사용 이미지:", image_path)
    print("metadata:", metadata_path)
    print("얼굴 메쉬 정점 수:", mesh_xyz.shape[0])
    print("포르피린 점 개수:", len(detected_points))
    print("HTML:", html_path)
    print("Points JSON:", points_json_path)
    print("Summary JSON:", summary_json_path)

    return summary


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="porphyrin_analysis 결과 metadata.json 경로"
    )
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="cam4_660nm 기준 폴더. metadata를 자동으로 찾을 때 사용"
    )
    parser.add_argument(
        "--face-image",
        type=str,
        default=None,
        help="3D 얼굴 메쉬를 만들 전면 얼굴 이미지 경로. 지정하지 않으면 metadata.saved_file 사용"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="결과 저장 폴더"
    )

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
