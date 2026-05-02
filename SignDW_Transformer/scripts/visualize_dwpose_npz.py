#!/usr/bin/env python3

import argparse
import importlib.util
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import matplotlib
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.draw_dw_lib import draw_pose


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".webm")
EPS = 0.01
STABLE_SIGNER_OPENPOSE_PATH = Path(
    "/research/cbim/vast/sf895/code/SignerX-inference-webui/plugins/StableSigner/easy_dwpose/draw/openpose.py"
)
_STABLE_SIGNER_OPENPOSE_DRAW = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Sign-DWPose NPZ outputs.")
    parser.add_argument("--video-dir", type=Path, required=True, help="Dataset video directory, e.g. dataset/<video_id>")
    parser.add_argument("--npz-dir", type=Path, default=None, help="Optional NPZ directory override")
    parser.add_argument("--raw-video", type=Path, default=None, help="Optional raw video path for overlay rendering")
    parser.add_argument("--fps", type=int, default=24, help="Visualization FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit the number of frames to render")
    parser.add_argument(
        "--draw-style",
        choices=("controlnext", "openpose", "dwpose"),
        default="controlnext",
        help="Rendering style. dwpose is kept as an alias of controlnext.",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.6, help="Confidence threshold for openpose filtering")
    parser.add_argument(
        "--frame-indices",
        default="1,2,3,4",
        help="Comma-separated 1-based frame indices for standalone single-frame previews",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Visualization output directory. Defaults to <video-dir>/visualization_dwpose",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing visualization outputs")
    return parser.parse_args()


def parse_frame_indices(value: str) -> List[int]:
    indices: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index > 0:
            indices.append(index)
    return sorted(set(indices))


def normalize_draw_style(value: str) -> str:
    return "controlnext" if value == "dwpose" else value


def get_stablesigner_openpose_draw():
    global _STABLE_SIGNER_OPENPOSE_DRAW  # noqa: PLW0603
    if _STABLE_SIGNER_OPENPOSE_DRAW is not None:
        return _STABLE_SIGNER_OPENPOSE_DRAW
    if not STABLE_SIGNER_OPENPOSE_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("stablesigner_openpose_draw", STABLE_SIGNER_OPENPOSE_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _STABLE_SIGNER_OPENPOSE_DRAW = getattr(module, "draw_pose", None)
    return _STABLE_SIGNER_OPENPOSE_DRAW


def load_npz_frame(npz_path: Path, aggregated_index: int = 0) -> Dict[str, object]:
    payload = np.load(npz_path, allow_pickle=True)
    if "frame_payloads" in payload.files:
        frame_payloads = payload["frame_payloads"]
        if aggregated_index >= len(frame_payloads):
            raise IndexError(f"Aggregated frame index {aggregated_index} out of range for {npz_path}")
        payload_dict = frame_payloads[aggregated_index]
        if hasattr(payload_dict, "item"):
            payload_dict = payload_dict.item()
        frame: Dict[str, object] = {}
        frame["num_persons"] = int(payload_dict["num_persons"])
        frame["frame_width"] = int(payload_dict["frame_width"])
        frame["frame_height"] = int(payload_dict["frame_height"])
        source = payload_dict
    else:
        frame = {}
        frame["num_persons"] = int(payload["num_persons"])
        frame["frame_width"] = int(payload["frame_width"])
        frame["frame_height"] = int(payload["frame_height"])
        source = payload

    for person_idx in range(frame["num_persons"]):
        source_prefix = f"person_{person_idx:03d}"
        target_prefix = f"person_{person_idx}"
        person_data: Dict[str, np.ndarray] = {}
        for suffix in (
            "body_keypoints",
            "body_scores",
            "face_keypoints",
            "face_scores",
            "left_hand_keypoints",
            "left_hand_scores",
            "right_hand_keypoints",
            "right_hand_scores",
        ):
            key = f"{source_prefix}_{suffix}"
            if key in source:
                person_data[suffix] = source[key]
        if person_data:
            frame[target_prefix] = person_data
    return frame


def to_openpose_frame(frame: Dict[str, object]) -> Dict[str, np.ndarray]:
    num_persons = int(frame["num_persons"])
    bodies: List[np.ndarray] = []
    body_scores: List[np.ndarray] = []
    hands: List[np.ndarray] = []
    hand_scores: List[np.ndarray] = []
    faces: List[np.ndarray] = []
    face_scores: List[np.ndarray] = []

    for person_idx in range(num_persons):
        person = frame.get(f"person_{person_idx}")
        if not isinstance(person, dict):
            continue
        bodies.append(np.asarray(person["body_keypoints"], dtype=np.float32))
        body_scores.append(np.asarray(person["body_scores"], dtype=np.float32))
        hands.extend(
            [
                np.asarray(person["left_hand_keypoints"], dtype=np.float32),
                np.asarray(person["right_hand_keypoints"], dtype=np.float32),
            ]
        )
        hand_scores.extend(
            [
                np.asarray(person["left_hand_scores"], dtype=np.float32),
                np.asarray(person["right_hand_scores"], dtype=np.float32),
            ]
        )
        faces.append(np.asarray(person["face_keypoints"], dtype=np.float32))
        face_scores.append(np.asarray(person["face_scores"], dtype=np.float32))

    if bodies:
        stacked_bodies = np.vstack(bodies)
        stacked_subset = np.vstack(body_scores)
    else:
        stacked_bodies = np.zeros((0, 2), dtype=np.float32)
        stacked_subset = np.zeros((0, 18), dtype=np.float32)

    return {
        "bodies": stacked_bodies,
        "body_scores": stacked_subset,
        "hands": np.asarray(hands, dtype=np.float32) if hands else np.zeros((0, 21, 2), dtype=np.float32),
        "hands_scores": np.asarray(hand_scores, dtype=np.float32) if hand_scores else np.zeros((0, 21), dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.float32) if faces else np.zeros((0, 68, 2), dtype=np.float32),
        "faces_scores": np.asarray(face_scores, dtype=np.float32) if face_scores else np.zeros((0, 68), dtype=np.float32),
    }


def filter_pose_for_openpose(frame: Dict[str, np.ndarray], conf_threshold: float, update_subset: bool) -> Dict[str, np.ndarray]:
    filtered = {key: np.array(value, copy=True) for key, value in frame.items()}

    bodies = filtered.get("bodies", None)
    body_scores = filtered.get("body_scores", None)
    if bodies is not None:
        bodies = bodies.copy()
        min_valid = 1e-6
        coord_mask = (bodies[:, 0] > min_valid) & (bodies[:, 1] > min_valid)

        conf_mask = None
        if body_scores is not None:
            scores = np.array(body_scores, copy=False)
            score_vec = scores.reshape(-1) if scores.ndim == 2 else scores
            score_vec = score_vec.astype(float)
            conf_mask = score_vec < conf_threshold
            if conf_mask.shape[0] < bodies.shape[0]:
                conf_mask = np.pad(conf_mask, (0, bodies.shape[0] - conf_mask.shape[0]), constant_values=False)
            elif conf_mask.shape[0] > bodies.shape[0]:
                conf_mask = conf_mask[: bodies.shape[0]]
        valid_mask = coord_mask if conf_mask is None else (coord_mask & (~conf_mask))
        bodies[~valid_mask, :] = 0
        filtered["bodies"] = bodies

        if update_subset:
            if body_scores is not None:
                subset = np.array(body_scores, copy=True)
                if subset.ndim == 1:
                    subset = subset.reshape(1, -1)
            else:
                subset = np.arange(bodies.shape[0], dtype=float).reshape(1, -1)
            if subset.shape[1] < bodies.shape[0]:
                subset = np.pad(subset, ((0, 0), (0, bodies.shape[0] - subset.shape[1])), constant_values=-1)
            elif subset.shape[1] > bodies.shape[0]:
                subset = subset[:, : bodies.shape[0]]
            subset[:, ~valid_mask] = -1
            filtered["body_scores"] = subset

    hands = filtered.get("hands", None)
    hand_scores = filtered.get("hands_scores", None)
    if hands is not None and hand_scores is not None:
        scores = np.array(hand_scores)
        hands = hands.copy()
        if hands.ndim == 3 and scores.ndim == 2:
            for hand_index in range(hands.shape[0]):
                mask = (scores[hand_index] < conf_threshold) | (scores[hand_index] <= 0)
                hands[hand_index][mask, :] = 0
        filtered["hands"] = hands

    faces = filtered.get("faces", None)
    face_scores = filtered.get("faces_scores", None)
    if faces is not None and face_scores is not None:
        scores = np.array(face_scores)
        faces = faces.copy()
        if faces.ndim == 3 and scores.ndim == 2:
            for face_index in range(faces.shape[0]):
                mask = (scores[face_index] < conf_threshold) | (scores[face_index] <= 0)
                faces[face_index][mask, :] = 0
        filtered["faces"] = faces

    return filtered


def draw_openpose_body(canvas: np.ndarray, candidate: np.ndarray, subset: np.ndarray, score: np.ndarray, conf_threshold: float) -> np.ndarray:
    height, width, _ = canvas.shape
    limb_seq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11],
        [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18],
    ]
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],
    ]

    for limb_index in range(17):
        for person_index in range(len(subset)):
            index = subset[person_index][np.array(limb_seq[limb_index]) - 1]
            if -1 in index:
                continue
            confidence = score[person_index][np.array(limb_seq[limb_index]) - 1]
            if confidence[0] < conf_threshold or confidence[1] < conf_threshold:
                continue
            coords = candidate[index.astype(int)]
            if np.any(coords <= EPS):
                continue
            y_coords = coords[:, 0] * float(width)
            x_coords = coords[:, 1] * float(height)
            mean_x = np.mean(x_coords)
            mean_y = np.mean(y_coords)
            length = ((x_coords[0] - x_coords[1]) ** 2 + (y_coords[0] - y_coords[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(x_coords[0] - x_coords[1], y_coords[0] - y_coords[1]))
            polygon = cv2.ellipse2Poly((int(mean_y), int(mean_x)), (int(length / 2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[limb_index])

    canvas = (canvas * 0.6).astype(np.uint8)
    for keypoint_index in range(18):
        for person_index in range(len(subset)):
            index = int(subset[person_index][keypoint_index])
            if index == -1 or score[person_index][keypoint_index] < conf_threshold:
                continue
            x_value, y_value = candidate[index][0:2]
            cv2.circle(canvas, (int(x_value * width), int(y_value * height)), 4, colors[keypoint_index], thickness=-1)
    return canvas


def draw_openpose_hands(canvas: np.ndarray, hand_peaks: np.ndarray, hand_scores: np.ndarray, conf_threshold: float) -> np.ndarray:
    height, width, _ = canvas.shape
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
        [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
    ]
    for hand_index, peaks in enumerate(hand_peaks):
        scores = hand_scores[hand_index] if len(hand_scores) > hand_index else None
        for edge_index, edge in enumerate(edges):
            x1, y1 = peaks[edge[0]]
            x2, y2 = peaks[edge[1]]
            if scores is not None and (scores[edge[0]] < conf_threshold or scores[edge[1]] < conf_threshold):
                continue
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            if x1 > EPS and y1 > EPS and x2 > EPS and y2 > EPS:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    matplotlib.colors.hsv_to_rgb([edge_index / float(len(edges)), 1.0, 1.0]) * 255,
                    thickness=2,
                )
        for point_index, point in enumerate(peaks):
            if scores is not None and scores[point_index] < conf_threshold:
                continue
            x_value = int(point[0] * width)
            y_value = int(point[1] * height)
            if x_value > EPS and y_value > EPS:
                cv2.circle(canvas, (x_value, y_value), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_openpose_faces(canvas: np.ndarray, face_points: np.ndarray, face_scores: np.ndarray, conf_threshold: float) -> np.ndarray:
    height, width, _ = canvas.shape
    for face_index, points in enumerate(face_points):
        scores = face_scores[face_index] if len(face_scores) > face_index else None
        for point_index, point in enumerate(points):
            if scores is not None and scores[point_index] < conf_threshold:
                continue
            x_value = int(point[0] * width)
            y_value = int(point[1] * height)
            if x_value > EPS and y_value > EPS:
                cv2.circle(canvas, (x_value, y_value), 3, (255, 255, 255), thickness=-1)
    return canvas


def draw_openpose_frame(frame: Dict[str, np.ndarray], width: int, height: int, conf_threshold: float) -> Image.Image:
    draw_func = get_stablesigner_openpose_draw()
    if draw_func is not None:
        canvas = draw_func(
            pose=frame,
            height=height,
            width=width,
            include_face=True,
            include_hands=True,
            conf_threshold=conf_threshold,
        )
        return Image.fromarray(canvas, "RGB")

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    bodies = frame["bodies"]
    subset = frame.get("body_scores", np.zeros((1, 18), dtype=np.float32))
    if subset.ndim == 1:
        subset = subset.reshape(1, -1)
    canvas = draw_openpose_body(canvas, bodies, subset, subset, conf_threshold)
    if len(frame.get("faces", [])) > 0:
        canvas = draw_openpose_faces(canvas, frame["faces"], frame.get("faces_scores", np.zeros((0, 68))), conf_threshold)
    if len(frame.get("hands", [])) > 0:
        canvas = draw_openpose_hands(canvas, frame["hands"], frame.get("hands_scores", np.zeros((0, 21))), conf_threshold)
    return Image.fromarray(canvas, "RGB")


def render_pose_image(frame: Dict[str, object], draw_style: str, transparent: bool, conf_threshold: float) -> Image.Image:
    width = int(frame["frame_width"])
    height = int(frame["frame_height"])
    if draw_style == "openpose":
        openpose_frame = filter_pose_for_openpose(
            to_openpose_frame(frame),
            conf_threshold=conf_threshold,
            update_subset=True,
        )
        image = draw_openpose_frame(openpose_frame, width, height, conf_threshold)
        if not transparent:
            return image
        rgba = image.convert("RGBA")
        alpha = np.where(np.array(image).sum(axis=2) > 0, 255, 0).astype(np.uint8)
        rgba.putalpha(Image.fromarray(alpha, "L"))
        return rgba

    rendered = draw_pose(
        frame,
        H=height,
        W=width,
        include_body=True,
        include_hand=True,
        include_face=True,
        transparent=transparent,
    )
    rendered = np.transpose(rendered, (1, 2, 0))
    if rendered.dtype != np.uint8:
        rendered = np.clip(rendered * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rendered, "RGBA" if transparent else "RGB")


def save_frame_previews(npz_paths: Iterable[Path], single_frame_dir: Path, draw_style: str, conf_threshold: float) -> None:
    single_frame_dir.mkdir(parents=True, exist_ok=True)
    for preview_index, npz_path in enumerate(npz_paths, start=1):
        frame = load_npz_frame(npz_path, aggregated_index=preview_index - 1 if npz_path.name == "poses.npz" else 0)
        image = render_pose_image(frame, draw_style=draw_style, transparent=False, conf_threshold=conf_threshold)
        image.save(single_frame_dir / f"{npz_path.stem}.png")


def render_pose_frames(npz_paths: List[Path], pose_frame_dir: Path, draw_style: str, conf_threshold: float) -> None:
    pose_frame_dir.mkdir(parents=True, exist_ok=True)
    total = len(npz_paths)
    for index, npz_path in enumerate(npz_paths, start=1):
        frame = load_npz_frame(npz_path, aggregated_index=index - 1 if npz_path.name == "poses.npz" else 0)
        image = render_pose_image(frame, draw_style=draw_style, transparent=False, conf_threshold=conf_threshold)
        image.save(pose_frame_dir / f"{npz_path.stem}.png")
        if index == 1 or index % 100 == 0 or index == total:
            print(f"Rendered pose frame {index}/{total}: {npz_path.name}")


def create_video_from_frames(frame_dir: Path, output_path: Path, fps: int) -> None:
    if not any(frame_dir.glob("*.png")):
        return
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "%08d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def resolve_raw_video(video_dir: Path, raw_video: Path | None) -> Path | None:
    if raw_video is not None and raw_video.exists():
        return raw_video
    video_id = video_dir.name
    raw_root = REPO_ROOT / "raw_video"
    for extension in VIDEO_EXTENSIONS:
        candidate = raw_root / f"{video_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def extract_video_frames(raw_video: Path, fps: int, temp_dir: Path) -> List[Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_video),
        "-vf",
        f"fps={fps}",
        str(temp_dir / "%08d.png"),
    ]
    subprocess.run(command, check=True)
    return sorted(temp_dir.glob("*.png"))


def render_overlay_frames(
    npz_paths: List[Path],
    raw_frame_paths: List[Path],
    overlay_dir: Path,
    draw_style: str,
    conf_threshold: float,
) -> None:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    frame_count = min(len(npz_paths), len(raw_frame_paths))
    for index, (npz_path, raw_frame_path) in enumerate(zip(npz_paths[:frame_count], raw_frame_paths[:frame_count]), start=1):
        frame = load_npz_frame(npz_path, aggregated_index=index - 1 if npz_path.name == "poses.npz" else 0)
        pose_rgba = render_pose_image(frame, draw_style=draw_style, transparent=True, conf_threshold=conf_threshold)
        with Image.open(raw_frame_path) as raw_image:
            base = raw_image.convert("RGBA")
        overlay = Image.alpha_composite(base, pose_rgba)
        overlay.save(overlay_dir / f"{npz_path.stem}.png")
        if index == 1 or index % 100 == 0 or index == frame_count:
            print(f"Rendered overlay frame {index}/{frame_count}: {npz_path.name}")


def main() -> None:
    args = parse_args()
    args.draw_style = normalize_draw_style(args.draw_style)
    video_dir = args.video_dir.resolve()
    npz_dir = (args.npz_dir or (video_dir / "npz")).resolve()
    output_dir = (args.output_dir or (video_dir / f"visualization_{args.draw_style}")).resolve()
    pose_frame_dir = output_dir / "pose_frames"
    single_frame_dir = output_dir / "single_frames"
    overlay_frame_dir = output_dir / "overlay_frames"
    pose_video_path = output_dir / f"visualization_{args.draw_style}.mp4"
    overlay_video_path = output_dir / f"visualization_{args.draw_style}_overlay.mp4"

    if not npz_dir.exists():
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")

    poses_npz_path = npz_dir / "poses.npz"
    if poses_npz_path.exists():
        npz_paths = [poses_npz_path]
    else:
        npz_paths = sorted(npz_dir.glob("*.npz"))
    if args.max_frames is not None:
        npz_paths = npz_paths[: args.max_frames]
    if not npz_paths:
        raise FileNotFoundError(f"No NPZ files found in {npz_dir}")

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_indices = parse_frame_indices(args.frame_indices)
    preview_paths = [
        npz_paths[index - 1]
        for index in preview_indices
        if 0 < index <= len(npz_paths)
    ]
    save_frame_previews(preview_paths, single_frame_dir, args.draw_style, args.conf_threshold)

    render_pose_frames(npz_paths, pose_frame_dir, args.draw_style, args.conf_threshold)
    create_video_from_frames(pose_frame_dir, pose_video_path, args.fps)

    raw_video = resolve_raw_video(video_dir, args.raw_video)
    if raw_video is None:
        print("No raw video found for overlay rendering. Pose-only outputs were created.")
        return

    temp_root = Path(tempfile.mkdtemp(prefix="sign_dwpose_overlay_"))
    try:
        raw_frame_paths = extract_video_frames(raw_video, args.fps, temp_root)
        render_overlay_frames(npz_paths, raw_frame_paths, overlay_frame_dir, args.draw_style, args.conf_threshold)
        create_video_from_frames(overlay_frame_dir, overlay_video_path, args.fps)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
