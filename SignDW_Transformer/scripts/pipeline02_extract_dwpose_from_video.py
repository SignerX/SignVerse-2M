#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision.ops as tv_ops
from easy_dwpose import DWposeDetector
from easy_dwpose.body_estimation import resize_image
from easy_dwpose.body_estimation.detector import inference_detector, preprocess as detector_preprocess, demo_postprocess as detector_demo_postprocess
from easy_dwpose.body_estimation.pose import (
    postprocess as pose_postprocess,
    preprocess as pose_preprocess,
)
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.stats_npz import load_stats, update_video_stats_best_effort
from utils.raw_video_pool import iter_raw_video_files
from utils.dataset_pool import dataset_dir_for_video, find_dataset_video_dir


DEFAULT_RAW_VIDEO_DIR = REPO_ROOT / "raw_video"
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_STATS_NPZ = REPO_ROOT / "stats.npz"
DEFAULT_STATUS_JOURNAL_PATH = REPO_ROOT / "upload_status_journal.jsonl"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov"}
COMPLETE_MARKER_NAME = ".complete"


def build_optimized_providers(device: str, optimized_provider: str, cache_dir: Path):
    device = str(device)
    gpu_id = 0
    if ":" in device:
        gpu_id = int(device.split(":", 1)[1])
    cache_dir.mkdir(parents=True, exist_ok=True)
    if optimized_provider == "tensorrt":
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [
            {
                "device_id": str(gpu_id),
                "trt_engine_cache_enable": "1",
                "trt_engine_cache_path": str(cache_dir),
                "trt_timing_cache_enable": "1",
                "trt_fp16_enable": "1",
            },
            {"device_id": str(gpu_id)},
            {},
        ]
    elif optimized_provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [
            {"device_id": str(gpu_id)},
            {},
        ]
    else:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    return providers, provider_options


def create_detector(device: str, optimized_mode: bool, optimized_provider: str, tmp_root: Path) -> DWposeDetector:
    detector = DWposeDetector(device=device)
    if not optimized_mode:
        return detector
    providers, provider_options = build_optimized_providers(device, optimized_provider, tmp_root / "ort_trt_cache")
    detector.pose_estimation.session_det = ort.InferenceSession(
        "checkpoints/yolox_l.onnx",
        providers=providers,
        provider_options=provider_options,
    )
    detector.pose_estimation.session_pose = ort.InferenceSession(
        "checkpoints/dw-ll_ucoco_384.onnx",
        providers=providers,
        provider_options=provider_options,
    )
    return detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DWpose NPZ files from raw videos."
    )
    parser.add_argument("--raw-video-dir", type=Path, default=DEFAULT_RAW_VIDEO_DIR)
    parser.add_argument("--scratch-raw-video-dir", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--scratch-dataset-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--delete-source-on-success", action="store_true")
    parser.add_argument("--tmp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--stats-npz", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--status-journal-path", type=Path, default=DEFAULT_STATUS_JOURNAL_PATH)
    parser.add_argument(
        "--single-poses-npz",
        dest="single_poses_npz",
        action="store_true",
        default=True,
        help="Save one aggregated poses.npz per video (default).",
    )
    parser.add_argument(
        "--per-frame-npz",
        dest="single_poses_npz",
        action="store_false",
        help="Save one NPZ file per frame under the npz directory.",
    )
    parser.add_argument(
        "--stream-frames",
        dest="stream_frames",
        action="store_true",
        default=True,
        help="Decode frames directly from ffmpeg stdout without JPG spill (default).",
    )
    parser.add_argument(
        "--spill-jpg-frames",
        dest="stream_frames",
        action="store_false",
        help="Use legacy ffmpeg-to-JPG spill path for comparison/debugging.",
    )
    parser.add_argument(
        "--optimized-mode",
        dest="optimized_mode",
        action="store_true",
        default=True,
        help="Enable optimized ndarray + batched pose inference path (default).",
    )
    parser.add_argument(
        "--legacy-mode",
        dest="optimized_mode",
        action="store_false",
        help="Disable optimized path and use legacy per-frame single-image inference.",
    )
    parser.add_argument(
        "--optimized-frame-batch-size",
        type=int,
        default=8,
        help="Frame micro-batch size for optimized pose inference.",
    )
    parser.add_argument(
        "--optimized-detect-resolution",
        type=int,
        default=512,
        help="Detect resolution used only in optimized mode.",
    )
    parser.add_argument(
        "--optimized-frame-stride",
        type=int,
        default=1,
        help="Process every Nth decoded frame in optimized mode.",
    )
    parser.add_argument(
        "--optimized-provider",
        choices=("tensorrt", "cuda", "cpu"),
        default="cuda",
        help="Execution provider used only in optimized mode.",
    )
    parser.add_argument(
        "--optimized-gpu-pose-preprocess",
        action="store_true",
        help="Experimental: move pose crop affine/normalize to GPU in optimized mode.",
    )
    parser.add_argument(
        "--optimized-gpu-detector-postprocess",
        action="store_true",
        help="Experimental: run detector postprocess and NMS on GPU in optimized mode.",
    )
    parser.add_argument(
        "--optimized-io-binding",
        action="store_true",
        help="Experimental: use ONNX Runtime IO binding in optimized mode.",
    )
    return parser.parse_args()


def select_video_paths(args: argparse.Namespace) -> List[Path]:
    args.raw_video_dir.mkdir(parents=True, exist_ok=True)
    if args.scratch_raw_video_dir is not None:
        args.scratch_raw_video_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    if args.scratch_dataset_dir is not None:
        args.scratch_dataset_dir.mkdir(parents=True, exist_ok=True)
    video_id_filter = set(args.video_ids or [])
    stats = load_stats(args.stats_npz)

    selected = []
    for path in sorted(iter_raw_video_files(args.raw_video_dir, args.scratch_raw_video_dir), key=lambda p: (p.stem, str(p))):
        video_id = path.stem
        if video_id_filter and video_id not in video_id_filter:
            continue
        dataset_root = find_dataset_video_dir(video_id, args.dataset_dir, args.scratch_dataset_dir)
        npz_dir = dataset_root / video_id / "npz"
        complete_marker = npz_dir / COMPLETE_MARKER_NAME
        if (
            not args.force
            and npz_dir.exists()
            and complete_marker.exists()
            and stats.get(video_id, {}).get("process_status") == "ok"
        ):
            continue
        selected.append(path)
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def extract_frames_to_jpg(video_path: Path, frame_dir: Path, fps: int) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        str(frame_dir / "%08d.jpg"),
    ]
    subprocess.run(command, check=True)


def probe_video_dimensions(video_path: Path) -> Tuple[int, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    dims = (proc.stdout or "").strip()
    if "x" not in dims:
        raise RuntimeError(f"Unable to parse ffprobe dimensions for {video_path.name}: {dims!r}")
    width_s, height_s = dims.split("x", 1)
    return int(width_s), int(height_s)


def iter_streamed_frames(video_path: Path, fps: int) -> Iterator[Tuple[int, np.ndarray, int, int]]:
    width, height = probe_video_dimensions(video_path)
    frame_bytes = width * height * 3
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        frame_index = 0
        while True:
            chunk = proc.stdout.read(frame_bytes)
            if not chunk:
                break
            if len(chunk) != frame_bytes:
                raise RuntimeError(
                    f"Short raw frame read for {video_path.name}: expected {frame_bytes} bytes, got {len(chunk)}"
                )
            frame_index += 1
            frame_array = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
            yield frame_index, frame_array, width, height
    finally:
        if proc.stdout:
            proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg raw frame stream failed for {video_path.name}: {stderr.strip()}")


def build_npz_payload(pose_data: Dict[str, np.ndarray], width: int, height: int) -> Dict[str, np.ndarray]:
    num_persons = int(pose_data["faces"].shape[0]) if "faces" in pose_data else 0
    payload: Dict[str, np.ndarray] = {
        "num_persons": np.asarray(num_persons, dtype=np.int32),
        "frame_width": np.asarray(width, dtype=np.int32),
        "frame_height": np.asarray(height, dtype=np.int32),
    }
    if num_persons == 0:
        return payload

    bodies = pose_data["bodies"].reshape(num_persons, 18, 2).astype(np.float32, copy=False)
    body_scores = pose_data["body_scores"].astype(np.float32, copy=False)
    faces = pose_data["faces"].astype(np.float32, copy=False)
    face_scores = pose_data["faces_scores"].astype(np.float32, copy=False)
    hands = pose_data["hands"].astype(np.float32, copy=False)
    hand_scores = pose_data["hands_scores"].astype(np.float32, copy=False)

    for person_idx in range(num_persons):
        prefix = f"person_{person_idx:03d}"
        payload[f"{prefix}_body_keypoints"] = bodies[person_idx]
        payload[f"{prefix}_body_scores"] = body_scores[person_idx]
        payload[f"{prefix}_face_keypoints"] = faces[person_idx]
        payload[f"{prefix}_face_scores"] = face_scores[person_idx]
        left_hand_idx = person_idx * 2
        right_hand_idx = left_hand_idx + 1
        if left_hand_idx < len(hands):
            payload[f"{prefix}_left_hand_keypoints"] = hands[left_hand_idx]
            payload[f"{prefix}_left_hand_scores"] = hand_scores[left_hand_idx]
        if right_hand_idx < len(hands):
            payload[f"{prefix}_right_hand_keypoints"] = hands[right_hand_idx]
            payload[f"{prefix}_right_hand_scores"] = hand_scores[right_hand_idx]
    return payload


def run_session_outputs(
    session: ort.InferenceSession,
    input_array: np.ndarray,
    use_io_binding: bool,
    device_id: int,
):
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    if not use_io_binding:
        return session.run(output_names, {input_name: input_array})
    io_binding = session.io_binding()
    io_binding.bind_cpu_input(input_name, input_array)
    for output_name in output_names:
        io_binding.bind_output(output_name, device_type="cuda", device_id=device_id)
    session.run_with_iobinding(io_binding)
    return io_binding.copy_outputs_to_cpu()


def inference_detector_gpu_postprocess(
    session: ort.InferenceSession,
    ori_img: np.ndarray,
    device: torch.device,
    use_io_binding: bool,
    device_id: int,
) -> np.ndarray:
    input_shape = (640, 640)
    img, ratio = detector_preprocess(ori_img, input_shape)
    outputs = run_session_outputs(session, img[None, :, :, :], use_io_binding, device_id)
    predictions = detector_demo_postprocess(outputs[0], input_shape)[0]
    pred = torch.from_numpy(np.ascontiguousarray(predictions)).to(device=device, dtype=torch.float32)
    boxes = pred[:, :4]
    score_obj = pred[:, 4]
    cls_scores = pred[:, 5:]
    if cls_scores.ndim == 1:
        cls_scores = cls_scores.unsqueeze(1)
    cls0 = score_obj * cls_scores[:, 0]
    mask = cls0 > 0.1
    if not torch.any(mask):
        return np.array([])
    boxes = boxes[mask]
    cls0 = cls0[mask]
    boxes_xyxy = torch.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    keep = tv_ops.nms(boxes_xyxy, cls0, 0.45)
    if keep.numel() == 0:
        return np.array([])
    final_boxes = boxes_xyxy[keep]
    final_scores = cls0[keep]
    final_boxes = final_boxes[final_scores > 0.3]
    if final_boxes.numel() == 0:
        return np.array([])
    return final_boxes.detach().cpu().numpy()


def optimized_detector_call(
    detector: DWposeDetector,
    frame: np.ndarray,
    detect_resolution: int,
    include_hands: bool = True,
    include_face: bool = True,
) -> Dict[str, np.ndarray]:
    del include_hands, include_face
    return optimized_process_frame_batch(detector, [(1, frame, 0, 0)], detect_resolution)[0][1]


def empty_pose_payload() -> Dict[str, np.ndarray]:
    empty_f = np.zeros((0,), dtype=np.float32)
    return {
        "bodies": empty_f.reshape(0, 2),
        "body_scores": empty_f.reshape(0, 18),
        "hands": empty_f.reshape(0, 21, 2),
        "hands_scores": empty_f.reshape(0, 21),
        "faces": empty_f.reshape(0, 68, 2),
        "faces_scores": empty_f.reshape(0, 68),
    }


def format_pose_output(
    detector: DWposeDetector,
    keypoints: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
) -> Dict[str, np.ndarray]:
    keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    neck[:, 2:4] = np.logical_and(
        keypoints_info[:, 5, 2:4] > 0.3,
        keypoints_info[:, 6, 2:4] > 0.3,
    ).astype(int)
    new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
    keypoints_info = new_keypoints_info

    keypoints = keypoints_info[..., :2]
    scores = keypoints_info[..., 2]
    return detector._format_pose(keypoints, scores, width, height)


def prepare_optimized_frame(
    detector: DWposeDetector,
    frame: np.ndarray,
    detect_resolution: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    image = frame
    if not isinstance(image, np.ndarray):
        image = np.asarray(image.convert("RGB"))
    image = resize_image(np.ascontiguousarray(image), target_resolution=detect_resolution)
    height, width = image.shape[:2]
    if getattr(detector, "_optimized_gpu_detector_postprocess", False):
        det_result = inference_detector_gpu_postprocess(
            detector.pose_estimation.session_det,
            image,
            detector._optimized_torch_device,
            getattr(detector, "_optimized_io_binding", False),
            getattr(detector, "_optimized_device_id", 0),
        )
    else:
        det_result = inference_detector(detector.pose_estimation.session_det, image)
    return image, det_result, width, height


def gpu_pose_preprocess(
    image: np.ndarray,
    out_bbox: np.ndarray,
    input_size: Tuple[int, int],
    device: torch.device,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    if len(out_bbox) == 0:
        return np.empty((0, 3, input_size[1], input_size[0]), dtype=np.float32), [], []
    img_t = torch.from_numpy(np.ascontiguousarray(image)).to(device=device, dtype=torch.float32)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)
    H, W = image.shape[:2]
    out_w, out_h = input_size
    boxes = np.asarray(out_bbox, dtype=np.float32)
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    centers = np.stack([(x0 + x1) * 0.5, (y0 + y1) * 0.5], axis=1).astype(np.float32)
    scales = np.stack([(x1 - x0) * 1.25, (y1 - y0) * 1.25], axis=1).astype(np.float32)
    aspect = out_w / out_h
    w = scales[:, 0:1]
    h = scales[:, 1:2]
    scales = np.where(w > h * aspect, np.concatenate([w, w / aspect], axis=1), np.concatenate([h * aspect, h], axis=1)).astype(np.float32)
    centers_t = torch.from_numpy(centers).to(device=device, dtype=torch.float32)
    scales_t = torch.from_numpy(scales).to(device=device, dtype=torch.float32)
    theta = torch.zeros((len(boxes), 2, 3), device=device, dtype=torch.float32)
    theta[:, 0, 0] = scales_t[:, 0] / max(W - 1, 1)
    theta[:, 1, 1] = scales_t[:, 1] / max(H - 1, 1)
    theta[:, 0, 2] = 2.0 * centers_t[:, 0] / max(W - 1, 1) - 1.0
    theta[:, 1, 2] = 2.0 * centers_t[:, 1] / max(H - 1, 1) - 1.0
    grid = F.affine_grid(theta, size=(len(boxes), 3, out_h, out_w), align_corners=True)
    crops = F.grid_sample(img_t.expand(len(boxes), -1, -1, -1), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    mean = torch.tensor([123.675, 116.28, 103.53], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    crops = (crops - mean) / std
    return crops.detach().cpu().numpy(), [c for c in centers], [s for s in scales]


def optimized_process_frame_batch(
    detector: DWposeDetector,
    frames: Sequence[Tuple[int, np.ndarray, int, int]],
    detect_resolution: int,
) -> List[Tuple[int, Dict[str, np.ndarray], int, int]]:
    session_pose = detector.pose_estimation.session_pose
    model_input = session_pose.get_inputs()[0].shape
    model_input_size = (model_input[3], model_input[2])

    prepared = []
    pose_inputs = []
    all_centers = []
    all_scales = []

    for frame_index, frame, width, height in frames:
        input_image, det_result, input_width, input_height = prepare_optimized_frame(detector, frame, detect_resolution)
        if len(det_result) == 0:
            prepared.append((frame_index, empty_pose_payload(), width, height, 0, input_width, input_height))
            continue
        torch_device = getattr(detector, "_optimized_torch_device", None)
        if torch_device is not None:
            batch_imgs, centers, scales = gpu_pose_preprocess(input_image, det_result, model_input_size, torch_device)
            count = int(batch_imgs.shape[0])
            prepared.append((frame_index, None, width, height, count, input_width, input_height))
            if count:
                pose_inputs.extend(list(batch_imgs))
                all_centers.extend(centers)
                all_scales.extend(scales)
            continue
        resized_imgs, centers, scales = pose_preprocess(input_image, det_result, model_input_size)
        count = len(resized_imgs)
        prepared.append((frame_index, None, width, height, count, input_width, input_height))
        pose_inputs.extend([img.transpose(2, 0, 1) for img in resized_imgs])
        all_centers.extend(centers)
        all_scales.extend(scales)

    if pose_inputs:
        batch = np.stack(pose_inputs, axis=0).astype(np.float32, copy=False)
        sess_input = {session_pose.get_inputs()[0].name: batch}
        sess_output = [out.name for out in session_pose.get_outputs()]
        simcc_x, simcc_y = session_pose.run(sess_output, sess_input)
        batched_outputs = [(simcc_x[i : i + 1], simcc_y[i : i + 1]) for i in range(batch.shape[0])]
        keypoints_all, scores_all = pose_postprocess(
            batched_outputs,
            model_input_size,
            all_centers,
            all_scales,
        )
    else:
        keypoints_all = scores_all = None

    results = []
    offset = 0
    for frame_index, pose_data, width, height, count, input_width, input_height in prepared:
        if count == 0:
            results.append((frame_index, pose_data, width, height))
            continue
        keypoints = keypoints_all[offset : offset + count]
        scores = scores_all[offset : offset + count]
        offset += count
        results.append(
            (
                frame_index,
                format_pose_output(detector, keypoints, scores, input_width, input_height),
                width,
                height,
            )
        )
    return results


def process_video(
    video_path: Path,
    dataset_dir: Path,
    scratch_dataset_dir: Path | None,
    raw_video_dir: Path,
    scratch_raw_video_dir: Path | None,
    fps: int,
    detector: DWposeDetector,
    tmp_root: Path,
    force: bool,
    single_poses_npz: bool,
    stream_frames: bool,
    optimized_mode: bool,
    optimized_frame_batch_size: int,
    optimized_detect_resolution: int,
    optimized_frame_stride: int,
) -> None:
    video_id = video_path.stem
    output_dataset_dir = dataset_dir_for_video(video_path, raw_video_dir, scratch_raw_video_dir, dataset_dir, scratch_dataset_dir)
    output_npz_dir = output_dataset_dir / video_id / "npz"
    complete_marker = output_npz_dir / COMPLETE_MARKER_NAME
    poses_npz_path = output_npz_dir / "poses.npz"
    if output_npz_dir.exists() and complete_marker.exists() and not force:
        print(f"Skip {video_id}: NPZ files already exist")
        return

    if output_npz_dir.exists() and (force or not complete_marker.exists()):
        shutil.rmtree(output_npz_dir)
    output_npz_dir.mkdir(parents=True, exist_ok=True)

    aggregated_payloads = []
    frame_widths = []
    frame_heights = []
    frame_indices = []
    total_frames = 0
    decode_start = time.perf_counter()
    process_start = decode_start

    if stream_frames:
        print(f"{video_id}: decoding mode=stream fps={fps} optimized={optimized_mode}")
        frame_batch = []
        batch_size = max(1, optimized_frame_batch_size)
        frame_stride = max(1, optimized_frame_stride)
        for frame_index, frame, width, height in iter_streamed_frames(video_path, fps):
            total_frames = frame_index
            if optimized_mode:
                if ((frame_index - 1) % frame_stride) != 0:
                    continue
                frame_batch.append((frame_index, frame, width, height))
                if len(frame_batch) < batch_size:
                    continue
                batch_results = optimized_process_frame_batch(detector, frame_batch, optimized_detect_resolution)
                frame_batch = []
                for result_index, pose_data, result_width, result_height in batch_results:
                    payload = build_npz_payload(pose_data, result_width, result_height)
                    if single_poses_npz:
                        aggregated_payloads.append(payload)
                        frame_widths.append(result_width)
                        frame_heights.append(result_height)
                        frame_indices.append(result_index)
                    else:
                        np.savez(output_npz_dir / f"{result_index:08d}.npz", **payload)
                    if result_index == 1 or result_index % 100 == 0:
                        print(f"{video_id}: processed {result_index} frames")
                continue
            pose_data = detector(frame, draw_pose=False, include_hands=True, include_face=True)
            payload = build_npz_payload(pose_data, width, height)
            if single_poses_npz:
                aggregated_payloads.append(payload)
                frame_widths.append(width)
                frame_heights.append(height)
            else:
                np.savez(output_npz_dir / f"{frame_index:08d}.npz", **payload)
            if frame_index == 1 or frame_index % 100 == 0:
                print(f"{video_id}: processed {frame_index} frames")
        if optimized_mode and frame_batch:
            for result_index, pose_data, result_width, result_height in optimized_process_frame_batch(detector, frame_batch, optimized_detect_resolution):
                payload = build_npz_payload(pose_data, result_width, result_height)
                if single_poses_npz:
                    aggregated_payloads.append(payload)
                    frame_widths.append(result_width)
                    frame_heights.append(result_height)
                    frame_indices.append(result_index)
                else:
                    np.savez(output_npz_dir / f"{result_index:08d}.npz", **payload)
                if result_index == 1 or result_index % 100 == 0:
                    print(f"{video_id}: processed {result_index} frames")
    else:
        print(f"{video_id}: decoding mode=jpg-spill fps={fps} optimized={optimized_mode}")
        tmp_root.mkdir(parents=True, exist_ok=True)
        frame_dir = Path(tempfile.mkdtemp(prefix=f"sign_dwpose_{video_id}_", dir=str(tmp_root)))
        try:
            extract_frames_to_jpg(video_path, frame_dir, fps)
            frame_paths = sorted(frame_dir.glob("*.jpg"))
            total_frames = len(frame_paths)
            print(f"{video_id}: extracted {total_frames} frames at {fps} fps")
            process_start = time.perf_counter()
            for frame_index, frame_path in enumerate(frame_paths, start=1):
                with Image.open(frame_path) as image:
                    frame = np.asarray(image.convert("RGB"))
                    height, width = frame.shape[:2]
                    if optimized_mode:
                        pose_data = optimized_detector_call(
                            detector,
                            frame,
                            optimized_detect_resolution,
                            include_hands=True,
                            include_face=True,
                        )
                    else:
                        pose_data = detector(frame, draw_pose=False, include_hands=True, include_face=True)
                payload = build_npz_payload(pose_data, width, height)
                if single_poses_npz:
                    aggregated_payloads.append(payload)
                    frame_widths.append(width)
                    frame_heights.append(height)
                    frame_indices.append(frame_index)
                    frame_indices.append(frame_index)
                else:
                    np.savez(output_npz_dir / f"{frame_index:08d}.npz", **payload)
                if frame_index == 1 or frame_index % 100 == 0 or frame_index == total_frames:
                    print(f"{video_id}: processed {frame_index}/{total_frames} frames")
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    decode_process_elapsed = time.perf_counter() - decode_start
    print(f"{video_id}: processed total_frames={total_frames} elapsed={decode_process_elapsed:.2f}s mode={'stream' if stream_frames else 'jpg-spill'} optimized={optimized_mode}")

    if single_poses_npz:
        np.savez(
            poses_npz_path,
            video_id=np.asarray(video_id),
            fps=np.asarray(fps, dtype=np.int32),
            total_frames=np.asarray(total_frames, dtype=np.int32),
            frame_widths=np.asarray(frame_widths, dtype=np.int32),
            frame_heights=np.asarray(frame_heights, dtype=np.int32),
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            frame_payloads=np.asarray(aggregated_payloads, dtype=object),
        )

    complete_marker.write_text(
        f"video_id={video_id}\nfps={fps}\nframes={total_frames}\noutput_mode={'single_poses_npy' if single_poses_npz else 'per_frame_npz'}\ndecode_mode={'stream' if stream_frames else 'jpg-spill'}\noptimized_mode={optimized_mode}\noptimized_detect_resolution={optimized_detect_resolution}\noptimized_frame_stride={optimized_frame_stride}\ncompleted_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )


def worker(rank: int, worker_count: int, video_paths: Sequence[Path], args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run DWpose on CPU")
    device_count = torch.cuda.device_count()
    if device_count <= rank:
        raise RuntimeError(
            f"CUDA device rank {rank} is unavailable; visible device_count={device_count}"
        )
    device = f"cuda:{rank}"
    detector = create_detector(
        device=device,
        optimized_mode=args.optimized_mode,
        optimized_provider=args.optimized_provider,
        tmp_root=args.tmp_root,
    )
    if args.optimized_mode:
        detector._optimized_device_id = int(device.split(":", 1)[1]) if ":" in device else 0
        detector._optimized_io_binding = bool(args.optimized_io_binding)
        detector._optimized_gpu_detector_postprocess = bool(args.optimized_gpu_detector_postprocess)
    if args.optimized_mode and (args.optimized_gpu_pose_preprocess or args.optimized_gpu_detector_postprocess):
        detector._optimized_torch_device = torch.device(device)
    print(f"Worker {rank}: device={device}, cuda_device_count={device_count}", flush=True)

    for index, video_path in enumerate(video_paths):
        if index % worker_count != rank:
            continue
        try:
            update_video_stats_best_effort(
                args.stats_npz,
                args.status_journal_path,
                video_path.stem,
                process_status="running",
                last_error="",
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            process_video(
                video_path=video_path,
                dataset_dir=args.dataset_dir,
                scratch_dataset_dir=args.scratch_dataset_dir,
                raw_video_dir=args.raw_video_dir,
                scratch_raw_video_dir=args.scratch_raw_video_dir,
                fps=args.fps,
                detector=detector,
                tmp_root=args.tmp_root,
                force=args.force,
                single_poses_npz=args.single_poses_npz,
                stream_frames=args.stream_frames,
                optimized_mode=args.optimized_mode,
                optimized_frame_batch_size=args.optimized_frame_batch_size,
                optimized_detect_resolution=args.optimized_detect_resolution,
                optimized_frame_stride=args.optimized_frame_stride,
            )
            update_video_stats_best_effort(
                args.stats_npz,
                args.status_journal_path,
                video_path.stem,
                process_status="ok",
                last_error="",
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            if args.delete_source_on_success and video_path.exists():
                video_path.unlink()
                print(f"Worker {rank}: deleted source video {video_path.name}")
        except Exception as exc:
            update_video_stats_best_effort(
                args.stats_npz,
                args.status_journal_path,
                video_path.stem,
                process_status="failed",
                last_error=str(exc),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            print(f"Worker {rank}: failed on {video_path.name}: {exc}")


def main() -> None:
    args = parse_args()
    video_paths = select_video_paths(args)
    if not video_paths:
        print("No videos need DWpose extraction.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run DWpose on CPU")

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count < 1:
        raise RuntimeError("No CUDA devices are visible to the DWpose worker")

    if args.workers is not None:
        worker_count = max(1, args.workers)
    else:
        worker_count = visible_gpu_count
        worker_count = max(1, worker_count)
    if worker_count > visible_gpu_count:
        raise RuntimeError(
            f"Requested workers={worker_count}, but only {visible_gpu_count} CUDA device(s) are visible"
        )
    worker_count = min(worker_count, len(video_paths))
    print(f"DWpose main: visible_cuda_devices={visible_gpu_count}, worker_count={worker_count}, stream_frames={args.stream_frames}, optimized_mode={args.optimized_mode}, optimized_frame_batch_size={args.optimized_frame_batch_size}, optimized_detect_resolution={args.optimized_detect_resolution}, optimized_frame_stride={args.optimized_frame_stride}, optimized_provider={args.optimized_provider}, optimized_gpu_pose_preprocess={args.optimized_gpu_pose_preprocess}, optimized_gpu_detector_postprocess={args.optimized_gpu_detector_postprocess}, optimized_io_binding={args.optimized_io_binding}", flush=True)

    if worker_count == 1:
        worker(0, 1, video_paths, args)
        return

    mp.set_start_method("spawn", force=True)
    processes = []
    for rank in range(worker_count):
        process = mp.Process(target=worker, args=(rank, worker_count, video_paths, args))
        process.start()
        processes.append(process)

    failed = False
    for process in processes:
        process.join()
        failed = failed or process.exitcode != 0
    if failed:
        raise SystemExit("One or more DWpose workers failed.")


if __name__ == "__main__":
    main()
