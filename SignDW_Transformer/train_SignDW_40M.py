#!/usr/bin/env python3
"""
Minimal text-to-DWPose training script for the local SignVerse-2M shard.

Default behavior:
  - reads videos under ./dataset/*/npz/poses.npz
  - uses caption cues from ./dataset/*/captions/*.vtt as text/pose segments
  - splits by video id: 80% train, 20% test
  - trains a compact Transformer that maps text tokens to a fixed-length
    ControlNext/DWPose-style pose sequence
  - writes predicted pose videos with utils.draw_dw_lib.draw_pose

This intentionally avoids the old torchtext/JoeyNMT-style training stack.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.draw_dw_lib import draw_pose, filter_pose_by_confidence


POSE_DIM = 384
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
RENDER_STYLES = {
    "bold": {"line_scale": 3.0, "point_scale": 2.0},
    "thin": {"line_scale": 1.0, "point_scale": 1.0},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_timestamp(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(value)


@dataclass(frozen=True)
class CaptionCue:
    start: float
    end: float
    text: str


def parse_vtt(path: Path) -> List[CaptionCue]:
    cues: List[CaptionCue] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" not in line:
            i += 1
            continue

        left, right = line.split("-->", 1)
        start = parse_timestamp(left)
        end = parse_timestamp(right.split()[0])
        i += 1
        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            text_line = re.sub(r"<[^>]+>", "", lines[i]).strip()
            if text_line and not text_line.startswith(("NOTE", "Kind:", "Language:")):
                text_lines.append(text_line)
            i += 1

        text = " ".join(text_lines).strip()
        text = re.sub(r"\s+", " ", text)
        if text and end > start:
            cues.append(CaptionCue(start=start, end=end, text=text))
        i += 1
    return cues


def tokenize(text: str) -> List[str]:
    # Multilingual-friendly enough for subtitles: keep letters, numbers, and apostrophes.
    tokens = re.findall(r"[\w']+|[^\w\s]", text.lower(), flags=re.UNICODE)
    return [tok for tok in tokens if tok.strip()]


class Vocabulary:
    def __init__(self, min_freq: int = 1, max_size: int = 30000) -> None:
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi: Dict[str, int] = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            BOS_TOKEN: 2,
            EOS_TOKEN: 3,
        }
        self.itos: List[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def build(self, texts: Iterable[str]) -> None:
        counts: Dict[str, int] = {}
        for text in texts:
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        for tok, freq in ordered:
            if freq < self.min_freq:
                continue
            if tok in self.stoi:
                continue
            if len(self.itos) >= self.max_size:
                break
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)

    def encode(self, text: str, max_len: int) -> Tuple[torch.Tensor, int]:
        ids = [self.stoi[BOS_TOKEN]]
        ids.extend(self.stoi.get(tok, self.stoi[UNK_TOKEN]) for tok in tokenize(text))
        ids.append(self.stoi[EOS_TOKEN])
        ids = ids[:max_len]
        length = len(ids)
        if length < max_len:
            ids.extend([self.stoi[PAD_TOKEN]] * (max_len - length))
        return torch.tensor(ids, dtype=torch.long), length

    def to_json(self) -> Dict[str, Any]:
        return {"itos": self.itos, "min_freq": self.min_freq, "max_size": self.max_size}

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "Vocabulary":
        vocab = cls(min_freq=int(payload.get("min_freq", 1)), max_size=int(payload.get("max_size", 30000)))
        vocab.itos = [str(x) for x in payload["itos"]]
        vocab.stoi = {tok: idx for idx, tok in enumerate(vocab.itos)}
        return vocab

    def __len__(self) -> int:
        return len(self.itos)


@dataclass(frozen=True)
class Segment:
    video_id: str
    npz_path: Path
    caption_path: Optional[Path]
    text: str
    start_frame: int
    end_frame: int
    fps: float
    sign_language: str = "unknown"


def load_sign_language_metadata(metadata_csv: Path) -> Dict[str, str]:
    if not metadata_csv.exists():
        print(f"Metadata CSV not found: {metadata_csv}. All videos will be treated as unknown.")
        return {}

    mapping: Dict[str, str] = {}
    with metadata_csv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            video_id = row[0].strip()
            sign_language = row[1].strip() or "unknown"
            if video_id:
                mapping[video_id] = sign_language
    return mapping


def summarize_sign_languages(segments: Sequence[Segment]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for seg in segments:
        code = seg.sign_language or "unknown"
        if code not in summary:
            summary[code] = {"videos": 0, "segments": 0}
        summary[code]["segments"] += 1
    by_language_videos: Dict[str, set] = {}
    for seg in segments:
        by_language_videos.setdefault(seg.sign_language or "unknown", set()).add(seg.video_id)
    for code, video_ids in by_language_videos.items():
        summary[code]["videos"] = len(video_ids)
    return summary


def choose_sign_language(segments: Sequence[Segment], requested: str) -> Optional[str]:
    summary = summarize_sign_languages(segments)
    if not summary:
        return None

    print("Available sign language groups:")
    for code, stats in sorted(summary.items(), key=lambda item: (-item[1]["segments"], item[0])):
        print(f"  {code}: {stats['segments']} segments, {stats['videos']} videos")

    if requested == "list":
        return None
    if requested == "auto":
        selected = max(summary.items(), key=lambda item: (item[1]["segments"], item[1]["videos"]))[0]
        print(f"No sign language explicitly selected. Using largest group: {selected}")
        return selected
    if requested not in summary:
        raise ValueError(f"Requested sign language '{requested}' not found. Available: {sorted(summary)}")
    print(f"Using requested sign language group: {requested}")
    return requested


def choose_caption_file(video_dir: Path) -> Optional[Path]:
    caption_dir = video_dir / "captions"
    if not caption_dir.exists():
        return None
    files = sorted(caption_dir.glob("*.vtt"))
    if not files:
        return None
    # Prefer English when present, otherwise use the first available subtitle.
    for path in files:
        if path.name.endswith(".en.vtt"):
            return path
    return files[0]


def load_npz_meta(npz_path: Path) -> Tuple[str, float, int]:
    with np.load(npz_path, allow_pickle=True) as data:
        video_id = str(data["video_id"].item()) if "video_id" in data else npz_path.parents[1].name
        fps = float(data["fps"].item()) if "fps" in data else 25.0
        total_frames = int(data["total_frames"].item()) if "total_frames" in data else len(data["frame_payloads"])
    return video_id, fps, total_frames


def build_segments(
    dataset_dir: Path,
    min_segment_frames: int,
    max_samples_per_video: int,
    sign_language_by_video: Optional[Dict[str, str]] = None,
) -> List[Segment]:
    sign_language_by_video = sign_language_by_video or {}
    segments: List[Segment] = []
    for npz_path in sorted(dataset_dir.glob("*/npz/poses.npz")):
        video_dir = npz_path.parents[1]
        video_id, fps, total_frames = load_npz_meta(npz_path)
        sign_language = sign_language_by_video.get(video_id, "unknown")
        caption_path = choose_caption_file(video_dir)
        cues = parse_vtt(caption_path) if caption_path is not None else []

        video_segments: List[Segment] = []
        for cue in cues:
            start_frame = max(0, int(math.floor(cue.start * fps)))
            end_frame = min(total_frames, int(math.ceil(cue.end * fps)))
            if end_frame - start_frame < min_segment_frames:
                continue
            video_segments.append(
                Segment(
                    video_id=video_id,
                    npz_path=npz_path,
                    caption_path=caption_path,
                    text=cue.text,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    fps=fps,
                    sign_language=sign_language,
                )
            )

        if not video_segments:
            video_segments.append(
                Segment(
                    video_id=video_id,
                    npz_path=npz_path,
                    caption_path=caption_path,
                    text=video_id,
                    start_frame=0,
                    end_frame=min(total_frames, max(min_segment_frames, int(fps * 4))),
                    fps=fps,
                    sign_language=sign_language,
                )
            )

        if max_samples_per_video > 0 and len(video_segments) > max_samples_per_video:
            idx = np.linspace(0, len(video_segments) - 1, max_samples_per_video).round().astype(int).tolist()
            video_segments = [video_segments[i] for i in idx]
        segments.extend(video_segments)
    return segments


def split_by_video(
    segments: Sequence[Segment],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Segment], List[Segment], List[str], List[str]]:
    video_ids = sorted({seg.video_id for seg in segments})
    rng = random.Random(seed)
    rng.shuffle(video_ids)
    n_train = max(1, int(round(len(video_ids) * train_ratio)))
    n_train = min(n_train, max(1, len(video_ids) - 1)) if len(video_ids) > 1 else len(video_ids)
    train_ids = set(video_ids[:n_train])
    train = [seg for seg in segments if seg.video_id in train_ids]
    test = [seg for seg in segments if seg.video_id not in train_ids]
    if not test and train:
        test = train[-1:]
        train = train[:-1] or test
    return train, test, sorted(train_ids), sorted(set(video_ids) - train_ids)


def _payload_get(payload: Dict[str, Any], name: str, default: np.ndarray) -> np.ndarray:
    for prefix in ("person_000", "person_0", "person_00"):
        key = f"{prefix}_{name}"
        if key in payload:
            return np.asarray(payload[key], dtype=np.float32)
    return default.astype(np.float32)


def payload_scalar(payload: Dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        return float(np.asarray(value).item())
    except Exception:
        return float(default)


def center_square_transform(
    points: np.ndarray,
    scores: Optional[np.ndarray],
    frame_width: float,
    frame_height: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Map full-frame normalized keypoints into the centered square crop space."""
    points = np.asarray(points, dtype=np.float32).copy()
    if points.size == 0 or frame_width <= 0 or frame_height <= 0:
        return points, scores

    scores_out = None if scores is None else np.asarray(scores, dtype=np.float32).copy()
    flat = points.reshape(-1, 2)
    score_flat = None if scores_out is None else scores_out.reshape(-1)

    crop_size = min(frame_width, frame_height)
    offset_x = max(0.0, (frame_width - crop_size) / 2.0)
    offset_y = max(0.0, (frame_height - crop_size) / 2.0)

    px = flat[:, 0] * frame_width
    py = flat[:, 1] * frame_height
    valid = (flat[:, 0] > 1e-6) & (flat[:, 1] > 1e-6)
    valid &= (px >= offset_x) & (px <= offset_x + crop_size)
    valid &= (py >= offset_y) & (py <= offset_y + crop_size)

    flat[:, 0] = (px - offset_x) / crop_size
    flat[:, 1] = (py - offset_y) / crop_size
    flat[~valid, :] = 0.0
    flat[:, :] = np.clip(flat, 0.0, 1.0)
    if score_flat is not None:
        score_flat[~valid] = 0.0

    return points, scores_out


def flatten_payload(payload: Any, center_square_crop: bool = True) -> np.ndarray:
    """Convert one SignVerse DWPose frame payload to the StableSigner 384-vector order."""
    if not isinstance(payload, dict):
        return np.zeros((POSE_DIM,), dtype=np.float32)

    num_persons = int(np.asarray(payload.get("num_persons", 0)).item())
    if num_persons <= 0:
        return np.zeros((POSE_DIM,), dtype=np.float32)

    body = _payload_get(payload, "body_keypoints", np.zeros((18, 2), dtype=np.float32)).reshape(18, 2)
    body_scores = _payload_get(payload, "body_scores", np.zeros((18,), dtype=np.float32)).reshape(18)
    left_hand = _payload_get(payload, "left_hand_keypoints", np.zeros((21, 2), dtype=np.float32)).reshape(21, 2)
    left_scores = _payload_get(payload, "left_hand_scores", np.zeros((21,), dtype=np.float32)).reshape(21)
    right_hand = _payload_get(payload, "right_hand_keypoints", np.zeros((21, 2), dtype=np.float32)).reshape(21, 2)
    right_scores = _payload_get(payload, "right_hand_scores", np.zeros((21,), dtype=np.float32)).reshape(21)
    face = _payload_get(payload, "face_keypoints", np.zeros((68, 2), dtype=np.float32)).reshape(68, 2)
    face_scores = _payload_get(payload, "face_scores", np.zeros((68,), dtype=np.float32)).reshape(68)

    if center_square_crop:
        frame_width = payload_scalar(payload, "frame_width", 1.0)
        frame_height = payload_scalar(payload, "frame_height", 1.0)
        body, body_scores = center_square_transform(body, body_scores, frame_width, frame_height)
        left_hand, left_scores = center_square_transform(left_hand, left_scores, frame_width, frame_height)
        right_hand, right_scores = center_square_transform(right_hand, right_scores, frame_width, frame_height)
        face, face_scores = center_square_transform(face, face_scores, frame_width, frame_height)

    vec = np.concatenate(
        [
            body.reshape(-1),
            body_scores.reshape(-1),
            left_hand.reshape(-1),
            right_hand.reshape(-1),
            left_scores.reshape(-1),
            right_scores.reshape(-1),
            face.reshape(-1),
            face_scores.reshape(-1),
        ]
    ).astype(np.float32)
    if vec.shape[0] != POSE_DIM:
        raise ValueError(f"Unexpected pose vector dim {vec.shape[0]}; expected {POSE_DIM}")
    return vec


def vector_to_pose_dict(vec: np.ndarray) -> Dict[str, Any]:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.shape[0] < POSE_DIM:
        vec = np.pad(vec, (0, POSE_DIM - vec.shape[0]))
    vec = vec[:POSE_DIM]

    body = vec[0:36].reshape(18, 2)
    body_scores = vec[36:54].reshape(18)
    left_hand = vec[54:96].reshape(21, 2)
    right_hand = vec[96:138].reshape(21, 2)
    left_scores = vec[138:159].reshape(21)
    right_scores = vec[159:180].reshape(21)
    face = vec[180:316].reshape(68, 2)
    face_scores = vec[316:384].reshape(68)

    # Draw code expects normalized coordinates and confidence-like scores.
    body = np.nan_to_num(body, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    left_hand = np.nan_to_num(left_hand, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    right_hand = np.nan_to_num(right_hand, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    face = np.nan_to_num(face, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    body_scores = np.nan_to_num(body_scores, nan=0.0).clip(0.0, 1.0)
    left_scores = np.nan_to_num(left_scores, nan=0.0).clip(0.0, 1.0)
    right_scores = np.nan_to_num(right_scores, nan=0.0).clip(0.0, 1.0)
    face_scores = np.nan_to_num(face_scores, nan=0.0).clip(0.0, 1.0)

    return {
        "bodies": body,
        "body_scores": body_scores,
        "hands": np.stack([left_hand, right_hand], axis=0),
        "hands_scores": np.stack([left_scores, right_scores], axis=0),
        "faces": face.reshape(1, 68, 2),
        "faces_scores": face_scores.reshape(1, 68),
    }


def prepare_pose_for_render(
    vec: np.ndarray,
    conf_threshold: float = 0.6,
) -> Dict[str, Any]:
    pose = vector_to_pose_dict(vec)
    original_bodies = np.array(pose["bodies"], copy=True)
    filtered = filter_pose_by_confidence(pose, conf_threshold=conf_threshold)

    # Visualization-only torso fix from Sign-X notes: DWpose can keep hip
    # coordinates while suppressing hip scores, which removes neck-to-hip lines.
    for hip_idx in (8, 11):
        if hip_idx < original_bodies.shape[0]:
            x, y = original_bodies[hip_idx]
            if x > 1e-6 and y > 1e-6:
                filtered["bodies"][hip_idx] = original_bodies[hip_idx]
                filtered["body_scores"][0, hip_idx] = hip_idx
    return filtered


class SignDWSegmentDataset(Dataset):
    def __init__(
        self,
        segments: Sequence[Segment],
        vocab: Vocabulary,
        max_text_len: int,
        max_pose_frames: int,
        normalize: bool = False,
        pose_mean: Optional[np.ndarray] = None,
        pose_std: Optional[np.ndarray] = None,
        center_square_crop: bool = True,
    ) -> None:
        self.segments = list(segments)
        self.vocab = vocab
        self.max_text_len = max_text_len
        self.max_pose_frames = max_pose_frames
        self.normalize = normalize
        self.pose_mean = pose_mean.astype(np.float32) if pose_mean is not None else None
        self.pose_std = pose_std.astype(np.float32) if pose_std is not None else None
        self.center_square_crop = center_square_crop
        self._npz_cache: Dict[Path, Any] = {}

    def __len__(self) -> int:
        return len(self.segments)

    def _load_payloads(self, npz_path: Path) -> np.ndarray:
        cached = self._npz_cache.get(npz_path)
        if cached is not None:
            return cached
        data = np.load(npz_path, allow_pickle=True)
        payloads = data["frame_payloads"]
        # Keep np.load handle alive by caching it too.
        self._npz_cache[npz_path] = payloads
        return payloads

    def _load_pose(self, seg: Segment) -> Tuple[torch.Tensor, torch.Tensor, int]:
        payloads = self._load_payloads(seg.npz_path)
        end_frame = min(seg.end_frame, len(payloads))
        start_frame = min(max(seg.start_frame, 0), max(0, end_frame - 1))
        frame_indices = np.arange(start_frame, end_frame, dtype=np.int64)
        if len(frame_indices) == 0:
            frame_indices = np.array([start_frame], dtype=np.int64)

        if len(frame_indices) > self.max_pose_frames:
            selected = np.linspace(0, len(frame_indices) - 1, self.max_pose_frames).round().astype(np.int64)
            frame_indices = frame_indices[selected]

        frames = np.stack([flatten_payload(payloads[int(idx)], center_square_crop=self.center_square_crop) for idx in frame_indices], axis=0)
        true_len = int(frames.shape[0])
        mask = np.zeros((self.max_pose_frames,), dtype=np.float32)
        mask[:true_len] = 1.0

        if frames.shape[0] < self.max_pose_frames:
            pad = np.zeros((self.max_pose_frames - frames.shape[0], POSE_DIM), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)

        if self.normalize:
            if self.pose_mean is None or self.pose_std is None:
                raise ValueError("normalize=True requires pose_mean and pose_std")
            frames = (frames - self.pose_mean) / self.pose_std

        return torch.from_numpy(frames.astype(np.float32)), torch.from_numpy(mask), true_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seg = self.segments[idx]
        text_ids, text_len = self.vocab.encode(seg.text, self.max_text_len)
        poses, pose_mask, pose_len = self._load_pose(seg)
        return {
            "text_ids": text_ids,
            "text_len": torch.tensor(text_len, dtype=torch.long),
            "poses": poses,
            "pose_mask": pose_mask,
            "pose_len": torch.tensor(pose_len, dtype=torch.long),
            "length_target": torch.tensor(pose_len / self.max_pose_frames, dtype=torch.float32),
            "text": seg.text,
            "video_id": seg.video_id,
        }


def collate_batch(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "text_ids": torch.stack([x["text_ids"] for x in items], dim=0),
        "text_len": torch.stack([x["text_len"] for x in items], dim=0),
        "poses": torch.stack([x["poses"] for x in items], dim=0),
        "pose_mask": torch.stack([x["pose_mask"] for x in items], dim=0),
        "pose_len": torch.stack([x["pose_len"] for x in items], dim=0),
        "length_target": torch.stack([x["length_target"] for x in items], dim=0),
        "text": [x["text"] for x in items],
        "video_id": [x["video_id"] for x in items],
    }


def sinusoidal_positions(max_len: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TextToDWPoseTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_text_len: int,
        max_pose_frames: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_text_len = max_text_len
        self.max_pose_frames = max_pose_frames
        self.hidden_dim = hidden_dim

        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.register_buffer("text_pos", sinusoidal_positions(max_text_len, hidden_dim), persistent=False)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

        self.pose_queries = nn.Parameter(torch.randn(max_pose_frames, hidden_dim) * 0.02)
        self.register_buffer("pose_pos", sinusoidal_positions(max_pose_frames, hidden_dim), persistent=False)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pose_head = nn.Linear(hidden_dim, POSE_DIM)
        self.length_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, text_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_mask = text_ids.eq(0)
        x = self.token_embed(text_ids) * math.sqrt(self.hidden_dim)
        x = x + self.text_pos[: x.size(1)].unsqueeze(0)
        memory = self.encoder(x, src_key_padding_mask=pad_mask)

        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        length_pred = torch.sigmoid(self.length_head(pooled)).squeeze(-1)

        queries = self.pose_queries.unsqueeze(0).expand(text_ids.size(0), -1, -1)
        queries = queries + self.pose_pos[: self.max_pose_frames].unsqueeze(0)
        y = self.decoder(tgt=queries, memory=memory, memory_key_padding_mask=pad_mask)
        poses = self.pose_head(self.output_norm(y))
        return poses, length_pred


def masked_pose_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1)
    loss = F.smooth_l1_loss(pred * mask, target * mask, reduction="sum")
    denom = mask.sum().clamp_min(1.0) * pred.size(-1)
    return loss / denom


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_pose = 0.0
    total_len = 0.0
    total = 0
    for batch in loader:
        text_ids = batch["text_ids"].to(device)
        poses = batch["poses"].to(device)
        pose_mask = batch["pose_mask"].to(device)
        length_target = batch["length_target"].to(device)
        pred, length_pred = model(text_ids)
        pose_loss = masked_pose_loss(pred, poses, pose_mask)
        length_loss = F.mse_loss(length_pred, length_target)
        n = text_ids.size(0)
        total_pose += float(pose_loss.detach().cpu()) * n
        total_len += float(length_loss.detach().cpu()) * n
        total += n
    total = max(total, 1)
    return {"pose_loss": total_pose / total, "length_loss": total_len / total}


def denormalize_pose(poses: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if mean is None or std is None:
        return poses
    return poses * std.reshape(1, -1) + mean.reshape(1, -1)


def save_pose_video(
    pose_vectors: np.ndarray,
    output_path: Path,
    fps: float = 12.0,
    height: int = 650,
    width: int = 650,
    render_style: str = "bold",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [render_pose_frame(vec, height=height, width=width, render_style=render_style) for vec in pose_vectors]
    write_rgb_video(frames, output_path, fps=fps)


def get_render_style(render_style: str) -> Dict[str, float]:
    if render_style not in RENDER_STYLES:
        raise ValueError(f"Unknown render style '{render_style}'. Choose from: {sorted(RENDER_STYLES)}")
    return RENDER_STYLES[render_style]


def render_pose_frame(
    vec: np.ndarray,
    height: int = 650,
    width: int = 650,
    render_style: str = "bold",
) -> np.ndarray:
    pose = prepare_pose_for_render(vec, conf_threshold=0.6)
    style = get_render_style(render_style)
    img = draw_pose(
        pose=pose,
        H=height,
        W=width,
        include_body=True,
        include_hand=True,
        include_face=True,
        line_scale=style["line_scale"],
        point_scale=style["point_scale"],
    )
    return img.transpose(1, 2, 0).astype(np.uint8)


def write_rgb_video(frames: Sequence[np.ndarray], output_path: Path, fps: float) -> None:
    if not frames:
        raise ValueError(f"No frames to write for {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0]
    height, width = first.shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                proc.stdin.write(np.ascontiguousarray(frame[:, :, :3]).tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            return_code = proc.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed for {output_path}: {stderr[-1000:]}")
            return
        except Exception:
            proc.kill()
            raise

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height), True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_comparison_video(
    pred_vectors: np.ndarray,
    gt_vectors: np.ndarray,
    output_path: Path,
    fps: float = 12.0,
    height: int = 650,
    width: int = 650,
    render_style: str = "bold",
) -> None:
    n_frames = max(len(pred_vectors), len(gt_vectors))
    if n_frames == 0:
        raise ValueError(f"No frames to write for {output_path}")

    frames: List[np.ndarray] = []
    for idx in range(n_frames):
        pred_idx = min(idx, len(pred_vectors) - 1)
        gt_idx = min(idx, len(gt_vectors) - 1)
        pred_frame = render_pose_frame(pred_vectors[pred_idx], height=height, width=width, render_style=render_style)
        gt_frame = render_pose_frame(gt_vectors[gt_idx], height=height, width=width, render_style=render_style)
        cv2.putText(pred_frame, "Prediction", (24, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(gt_frame, "Ground Truth", (24, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        frames.append(np.concatenate([pred_frame, gt_frame], axis=1))

    write_rgb_video(frames, output_path, fps=fps)


def resample_pose_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if target_len <= 0:
        return sequence[:0]
    if len(sequence) == 0:
        return np.zeros((target_len, POSE_DIM), dtype=np.float32)
    if len(sequence) == target_len:
        return sequence
    indices = np.linspace(0, len(sequence) - 1, target_len).round().astype(np.int64)
    return sequence[indices]


def load_raw_segment_poses(
    seg: Segment,
    max_frames: int = 0,
    center_square_crop: bool = True,
) -> np.ndarray:
    with np.load(seg.npz_path, allow_pickle=True) as data:
        payloads = data["frame_payloads"]
        start = max(0, min(seg.start_frame, len(payloads) - 1))
        end = max(start + 1, min(seg.end_frame, len(payloads)))
        frame_indices = np.arange(start, end, dtype=np.int64)
        if max_frames > 0 and len(frame_indices) > max_frames:
            selected = np.linspace(0, len(frame_indices) - 1, max_frames).round().astype(np.int64)
            frame_indices = frame_indices[selected]
        return np.stack([flatten_payload(payloads[int(idx)], center_square_crop=center_square_crop) for idx in frame_indices], axis=0)


def plot_stats(stats: Sequence[Dict[str, float]], output_path: Path) -> None:
    if not stats:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not update stats plot: {exc}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_points = [(int(row["step"]), float(row["train_loss"])) for row in stats if "train_loss" in row]
    test_points = [(int(row["step"]), float(row["test_total"])) for row in stats if "test_total" in row]
    pose_points = [(int(row["step"]), float(row["test_pose"])) for row in stats if "test_pose" in row]

    plt.figure(figsize=(10, 5))
    if train_points:
        x, y = zip(*train_points)
        plt.plot(x, y, label="train loss", linewidth=1.8)
    if test_points:
        x, y = zip(*test_points)
        plt.plot(x, y, label="test total", linewidth=1.8)
    if pose_points:
        x, y = zip(*pose_points)
        plt.plot(x, y, label="test pose", linewidth=1.4, linestyle="--")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    tmp_path = output_path.with_name(output_path.name + ".tmp.png")
    plt.savefig(tmp_path, dpi=150)
    plt.close()
    os.replace(tmp_path, output_path)


@torch.no_grad()
def predict_text(
    model: TextToDWPoseTransformer,
    vocab: Vocabulary,
    text: str,
    max_text_len: int,
    max_pose_frames: int,
    device: torch.device,
    pose_mean: Optional[np.ndarray],
    pose_std: Optional[np.ndarray],
    min_frames: int = 8,
) -> np.ndarray:
    model.eval()
    ids, _ = vocab.encode(text, max_text_len)
    pred, length_pred = model(ids.unsqueeze(0).to(device))
    pred_np = pred[0].detach().cpu().numpy()
    pred_np = denormalize_pose(pred_np, pose_mean, pose_std)
    pred_len = int(round(float(length_pred[0].detach().cpu()) * max_pose_frames))
    pred_len = max(min_frames, min(max_pose_frames, pred_len))
    return pred_np[:pred_len]


def compute_pose_stats(
    segments: Sequence[Segment],
    max_pose_frames: int,
    sample_limit: int,
    center_square_crop: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(1234)
    sampled = list(segments)
    rng.shuffle(sampled)
    if sample_limit > 0:
        sampled = sampled[:sample_limit]

    sums = np.zeros((POSE_DIM,), dtype=np.float64)
    sq_sums = np.zeros((POSE_DIM,), dtype=np.float64)
    count = 0
    cache: Dict[Path, np.ndarray] = {}
    for seg in sampled:
        payloads = cache.get(seg.npz_path)
        if payloads is None:
            data = np.load(seg.npz_path, allow_pickle=True)
            payloads = data["frame_payloads"]
            cache[seg.npz_path] = payloads

        start = max(0, min(seg.start_frame, len(payloads) - 1))
        end = max(start + 1, min(seg.end_frame, len(payloads)))
        frame_indices = np.arange(start, end, dtype=np.int64)
        if len(frame_indices) > max_pose_frames:
            selected = np.linspace(0, len(frame_indices) - 1, max_pose_frames).round().astype(np.int64)
            frame_indices = frame_indices[selected]
        frames = np.stack([flatten_payload(payloads[int(idx)], center_square_crop=center_square_crop) for idx in frame_indices], axis=0)
        sums += frames.sum(axis=0)
        sq_sums += np.square(frames).sum(axis=0)
        count += frames.shape[0]

    if count == 0:
        return np.zeros((POSE_DIM,), dtype=np.float32), np.ones((POSE_DIM,), dtype=np.float32)
    mean = sums / count
    var = np.maximum(sq_sums / count - np.square(mean), 1e-6)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def save_checkpoint(
    path: Path,
    model: TextToDWPoseTransformer,
    vocab: Vocabulary,
    args: argparse.Namespace,
    step: int,
    best_loss: float,
    pose_mean: Optional[np.ndarray],
    pose_std: Optional[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "vocab": vocab.to_json(),
        "args": vars(args),
        "step": step,
        "best_loss": best_loss,
        "pose_mean": None if pose_mean is None else pose_mean.tolist(),
        "pose_std": None if pose_std is None else pose_std.tolist(),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Tuple[TextToDWPoseTransformer, Vocabulary, Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["args"]
    vocab = Vocabulary.from_json(ckpt["vocab"])
    model = TextToDWPoseTransformer(
        vocab_size=len(vocab),
        max_text_len=int(cfg["max_text_len"]),
        max_pose_frames=int(cfg["max_pose_frames"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_heads=int(cfg["num_heads"]),
        encoder_layers=int(cfg["encoder_layers"]),
        decoder_layers=int(cfg["decoder_layers"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    pose_mean = None if ckpt.get("pose_mean") is None else np.asarray(ckpt["pose_mean"], dtype=np.float32)
    pose_std = None if ckpt.get("pose_std") is None else np.asarray(ckpt["pose_std"], dtype=np.float32)
    return model, vocab, cfg, pose_mean, pose_std


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
        if device.type != "cuda":
            print("Training requires a CUDA GPU. Remove --device cpu or use --mode infer for CPU inference.", file=sys.stderr)
            sys.exit(2)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("Training requires a CUDA GPU, but torch.cuda.is_available() is False. Exiting.", file=sys.stderr)
        sys.exit(2)

    root = Path(args.root).resolve()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = root / dataset_dir
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        # Keep the run layout simple: log/<timestamp>/.
        if len(output_root.parts) > 1 and output_root.parts[0] == "log":
            print(f"Using {root / 'log'} as log root; ignoring intermediate output path: {output_root}")
            output_root = Path("log")
        output_base_dir = root / output_root
    else:
        output_base_dir = output_root

    metadata_csv = Path(args.metadata_csv)
    if not metadata_csv.is_absolute():
        metadata_csv = root / metadata_csv
    sign_language_by_video = load_sign_language_metadata(metadata_csv)

    segments = build_segments(
        dataset_dir=dataset_dir,
        min_segment_frames=args.min_segment_frames,
        max_samples_per_video=args.max_samples_per_video,
        sign_language_by_video=sign_language_by_video,
    )
    if not segments:
        raise RuntimeError(f"No training segments found under {dataset_dir}")

    selected_sign_language = choose_sign_language(segments, args.sign_language)
    if args.sign_language == "list":
        return
    if selected_sign_language is not None:
        segments = [seg for seg in segments if seg.sign_language == selected_sign_language]
        if not segments:
            raise RuntimeError(f"No segments left after filtering by sign language '{selected_sign_language}'")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {output_dir}")

    train_segments, test_segments, train_ids, test_ids = split_by_video(segments, args.train_ratio, args.seed)
    vocab = Vocabulary(min_freq=args.min_freq, max_size=args.vocab_size)
    vocab.build(seg.text for seg in train_segments)

    pose_mean: Optional[np.ndarray] = None
    pose_std: Optional[np.ndarray] = None
    if args.normalize_pose:
        print("Computing pose normalization statistics...")
        pose_mean, pose_std = compute_pose_stats(
            train_segments,
            args.max_pose_frames,
            args.stats_sample_limit,
            center_square_crop=args.center_square_crop,
        )

    manifest = {
        "dataset_dir": str(dataset_dir),
        "metadata_csv": str(metadata_csv),
        "selected_sign_language": selected_sign_language,
        "sign_language_summary": summarize_sign_languages(segments),
        "center_square_crop": args.center_square_crop,
        "num_segments": len(segments),
        "train_segments": len(train_segments),
        "test_segments": len(test_segments),
        "train_video_ids": train_ids,
        "test_video_ids": test_ids,
        "vocab_size": len(vocab),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    train_ds = SignDWSegmentDataset(
        train_segments,
        vocab=vocab,
        max_text_len=args.max_text_len,
        max_pose_frames=args.max_pose_frames,
        normalize=args.normalize_pose,
        pose_mean=pose_mean,
        pose_std=pose_std,
        center_square_crop=args.center_square_crop,
    )
    test_ds = SignDWSegmentDataset(
        test_segments,
        vocab=vocab,
        max_text_len=args.max_text_len,
        max_pose_frames=args.max_pose_frames,
        normalize=args.normalize_pose,
        pose_mean=pose_mean,
        pose_std=pose_std,
        center_square_crop=args.center_square_crop,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    model = TextToDWPoseTransformer(
        vocab_size=len(vocab),
        max_text_len=args.max_text_len,
        max_pose_frames=args.max_pose_frames,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.max_steps, 1), eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_loss = float("inf")
    model.train()
    running = 0.0
    running_pose = 0.0
    running_len = 0.0
    running_items = 0
    train_iter = itertools.cycle(train_loader)
    stats_history: List[Dict[str, float]] = []
    latest_step_ckpt: Optional[Path] = None

    for step in range(1, args.max_steps + 1):
        batch = next(train_iter)
        text_ids = batch["text_ids"].to(device, non_blocking=True)
        poses = batch["poses"].to(device, non_blocking=True)
        pose_mask = batch["pose_mask"].to(device, non_blocking=True)
        length_target = batch["length_target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            pred, length_pred = model(text_ids)
            pose_loss = masked_pose_loss(pred, poses, pose_mask)
            length_loss = F.mse_loss(length_pred, length_target)
            loss = pose_loss + args.length_loss_weight * length_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = text_ids.size(0)
        running += float(loss.detach().cpu()) * batch_size
        running_pose += float(pose_loss.detach().cpu()) * batch_size
        running_len += float(length_loss.detach().cpu()) * batch_size
        running_items += batch_size

        if step % args.log_every == 0:
            train_loss_avg = running / max(running_items, 1)
            train_pose_avg = running_pose / max(running_items, 1)
            train_len_avg = running_len / max(running_items, 1)
            print(
                f"step={step:06d}/{args.max_steps} "
                f"loss={train_loss_avg:.5f} "
                f"pose={train_pose_avg:.5f} "
                f"len={train_len_avg:.5f} "
                f"lr={optimizer.param_groups[0]['lr']:.6g}"
            )
            stats_history.append(
                {
                    "step": float(step),
                    "train_loss": float(train_loss_avg),
                    "train_pose": float(train_pose_avg),
                    "train_len": float(train_len_avg),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
            plot_stats(stats_history, output_dir / "stats.png")
            running = running_pose = running_len = 0.0
            running_items = 0

        should_eval = args.eval_every > 0 and (step % args.eval_every == 0 or step == args.max_steps)
        if should_eval:
            metrics = evaluate(model, test_loader, device)
            test_loss = metrics["pose_loss"] + args.length_loss_weight * metrics["length_loss"]
            print(
                f"step={step:06d} "
                f"test_pose={metrics['pose_loss']:.5f} test_len={metrics['length_loss']:.5f} "
                f"test_total={test_loss:.5f}"
            )
            stats_history.append(
                {
                    "step": float(step),
                    "test_pose": float(metrics["pose_loss"]),
                    "test_len": float(metrics["length_loss"]),
                    "test_total": float(test_loss),
                }
            )
            (output_dir / "stats.json").write_text(json.dumps(stats_history, indent=2), encoding="utf-8")
            plot_stats(stats_history, output_dir / "stats.png")
            model.train()

            latest_path = output_dir / f"step_{step:08d}.pth"
            save_checkpoint(latest_path, model, vocab, args, step, best_loss, pose_mean, pose_std)
            if latest_step_ckpt is not None and latest_step_ckpt != latest_path and latest_step_ckpt.exists():
                latest_step_ckpt.unlink()
            latest_step_ckpt = latest_path
            if test_loss < best_loss:
                best_loss = test_loss
                save_checkpoint(output_dir / "best.pth", model, vocab, args, step, best_loss, pose_mean, pose_std)
                print(f"Saved new best checkpoint: {output_dir / 'best.pth'}")

        should_render = args.render_every > 0 and (step % args.render_every == 0 or step == args.max_steps)
        if should_render:
            sample_seg = test_segments[0] if test_segments else train_segments[0]
            sample_text = sample_seg.text
            pred_poses = predict_text(
                model,
                vocab,
                sample_text,
                args.max_text_len,
                args.max_pose_frames,
                device,
                pose_mean,
                pose_std,
            )
            step_dir = output_dir / f"step_{step:08d}"
            prediction_path = step_dir / "prediction.mp4"
            comparison_path = step_dir / "comparison.mp4"
            gt_poses = load_raw_segment_poses(
                sample_seg,
                max_frames=args.max_render_gt_frames,
                center_square_crop=args.center_square_crop,
            )
            pred_for_comparison = resample_pose_sequence(pred_poses, len(gt_poses))
            save_pose_video(pred_poses, prediction_path, fps=args.render_fps, render_style=args.render_style)
            save_comparison_video(
                pred_for_comparison,
                gt_poses,
                comparison_path,
                fps=sample_seg.fps,
                render_style=args.render_style,
            )
            print(f"Rendered sample: {prediction_path}")
            print(f"Rendered comparison: {comparison_path}")
            model.train()


def infer(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = Path(args.checkpoint)
    model, vocab, cfg, pose_mean, pose_std = load_checkpoint(checkpoint, device)
    text = args.text.strip()
    if not text:
        raise ValueError("--text is required for --mode infer")
    pred_poses = predict_text(
        model,
        vocab,
        text,
        int(cfg["max_text_len"]),
        int(cfg["max_pose_frames"]),
        device,
        pose_mean,
        pose_std,
    )
    output_path = Path(args.output_video)
    if not output_path.is_absolute():
        output_path = Path(args.root).resolve() / output_path
    save_pose_video(pred_poses, output_path, fps=args.render_fps, render_style=args.render_style)
    print(f"Wrote prediction video: {output_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a compact text-to-DWPose Transformer on local SignVerse DWPose NPZ data.")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="log")
    parser.add_argument("--metadata-csv", default="utils/Sign-DWPose-2M-metadata_ori.csv")
    parser.add_argument("--sign-language", default="auto", help="Use 'auto' for the largest group, 'list' to only print groups, or a metadata code such as 'aed' or '???'.")
    parser.add_argument("--checkpoint", default="log/best.pth")
    parser.add_argument("--output-video", default="log/inference.mp4")
    parser.add_argument("--text", default="")

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--min-segment-frames", type=int, default=8)
    parser.add_argument("--max-samples-per-video", type=int, default=128)
    parser.add_argument("--max-text-len", type=int, default=96)
    parser.add_argument("--max-pose-frames", type=int, default=96)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=1)

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--length-loss-weight", type=float, default=0.1)
    parser.add_argument("--normalize-pose", action="store_true")
    parser.add_argument("--stats-sample-limit", type=int, default=512)
    parser.add_argument("--center-square-crop", dest="center_square_crop", action="store_true", default=True)
    parser.add_argument("--no-center-square-crop", dest="center_square_crop", action="store_false")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--render-every", type=int, default=500)
    parser.add_argument("--render-fps", type=float, default=12.0)
    parser.add_argument("--render-style", choices=["bold", "thin"], default="bold")
    parser.add_argument("--max-render-gt-frames", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
