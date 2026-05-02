from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple


COMPLETE_MARKER_NAME = ".complete"


def dataset_dir_for_video(
    video_path: Path,
    home_raw_dir: Path,
    scratch_raw_dir: Path | None,
    home_dataset_dir: Path,
    scratch_dataset_dir: Path | None,
) -> Path:
    if scratch_raw_dir is not None and scratch_dataset_dir is not None:
        try:
            if video_path.parent.resolve() == scratch_raw_dir.resolve():
                return scratch_dataset_dir
        except FileNotFoundError:
            pass
    return home_dataset_dir


def iter_dataset_video_dirs(home_dataset_dir: Path, scratch_dataset_dir: Path | None = None) -> Iterator[Tuple[str, Path]]:
    seen: Dict[str, Path] = {}
    for dataset_dir in [home_dataset_dir, scratch_dataset_dir]:
        if dataset_dir is None or not dataset_dir.exists():
            continue
        for folder_path in sorted(dataset_dir.iterdir()):
            if not folder_path.is_dir():
                continue
            seen.setdefault(folder_path.name, folder_path)
    for video_id, folder_path in sorted(seen.items()):
        yield video_id, folder_path


def complete_video_ids(home_dataset_dir: Path, scratch_dataset_dir: Path | None = None) -> set[str]:
    complete: set[str] = set()
    for video_id, folder_path in iter_dataset_video_dirs(home_dataset_dir, scratch_dataset_dir):
        if (folder_path / "npz" / COMPLETE_MARKER_NAME).exists():
            complete.add(video_id)
    return complete


def count_complete(home_dataset_dir: Path, scratch_dataset_dir: Path | None = None) -> int:
    return len(complete_video_ids(home_dataset_dir, scratch_dataset_dir))


def find_dataset_video_dir(video_id: str, home_dataset_dir: Path, scratch_dataset_dir: Path | None = None) -> Path:
    home_path = home_dataset_dir / video_id
    if home_path.exists():
        return home_path
    if scratch_dataset_dir is not None:
        scratch_path = scratch_dataset_dir / video_id
        if scratch_path.exists():
            return scratch_path
    return home_path


def list_unuploaded_folder_paths(
    home_dataset_dir: Path,
    scratch_dataset_dir: Path | None,
    uploaded_folders: Dict[str, object],
) -> List[Tuple[str, Path]]:
    folders: List[Tuple[str, Path]] = []
    for video_id, folder_path in iter_dataset_video_dirs(home_dataset_dir, scratch_dataset_dir):
        if video_id in uploaded_folders:
            continue
        if not (folder_path / "npz" / COMPLETE_MARKER_NAME).exists():
            continue
        folders.append((video_id, folder_path))
    return folders
