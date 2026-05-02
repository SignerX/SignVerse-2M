from __future__ import annotations

import contextlib
import fcntl
from pathlib import Path
from typing import Iterable, Iterator, Sequence

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov"}


def existing_raw_dirs(*dirs: Path | None) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        if directory is None:
            continue
        key = str(directory)
        if key in seen:
            continue
        seen.add(key)
        result.append(directory)
    return result


def collect_raw_videos(*dirs: Path | None) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for directory in existing_raw_dirs(*dirs):
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            videos.setdefault(path.stem, path)
    return videos


def count_raw_videos(*dirs: Path | None) -> int:
    return len(collect_raw_videos(*dirs))


def sum_raw_video_sizes(*dirs: Path | None) -> int:
    return sum(path.stat().st_size for path in collect_raw_videos(*dirs).values() if path.exists())


def iter_raw_video_files(*dirs: Path | None) -> Iterator[Path]:
    for path in collect_raw_videos(*dirs).values():
        yield path


def find_video_file(video_id: str, *dirs: Path | None) -> Path | None:
    for directory in existing_raw_dirs(*dirs):
        if not directory.exists():
            continue
        candidates = []
        for path in directory.glob(f"{video_id}.*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                candidates.append(path)
        if candidates:
            return sorted(candidates)[0]
    return None


def iter_partial_download_files(video_id: str, *dirs: Path | None) -> Iterator[Path]:
    seen: set[Path] = set()
    for directory in existing_raw_dirs(*dirs):
        if not directory.exists():
            continue
        for path in directory.glob(f"{video_id}*"):
            if not path.is_file():
                continue
            suffixes = set(path.suffixes)
            if '.part' in suffixes or '.ytdl' in suffixes or path.suffix in {'.part', '.ytdl'}:
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield path


def cleanup_partial_downloads(video_id: str, *dirs: Path | None) -> None:
    for partial_path in iter_partial_download_files(video_id, *dirs):
        partial_path.unlink(missing_ok=True)


def _count_reservations(reservation_dir: Path | None, pool_name: str) -> int:
    if reservation_dir is None:
        return 0
    pool_dir = reservation_dir / pool_name
    if not pool_dir.exists():
        return 0
    return sum(1 for path in pool_dir.iterdir() if path.is_file() and path.suffix == ".reserve")


def _create_reservation(reservation_dir: Path | None, pool_name: str, reservation_key: str | None) -> Path | None:
    if reservation_dir is None or not reservation_key:
        return None
    pool_dir = reservation_dir / pool_name
    pool_dir.mkdir(parents=True, exist_ok=True)
    reservation_path = pool_dir / f"{reservation_key}.reserve"
    reservation_path.write_text(f"pool={pool_name}\nkey={reservation_key}\n", encoding="utf-8")
    return reservation_path


def release_download_reservation(reservation_path: Path | None) -> None:
    if reservation_path is not None:
        reservation_path.unlink(missing_ok=True)


def choose_download_target(
    primary_dir: Path,
    scratch_dir: Path | None,
    primary_limit: int,
    scratch_limit: int,
    reservation_dir: Path | None = None,
    reservation_key: str | None = None,
) -> tuple[Path, Path | None]:
    primary_dir.mkdir(parents=True, exist_ok=True)
    if reservation_dir is None:
        primary_count = count_raw_videos(primary_dir)
        if primary_count < primary_limit:
            return primary_dir, None
        if scratch_dir is None:
            raise RuntimeError(
                f"raw backlog full in primary pool ({primary_count}/{primary_limit}) and no scratch raw pool configured"
            )
        scratch_dir.mkdir(parents=True, exist_ok=True)
        scratch_count = count_raw_videos(scratch_dir)
        if scratch_count < scratch_limit:
            return scratch_dir, None
        raise RuntimeError(
            f"raw backlog full in both pools: primary {primary_count}/{primary_limit}, scratch {scratch_count}/{scratch_limit}"
        )

    reservation_dir.mkdir(parents=True, exist_ok=True)
    lock_path = reservation_dir / ".target_selection.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            primary_count = count_raw_videos(primary_dir) + _count_reservations(reservation_dir, "home")
            if primary_count < primary_limit:
                return primary_dir, _create_reservation(reservation_dir, "home", reservation_key)
            if scratch_dir is None:
                raise RuntimeError(
                    f"raw backlog full in primary pool ({primary_count}/{primary_limit}) and no scratch raw pool configured"
                )
            scratch_dir.mkdir(parents=True, exist_ok=True)
            scratch_count = count_raw_videos(scratch_dir) + _count_reservations(reservation_dir, "scratch")
            if scratch_count < scratch_limit:
                return scratch_dir, _create_reservation(reservation_dir, "scratch", reservation_key)
            raise RuntimeError(
                f"raw backlog full in both pools: primary {primary_count}/{primary_limit}, scratch {scratch_count}/{scratch_limit}"
            )
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def remove_video_files(video_id: str, *dirs: Path | None) -> None:
    for directory in existing_raw_dirs(*dirs):
        if not directory.exists():
            continue
        for path in directory.glob(f"{video_id}.*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                path.unlink(missing_ok=True)
