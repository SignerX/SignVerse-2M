import fcntl
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np


STATUS_FIELDS = [
    "sign_language",
    "title",
    "duration_sec",
    "start_sec",
    "end_sec",
    "subtitle_languages",
    "subtitle_dir_path",
    "subtitle_en_source",
    "raw_video_path",
    "raw_metadata_path",
    "metadata_status",
    "subtitle_status",
    "download_status",
    "process_status",
    "upload_status",
    "local_cleanup_status",
    "archive_name",
    "last_error",
    "updated_at",
]


def _lock_path(stats_path: Path) -> Path:
    return stats_path.with_suffix(stats_path.suffix + ".lock")


def _load_stats_unlocked(stats_path: Path) -> Dict[str, Dict[str, str]]:
    if not stats_path.exists() or stats_path.stat().st_size == 0:
        return {}

    stats: Dict[str, Dict[str, str]] = {}
    with np.load(stats_path, allow_pickle=True) as data:
        video_ids = [str(item) for item in data.get("video_ids", np.asarray([], dtype=object)).tolist()]
        for index, video_id in enumerate(video_ids):
            record = {}
            for field in STATUS_FIELDS:
                values = data.get(field)
                record[field] = str(values[index]) if values is not None and index < len(values) else ""
            stats[video_id] = record
    return stats


def load_stats(stats_path: Path, retries: int = 8, retry_delay: float = 0.2) -> Dict[str, Dict[str, str]]:
    if not stats_path.exists():
        return {}

    lock_path = _lock_path(stats_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _ in range(retries):
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                return _load_stats_unlocked(stats_path)
            except (EOFError, ValueError, OSError, zipfile.BadZipFile) as exc:
                last_error = exc
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        time.sleep(retry_delay)
    if last_error is not None:
        raise last_error
    return {}


def save_stats(stats_path: Path, stats: Dict[str, Dict[str, str]]) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    video_ids = sorted(stats)
    payload = {"video_ids": np.asarray(video_ids, dtype=object)}
    for field in STATUS_FIELDS:
        payload[field] = np.asarray([stats[video_id].get(field, "") for video_id in video_ids], dtype=object)
    tmp_path = stats_path.parent / f".{stats_path.stem}.{os.getpid()}.tmp.npz"
    try:
        np.savez(tmp_path, **payload)
        os.replace(tmp_path, stats_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def ensure_record(stats: Dict[str, Dict[str, str]], video_id: str) -> Dict[str, str]:
    if video_id not in stats:
        stats[video_id] = {field: "" for field in STATUS_FIELDS}
    return stats[video_id]


def update_video_stats(stats_path: Path, video_id: str, **updates: str) -> Dict[str, str]:
    lock_path = _lock_path(stats_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        stats = _load_stats_unlocked(stats_path)
        record = ensure_record(stats, video_id)
        for key, value in updates.items():
            if key in STATUS_FIELDS:
                record[key] = "" if value is None else str(value)
        save_stats(stats_path, stats)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return dict(record)


def update_many_video_stats(stats_path: Path, video_ids: Iterable[str], **updates: str) -> None:
    lock_path = _lock_path(stats_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        stats = _load_stats_unlocked(stats_path)
        for video_id in video_ids:
            record = ensure_record(stats, video_id)
            for key, value in updates.items():
                if key in STATUS_FIELDS:
                    record[key] = "" if value is None else str(value)
        save_stats(stats_path, stats)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def update_many_video_stats_with_retry(stats_path: Path, video_ids: Iterable[str], retries: int = 8, retry_delay: float = 0.2, **updates: str) -> None:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            update_many_video_stats(stats_path, video_ids, **updates)
            return
        except (EOFError, ValueError, OSError, zipfile.BadZipFile) as exc:
            last_error = exc
            time.sleep(retry_delay)
    if last_error is not None:
        raise last_error



def journal_lock_path(journal_path: Path) -> Path:
    return journal_path.with_suffix(journal_path.suffix + ".lock")


def sidecar_status_dir(stats_path: Path) -> Path:
    return stats_path.parent / "video_status"


def sidecar_status_path(stats_path: Path, video_id: str) -> Path:
    return sidecar_status_dir(stats_path) / f"{video_id}.jsonl"


def append_status_journal(journal_path: Path, video_ids: Sequence[str], **updates: str) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_ids": list(video_ids),
        "updates": {key: ("" if value is None else str(value)) for key, value in updates.items()},
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    lock_path = journal_lock_path(journal_path)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            with journal_path.open("a", encoding="utf-8") as journal_handle:
                journal_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_status_sidecar(stats_path: Path, video_id: str, **updates: str) -> None:
    payload = {
        "video_ids": [video_id],
        "updates": {key: ("" if value is None else str(value)) for key, value in updates.items()},
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    sidecar_path = sidecar_status_path(stats_path, video_id)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = journal_lock_path(sidecar_path)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            with sidecar_path.open("a", encoding="utf-8") as sidecar_handle:
                sidecar_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def update_video_stats_best_effort(stats_path: Path, journal_path: Path, video_id: str, **updates: str) -> Dict[str, str]:
    append_status_sidecar(stats_path, video_id, **updates)
    return {key: ("" if value is None else str(value)) for key, value in updates.items() if key in STATUS_FIELDS}


def update_many_video_stats_best_effort(stats_path: Path, journal_path: Path, video_ids: Sequence[str], **updates: str) -> None:
    try:
        update_many_video_stats_with_retry(stats_path, video_ids, **updates)
    except Exception:
        append_status_journal(journal_path, video_ids, **updates)


def _apply_payloads_to_stats(stats_path: Path, payloads: Sequence[dict]) -> int:
    if not payloads:
        return 0
    update_count = 0
    stats_lock_path = _lock_path(stats_path)
    stats_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_lock_path.open("a+", encoding="utf-8") as stats_handle:
        fcntl.flock(stats_handle.fileno(), fcntl.LOCK_EX)
        try:
            stats = _load_stats_unlocked(stats_path)
            for payload in payloads:
                video_ids = [str(item) for item in payload.get("video_ids", []) if str(item)]
                updates = payload.get("updates", {})
                if not video_ids or not isinstance(updates, dict):
                    continue
                for video_id in video_ids:
                    record = ensure_record(stats, video_id)
                    for key, value in updates.items():
                        if key in STATUS_FIELDS:
                            record[key] = "" if value is None else str(value)
                    update_count += 1
            if update_count:
                save_stats(stats_path, stats)
        finally:
            fcntl.flock(stats_handle.fileno(), fcntl.LOCK_UN)
    return update_count


def apply_status_journal_to_stats(stats_path: Path, journal_path: Path, remove_applied: bool = True) -> int:
    payloads = []

    if journal_path.exists() and journal_path.stat().st_size > 0:
        lock_path = journal_lock_path(journal_path)
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                if journal_path.exists() and journal_path.stat().st_size > 0:
                    for line in journal_path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            payloads.append(json.loads(line))
                    if remove_applied:
                        journal_path.unlink(missing_ok=True)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    sidecar_dir = sidecar_status_dir(stats_path)
    if sidecar_dir.exists():
        for sidecar_path in sorted(sidecar_dir.glob("*.jsonl")):
            lock_path = journal_lock_path(sidecar_path)
            with lock_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    if sidecar_path.exists() and sidecar_path.stat().st_size > 0:
                        for line in sidecar_path.read_text(encoding="utf-8").splitlines():
                            if line.strip():
                                payloads.append(json.loads(line))
                    if remove_applied:
                        sidecar_path.unlink(missing_ok=True)
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            if remove_applied:
                lock_path.unlink(missing_ok=True)

    return _apply_payloads_to_stats(stats_path, payloads)
