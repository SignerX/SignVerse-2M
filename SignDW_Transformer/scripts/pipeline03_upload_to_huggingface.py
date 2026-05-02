#!/usr/bin/env python3

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import threading
import traceback
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.stats_npz import apply_status_journal_to_stats, update_many_video_stats_with_retry
from utils.dataset_pool import find_dataset_video_dir, list_unuploaded_folder_paths as list_unuploaded_folder_paths_pool


DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_RAW_VIDEO_DIR = REPO_ROOT / "raw_video"
DEFAULT_RAW_CAPTION_DIR = REPO_ROOT / "raw_caption"
DEFAULT_RAW_METADATA_DIR = REPO_ROOT / "raw_metadata"
DEFAULT_ARCHIVE_DIR = REPO_ROOT / "archives"
DEFAULT_PROGRESS_PATH = REPO_ROOT / "archive_upload_progress.json"
DEFAULT_STATUS_JOURNAL_PATH = REPO_ROOT / "upload_status_journal.jsonl"
DEFAULT_STATS_NPZ = REPO_ROOT / "stats.npz"
DEFAULT_GIT_CLONE_DIR = DEFAULT_ARCHIVE_DIR / ".hf_git_repo"
DEFAULT_TOKEN_PATH = Path.home() / ".hf_token.txt"
DEFAULT_TARGET_BYTES = 10 * 1024 * 1024 * 1024
DEFAULT_TARGET_FOLDERS = 40
COMPLETE_MARKER_NAME = ".complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive NPZ folders into 14GB tar files and upload them to Hugging Face."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--scratch-dataset-dir", type=Path, default=None)
    parser.add_argument("--raw-video-dir", type=Path, default=DEFAULT_RAW_VIDEO_DIR)
    parser.add_argument("--scratch-raw-video-dir", type=Path, default=None)
    parser.add_argument("--raw-caption-dir", type=Path, default=DEFAULT_RAW_CAPTION_DIR)
    parser.add_argument("--raw-metadata-dir", type=Path, default=DEFAULT_RAW_METADATA_DIR)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--progress-path", type=Path, default=DEFAULT_PROGRESS_PATH)
    parser.add_argument("--stats-npz", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--status-journal-path", type=Path, default=DEFAULT_STATUS_JOURNAL_PATH)
    parser.add_argument("--processed-csv-path", type=Path, default=None)
    parser.add_argument("--repo-id", default="SignerX/SignVerse-2M")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--repo-revision", default=os.environ.get("HF_REPO_REVISION", "main"))
    parser.add_argument("--target-bytes", type=int, default=DEFAULT_TARGET_BYTES)
    parser.add_argument("--target-folders", type=int, default=DEFAULT_TARGET_FOLDERS)
    parser.add_argument("--parallel-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--start-stagger-min", type=int, default=0)
    parser.add_argument("--start-stagger-max", type=int, default=0)
    parser.add_argument("--skip-stats-write", action="store_true")
    parser.add_argument("--require-target-bytes", action="store_true", default=True)
    parser.add_argument("--allow-small-final-batch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upload-mode", choices=["git-ssh", "api", "api-stream"], default=os.environ.get("HF_UPLOAD_MODE", "api"))
    parser.add_argument("--git-clone-dir", type=Path, default=DEFAULT_GIT_CLONE_DIR)
    parser.add_argument("--token", default=None)
    return parser.parse_args()




def resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token
    if DEFAULT_TOKEN_PATH.exists():
        token = DEFAULT_TOKEN_PATH.read_text(encoding="utf-8").strip()
        return token or None
    return None



def journal_lock_path(journal_path: Path) -> Path:
    return journal_path.with_suffix(journal_path.suffix + '.lock')


@contextlib.contextmanager
def locked_journal(journal_path: Path):
    lock_path = journal_lock_path(journal_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_status_journal(journal_path: Path, video_ids: Sequence[str], **updates: str) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_ids": list(video_ids),
        "updates": {k: ("" if v is None else str(v)) for k, v in updates.items()},
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with locked_journal(journal_path):
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def update_many_video_stats_best_effort(stats_path: Path, journal_path: Path, video_ids: Sequence[str], skip_stats_write: bool = False, **updates: str) -> None:
    if skip_stats_write:
        append_status_journal(journal_path, video_ids, **updates)
        return
    try:
        update_many_video_stats_with_retry(stats_path, video_ids, **updates)
    except Exception as exc:
        payload = dict(updates)
        payload["last_error"] = str(exc) if not payload.get("last_error") else payload["last_error"]
        append_status_journal(journal_path, video_ids, **payload)
        print(f"Warning: stats.npz update deferred to journal due to: {exc}")

def progress_lock_path(progress_path: Path) -> Path:
    return progress_path.with_suffix(progress_path.suffix + '.lock')


@contextlib.contextmanager
def locked_progress(progress_path: Path):
    lock_path = progress_lock_path(progress_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def shard_for_video_id(video_id: str, shard_count: int) -> int:
    if shard_count <= 1:
        return 0
    digest = hashlib.sha1(video_id.encode('utf-8')).hexdigest()
    return int(digest[:8], 16) % shard_count


def filter_folders_for_shard(folders: Sequence[Tuple[str, Path]], shard_count: int, shard_index: int) -> List[Tuple[str, Path]]:
    if shard_count <= 1:
        return list(folders)
    return [(video_id, folder_path) for video_id, folder_path in folders if shard_for_video_id(video_id, shard_count) == shard_index]


def load_progress(progress_path: Path, retries: int = 8, retry_delay: float = 0.2) -> Dict[str, object]:
    if not progress_path.exists():
        return {"archives": {}, "uploaded_folders": {}}
    last_error = None
    for _ in range(retries):
        try:
            with progress_path.open("r", encoding="utf-8") as handle:
                data = handle.read()
            if not data.strip():
                raise json.JSONDecodeError("empty progress file", data, 0)
            return json.loads(data)
        except (json.JSONDecodeError, OSError) as exc:
            last_error = exc
            time.sleep(retry_delay)
    if last_error is not None:
        raise last_error
    return {"archives": {}, "uploaded_folders": {}}


def save_progress(progress_path: Path, progress: Dict[str, object]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.parent / f'.{progress_path.name}.{os.getpid()}.tmp'
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(progress, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, progress_path)


def folder_size_bytes(folder_path: Path) -> int:
    total = 0
    for path in folder_path.rglob("*"):
        if not path.is_file():
            continue
        stat_result = path.stat()
        # Use allocated blocks so upload triggering follows quota pressure,
        # not just logical payload bytes. This avoids undercounting millions
        # of tiny NPZ files.
        allocated_bytes = getattr(stat_result, "st_blocks", 0) * 512
        total += allocated_bytes if allocated_bytes > 0 else stat_result.st_size
    return total


def list_unuploaded_folder_paths(dataset_dir: Path, scratch_dataset_dir: Path | None, progress: Dict[str, object]) -> List[Tuple[str, Path]]:
    uploaded_folders = progress.get("uploaded_folders", {})
    return list_unuploaded_folder_paths_pool(dataset_dir, scratch_dataset_dir, uploaded_folders)


def enrich_folder_sizes(folders: Sequence[Tuple[str, Path]]) -> List[Tuple[str, Path, int]]:
    return [(folder_name, folder_path, folder_size_bytes(folder_path)) for folder_name, folder_path in folders]


def build_batch(folders: Sequence[Tuple[str, Path, int]], target_bytes: int) -> List[Tuple[str, Path, int]]:
    batch = []
    total_bytes = 0
    for folder_info in folders:
        _, _, folder_bytes = folder_info
        if batch and total_bytes + folder_bytes > target_bytes:
            break
        batch.append(folder_info)
        total_bytes += folder_bytes
        if total_bytes >= target_bytes:
            break
    return batch


def total_batchable_bytes(folders: Sequence[Tuple[str, Path, int]]) -> int:
    return sum(folder_bytes for _, _, folder_bytes in folders)


def next_archive_index(progress: Dict[str, object], repo_files: Sequence[str]) -> int:
    indices = []
    for archive_name in progress.get("archives", {}):
        if archive_name.startswith("Sign_DWPose_NPZ_") and archive_name.endswith(".tar"):
            indices.append(int(archive_name[-10:-4]))
    for repo_file in repo_files:
        name = Path(repo_file).name
        if name.startswith("Sign_DWPose_NPZ_") and name.endswith(".tar"):
            indices.append(int(name[-10:-4]))
    return (max(indices) + 1) if indices else 1




def preferred_temp_archive_dir() -> Path:
    for key in ("SLURM_TMPDIR", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            path = Path(value)
            path.mkdir(parents=True, exist_ok=True)
            return path
    path = Path("/tmp")
    path.mkdir(parents=True, exist_ok=True)
    return path

def create_tar_archive(archive_path: Path, folder_paths: Sequence[Tuple[str, Path]]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w") as tar:
        for folder_name, folder_path in folder_paths:
            tar.add(folder_path, arcname=folder_name, recursive=True)


def upload_archive(api: HfApi, repo_id: str, repo_type: str, repo_revision: str, archive_path: Path) -> None:
    api.upload_file(
        path_or_fileobj=str(archive_path),
        path_in_repo=f"dataset/{archive_path.name}",
        repo_id=repo_id,
        repo_type=repo_type,
        revision=repo_revision,
    )


def upload_runtime_state_files(api: HfApi | None, repo_id: str, repo_type: str, repo_revision: str, progress_path: Path, journal_path: Path, stats_path: Path | None = None, processed_csv_path: Path | None = None) -> None:
    if api is None:
        return
    api.upload_file(
        path_or_fileobj=str(progress_path),
        path_in_repo="runtime_state/archive_upload_progress.json",
        repo_id=repo_id,
        repo_type=repo_type,
        revision=repo_revision,
    )
    if journal_path.exists():
        api.upload_file(
            path_or_fileobj=str(journal_path),
            path_in_repo="runtime_state/upload_status_journal.jsonl",
            repo_id=repo_id,
            repo_type=repo_type,
            revision=repo_revision,
        )
    if stats_path is not None and stats_path.exists():
        api.upload_file(
            path_or_fileobj=str(stats_path),
            path_in_repo="runtime_state/stats.npz",
            repo_id=repo_id,
            repo_type=repo_type,
            revision=repo_revision,
        )
    if processed_csv_path is not None and processed_csv_path.exists():
        api.upload_file(
            path_or_fileobj=str(processed_csv_path),
            path_in_repo="runtime_state/SignVerse-2M-metadata_processed.csv",
            repo_id=repo_id,
            repo_type=repo_type,
            revision=repo_revision,
        )


def upload_archive_streaming(api: HfApi, repo_id: str, repo_type: str, repo_revision: str, folder_paths: Sequence[Tuple[str, Path]], archive_name: str) -> None:
    if api is None:
        raise RuntimeError('api-stream upload requires a Hugging Face token')
    command = ['tar', '-cf', '-']
    for folder_name, folder_path in folder_paths:
        command.extend(['-C', str(folder_path.parent), folder_name])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    stderr_chunks = []

    def _read_stderr() -> None:
        assert process.stderr is not None
        data = process.stderr.read()
        if data:
            stderr_chunks.append(data.decode('utf-8', errors='replace'))

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()
    try:
        api.upload_file(
            path_or_fileobj=process.stdout,
            path_in_repo=f"dataset/{archive_name}",
            repo_id=repo_id,
            repo_type=repo_type,
            revision=repo_revision,
        )
    finally:
        try:
            process.stdout.close()
        except Exception:
            pass
    return_code = process.wait()
    stderr_thread.join(timeout=5)
    if return_code != 0:
        stderr_text = ''.join(stderr_chunks).strip()
        raise RuntimeError(stderr_text or f'tar streaming command failed with exit code {return_code}')


def repo_git_url(repo_id: str, repo_type: str) -> str:
    prefix = ""
    if repo_type == "dataset":
        prefix = "datasets/"
    elif repo_type == "space":
        prefix = "spaces/"
    return f"git@hf.co:{prefix}{repo_id}"


def run_git(command: Sequence[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or f"git command failed: {' '.join(command)}").strip())
    return (result.stdout or result.stderr or "").strip()


def ensure_git_upload_repo(clone_dir: Path, repo_id: str, repo_type: str, repo_revision: str) -> Path:
    remote_url = repo_git_url(repo_id, repo_type)
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (clone_dir / '.git').exists():
        subprocess.run(["git", "clone", remote_url, str(clone_dir)], check=True, capture_output=True, text=True)
    else:
        configured = run_git(["git", "remote", "get-url", "origin"], clone_dir)
        if configured != remote_url:
            raise RuntimeError(f"Git upload clone remote mismatch: {configured} != {remote_url}")
        run_git(["git", "fetch", "origin"], clone_dir)
        remote_branch = f"origin/{repo_revision}"
        existing = subprocess.run(["git", "show-ref", "--verify", f"refs/remotes/{remote_branch}"], cwd=str(clone_dir))
        if existing.returncode == 0:
            run_git(["git", "checkout", "-B", repo_revision, remote_branch], clone_dir)
        else:
            run_git(["git", "checkout", "-B", repo_revision, "origin/main"], clone_dir)
    run_git(["git", "lfs", "install", "--local"], clone_dir)
    try:
        run_git(["git", "config", "user.name"], clone_dir)
    except Exception:
        run_git(["git", "config", "user.name", os.environ.get("HF_GIT_USER_NAME", "sf895")], clone_dir)
    try:
        run_git(["git", "config", "user.email"], clone_dir)
    except Exception:
        run_git(["git", "config", "user.email", os.environ.get("HF_GIT_USER_EMAIL", "sf895@rutgers.edu")], clone_dir)
    return clone_dir


def list_repo_files_via_git(clone_dir: Path, repo_id: str, repo_type: str, repo_revision: str) -> List[str]:
    repo_dir = ensure_git_upload_repo(clone_dir, repo_id, repo_type, repo_revision)
    return [path.name for path in repo_dir.iterdir() if path.is_file()]


def upload_archive_via_git(clone_dir: Path, repo_id: str, repo_type: str, repo_revision: str, archive_path: Path) -> None:
    repo_dir = ensure_git_upload_repo(clone_dir, repo_id, repo_type, repo_revision)
    target_path = repo_dir / archive_path.name
    shutil.copy2(archive_path, target_path)
    run_git(["git", "add", archive_path.name], repo_dir)
    diff_result = subprocess.run(["git", "diff", "--cached", "--quiet", "--", archive_path.name], cwd=str(repo_dir))
    if diff_result.returncode == 0:
        return
    if diff_result.returncode != 1:
        raise RuntimeError(f"git diff --cached failed for {archive_path.name}")
    run_git(["git", "commit", "-m", f"Add {archive_path.name}"], repo_dir)
    run_git(["git", "push", "origin", repo_revision], repo_dir)


def cleanup_local_assets(
    video_ids: Sequence[str],
    dataset_dir: Path,
    scratch_dataset_dir: Path | None,
    raw_video_dir: Path,
    scratch_raw_video_dir: Path | None,
    raw_caption_dir: Path,
    raw_metadata_dir: Path,
) -> None:
    for video_id in video_ids:
        dataset_video_dir = find_dataset_video_dir(video_id, dataset_dir, scratch_dataset_dir)
        if dataset_video_dir.exists():
            shutil.rmtree(dataset_video_dir, ignore_errors=True)
        for raw_dir in [raw_video_dir, scratch_raw_video_dir]:
            if raw_dir is None:
                continue
            for path in raw_dir.glob(f"{video_id}.*"):
                if path.is_file():
                    path.unlink()
        caption_dir = raw_caption_dir / video_id
        if caption_dir.exists():
            shutil.rmtree(caption_dir, ignore_errors=True)
        metadata_path = raw_metadata_dir / f"{video_id}.json"
        if metadata_path.exists():
            metadata_path.unlink(missing_ok=True)


def prune_uploaded_runtime_residue(
    progress: Dict[str, object],
    dataset_dir: Path,
    scratch_dataset_dir: Path | None,
    raw_video_dir: Path,
    scratch_raw_video_dir: Path | None,
    raw_caption_dir: Path,
    raw_metadata_dir: Path,
) -> None:
    uploaded = set(progress.get("uploaded_folders", {}))
    for video_id in uploaded:
        for raw_dir in [raw_video_dir, scratch_raw_video_dir]:
            if raw_dir is None:
                continue
            for path in raw_dir.glob(f"{video_id}.*"):
                if path.is_file():
                    path.unlink(missing_ok=True)
        caption_dir = raw_caption_dir / video_id
        if caption_dir.exists():
            shutil.rmtree(caption_dir, ignore_errors=True)
        metadata_path = raw_metadata_dir / f"{video_id}.json"
        if metadata_path.exists():
            metadata_path.unlink(missing_ok=True)
        dataset_video_dir = find_dataset_video_dir(video_id, dataset_dir, scratch_dataset_dir)
        if dataset_video_dir.exists() and not (dataset_video_dir / "npz" / COMPLETE_MARKER_NAME).exists():
            shutil.rmtree(dataset_video_dir, ignore_errors=True)


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def main() -> None:
    args = parse_args()
    if args.parallel_shards < 1:
        raise ValueError('--parallel-shards must be >= 1')
    if not (0 <= args.shard_index < args.parallel_shards):
        raise ValueError('--shard-index must satisfy 0 <= shard-index < parallel-shards')
    if args.start_stagger_max < args.start_stagger_min:
        raise ValueError('--start-stagger-max must be >= --start-stagger-min')
    if args.start_stagger_max > 0:
        delay = random.randint(args.start_stagger_min, args.start_stagger_max)
        if delay > 0:
            print(f"[pipeline03] stagger sleep {delay}s shard={args.shard_index}/{args.parallel_shards}", flush=True)
            time.sleep(delay)
    print(f"[pipeline03] start upload_mode={args.upload_mode} repo_id={args.repo_id} shard={args.shard_index}/{args.parallel_shards}", flush=True)
    skip_stats_write = args.skip_stats_write or args.parallel_shards > 1
    progress = load_progress(args.progress_path)
    print(
        f"[pipeline03] loaded progress archives={len(progress.get('archives', {}))} "
        f"uploaded_folders={len(progress.get('uploaded_folders', {}))}",
        flush=True,
    )
    resolved_token = resolve_token(args.token)
    print(f"[pipeline03] token_present={bool(resolved_token)}", flush=True)
    api = HfApi(token=resolved_token) if args.upload_mode in {"api", "api-stream"} else None
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    if args.scratch_dataset_dir is not None:
        args.scratch_dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.upload_mode in {"api", "api-stream"}:
            print("[pipeline03] skipping repo file listing for api mode; using local progress for archive index", flush=True)
            repo_files = []
        else:
            print("[pipeline03] listing repo files via git", flush=True)
            repo_files = list_repo_files_via_git(args.git_clone_dir, args.repo_id, args.repo_type, args.repo_revision)
    except Exception as exc:
        print(f"[pipeline03] repo file listing failed: {exc}", flush=True)
        traceback.print_exc()
        repo_files = []

    while True:
        with locked_progress(args.progress_path):
            progress = load_progress(args.progress_path)
            prune_uploaded_runtime_residue(
                progress,
                args.dataset_dir,
                args.scratch_dataset_dir,
                args.raw_video_dir,
                args.scratch_raw_video_dir,
                args.raw_caption_dir,
                args.raw_metadata_dir,
            )
            remaining_folder_paths = list_unuploaded_folder_paths(args.dataset_dir, args.scratch_dataset_dir, progress)
            remaining_folder_paths = filter_folders_for_shard(remaining_folder_paths, args.parallel_shards, args.shard_index)
            if not remaining_folder_paths:
                print(f"No unuploaded dataset folders remain for shard {args.shard_index}/{args.parallel_shards}.")
                break
            remaining_count = len(remaining_folder_paths)
            print(f"[pipeline03] remaining completed folders available={remaining_count} shard={args.shard_index}/{args.parallel_shards}", flush=True)
            if remaining_count >= args.target_folders:
                selected_folder_paths = remaining_folder_paths[: args.target_folders]
                batch = enrich_folder_sizes(selected_folder_paths)
                batch_names = [name for name, _, _ in batch]
                batch_bytes = sum(folder_bytes for _, _, folder_bytes in batch)
                print(f"[pipeline03] folder threshold reached; selecting first {len(batch_names)} folders without global size scan", flush=True)
            else:
                remaining_folders = enrich_folder_sizes(remaining_folder_paths)
                remaining_bytes = total_batchable_bytes(remaining_folders)
                require_target_bytes = args.require_target_bytes and not args.allow_small_final_batch
                if require_target_bytes and remaining_bytes < args.target_bytes:
                    print(
                        f"Skip upload: only {format_size(remaining_bytes)} across {remaining_count} completed NPZ folders available, below targets {format_size(args.target_bytes)} or {args.target_folders} folders."
                    )
                    break
                batch = build_batch(remaining_folders, args.target_bytes)
                batch_names = [name for name, _, _ in batch]
                batch_bytes = sum(folder_bytes for _, _, folder_bytes in batch)
            archive_index = next_archive_index(progress, repo_files)
            archive_name = f"Sign_DWPose_NPZ_{archive_index:06d}.tar"
            progress["archives"][archive_name] = {
                "folders": batch_names,
                "size_bytes": batch_bytes,
                "status": "uploading",
                "reserved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "shard_index": args.shard_index,
                "parallel_shards": args.parallel_shards,
            }
            save_progress(args.progress_path, progress)

        archive_root = args.archive_dir if args.upload_mode == "git-ssh" else preferred_temp_archive_dir()
        archive_path = archive_root / archive_name

        print(f"Create archive {archive_name} with {len(batch_names)} folders ({format_size(batch_bytes)})")
        for folder_name in batch_names:
            print(f"  - {folder_name}")

        if args.dry_run:
            break

        args.archive_dir.mkdir(parents=True, exist_ok=True)
        try:
            apply_status_journal_to_stats(args.stats_npz, args.status_journal_path)
        except Exception as exc:
            print(f"Warning: pre-upload status compaction skipped due to: {exc}")
        append_status_journal(
            args.status_journal_path,
            batch_names,
            upload_status="uploading",
            archive_name=archive_name,
            local_cleanup_status="pending",
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        try:
            if args.upload_mode == "api-stream":
                upload_archive_streaming(api, args.repo_id, args.repo_type, [(name, path) for name, path, _ in batch], archive_name)
            else:
                create_tar_archive(archive_path, [(name, path) for name, path, _ in batch])
                if args.upload_mode == "api":
                    upload_archive(api, args.repo_id, args.repo_type, args.repo_revision, archive_path)
                else:
                    upload_archive_via_git(args.git_clone_dir, args.repo_id, args.repo_type, args.repo_revision, archive_path)
        except Exception as exc:
            with locked_progress(args.progress_path):
                progress = load_progress(args.progress_path)
                archive_meta = progress.get("archives", {}).get(archive_name, {})
                if isinstance(archive_meta, dict):
                    archive_meta["status"] = "failed"
                    archive_meta["last_error"] = str(exc)
                    archive_meta["failed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    progress["archives"][archive_name] = archive_meta
                    save_progress(args.progress_path, progress)
            update_many_video_stats_best_effort(
                args.stats_npz,
                args.status_journal_path,
                batch_names,
                upload_status="failed",
                local_cleanup_status="pending",
                archive_name=archive_name,
                skip_stats_write=skip_stats_write,
                last_error=str(exc),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            try:
                apply_status_journal_to_stats(args.stats_npz, args.status_journal_path)
            except Exception as compact_exc:
                print(f"Warning: failed-upload compaction skipped due to: {compact_exc}")
            if archive_path.exists():
                archive_path.unlink(missing_ok=True)
            raise

        with locked_progress(args.progress_path):
            progress = load_progress(args.progress_path)
            progress["archives"][archive_name] = {
                "folders": batch_names,
                "size_bytes": batch_bytes,
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "uploaded",
                "shard_index": args.shard_index,
                "parallel_shards": args.parallel_shards,
            }
            for folder_name in batch_names:
                progress["uploaded_folders"][folder_name] = archive_name
            save_progress(args.progress_path, progress)

        repo_files.append(f"dataset/{archive_name}")
        update_many_video_stats_best_effort(
            args.stats_npz,
            args.status_journal_path,
            batch_names,
            upload_status="uploaded",
            local_cleanup_status="pending",
            archive_name=archive_name,
            skip_stats_write=skip_stats_write,
            last_error="",
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        try:
            apply_status_journal_to_stats(args.stats_npz, args.status_journal_path)
        except Exception as exc:
            print(f"Warning: post-upload status compaction skipped due to: {exc}")

        cleanup_error = ""
        try:
            cleanup_local_assets(
                batch_names,
                args.dataset_dir,
                args.scratch_dataset_dir,
                args.raw_video_dir,
                args.scratch_raw_video_dir,
                args.raw_caption_dir,
                args.raw_metadata_dir,
            )
            if archive_path.exists():
                archive_path.unlink(missing_ok=True)
        except Exception as exc:
            cleanup_error = str(exc)
        update_many_video_stats_best_effort(
            args.stats_npz,
            args.status_journal_path,
            batch_names,
            upload_status="uploaded",
            local_cleanup_status="deleted" if not cleanup_error else "failed",
            archive_name=archive_name,
            skip_stats_write=skip_stats_write,
            last_error=cleanup_error,
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        try:
            apply_status_journal_to_stats(args.stats_npz, args.status_journal_path)
        except Exception as exc:
            print(f"Warning: post-cleanup status compaction skipped due to: {exc}")
        upload_runtime_state_files(api, args.repo_id, args.repo_type, args.repo_revision, args.progress_path, args.status_journal_path, args.stats_npz, args.processed_csv_path if hasattr(args, "processed_csv_path") else None)
        with locked_progress(args.progress_path):
            progress = load_progress(args.progress_path)
            prune_uploaded_runtime_residue(
                progress,
                args.dataset_dir,
                args.scratch_dataset_dir,
                args.raw_video_dir,
                args.scratch_raw_video_dir,
                args.raw_caption_dir,
                args.raw_metadata_dir,
            )
        if cleanup_error:
            raise RuntimeError(f"Uploaded {archive_name} but local cleanup failed: {cleanup_error}")
        print(f"Uploaded {archive_name} and cleaned raw assets for {len(batch_names)} videos. shard={args.shard_index}/{args.parallel_shards}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[pipeline03] fatal: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
