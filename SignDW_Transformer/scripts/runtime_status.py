#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

from utils.dataset_pool import count_complete

VIDEO_EXTS = {'.mp4', '.mkv', '.webm', '.mov'}
ARRAY_RANGE_RE = re.compile(r'^(\d+)_\[(.+)\]$')
PROCESSED_REQUIRED_COLUMNS = {
    'video_id',
    'download_status',
    'process_status',
    'upload_status',
    'archive_name',
}
GPU_PARTITIONS = ['gpu', 'gpu-redhat', 'cgpu']
DEFAULT_VIDEOS_PER_DWPOSE_JOB = 20


def run_command(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError:
        return ''
    return (proc.stdout or '').strip()


def count_claims(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob('*.claim'))


def aggregate_claims_by_job_key(directory: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not directory.exists():
        return {}
    for path in directory.glob('*.claim'):
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except OSError:
            continue
        job_key = ''
        for line in lines:
            if line.startswith('job_key='):
                job_key = line.split('=', 1)[1].strip()
                break
        if job_key:
            counts[job_key] += 1
    return dict(counts)


def read_videos_per_dwpose_job(root_dir: Path) -> int:
    worker = root_dir / 'slurm' / 'process_dwpose_array.slurm'
    if not worker.exists():
        return DEFAULT_VIDEOS_PER_DWPOSE_JOB
    try:
        for line in worker.read_text().splitlines():
            if line.startswith('VIDEOS_PER_JOB='):
                m = re.search(r'\$\{VIDEOS_PER_JOB:-([0-9]+)\}', line)
                if m:
                    return int(m.group(1))
    except Exception:
        pass
    return DEFAULT_VIDEOS_PER_DWPOSE_JOB


def sum_file_sizes(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def count_uploaded(progress_path: Path) -> tuple[int, int]:
    if not progress_path.exists():
        return 0, 0
    try:
        data = json.loads(progress_path.read_text())
    except Exception:
        return 0, 0
    archives = data.get('archives', {})
    uploaded_folders = data.get('uploaded_folders', {})
    return len(archives), len(uploaded_folders)


def expand_task_count(jobid_token: str) -> int:
    m = ARRAY_RANGE_RE.match(jobid_token)
    if not m:
        return 1
    body = m.group(2)
    if '%' in body:
        body = body.split('%', 1)[0]
    total = 0
    for part in body.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                total += int(b) - int(a) + 1
            except ValueError:
                total += 1
        else:
            total += 1
    return max(total, 1)


def queue_status(username: str) -> dict[str, object]:
    output = run_command(['squeue', '-u', username, '-h', '-o', '%i|%j|%T|%P'])
    job_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    active_tasks_by_partition: Counter[str] = Counter()
    running_dwpose = 0
    running_download = 0
    pending_download = 0
    pending_dwpose = 0
    if output:
        for line in output.splitlines():
            parts = line.split('|')
            if len(parts) != 4:
                continue
            jobid_token, job, state, partition = parts
            count = expand_task_count(jobid_token)
            job_counts[f'{job}|{state}'] += count
            partition_counts[f'{job}|{partition}|{state}'] += count
            if state in {'RUNNING', 'PENDING', 'CONFIGURING'}:
                active_tasks_by_partition[partition] += count
            if job == 'dwpose' and state == 'RUNNING':
                running_dwpose += count
            if job == 'download' and state == 'RUNNING':
                running_download += count
            if job == 'download' and state in {'PENDING', 'CONFIGURING'}:
                pending_download += count
            if job == 'dwpose' and state in {'PENDING', 'CONFIGURING'}:
                pending_dwpose += count
    total_download = running_download + pending_download
    return {
        'running_dwpose': running_dwpose,
        'running_download': running_download,
        'pending_dwpose_jobs': pending_dwpose,
        'pending_download_jobs': pending_download,
        'total_download_jobs': total_download,
        'job_state_counts': dict(job_counts),
        'job_partition_state_counts': dict(partition_counts),
        'active_tasks_by_partition': dict(active_tasks_by_partition),
    }


def gpu_partition_capacity(partitions: list[str], active_tasks_by_partition: dict[str, int]) -> list[dict[str, object]]:
    qos_limit_by_part: dict[str, int] = {}
    qos_output = run_command(['sacctmgr', 'show', 'qos', 'format=Name,MaxSubmitPU', '-P'])
    if qos_output:
        for line in qos_output.splitlines():
            if not line.strip() or '|' not in line:
                continue
            name, max_submit = line.split('|', 1)
            name = name.strip()
            max_submit = max_submit.strip()
            if name in partitions and max_submit:
                try:
                    qos_limit_by_part[name] = int(max_submit)
                except ValueError:
                    pass

    rows: list[dict[str, object]] = []
    for partition in partitions:
        free_gpus = 0
        nodes_output = run_command(['sinfo', '-h', '-N', '-p', partition, '-o', '%N'])
        nodes = [line.strip() for line in nodes_output.splitlines() if line.strip()]
        for node in nodes:
            node_line = run_command(['scontrol', 'show', 'node', node, '-o'])
            if not node_line:
                continue
            state_m = re.search(r'\bState=([^ ]+)', node_line)
            state = state_m.group(1).lower() if state_m else ''
            if any(flag in state for flag in ('drain', 'drained', 'down', 'fail', 'inval')):
                continue
            cfg_m = re.search(r'\bCfgTRES=.*?(?:,|^)gres/gpu=(\d+)', node_line)
            alloc_m = re.search(r'\bAllocTRES=.*?(?:,|^)gres/gpu=(\d+)', node_line)
            total = int(cfg_m.group(1)) if cfg_m else 0
            used = int(alloc_m.group(1)) if alloc_m else 0
            free = total - used
            if free > 0:
                free_gpus += free
        active_tasks = int(active_tasks_by_partition.get(partition, 0))
        qos_limit = qos_limit_by_part.get(partition)
        submit_slots = free_gpus
        if qos_limit is not None:
            submit_slots = min(submit_slots, max(0, qos_limit - active_tasks))
        rows.append({
            'partition': partition,
            'free_gpus': free_gpus,
            'active_tasks': active_tasks,
            'qos_limit': qos_limit,
            'submit_slots': submit_slots,
        })
    return rows


def filesystem_avail_bytes(path: Path) -> int:
    try:
        proc = subprocess.run(['df', '-B1', str(path)], check=False, capture_output=True, text=True)
        lines = (proc.stdout or '').splitlines()
        if len(lines) < 2:
            return 0
        fields = lines[1].split()
        if len(fields) < 4:
            return 0
        return int(fields[3])
    except Exception:
        return 0


def human_bytes(num: int) -> str:
    value = float(num)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if value < 1024.0:
            return f'{value:.1f}{unit}'
        value /= 1024.0
    return f'{value:.1f}EB'


def read_source_manifest_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if not (row[0] or '').strip():
                continue
            count += 1
    return count


def read_processed_progress(path: Path) -> dict[str, object]:
    result = {
        'csv_exists': path.exists(),
        'csv_ok': False,
        'csv_error': '',
        'processed_rows': 0,
        'download_ok_rows': 0,
        'download_skipped_rows': 0,
        'download_running_rows': 0,
        'download_pending_rows': 0,
        'process_ok_rows': 0,
        'process_running_rows': 0,
        'upload_uploaded_rows': 0,
    }
    if not path.exists():
        result['csv_error'] = 'missing'
        return result
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            missing = sorted(PROCESSED_REQUIRED_COLUMNS - set(fieldnames))
            if missing:
                result['csv_error'] = f'missing_columns:{",".join(missing)}'
                return result
            rows = list(reader)
        result['processed_rows'] = len(rows)
        for row in rows:
            d = (row.get('download_status') or '').strip()
            p = (row.get('process_status') or '').strip()
            u = (row.get('upload_status') or '').strip()
            if d == 'ok':
                result['download_ok_rows'] += 1
            elif d == 'skipped':
                result['download_skipped_rows'] += 1
            elif d == 'running':
                result['download_running_rows'] += 1
            else:
                result['download_pending_rows'] += 1
            if p == 'ok':
                result['process_ok_rows'] += 1
            elif p == 'running':
                result['process_running_rows'] += 1
            if u == 'uploaded':
                result['upload_uploaded_rows'] += 1
        result['csv_ok'] = True
        return result
    except Exception as exc:
        result['csv_error'] = str(exc)
        return result


def run_sync(runtime_root: Path) -> str:
    sync_script = Path('/cache/home/sf895/SignVerse-2M/scripts/sync_processed_csv_from_runtime.py')
    if not sync_script.exists():
        return 'missing_sync_script'
    cmd = [
        'python3', str(sync_script),
        '--source-metadata-csv', str(runtime_root / 'SignVerse-2M-metadata_ori.csv'),
        '--output-metadata-csv', str(runtime_root / 'SignVerse-2M-metadata_processed.csv'),
        '--raw-video-dir', str(runtime_root / 'raw_video'),
        '--scratch-raw-video-dir', str(Path(f'/scratch/{os.environ.get("USER", "sf895")}/SignVerse-2M-runtime/raw_video')),
        '--raw-caption-dir', str(runtime_root / 'raw_caption'),
        '--raw-metadata-dir', str(runtime_root / 'raw_metadata'),
        '--dataset-dir', str(runtime_root / 'dataset'),
        '--scratch-dataset-dir', str(Path(f'/scratch/{os.environ.get("USER", "sf895")}/SignVerse-2M-runtime/dataset')),
        '--progress-path', str(runtime_root / 'archive_upload_progress.json'),
        '--status-journal-path', str(runtime_root / 'upload_status_journal.jsonl'),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError as exc:
        return f'error:{exc}'
    if proc.returncode == 0:
        return (proc.stdout or '').strip() or 'ok'
    err = (proc.stderr or proc.stdout or '').strip()
    return f'failed:{err}'


def main() -> None:
    parser = argparse.ArgumentParser(description='Report SignVerse runtime status.')
    parser.add_argument('--runtime-root', default='/home/sf895/SignVerse-2M-runtime')
    parser.add_argument('--username', default='sf895')
    parser.add_argument('--no-sync', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--include-partitions', action='store_true')
    parser.add_argument('--scan-complete', action='store_true')
    parser.add_argument('--scan-runtime-size', action='store_true')
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root)
    root_dir = Path('/cache/home/sf895/SignVerse-2M')
    raw_dir = runtime_root / 'raw_video'
    scratch_raw_dir = Path(f'/scratch/{os.environ.get("USER", "sf895")}/SignVerse-2M-runtime/raw_video')
    dataset_dir = runtime_root / 'dataset'
    scratch_dataset_dir = Path(f'/scratch/{os.environ.get("USER", "sf895")}/SignVerse-2M-runtime/dataset')
    claims_dir = runtime_root / 'slurm' / 'state' / 'claims'
    download_claims_dir = runtime_root / 'slurm' / 'state' / 'download_claims'
    progress_path = runtime_root / 'archive_upload_progress.json'
    source_csv = runtime_root / 'SignVerse-2M-metadata_ori.csv'
    processed_csv = runtime_root / 'SignVerse-2M-metadata_processed.csv'

    sync_result = 'skipped'
    if not args.no_sync:
        sync_result = run_sync(runtime_root)

    raw_complete: dict[str, Path] = {}
    raw_temp: list[Path] = []
    for current_raw_dir in [raw_dir, scratch_raw_dir]:
        if not current_raw_dir.exists():
            continue
        for path in current_raw_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() in VIDEO_EXTS:
                raw_complete.setdefault(path.stem, path)
            else:
                raw_temp.append(path)

    raw_size = sum_file_sizes(list(raw_complete.values()))
    runtime_size = 0
    if runtime_root.exists():
        proc = subprocess.run(['du', '-sb', str(runtime_root)], check=False, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                runtime_size = int(proc.stdout.split()[0])
            except Exception:
                runtime_size = 0

    source_rows = read_source_manifest_count(source_csv)
    progress = read_processed_progress(processed_csv)
    videos_per_dwpose_job = read_videos_per_dwpose_job(root_dir)

    payload = {
        'sync_result': sync_result,
        'download_normal': len(raw_temp) == 0,
        'raw_videos': len(raw_complete),
        'raw_temp_files': len(raw_temp),
        'sent_to_gpu': count_claims(claims_dir),
        'processed_complete': count_complete(dataset_dir, scratch_dataset_dir),
        'active_downloads': count_claims(download_claims_dir),
        'uploaded_archives': 0,
        'uploaded_folders': 0,
        'raw_size_bytes': raw_size,
        'runtime_size_bytes': runtime_size,
        'filesystem_avail_bytes': filesystem_avail_bytes(runtime_root),
        'source_rows': source_rows,
        'csv_exists': progress['csv_exists'],
        'csv_ok': progress['csv_ok'],
        'csv_error': progress['csv_error'],
        'processed_rows': progress['processed_rows'],
        'download_ok_rows': progress['download_ok_rows'],
        'download_skipped_rows': progress['download_skipped_rows'],
        'download_running_rows': progress['download_running_rows'],
        'download_pending_rows': progress['download_pending_rows'],
        'process_ok_rows': progress['process_ok_rows'],
        'process_running_rows': progress['process_running_rows'],
        'upload_uploaded_rows': progress['upload_uploaded_rows'],
    }
    uploaded_archives, uploaded_folders = count_uploaded(progress_path)
    payload['uploaded_archives'] = uploaded_archives
    payload['uploaded_folders'] = uploaded_folders
    payload.update(queue_status(args.username))
    process_claims_by_job = aggregate_claims_by_job_key(claims_dir)
    payload['videos_per_dwpose_job'] = videos_per_dwpose_job
    payload['process_claim_job_keys'] = len(process_claims_by_job)
    payload['process_claim_videos_actual'] = sum(process_claims_by_job.values())
    payload['process_claim_videos_max_per_job'] = max(process_claims_by_job.values(), default=0)
    payload['running_dwpose_jobs'] = payload['running_dwpose']
    payload['running_dwpose_videos_estimated'] = payload['running_dwpose'] * videos_per_dwpose_job
    payload['pending_dwpose_videos_estimated'] = payload['pending_dwpose_jobs'] * videos_per_dwpose_job
    payload['total_dwpose_jobs'] = payload['running_dwpose'] + payload['pending_dwpose_jobs']
    payload['total_dwpose_videos_estimated'] = payload['running_dwpose_videos_estimated'] + payload['pending_dwpose_videos_estimated']
    if args.include_partitions:
        payload['gpu_partition_capacity'] = gpu_partition_capacity(GPU_PARTITIONS, payload.get('active_tasks_by_partition', {}))
    else:
        payload['gpu_partition_capacity'] = []
        payload['job_partition_state_counts'] = {}
        payload['active_tasks_by_partition'] = {}
    payload['csv_row_match'] = (payload['processed_rows'] == payload['source_rows']) if payload['csv_ok'] else False

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    print(f"sync_result={payload['sync_result']}")
    print(f"download_normal={payload['download_normal']}")
    print(f"raw_videos={payload['raw_videos']}")
    print(f"raw_temp_files={payload['raw_temp_files']}")
    print(f"sent_to_gpu={payload['sent_to_gpu']}")
    print(f"running_dwpose={payload['running_dwpose']}")
    print(f"running_dwpose_jobs={payload['running_dwpose_jobs']}")
    print(f"pending_dwpose_jobs={payload['pending_dwpose_jobs']}")
    print(f"total_dwpose_jobs={payload['total_dwpose_jobs']}")
    print(f"videos_per_dwpose_job={payload['videos_per_dwpose_job']}")
    print(f"process_claim_job_keys={payload['process_claim_job_keys']}")
    print(f"process_claim_videos_actual={payload['process_claim_videos_actual']}")
    print(f"process_claim_videos_max_per_job={payload['process_claim_videos_max_per_job']}")
    print(f"running_dwpose_videos_estimated={payload['running_dwpose_videos_estimated']}")
    print(f"pending_dwpose_videos_estimated={payload['pending_dwpose_videos_estimated']}")
    print(f"total_dwpose_videos_estimated={payload['total_dwpose_videos_estimated']}")
    print(f"processed_complete={payload['processed_complete']}")
    print(f"active_downloads={payload['active_downloads']}")
    print(f"running_download_jobs={payload['running_download']}")
    print(f"pending_download_jobs={payload['pending_download_jobs']}")
    print(f"total_download_jobs={payload['total_download_jobs']}")
    print(f"uploaded_archives={payload['uploaded_archives']}")
    print(f"uploaded_folders={payload['uploaded_folders']}")
    print(f"source_rows={payload['source_rows']}")
    print(f"processed_rows={payload['processed_rows']}")
    print(f"csv_ok={payload['csv_ok']}")
    print(f"csv_row_match={payload['csv_row_match']}")
    print(f"csv_error={payload['csv_error']}")
    print(f"download_ok_rows={payload['download_ok_rows']}")
    print(f"download_skipped_rows={payload['download_skipped_rows']}")
    print(f"download_running_rows={payload['download_running_rows']}")
    print(f"download_pending_rows={payload['download_pending_rows']}")
    print(f"process_ok_rows={payload['process_ok_rows']}")
    print(f"process_running_rows={payload['process_running_rows']}")
    print(f"upload_uploaded_rows={payload['upload_uploaded_rows']}")
    for key in sorted(payload.get('job_partition_state_counts', {})):
        print(f"job_partition_state[{key}]={payload['job_partition_state_counts'][key]}")
    for row in payload.get('gpu_partition_capacity', []):
        qos_limit = row['qos_limit'] if row['qos_limit'] is not None else 'na'
        print(
            f"gpu_partition[{row['partition']}]=free_gpus={row['free_gpus']},"
            f"active_tasks={row['active_tasks']},qos_limit={qos_limit},submit_slots={row['submit_slots']}"
        )
    print(f"raw_size={human_bytes(payload['raw_size_bytes'])}")
    print(f"runtime_size={human_bytes(payload['runtime_size_bytes'])}")
    print(f"filesystem_avail={human_bytes(payload['filesystem_avail_bytes'])}")


if __name__ == '__main__':
    main()
