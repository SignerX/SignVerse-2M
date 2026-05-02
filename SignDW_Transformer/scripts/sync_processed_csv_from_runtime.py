#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_COLUMNS = [
    'video_id',
    'sign_language',
    'title',
    'duration_sec',
    'start_sec',
    'end_sec',
    'subtitle_languages',
    'subtitle_dir_path',
    'subtitle_en_source',
    'raw_video_path',
    'raw_metadata_path',
    'metadata_status',
    'subtitle_status',
    'download_status',
    'failure_count',
    'error',
    'processed_at',
    'subtitle_json_path',
    'subtitle_en',
    'subtitle_texts_json',
    'process_status',
    'upload_status',
    'local_cleanup_status',
    'archive_name',
]

VIDEO_EXTS = {'.mp4', '.mkv', '.webm', '.mov'}


from utils.dataset_pool import complete_video_ids


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    text = path.read_text(encoding='utf-8-sig')
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return [], []

    first = next(csv.reader([lines[0]]), [])
    first0 = first[0].strip() if first else ''
    if first0 == 'video_id' or 'download_status' in first or 'process_status' in first:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]
            fieldnames = list(reader.fieldnames or [])
        return rows, fieldnames

    rows: List[Dict[str, str]] = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for parts in reader:
            if not parts:
                continue
            video_id = (parts[0] or '').strip()
            sign_language = (parts[1] or '').strip() if len(parts) > 1 else ''
            if not video_id:
                continue
            rows.append({'video_id': video_id, 'sign_language': sign_language})
    return rows, ['video_id', 'sign_language']


def write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    tmp.replace(path)


def load_progress(progress_path: Path):
    if not progress_path.exists():
        return {}, {}
    obj = json.loads(progress_path.read_text())
    return obj.get('uploaded_folders', {}), obj.get('archives', {})


def load_journal(journal_path: Path):
    updates: Dict[str, Dict[str, str]] = {}
    if not journal_path.exists():
        return updates
    for line in journal_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        row_updates = {k: str(v) for k, v in (obj.get('updates') or {}).items()}
        for vid in obj.get('video_ids') or []:
            updates[str(vid)] = row_updates
    return updates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-metadata-csv', type=Path, required=True)
    ap.add_argument('--output-metadata-csv', type=Path, required=True)
    ap.add_argument('--raw-video-dir', type=Path, required=True)
    ap.add_argument('--scratch-raw-video-dir', type=Path, default=None)
    ap.add_argument('--raw-caption-dir', type=Path, required=True)
    ap.add_argument('--raw-metadata-dir', type=Path, required=True)
    ap.add_argument('--dataset-dir', type=Path, required=True)
    ap.add_argument('--scratch-dataset-dir', type=Path, default=None)
    ap.add_argument('--progress-path', type=Path, required=True)
    ap.add_argument('--status-journal-path', type=Path, required=True)
    args = ap.parse_args()

    source_rows, source_fields = read_csv_rows(args.source_metadata_csv)
    output_rows, output_fields = (read_csv_rows(args.output_metadata_csv) if args.output_metadata_csv.exists() else ([], []))
    fields: List[str] = []
    for col in DEFAULT_COLUMNS + source_fields + output_fields:
        if col and col not in fields:
            fields.append(col)

    out_by_id = {r.get('video_id', '').strip(): r for r in output_rows if r.get('video_id', '').strip()}
    rows: List[Dict[str, str]] = []
    for src in source_rows:
        vid = (src.get('video_id') or '').strip()
        if not vid:
            continue
        merged = {k: src.get(k, '') for k in fields}
        if vid in out_by_id:
            for k in fields:
                if k in out_by_id[vid]:
                    merged[k] = out_by_id[vid].get(k, '')
        rows.append(merged)

    raw_videos = {}
    if args.raw_video_dir.exists():
        raw_videos.update({p.stem: p for p in args.raw_video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS})
    if args.scratch_raw_video_dir is not None and args.scratch_raw_video_dir.exists():
        for p in args.scratch_raw_video_dir.iterdir():
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.stem not in raw_videos:
                raw_videos[p.stem] = p
    raw_metadata = {p.stem: p for p in args.raw_metadata_dir.glob('*.json')} if args.raw_metadata_dir.exists() else {}
    complete = complete_video_ids(args.dataset_dir, args.scratch_dataset_dir)
    process_claims_dir = args.dataset_dir.parent / 'slurm' / 'state' / 'claims'
    download_claims_dir = args.dataset_dir.parent / 'slurm' / 'state' / 'download_claims'
    process_claims = {p.stem for p in process_claims_dir.glob('*.claim')} if process_claims_dir.exists() else set()
    download_claims = {p.stem for p in download_claims_dir.glob('*.claim')} if download_claims_dir.exists() else set()
    uploaded_folders, _archives = load_progress(args.progress_path)
    journal_updates = load_journal(args.status_journal_path)

    for row in rows:
        vid = (row.get('video_id') or '').strip()
        if not vid:
            continue
        if vid in raw_metadata:
            row['raw_metadata_path'] = str(raw_metadata[vid])
            row['metadata_status'] = 'ok'
        if vid in raw_videos:
            row['raw_video_path'] = str(raw_videos[vid])
            row['download_status'] = 'ok'
        elif vid in download_claims and row.get('download_status', '') not in {'ok', 'skipped'}:
            row['download_status'] = 'running'
        if vid in complete:
            row['process_status'] = 'ok'
        elif vid in process_claims and row.get('process_status', '') != 'ok':
            row['process_status'] = 'running'
        if vid in uploaded_folders:
            row['upload_status'] = 'uploaded'
            row['archive_name'] = uploaded_folders[vid]
            row['local_cleanup_status'] = 'deleted'
            row['process_status'] = 'ok'
            row['download_status'] = 'ok'
            if not row.get('metadata_status'):
                row['metadata_status'] = 'ok'
        elif vid in complete:
            row['upload_status'] = ''
            row['archive_name'] = ''
            row['local_cleanup_status'] = ''
        elif vid in journal_updates:
            for k, v in journal_updates[vid].items():
                if k in {'upload_status', 'archive_name', 'local_cleanup_status'}:
                    row[k] = v

    write_csv_rows(args.output_metadata_csv, rows, fields)
    print(f'synced_rows={len(rows)}')

if __name__ == '__main__':
    main()
