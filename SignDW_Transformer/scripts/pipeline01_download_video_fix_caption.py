#!/usr/bin/env python3

import argparse
import csv
import fcntl
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.stats_npz import update_video_stats_best_effort
from utils.raw_video_pool import (
    choose_download_target,
    cleanup_partial_downloads as cleanup_partial_downloads_pool,
    find_video_file as find_video_file_pool,
    release_download_reservation,
)


DEFAULT_SOURCE_METADATA_CSV = REPO_ROOT / "SignVerse-2M-metadata_ori.csv"
DEFAULT_OUTPUT_METADATA_CSV = REPO_ROOT / "SignVerse-2M-metadata_processed.csv"
DEFAULT_RAW_VIDEO_DIR = REPO_ROOT / "raw_video"
DEFAULT_RAW_CAPTION_DIR = REPO_ROOT / "raw_caption"
DEFAULT_RAW_METADATA_DIR = REPO_ROOT / "raw_metadata"
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_STATS_NPZ = REPO_ROOT / "stats.npz"
DEFAULT_STATUS_JOURNAL_PATH = REPO_ROOT / "upload_status_journal.jsonl"
DEFAULT_YT_DLP_EXTRACTOR_ARGS = "youtube:player_client=web_safari,web"
COOKIE_DOMAINS = ("youtube.com", "google.com", "googlevideo.com", "ytimg.com")
TIMESTAMP_LINE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]+")
DEFAULT_COLUMNS = [
    "video_id",
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
    "failure_count",
    "error",
    "processed_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw videos and enrich the SignVerse-2M metadata CSV."
    )
    parser.add_argument("--source-metadata-csv", type=Path, default=DEFAULT_SOURCE_METADATA_CSV)
    parser.add_argument("--output-metadata-csv", type=Path, default=DEFAULT_OUTPUT_METADATA_CSV)
    parser.add_argument("--raw-video-dir", type=Path, default=DEFAULT_RAW_VIDEO_DIR)
    parser.add_argument("--scratch-raw-video-dir", type=Path, default=None)
    parser.add_argument("--home-raw-video-limit", type=int, default=180)
    parser.add_argument("--scratch-raw-video-limit", type=int, default=2800)
    parser.add_argument("--raw-caption-dir", type=Path, default=DEFAULT_RAW_CAPTION_DIR)
    parser.add_argument("--raw-metadata-dir", type=Path, default=DEFAULT_RAW_METADATA_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--stats-npz", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--status-journal-path", type=Path, default=DEFAULT_STATUS_JOURNAL_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--force-metadata", action="store_true")
    parser.add_argument("--force-subtitles", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-video-download", action="store_true")
    parser.add_argument("--skip-subtitles", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--cookies", type=Path, default=None)
    parser.add_argument("--cookies-from-browser", default=None)
    parser.add_argument("--extractor-args", default=DEFAULT_YT_DLP_EXTRACTOR_ARGS)
    parser.add_argument("--max-failures-before-skip", type=int, default=2)
    parser.add_argument("--claim-dir", type=Path, default=None)
    parser.add_argument("--csv-lock-path", type=Path, default=None)
    return parser.parse_args()


def read_manifest(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    csv.field_size_limit(min(sys.maxsize, max(csv.field_size_limit(), 10 * 1024 * 1024)))

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return [], DEFAULT_COLUMNS.copy()

    first = rows[0]
    if first and first[0] == "video_id":
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            manifest_rows = [dict(row) for row in reader]
            fieldnames = list(reader.fieldnames or [])
    else:
        manifest_rows = []
        for row in rows:
            if not row:
                continue
            manifest_rows.append(
                {
                    "video_id": row[0].strip(),
                    "sign_language": row[1].strip() if len(row) > 1 else "",
                }
            )
        fieldnames = []

    ordered_fieldnames = []
    for column in DEFAULT_COLUMNS + fieldnames:
        if column and column not in ordered_fieldnames:
            ordered_fieldnames.append(column)

    for row in manifest_rows:
        for column in ordered_fieldnames:
            row.setdefault(column, "")

    return manifest_rows, ordered_fieldnames


def read_state_manifest(source_csv: Path, output_csv: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    source_rows, source_fieldnames = read_manifest(source_csv)
    if not output_csv.exists():
        return source_rows, source_fieldnames

    output_rows, output_fieldnames = read_manifest(output_csv)
    ordered_fieldnames: List[str] = []
    for column in list(source_fieldnames) + list(output_fieldnames):
        if column and column not in ordered_fieldnames:
            ordered_fieldnames.append(column)

    merged_rows: List[Dict[str, str]] = []
    output_by_id = {row.get("video_id", "").strip(): row for row in output_rows if row.get("video_id", "").strip()}
    seen = set()
    for source_row in source_rows:
        video_id = source_row.get("video_id", "").strip()
        if not video_id:
            continue
        merged = {column: source_row.get(column, "") for column in ordered_fieldnames}
        if video_id in output_by_id:
            merge_row_values(merged, output_by_id[video_id], ordered_fieldnames)
        merged_rows.append(merged)
        seen.add(video_id)

    for video_id, row in output_by_id.items():
        if video_id in seen:
            continue
        merged = {column: row.get(column, "") for column in ordered_fieldnames}
        merged_rows.append(merged)

    return merged_rows, ordered_fieldnames


def write_manifest(csv_path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})
    tmp_path.replace(csv_path)


def lock_path_for_manifest(output_csv: Path, explicit_lock_path: Path | None) -> Path:
    return explicit_lock_path or output_csv.with_suffix(output_csv.suffix + ".lock")


def claim_path_for_video(claim_dir: Path, video_id: str) -> Path:
    return claim_dir / f"{video_id}.claim"


def with_manifest_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def merge_row_values(target_row: Dict[str, str], updates: Dict[str, str], fieldnames: Sequence[str]) -> None:
    for field in fieldnames:
        if field in updates:
            target_row[field] = updates.get(field, "")


def claim_target_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str], Path]:
    manifest_input_path = args.output_metadata_csv if args.output_metadata_csv.exists() else args.source_metadata_csv
    claim_dir = args.claim_dir
    if claim_dir is None:
        rows, fieldnames = read_state_manifest(args.source_metadata_csv, args.output_metadata_csv)
        local_video_ids = collect_local_video_ids(args) if args.local_only else None
        selected_rows = iter_target_rows(rows, args.video_ids, args.limit, local_video_ids, args)
        return rows, selected_rows, fieldnames, manifest_input_path

    claim_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_path_for_manifest(args.output_metadata_csv, args.csv_lock_path)
    handle = with_manifest_lock(lock_path)
    try:
        rows, fieldnames = read_state_manifest(args.source_metadata_csv, args.output_metadata_csv)
        local_video_ids = collect_local_video_ids(args) if args.local_only else None
        selected_rows: List[Dict[str, str]] = []
        video_id_filter = set(args.video_ids or [])
        limit = args.limit
        for row in rows:
            video_id = row["video_id"].strip()
            if not video_id:
                continue
            if video_id_filter and video_id not in video_id_filter:
                continue
            if local_video_ids is not None and video_id not in local_video_ids:
                continue
            if not row_needs_processing(row, args):
                continue
            claim_path = claim_path_for_video(claim_dir, video_id)
            try:
                fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as claim_handle:
                claim_handle.write(f"pid={os.getpid()}\n")
                claim_handle.write(f"video_id={video_id}\n")
                claim_handle.write(f"claimed_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            selected_rows.append(dict(row))
            if limit is not None and len(selected_rows) >= limit:
                break
        return rows, selected_rows, fieldnames, manifest_input_path
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def release_claim(args: argparse.Namespace, video_id: str) -> None:
    if args.claim_dir is None:
        return
    claim_path_for_video(args.claim_dir, video_id).unlink(missing_ok=True)


def persist_row_update(args: argparse.Namespace, video_id: str, updated_row: Dict[str, str], fieldnames: Sequence[str]) -> None:
    lock_path = lock_path_for_manifest(args.output_metadata_csv, args.csv_lock_path)
    handle = with_manifest_lock(lock_path)
    try:
        rows, current_fieldnames = read_state_manifest(args.source_metadata_csv, args.output_metadata_csv)
        ordered_fieldnames = []
        for column in list(current_fieldnames) + list(fieldnames):
            if column and column not in ordered_fieldnames:
                ordered_fieldnames.append(column)
        found = False
        for row in rows:
            if row.get("video_id", "").strip() == video_id:
                merge_row_values(row, updated_row, ordered_fieldnames)
                found = True
                break
        if not found:
            new_row = {column: updated_row.get(column, "") for column in ordered_fieldnames}
            rows.append(new_row)
        write_manifest(args.output_metadata_csv, rows, ordered_fieldnames)
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def repo_relative_or_absolute(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved_path)


def build_yt_dlp_base_command(args: argparse.Namespace) -> List[str]:
    command = ["yt-dlp", "--newline", "--remote-components", "ejs:github"]
    js_runtime = resolve_js_runtime()
    if js_runtime is not None:
        command.extend(["--js-runtimes", js_runtime])
    if getattr(args, "_effective_cookies", None):
        command.extend(["--cookies", str(args._effective_cookies)])
    if args.cookies_from_browser:
        command.extend(["--cookies-from-browser", args.cookies_from_browser])
    if args.extractor_args:
        command.extend(["--extractor-args", args.extractor_args])
    return command


def resolve_runtime_binary(name: str) -> Path | None:
    runtime_path = shutil.which(name)
    if runtime_path:
        return Path(runtime_path)

    python_path = Path(sys.executable).resolve()
    for parent in python_path.parents:
        fallback = parent / "bin" / name
        if fallback.exists():
            return fallback
    return None


def node_version_is_supported(node_binary: Path) -> bool:
    try:
        result = subprocess.run([str(node_binary), "--version"], capture_output=True, text=True, check=False)
    except OSError:
        return False
    version_text = (result.stdout or result.stderr).strip().lstrip("v")
    if not version_text:
        return False
    major_text = version_text.split(".", 1)[0]
    try:
        return int(major_text) >= 20
    except ValueError:
        return False


def resolve_js_runtime() -> str | None:
    deno_binary = resolve_runtime_binary("deno")
    if deno_binary is not None:
        return f"deno:{deno_binary}"

    node_binary = resolve_runtime_binary("node")
    if node_binary is not None and node_version_is_supported(node_binary):
        return f"node:{node_binary}"
    return None


def sanitize_cookie_file(cookie_path: Path) -> Path:
    tmp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        prefix="sign_dwpose_cookies_",
        delete=False,
    )
    with cookie_path.open("r", encoding="utf-8", errors="ignore") as src, tmp_handle as dst:
        wrote_header = False
        for raw_line in src:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                if not wrote_header:
                    dst.write("# Netscape HTTP Cookie File\n")
                    wrote_header = True
                continue

            parts = line.split("\t")
            if len(parts) < 7:
                continue
            domain = parts[0].strip()
            if not any(token in domain for token in COOKIE_DOMAINS):
                continue
            parts[1] = "TRUE" if domain.startswith(".") else "FALSE"
            dst.write("\t".join(parts[:7]) + "\n")
    return Path(tmp_handle.name)


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True)


def youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def fetch_metadata(video_id: str, args: argparse.Namespace) -> Tuple[Dict[str, object], str]:
    command = build_yt_dlp_base_command(args)
    command.extend(["-J", "--skip-download", youtube_url(video_id)])
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "yt-dlp metadata failed")
    return json.loads(result.stdout), ""


def caption_entry_has_translation(entry: Dict[str, object]) -> bool:
    url = str(entry.get("url") or "")
    if not url:
        return False
    return "tlang" in parse_qs(urlparse(url).query)


def filter_caption_languages(metadata: Dict[str, object]) -> Tuple[List[str], List[str], str]:
    manual = sorted(
        lang
        for lang in (metadata.get("subtitles") or {}).keys()
        if lang and lang != "live_chat"
    )
    native_automatic: List[str] = []
    english_translations: List[str] = []
    for lang, entries in sorted((metadata.get("automatic_captions") or {}).items()):
        if not lang or lang == "live_chat" or lang in manual:
            continue
        entry_list = entries if isinstance(entries, list) else []
        if any(isinstance(entry, dict) and caption_entry_has_translation(entry) for entry in entry_list):
            if lang.startswith("en-"):
                english_translations.append(lang)
            continue
        native_automatic.append(lang)

    english_translation = ""
    if english_translations:
        preferred_sources = manual + native_automatic
        for source_lang in preferred_sources:
            candidate = f"en-{source_lang}"
            if candidate in english_translations:
                english_translation = candidate
                break
        if not english_translation:
            english_translation = sorted(english_translations)[0]
    elif manual or native_automatic:
        english_translation = f"en-{(manual + native_automatic)[0]}"
    return manual, native_automatic, english_translation


def download_subtitles(
    video_id: str,
    subtitle_dir: Path,
    manual_langs: Sequence[str],
    native_automatic_langs: Sequence[str],
    english_translation_lang: str,
    args: argparse.Namespace,
) -> str:
    subtitle_dir.mkdir(parents=True, exist_ok=True)
    automatic_langs = list(native_automatic_langs)
    if english_translation_lang:
        automatic_langs.append(english_translation_lang)
    requested_langs = list(manual_langs) + automatic_langs
    if not requested_langs:
        return ""

    command = build_yt_dlp_base_command(args)
    command.extend(
        [
            "--skip-download",
            "--sub-format",
            "vtt",
            "--convert-subs",
            "vtt",
            "--output",
            str(subtitle_dir / "%(id)s.%(ext)s"),
            "--sub-langs",
            ",".join(requested_langs),
        ]
    )
    if manual_langs:
        command.append("--write-subs")
    if automatic_langs:
        command.append("--write-auto-subs")
    command.append(youtube_url(video_id))

    result = run_command(command)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        if any(subtitle_dir.glob(f"{video_id}.*.vtt")):
            return stderr or stdout
        raise RuntimeError(stderr or stdout or "subtitle download failed")
    return ""


def subtitle_dir_for_video(dataset_dir: Path, video_id: str) -> Path:
    return dataset_dir / video_id / "captions"


def find_video_file(raw_video_dir: Path, video_id: str, scratch_raw_video_dir: Path | None = None) -> Path | None:
    return find_video_file_pool(video_id, raw_video_dir, scratch_raw_video_dir)


def iter_partial_download_files(raw_video_dir: Path, video_id: str, scratch_raw_video_dir: Path | None = None) -> Iterable[Path]:
    yield from cleanup_partial_downloads_pool.__globals__['iter_partial_download_files'](video_id, raw_video_dir, scratch_raw_video_dir)


def cleanup_partial_downloads(raw_video_dir: Path, video_id: str, scratch_raw_video_dir: Path | None = None) -> None:
    cleanup_partial_downloads_pool(video_id, raw_video_dir, scratch_raw_video_dir)


def download_video(video_id: str, raw_video_dir: Path, args: argparse.Namespace) -> Tuple[str, str]:
    target_raw_video_dir, reservation_path = choose_download_target(
        raw_video_dir,
        args.scratch_raw_video_dir,
        args.home_raw_video_limit,
        args.scratch_raw_video_limit,
        args.claim_dir,
        video_id,
    )
    target_raw_video_dir.mkdir(parents=True, exist_ok=True)
    try:
        cleanup_partial_downloads(raw_video_dir, video_id, args.scratch_raw_video_dir)
        command = build_yt_dlp_base_command(args)
        command.extend(
            [
                "--output",
                str(target_raw_video_dir / "%(id)s.%(ext)s"),
                "--format",
                "worstvideo*+worstaudio/worst",
                "--format-sort",
                "+res,+size,+br,+fps",
                "--merge-output-format",
                "mp4",
                youtube_url(video_id),
            ]
        )
        result = run_command(command)
        video_path = find_video_file(raw_video_dir, video_id, args.scratch_raw_video_dir)
        cleanup_partial_downloads(raw_video_dir, video_id, args.scratch_raw_video_dir)
        if result.returncode != 0 and not video_path:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "video download failed")
        return repo_relative_or_absolute(video_path) if video_path else "", ""
    finally:
        release_download_reservation(reservation_path)


def subtitle_file_language(path: Path, video_id: str) -> str:
    name = path.name
    prefix = f"{video_id}."
    if not name.startswith(prefix):
        return ""
    middle = name[len(prefix):]
    if middle.endswith(".vtt"):
        middle = middle[:-4]
    return middle


def clean_vtt_to_text(path: Path) -> str:
    return normalize_subtitle_lines(extract_vtt_text_lines(path))


def extract_vtt_text_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "WEBVTT" or line.startswith("NOTE") or line.startswith("Kind:") or line.startswith("Language:"):
                continue
            if TIMESTAMP_LINE_RE.match(line) or "-->" in line:
                continue
            if line.isdigit():
                continue
            line = TAG_RE.sub("", line)
            line = ZERO_WIDTH_RE.sub("", html.unescape(line)).strip()
            if not line:
                continue
            lines.append(line)
    return lines


def normalize_subtitle_lines(lines: Sequence[str]) -> str:
    normalized_lines: List[str] = []
    for line in lines:
        if not line:
            continue
        if normalized_lines and normalized_lines[-1] == line:
            continue
        if normalized_lines and line[:1] in {",", ".", "!", "?", ";", ":", "%", ")", "]", "}"}:
            normalized_lines[-1] = normalized_lines[-1].rstrip() + line
            continue
        if normalized_lines and normalized_lines[-1].endswith(("-", "–", "—", "/")):
            line = line.lstrip("-–—").lstrip()
            normalized_lines[-1] = normalized_lines[-1].rstrip() + " " + line
            continue
        if normalized_lines and normalized_lines[-1].endswith("..."):
            line = line.lstrip(".").lstrip()
            normalized_lines[-1] = normalized_lines[-1].rstrip() + " " + line
            continue
        normalized_lines.append(line)

    text = " ".join(normalized_lines)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*-\s*-\s*", " - ", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"\s*\.\.\.\s*", " ... ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text


def timestamp_to_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_vtt_segments(path: Path) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    current_start = ""
    current_end = ""
    current_lines: List[str] = []

    def flush_segment() -> None:
        nonlocal current_start, current_end, current_lines
        if not current_start or not current_end:
            current_lines = []
            return
        text = normalize_subtitle_lines(current_lines)
        if text:
            segments.append(
                {
                    "start_sec": round(timestamp_to_seconds(current_start), 3),
                    "end_sec": round(timestamp_to_seconds(current_end), 3),
                    "text": text,
                }
            )
        current_start = ""
        current_end = ""
        current_lines = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                flush_segment()
                continue
            if line == "WEBVTT" or line.startswith("NOTE") or line.startswith("Kind:") or line.startswith("Language:"):
                continue
            match = TIMESTAMP_LINE_RE.match(line)
            if match:
                flush_segment()
                current_start = match.group("start")
                current_end = match.group("end")
                continue
            if line.isdigit():
                continue
            line = TAG_RE.sub("", line)
            line = ZERO_WIDTH_RE.sub("", html.unescape(line)).strip()
            if not line:
                continue
            current_lines.append(line)
    flush_segment()
    return segments


def load_subtitle_payloads(subtitle_dir: Path, video_id: str) -> Dict[str, Dict[str, object]]:
    subtitle_payloads: Dict[str, Dict[str, object]] = {}
    for path in sorted(subtitle_dir.glob(f"{video_id}.*.vtt")):
        lang = subtitle_file_language(path, video_id)
        if not lang:
            continue
        segments = parse_vtt_segments(path)
        text = normalize_subtitle_lines(segment["text"] for segment in segments)
        if text:
            subtitle_payloads[lang] = {
                "text": text,
                "segments": segments,
                "vtt_path": path.name,
            }
    return subtitle_payloads


def select_english_subtitle(subtitle_payloads: Dict[str, Dict[str, object]]) -> Tuple[str, str]:
    if "en" in subtitle_payloads:
        return str(subtitle_payloads["en"]["text"]), "manual_or_auto_en"
    english_variants = sorted(
        lang for lang in subtitle_payloads if lang.startswith("en-") or lang.lower().startswith("en_")
    )
    if english_variants:
        lang = english_variants[0]
        source_lang = lang[3:]
        return str(subtitle_payloads[lang]["text"]), f"translated_from_{source_lang}"
    translated_candidates = sorted(lang for lang in subtitle_payloads if lang.endswith("-en"))
    if translated_candidates:
        lang = translated_candidates[0]
        return str(subtitle_payloads[lang]["text"]), f"translated_from_{lang[:-3]}"
    return "", ""


def subtitle_json_path_for_video(subtitle_dir: Path, video_id: str) -> Path:
    return subtitle_dir / f"{video_id}.captions.json"


def build_subtitle_json_payload(
    video_id: str,
    subtitle_payloads: Dict[str, Dict[str, object]],
    subtitle_en: str,
    subtitle_en_source: str,
) -> Dict[str, object]:
    languages: Dict[str, Dict[str, object]] = {}
    compact_texts: Dict[str, str] = {}
    for lang, payload in sorted(subtitle_payloads.items()):
        compact_texts[lang] = str(payload.get("text") or "")
        languages[lang] = {
            "text": compact_texts[lang],
            "segments": payload.get("segments") or [],
            "vtt_path": payload.get("vtt_path") or "",
        }
    return {
        "video_id": video_id,
        "subtitle_languages": sorted(subtitle_payloads),
        "subtitle_en": subtitle_en,
        "subtitle_en_source": subtitle_en_source,
        "subtitle_texts": compact_texts,
        "languages": languages,
    }


def write_subtitle_json(
    subtitle_dir: Path,
    video_id: str,
    subtitle_payloads: Dict[str, Dict[str, object]],
    subtitle_en: str,
    subtitle_en_source: str,
) -> Path | None:
    json_path = subtitle_json_path_for_video(subtitle_dir, video_id)
    if not subtitle_payloads:
        json_path.unlink(missing_ok=True)
        return None
    payload = build_subtitle_json_payload(video_id, subtitle_payloads, subtitle_en, subtitle_en_source)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def persist_raw_metadata(raw_metadata_dir: Path, video_id: str, metadata: Dict[str, object]) -> str:
    raw_metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = raw_metadata_dir / f"{video_id}.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return repo_relative_or_absolute(metadata_path)


def iter_target_rows(
    rows: Sequence[Dict[str, str]],
    video_ids: Iterable[str] | None,
    limit: int | None,
    local_video_ids: set[str] | None = None,
    args: argparse.Namespace | None = None,
) -> List[Dict[str, str]]:
    video_id_filter = set(video_ids or [])
    selected = []
    for row in rows:
        video_id = row["video_id"]
        if video_id_filter and video_id not in video_id_filter:
            continue
        if local_video_ids is not None and video_id not in local_video_ids:
            continue
        if args is not None and not row_needs_processing(row, args):
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def collect_local_video_ids(args: argparse.Namespace) -> set[str]:
    local_video_ids: set[str] = set()
    if args.dataset_dir.exists():
        for path in args.dataset_dir.iterdir():
            if not path.is_dir():
                continue
            if (path / "captions").exists():
                local_video_ids.add(path.name)
    if args.raw_metadata_dir.exists():
        local_video_ids.update(path.stem for path in args.raw_metadata_dir.glob("*.json"))
    return local_video_ids


def row_failure_count(row: Dict[str, str]) -> int:
    try:
        return int((row.get("failure_count") or "0").strip() or "0")
    except ValueError:
        return 0


def record_row_failure(row: Dict[str, str], stats_record: Dict[str, object], error_text: str, max_failures_before_skip: int) -> tuple[int, bool]:
    failure_count = row_failure_count(row) + 1
    row["failure_count"] = str(failure_count)
    stats_record["failure_count"] = row["failure_count"]
    should_skip = failure_count >= max_failures_before_skip
    if should_skip:
        row["download_status"] = "skipped"
        row["error"] = f"{error_text} | skipped after {failure_count} failures" if error_text else f"skipped after {failure_count} failures"
        stats_record["download_status"] = row["download_status"]
        stats_record["last_error"] = row["error"]
    return failure_count, should_skip


def reset_row_failures(row: Dict[str, str], stats_record: Dict[str, object]) -> None:
    row["failure_count"] = "0"
    stats_record["failure_count"] = row["failure_count"]


def row_needs_processing(row: Dict[str, str], args: argparse.Namespace) -> bool:
    if args.force_metadata or args.force_subtitles or args.force_download:
        return True
    if row_failure_count(row) >= args.max_failures_before_skip:
        return False
    metadata_status = (row.get("metadata_status") or "").strip()
    subtitle_status = (row.get("subtitle_status") or "").strip()
    download_status = (row.get("download_status") or "").strip()

    needs_metadata = metadata_status != "ok"
    # Treat missing subtitles as a terminal state so videos without captions are not retried forever.
    needs_subtitles = (not args.skip_subtitles) and subtitle_status not in {"ok", "missing", "skipped"}
    needs_download = (not args.skip_video_download) and download_status != "ok"
    return needs_metadata or needs_subtitles or needs_download


def main() -> None:
    args = parse_args()
    temp_cookie_path: Path | None = None
    if args.cookies:
        temp_cookie_path = sanitize_cookie_file(args.cookies)
        args._effective_cookies = temp_cookie_path
    else:
        args._effective_cookies = None
    rows, selected_rows, fieldnames, _manifest_input_path = claim_target_rows(args)

    try:
        args.raw_video_dir.mkdir(parents=True, exist_ok=True)
        args.raw_metadata_dir.mkdir(parents=True, exist_ok=True)
        args.dataset_dir.mkdir(parents=True, exist_ok=True)

        for index, row in enumerate(selected_rows, start=1):
            video_id = row["video_id"].strip()
            if not video_id:
                continue
            stats_record = {}

            print(f"[{index}/{len(selected_rows)}] Processing {video_id}")
            metadata_error = ""
            subtitle_error = ""
            download_error = ""
            metadata: Dict[str, object] | None = None

            try:
                metadata_path = args.raw_metadata_dir / f"{video_id}.json"
                if metadata_path.exists() and not args.force_metadata:
                    with metadata_path.open("r", encoding="utf-8") as handle:
                        metadata = json.load(handle)
                else:
                    metadata, metadata_error = fetch_metadata(video_id, args)
                    row["raw_metadata_path"] = persist_raw_metadata(args.raw_metadata_dir, video_id, metadata)
                row["metadata_status"] = "ok"
            except Exception as exc:
                metadata_error = str(exc)
                row["metadata_status"] = "failed"
                row["error"] = metadata_error
                row["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                stats_record["metadata_status"] = "failed"
                stats_record["updated_at"] = row["processed_at"]
                failure_count, should_skip = record_row_failure(row, stats_record, metadata_error, args.max_failures_before_skip)
                stats_record["last_error"] = row["error"]
                persist_row_update(args, video_id, row, fieldnames)
                update_video_stats_best_effort(args.stats_npz, args.status_journal_path, video_id, **stats_record)
                print(f"  metadata failed: {metadata_error}")
                if should_skip:
                    print(f"  skipping after {failure_count} failures")
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
                continue

            assert metadata is not None
            row["raw_metadata_path"] = row.get("raw_metadata_path") or persist_raw_metadata(args.raw_metadata_dir, video_id, metadata)
            row["title"] = str(metadata.get("title") or "")
            duration = metadata.get("duration")
            row["duration_sec"] = str(duration or "")
            row["start_sec"] = "0"
            row["end_sec"] = str(duration or "")

            manual_langs, native_automatic_langs, english_translation_lang = filter_caption_languages(metadata)
            subtitle_dir = subtitle_dir_for_video(args.dataset_dir, video_id)
            subtitle_payloads: Dict[str, Dict[str, object]] = {}

            if args.skip_subtitles:
                row["subtitle_status"] = "skipped"
            else:
                try:
                    need_subtitles = args.force_subtitles or not any(subtitle_dir.glob(f"{video_id}.*.vtt"))
                    if need_subtitles:
                        subtitle_error = download_subtitles(
                            video_id,
                            subtitle_dir,
                            manual_langs,
                            native_automatic_langs,
                            english_translation_lang,
                            args,
                        )
                    subtitle_payloads = load_subtitle_payloads(subtitle_dir, video_id)
                    row["subtitle_status"] = "ok" if subtitle_payloads else "missing"
                except Exception as exc:
                    subtitle_error = str(exc)
                    subtitle_payloads = load_subtitle_payloads(subtitle_dir, video_id)
                    row["subtitle_status"] = "partial" if subtitle_payloads else "failed"

            row["subtitle_languages"] = "|".join(sorted(subtitle_payloads))
            row["subtitle_dir_path"] = repo_relative_or_absolute(subtitle_dir) if subtitle_payloads else ""
            subtitle_en, subtitle_en_source = select_english_subtitle(subtitle_payloads)
            row["subtitle_en_source"] = subtitle_en_source
            if "subtitle_texts_json" in row:
                row["subtitle_texts_json"] = ""
            if "subtitle_en" in row:
                row["subtitle_en"] = ""
            if "subtitle_json_path" in row:
                row["subtitle_json_path"] = ""
            if "raw_caption_dir" in row:
                row["raw_caption_dir"] = ""

            try:
                existing_video = find_video_file(args.raw_video_dir, video_id, args.scratch_raw_video_dir)
                if args.skip_video_download:
                    row["download_status"] = "skipped"
                    row["raw_video_path"] = repo_relative_or_absolute(existing_video) if existing_video else ""
                else:
                    if existing_video is None or args.force_download:
                        row["raw_video_path"], download_error = download_video(video_id, args.raw_video_dir, args)
                    else:
                        row["raw_video_path"] = repo_relative_or_absolute(existing_video)
                    row["download_status"] = "ok" if row["raw_video_path"] else "failed"
            except Exception as exc:
                download_error = str(exc)
                row["download_status"] = "failed"

            errors = [value for value in [metadata_error, subtitle_error, download_error] if value]
            row["error"] = " | ".join(errors)
            row["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            stats_record["sign_language"] = row.get("sign_language", "")
            stats_record["title"] = row["title"]
            stats_record["duration_sec"] = row["duration_sec"]
            stats_record["start_sec"] = row["start_sec"]
            stats_record["end_sec"] = row["end_sec"]
            stats_record["subtitle_languages"] = row["subtitle_languages"]
            stats_record["subtitle_dir_path"] = row["subtitle_dir_path"]
            stats_record["subtitle_en_source"] = row["subtitle_en_source"]
            stats_record["raw_video_path"] = row["raw_video_path"]
            stats_record["raw_metadata_path"] = row["raw_metadata_path"]
            stats_record["metadata_status"] = row["metadata_status"]
            stats_record["subtitle_status"] = row["subtitle_status"]
            stats_record["download_status"] = row["download_status"]
            if errors:
                failure_count, should_skip = record_row_failure(row, stats_record, row["error"], args.max_failures_before_skip)
            else:
                reset_row_failures(row, stats_record)
                failure_count, should_skip = 0, False
            stats_record["download_status"] = row["download_status"]
            stats_record["last_error"] = row["error"]
            stats_record["updated_at"] = row["processed_at"]
            persist_row_update(args, video_id, row, fieldnames)
            update_video_stats_best_effort(args.stats_npz, args.status_journal_path, video_id, **stats_record)

            if row["download_status"] == "failed":
                print(f"  video download failed: {download_error}")
            if row["subtitle_status"] in {"failed", "partial"}:
                print(f"  subtitle status: {row['subtitle_status']} {subtitle_error}")
            if should_skip:
                print(f"  skipping after {failure_count} failures")

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    finally:
        if temp_cookie_path is not None:
            temp_cookie_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
