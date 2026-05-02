#!/usr/bin/env bash
#SBATCH --job-name=dwpose-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sf895/SignVerse-2M-runtime/slurm/logs/dwpose_bench_%j.out
#SBATCH --error=/home/sf895/SignVerse-2M-runtime/slurm/logs/dwpose_bench_%j.err

set -euo pipefail

ROOT_DIR="/cache/home/sf895/SignVerse-2M"
RUNTIME_ROOT="/home/sf895/SignVerse-2M-runtime"
CONDA_SH="/home/sf895/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="dwpose"
VIDEO_ID="${1:?video id required}"
FPS="${FPS:-24}"
RAW_VIDEO_DIR="$RUNTIME_ROOT/raw_video"
BENCH_ROOT="$RUNTIME_ROOT/bench_pipeline02/$VIDEO_ID"
OLD_DATASET_DIR="$BENCH_ROOT/jpg_spill"
NEW_DATASET_DIR="$BENCH_ROOT/stream"
OLD_STATS="$BENCH_ROOT/stats_old.npz"
NEW_STATS="$BENCH_ROOT/stats_new.npz"
TMP_ROOT="${SLURM_TMPDIR:-/tmp}"
VIDEO_PATH=""
for ext in mp4 mkv webm mov; do
  candidate="$RAW_VIDEO_DIR/$VIDEO_ID.$ext"
  if [[ -f "$candidate" ]]; then
    VIDEO_PATH="$candidate"
    break
  fi
done
if [[ -z "$VIDEO_PATH" ]]; then
  echo "Video not found for $VIDEO_ID" >&2
  exit 1
fi

mkdir -p "$BENCH_ROOT"
rm -rf "$OLD_DATASET_DIR" "$NEW_DATASET_DIR"
rm -f "$OLD_STATS" "$NEW_STATS"

echo "video_id=$VIDEO_ID"
echo "video_path=$VIDEO_PATH"
echo "hostname=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "fps=$FPS"
echo "tmp_root=$TMP_ROOT"

source "$CONDA_SH"

run_case() {
  local mode="$1"
  local dataset_dir="$2"
  local stats_path="$3"
  shift 3
  local start end elapsed
  start=$(python3 - <<'PY'
import time
print(time.perf_counter())
PY
)
  conda run -n "$CONDA_ENV" python -u "$ROOT_DIR/scripts/pipeline02_extract_dwpose_from_video.py" \
    --raw-video-dir "$RAW_VIDEO_DIR" \
    --dataset-dir "$dataset_dir" \
    --stats-npz "$stats_path" \
    --fps "$FPS" \
    --workers 1 \
    --video-ids="$VIDEO_ID" \
    --force \
    --tmp-root "$TMP_ROOT" \
    "$@"
  end=$(python3 - <<'PY'
import time
print(time.perf_counter())
PY
)
  elapsed=$(python3 - <<PY
start = float("$start")
end = float("$end")
print(f"{end-start:.3f}")
PY
)
  local poses_npz="$dataset_dir/$VIDEO_ID/npz/poses.npz"
  local complete_marker="$dataset_dir/$VIDEO_ID/npz/.complete"
  local size_bytes=0
  if [[ -f "$poses_npz" ]]; then
    size_bytes=$(stat -c %s "$poses_npz")
  fi
  echo "benchmark_result mode=$mode elapsed_seconds=$elapsed poses_npz_bytes=$size_bytes complete=$([[ -f "$complete_marker" ]] && echo yes || echo no)"
}

run_case jpg_spill "$OLD_DATASET_DIR" "$OLD_STATS" --spill-jpg-frames
run_case stream "$NEW_DATASET_DIR" "$NEW_STATS"
