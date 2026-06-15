#!/bin/bash
# Stage 2 (API explanations) using a LOCAL Gemma-3 annotator served by SGLang,
# instead of the Anthropic API. Brings up the server, waits for health, runs
# stage 2 on the two SFT splits (av_sft, ar_sft), then tears the server down.
#
# RL split is intentionally skipped — RL never uses explanations.
#
# Submit:  sbatch scripts/datagen/stage2_local_gemma.sbatch
# Smoke:   SMOKE_ROWS=500 sbatch scripts/datagen/stage2_local_gemma.sbatch
# Full:    sbatch --time=06:00:00 scripts/datagen/stage2_local_gemma.sbatch
#
# ONE-TIME ENV SETUP (login node, needs internet):
#   module load Miniconda3
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   conda create -y -p /projects/sanghani_nlp/envs/nla-stage2 python=3.11
#   E=/projects/sanghani_nlp/envs/nla-stage2
#   "$E/bin/python" --version    # must say 3.11.x
#   "$E/bin/pip" install --upgrade pip
#   "$E/bin/pip" install "sglang[all]>=0.5.6" pyarrow pyyaml orjson httpx anthropic tqdm
# Then submit with: CONDA_ENV=/projects/sanghani_nlp/envs/nla-stage2 sbatch ...
#
# NOTE: conda activate / source activate are broken on this cluster (change the
# prompt but not PATH). The script activates by PATH-prepend instead.
#
# NOTE: CUDA_HOME must be set for sglang JIT kernel compilation. The script
# tries common CUDA module names and falls back to --disable-cuda-graph if none
# work. Find the right name with: module avail cuda
#
#SBATCH --job-name=nla_stage2_gemma
#SBATCH --account=sanghani_nlp
#SBATCH --partition=a100_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
# ^^^ 30 min default: enough for smoke (server ~3 min + small slice).
# Full ~500k-row run: sbatch --time=06:00:00 ... (shorter limits queue faster).
#SBATCH --output=/projects/sanghani_nlp/logs/%x_%j.out

set -eo pipefail

# ---- knobs (override via env at submit time) --------------------------------
PROJ="${PROJ:-/projects/sanghani_nlp}"
REPO="${REPO:-$PROJ/NLA}"
SPL="${SPL:-$PROJ/activations/qwen7b_fineweb_100k/splits}"
MODEL="${MODEL:-google/gemma-3-27b-it}"
PORT="${PORT:-30000}"
CONCURRENCY="${CONCURRENCY:-64}"
# Full path to the Python 3.11 conda prefix env. Do NOT use env name — conda
# activate is broken on this cluster (changes prompt, not PATH).
CONDA_ENV="${CONDA_ENV:-/projects/sanghani_nlp/envs/nla-stage2}"
# Tensor-parallel degree. 1 for 80 GB A100 (27B bf16 ~54 GB fits).
# Set to 2 (and --gres=gpu:2) for 40 GB nodes.
TP="${TP:-1}"
# Set SMOKE_ROWS=N to run a tiny N-row slice before the full pass.
SMOKE_ROWS="${SMOKE_ROWS:-}"
# ---------------------------------------------------------------------------

mkdir -p "$PROJ/logs"

# ---- CUDA: load toolkit so sglang can JIT-compile kernels and use CUDA graphs
# Without CUDA_HOME the server crashes during graph capture. Try common module
# names; if none work, fall back to --disable-cuda-graph (server still runs,
# just slower). Find the right name with: module avail cuda
module load Miniconda3    # load this first — some clusters need it before cuda
module load CUDA/12.6.0 2>/dev/null \
    || module load cuda/12.6.0 2>/dev/null \
    || module load CUDA 2>/dev/null \
    || module load cuda 2>/dev/null \
    || true
if [ -z "$CUDA_HOME" ] && command -v nvcc &>/dev/null; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [ -n "$CUDA_HOME" ]; then
    echo "[stage2] CUDA_HOME=$CUDA_HOME"
else
    echo "[stage2] WARNING: CUDA_HOME not found — using --disable-cuda-graph (reduced throughput)"
fi

# ---- activate env by PATH-prepend (conda activate is broken on this cluster)
export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"
if [ "$(command -v python)" != "$CONDA_ENV/bin/python" ]; then
    echo "[stage2] ERROR: env not usable — python is $(command -v python), expected $CONDA_ENV/bin/python" >&2
    echo "[stage2] Check that $CONDA_ENV/bin/python exists (was the env created with python=3.11?)." >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$PROJ/hf_cache}"
# Weights pre-downloaded on login node; force offline so compute nodes (which
# may have no internet) don't try to reach HF. Override with HF_HUB_OFFLINE=0.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO"

# Pre-flight: fail fast before wasting GPU time if deps are missing.
python -c "import sglang, pyarrow, httpx, nla.datagen.stage2_api_explain" 2>/dev/null || {
    echo "[stage2] ERROR: env '$CONDA_ENV' is missing deps (sglang / pyarrow / httpx / nla)." >&2
    echo "[stage2] Run the ONE-TIME ENV SETUP from this script's header on the LOGIN node." >&2
    exit 1
}

# ---- launch the SGLang server ---------------------------------------------
# Gemma-3 REQUIRES --attention-backend fa3 (flashinfer OOMs on head_dim=256).
# context-length capped at 4096: stage-2 prompts are ~800 in + 300 out.
echo "[stage2] launching SGLang server for $MODEL (tp=$TP) on port $PORT"
SGLANG_EXTRA_ARGS=""
[ -z "$CUDA_HOME" ] && SGLANG_EXTRA_ARGS="--disable-cuda-graph"
# shellcheck disable=SC2086
python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --tp "$TP" \
    --attention-backend fa3 \
    --context-length 4096 \
    --mem-fraction-static 0.9 \
    $SGLANG_EXTRA_ARGS &
SERVER_PID=$!

# Always kill the server when this script exits (success, failure, or timeout).
cleanup() {
    echo "[stage2] stopping SGLang server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---- wait for the server to be ready (timeout 900s) ----------------------
echo "[stage2] waiting for server health..."
for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "[stage2] server is up after ~$((i*5))s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[stage2] ERROR: server process died during startup — check log above" >&2
        exit 1
    fi
    [ "$i" -eq 180 ] && { echo "[stage2] ERROR: server did not become healthy within 900s" >&2; exit 1; }
    sleep 5
done

# ---- optional smoke slice --------------------------------------------------
SUFFIX=""
if [ -n "$SMOKE_ROWS" ]; then
    echo "[stage2] SMOKE mode: slicing first $SMOKE_ROWS rows of each SFT split"
    SUFFIX="_smoke${SMOKE_ROWS}"
    for s in av_sft ar_sft; do
        python - "$SPL" "$s" "$SMOKE_ROWS" "$SUFFIX" <<'PY'
import shutil, sys
import pyarrow.parquet as pq
spl, s, n, suffix = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
src = f"{spl}/{s}_raw.parquet"
dst = f"{spl}/{s}_raw{suffix}.parquet"
pq.write_table(pq.read_table(src).slice(0, n), dst)
shutil.copy(f"{src}.nla_meta.yaml", f"{dst}.nla_meta.yaml")
print(f"  wrote {dst}")
PY
    done
fi

# ---- run stage 2 on the two SFT splits (NOT rl) ---------------------------
for s in av_sft ar_sft; do
    IN="$SPL/${s}_raw${SUFFIX}.parquet"
    OUT="$SPL/${s}_explained${SUFFIX}.parquet"
    echo "[stage2] explaining $IN -> $OUT"
    python -m nla.datagen.stage2_api_explain \
        --input "$IN" \
        --output "$OUT" \
        --provider-cls nla.datagen.providers.LocalChatProvider \
        --provider-kwargs "{\"base_url\":\"http://localhost:$PORT/v1\",\"model\":\"$MODEL\",\"concurrency\":$CONCURRENCY}"
done

echo "[stage2] done."
