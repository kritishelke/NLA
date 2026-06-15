#!/bin/bash
# Stage 2 (API explanations) using a LOCAL Gemma-3 annotator served by SGLang,
# instead of the Anthropic API. Brings up the server, waits for health, runs
# stage 2 on the two SFT splits (av_sft, ar_sft), then tears the server down.
#
# RL split is intentionally skipped — RL never uses explanations.
#
# Submit:  sbatch scripts/datagen/stage2_local_gemma.sbatch
# Smoke:   SMOKE_ROWS=500 sbatch scripts/datagen/stage2_local_gemma.sbatch
#
# ONE-TIME ENV SETUP (run on the LOGIN node — it has internet; compute nodes
# may not). The env MUST be Python 3.11 — sglang's deps (outlines_core,
# sgl-kernel, flashinfer) have no prebuilt wheels for 3.13 and fall back to
# source builds that need Rust/nvcc. Use `conda activate` (not `source
# activate`, which is a broken shim here):
#   module load Miniconda3
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   conda create -y -p /projects/sanghani_nlp/envs/nla-stage2 python=3.11
#   conda activate /projects/sanghani_nlp/envs/nla-stage2
#   pip install "sglang[all]>=0.5.6" pyarrow pyyaml orjson httpx anthropic tqdm
# Then submit with: CONDA_ENV=/projects/sanghani_nlp/envs/nla-stage2 sbatch ...
#
#SBATCH --job-name=nla_stage2_gemma
#SBATCH --account=sanghani_nlp
#SBATCH --partition=a100_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/projects/sanghani_nlp/logs/%x_%j.out

set -eo pipefail

# ---- knobs (override via env at submit time) --------------------------------
PROJ="${PROJ:-/projects/sanghani_nlp}"
REPO="${REPO:-$PROJ/NLA}"
SPL="${SPL:-$PROJ/activations/qwen7b_fineweb_100k/splits}"
MODEL="${MODEL:-google/gemma-3-27b-it}"
PORT="${PORT:-30000}"
CONCURRENCY="${CONCURRENCY:-64}"
# Prefix env (activate by full path, NOT by name — `source activate hf` fails).
CONDA_ENV="${CONDA_ENV:-/projects/sanghani_nlp/envs/hf}"
# Tensor-parallel degree. Keep 1 for an 80GB A100 (27B bf16 ~54GB fits). Bump to
# 2 (and request --gres=gpu:2) if the node is a 40GB A100.
TP="${TP:-1}"
# Set SMOKE_ROWS=N to run a tiny N-row slice end-to-end before the full pass.
SMOKE_ROWS="${SMOKE_ROWS:-}"
# ---------------------------------------------------------------------------

mkdir -p "$PROJ/logs"

module load Miniconda3
# Use the conda hook + `conda activate`, NOT `source activate`: on this module
# `source activate` is a broken shim in non-interactive shells — it sets the
# prompt but does not prepend the env's bin/ to PATH, silently leaving you on
# the base (3.13) interpreter. The hook makes `conda activate` work properly.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
# Hard guard: confirm we actually switched interpreters (catches a stale prompt
# / failed activation before we waste a GPU allocation).
case "$(command -v python)" in
    "$CONDA_ENV"/bin/python) : ;;
    *) echo "[stage2] ERROR: activation failed — python is $(command -v python), expected $CONDA_ENV/bin/python" >&2; exit 1 ;;
esac

export HF_HOME="${HF_HOME:-$PROJ/hf_cache}"
# Weights are pre-downloaded to HF_HOME on the login node; force offline so the
# compute node never reaches out to HF (it may have no internet). Override with
# HF_HUB_OFFLINE=0 if you intend the job to download.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO"

# Pre-flight: fail fast with an actionable message if the env isn't provisioned,
# rather than dying halfway through after allocating a GPU. See ONE-TIME ENV
# SETUP in the header.
python -c "import sglang, pyarrow, httpx, nla.datagen.stage2_api_explain" 2>/dev/null || {
    echo "[stage2] ERROR: env '$CONDA_ENV' is missing deps (sglang / pyarrow / httpx / nla)." >&2
    echo "[stage2] Run the ONE-TIME ENV SETUP from this script's header on the LOGIN node." >&2
    exit 1
}

# ---- launch the SGLang server ---------------------------------------------
# Gemma-3 REQUIRES --attention-backend fa3 (flashinfer OOMs on head_dim=256).
# context-length is capped low: stage-2 prompts are ~800 in + 300 out, so a big
# KV cache is wasted memory that would otherwise cut achievable concurrency.
echo "[stage2] launching SGLang server for $MODEL (tp=$TP) on port $PORT"
python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --tp "$TP" \
    --attention-backend fa3 \
    --context-length 4096 \
    --mem-fraction-static 0.9 &
SERVER_PID=$!

# Always kill the server when this script exits (success, failure, or timeout).
cleanup() {
    echo "[stage2] stopping SGLang server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---- wait for the server to be ready --------------------------------------
echo "[stage2] waiting for server health (timeout 900s)..."
for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "[stage2] server is up after ~$((i*5))s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[stage2] ERROR: server process died during startup — check log above" >&2
        exit 1
    fi
    if [ "$i" -eq 180 ]; then
        echo "[stage2] ERROR: server did not become healthy within 900s" >&2
        exit 1
    fi
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
