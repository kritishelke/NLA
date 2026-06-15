#!/bin/bash
#SBATCH --job-name=nla_stage3
#SBATCH --account=sanghani_nlp
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/sanghani_nlp/logs/%x_%j.out

set -eo pipefail
mkdir -p /projects/sanghani_nlp/logs

PROJ=/projects/sanghani_nlp
REPO="$PROJ/NLA"
SPL="$PROJ/activations/qwen7b_fineweb_100k/splits"
OUT="$PROJ/activations/qwen7b_fineweb_100k"
CONDA_ENV="$PROJ/envs/nla-stage2"

module load Miniconda3
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO"

# Smoke test: verify the injection character survives tokenization as a
# standalone token inside the actor template. Catches BPE merge issues
# (e.g. >X< collapsing) before wasting a full stage 3 run.
echo "[stage3] verifying injection token..."
python - <<'PY'
import sys
from transformers import AutoTokenizer
from nla.datagen.injection_tokens import find_injection_token
from nla.datagen.stage3_build import _DEFAULT_ACTOR_TEMPLATE

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
# Pass actor_template so find_injection_token returns the IN-CONTEXT token ID.
# Qwen tiktoken merges the preceding space into the char → isolation ID != context ID.
inj_char, inj_id = find_injection_token(tok, _DEFAULT_ACTOR_TEMPLATE)
rendered = tok.apply_chat_template(
    [{"role": "user", "content": _DEFAULT_ACTOR_TEMPLATE.format(injection_char=inj_char)}],
    tokenize=False, add_generation_prompt=True,
)
ids = tok.encode(rendered, add_special_tokens=False)
matches = [i for i, tid in enumerate(ids) if tid == inj_id]
assert len(matches) == 1, (
    f"injection token {inj_char!r} (id={inj_id}) appears {len(matches)}x "
    f"in tokenized prompt — expected 1."
)
print(f"[stage3] injection token OK: {inj_char!r} id={inj_id} pos={matches[0]}")
PY

echo "[stage3] building av_sft..."
python -m nla.datagen.stage3_build \
    --input "$SPL/av_sft_explained.parquet" \
    --stage av_sft \
    --output "$OUT/av_sft.parquet"

echo "[stage3] building ar_sft..."
python -m nla.datagen.stage3_build \
    --input "$SPL/ar_sft_explained.parquet" \
    --stage ar_sft \
    --output "$OUT/ar_sft.parquet"

echo "[stage3] building rl..."
python -m nla.datagen.stage3_build \
    --input "$SPL/rl_raw.parquet" \
    --stage rl \
    --output "$OUT/rl.parquet"

echo "[stage3] done."
echo "  av_sft: $OUT/av_sft.parquet"
echo "  ar_sft: $OUT/ar_sft.parquet"
echo "  rl:     $OUT/rl.parquet"
