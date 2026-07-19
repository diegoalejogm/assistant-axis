#!/bin/bash
# Fix 3 (thinking-ON): Assistant Axis layer-fairness sweep on the ORIGINAL
# Qwen3-32B reasoning-on data (ASSISTANT_AXIS_PLAN.md's numbers), mirroring
# run_qwen3_nothink_axis_layer_sweep.sh but pointed at the non-_nothink CSVs
# and enable_thinking=True (the default, matching how those responses were
# actually generated). Even layers 0-62 (32 values) - see the off-by-one
# note in run_qwen3_nothink_axis_layer_sweep.sh for why not 0-64.
set -uo pipefail
export HF_HOME=/workspace/.cache/huggingface
cd "$(dirname "$0")/.."
PERSONA_VECTORS_DIR="${PERSONA_VECTORS_DIR:-/workspace/persona_vectors}"
AXIS_PATH="${AXIS_PATH:-/workspace/assistant-axis-cache/qwen-3-32b/assistant_axis.pt}"
EVAL_DIR="$PERSONA_VECTORS_DIR/eval_persona_eval/Qwen3-32B"
OUT_DIR="results/persona_vectors_projections_layersweep"
mkdir -p "$OUT_DIR"

for trait in evil sycophantic hallucinating; do
  for cond in explicit implicit_described implicit_contextual; do
    in_csv="$EVAL_DIR/${trait}_${cond}.csv"
    out_csv="$OUT_DIR/${trait}_${cond}_axis_sweep.csv"
    if [ -s "$out_csv" ]; then
      echo ">>> $(date +%H:%M:%S) $trait / $cond -> $out_csv (already exists, skipping)"
      continue
    fi
    echo ">>> $(date +%H:%M:%S) $trait / $cond -> $out_csv"
    .venv/bin/python scripts/project_persona_vectors_responses_layer_sweep.py \
      --input_csv "$in_csv" \
      --output_csv "$out_csv" \
      --axis_path "$AXIS_PATH" \
      --enable_thinking True \
      --batch_size 2 || echo "!!! FAILED: $trait / $cond"
  done
done
echo "ALL_THINKON_AXIS_LAYER_SWEEP_DONE $(date +%H:%M:%S)"
