#!/bin/bash
# Fix 3: Assistant Axis layer-fairness sweep on the Qwen3-32B thinking-off
# nothink data. Extracts all even layers (0-64) per conversation in one
# forward pass, instead of only the fixed target_layer (32), so the
# per-condition best-Axis-layer can be found and compared against the
# fixed-layer numbers already computed.
set -uo pipefail
export HF_HOME=/workspace/.cache/huggingface
cd "$(dirname "$0")/.."
PERSONA_VECTORS_DIR="${PERSONA_VECTORS_DIR:-/workspace/persona_vectors}"
AXIS_PATH="${AXIS_PATH:-/workspace/assistant-axis-cache/qwen-3-32b/assistant_axis.pt}"
EVAL_DIR="$PERSONA_VECTORS_DIR/eval_persona_eval/Qwen3-32B"
OUT_DIR="results/persona_vectors_projections_nothink_layersweep"
mkdir -p "$OUT_DIR"

for trait in evil sycophantic hallucinating; do
  for cond in explicit implicit_described implicit_contextual; do
    in_csv="$EVAL_DIR/${trait}_${cond}_nothink.csv"
    out_csv="$OUT_DIR/${trait}_${cond}_nothink_axis_sweep.csv"
    if [ -s "$out_csv" ]; then
      echo ">>> $(date +%H:%M:%S) $trait / $cond -> $out_csv (already exists, skipping)"
      continue
    fi
    echo ">>> $(date +%H:%M:%S) $trait / $cond -> $out_csv"
    .venv/bin/python scripts/project_persona_vectors_responses_layer_sweep.py \
      --input_csv "$in_csv" \
      --output_csv "$out_csv" \
      --axis_path "$AXIS_PATH" \
      --enable_thinking False || echo "!!! FAILED: $trait / $cond"
  done
done
echo "ALL_NOTHINK_AXIS_LAYER_SWEEP_DONE $(date +%H:%M:%S)"
