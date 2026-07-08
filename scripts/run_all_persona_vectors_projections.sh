#!/usr/bin/env bash
# Run project_persona_vectors_responses.py over all 6 implicit-elicitation CSVs
# from the persona_vectors Qwen3-32B run (3 traits x described/contextual).
# Assumes persona_vectors and assistant-axis are sibling directories under /workspace,
# and the axis has already been downloaded (see README note / axis_path below).
set -euo pipefail

PERSONA_VECTORS_DIR="${PERSONA_VECTORS_DIR:-/workspace/persona_vectors}"
AXIS_PATH="${AXIS_PATH:-/workspace/assistant-axis-cache/qwen-3-32b/assistant_axis.pt}"
EVAL_DIR="$PERSONA_VECTORS_DIR/eval_persona_eval/Qwen3-32B"

for trait in evil sycophantic hallucinating; do
  for cond in implicit_described implicit_contextual; do
    in_csv="$EVAL_DIR/${trait}_${cond}.csv"
    out_csv="$EVAL_DIR/${trait}_${cond}_axis.csv"
    echo "=== $trait $cond ==="
    uv run scripts/project_persona_vectors_responses.py \
      --input_csv "$in_csv" \
      --output_csv "$out_csv" \
      --axis_path "$AXIS_PATH"
  done
done
