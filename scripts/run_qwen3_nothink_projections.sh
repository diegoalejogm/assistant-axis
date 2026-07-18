#!/bin/bash
# Fix 2 (G): project the Qwen3-32B thinking-off responses (persona_vectors'
# *_nothink.csv, already judged with the theatrical rubric) onto both the
# pre-computed Assistant Axis and the existing persona vectors. Reads only
# from persona_vectors' directory; never writes there - all outputs land in
# this repo's own results/ directory, mirroring run_all_persona_vectors_projections.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
PERSONA_VECTORS_DIR="${PERSONA_VECTORS_DIR:-/workspace/persona_vectors}"
AXIS_PATH="${AXIS_PATH:-/workspace/assistant-axis-cache/qwen-3-32b/assistant_axis.pt}"
EVAL_DIR="$PERSONA_VECTORS_DIR/eval_persona_eval/Qwen3-32B"
OUT_DIR="results/persona_vectors_projections_nothink"
mkdir -p "$OUT_DIR"

for trait in evil sycophantic hallucinating; do
  for cond in explicit implicit_described implicit_contextual; do
    in_csv="$EVAL_DIR/${trait}_${cond}_nothink.csv"
    out_csv="$OUT_DIR/${trait}_${cond}_nothink_axis.csv"
    echo ">>> $(date +%H:%M:%S) $trait / $cond -> $out_csv"
    .venv/bin/python scripts/project_persona_vectors_responses.py \
      --input_csv "$in_csv" \
      --output_csv "$out_csv" \
      --axis_path "$AXIS_PATH" \
      --enable_thinking False || echo "!!! FAILED: $trait / $cond"
  done
done
echo "ALL_NOTHINK_PROJECTIONS_DONE $(date +%H:%M:%S)"
