#!/usr/bin/env python3
"""
Project persona_vectors implicit-elicitation responses onto the pre-computed
Assistant Axis for Qwen3-32B.

Answers: can the Assistant Axis (a general, non-trait-specific persona-drift
detector) monitor implicit trait elicitation as well as trait-specific persona
vectors do? Reads response CSVs from the diegoalejogm/persona_vectors repo
(sibling directory), reconstructs conversations from the `prompt`/`answer`
columns, extracts mean assistant-turn activations at Qwen3-32B's target layer
(32), projects onto the pre-computed axis, and writes the projection back as a
new column alongside the existing judge scores.

Reads only from persona_vectors' directory; never writes there - all outputs
go to this repo's own results/ directory (see run_all_persona_vectors_projections.sh).

Usage:
    uv run scripts/project_persona_vectors_responses.py \
        --input_csv /workspace/persona_vectors/eval_persona_eval/Qwen3-32B/evil_implicit_contextual.csv \
        --output_csv results/persona_vectors_projections/evil_implicit_contextual_axis.csv \
        --axis_path /workspace/assistant-axis-cache/qwen-3-32b/assistant_axis.pt
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis import load_axis, project
from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor, SpanMapper

MODEL_NAME = "Qwen/Qwen3-32B"
TARGET_LAYER = 32  # from assistant_axis.models.MODEL_CONFIGS["Qwen/Qwen3-32B"]

TURN_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>|\Z)", re.DOTALL)


def parse_prompt_to_conversation(prompt: str, answer: str) -> list[dict]:
    """Reconstruct a role/content conversation from persona_vectors' rendered
    Qwen chat-template `prompt` column (system+user, no assistant content) plus
    the separate `answer` column (the assistant's response)."""
    turns = []
    for role, content in TURN_RE.findall(prompt):
        content = content.strip()
        if content:
            turns.append({"role": role, "content": content})
    turns.append({"role": "assistant", "content": answer})
    return turns


def extract_activations_batch(pm, conversations, layer, batch_size=16, max_length=4096, enable_thinking=False):
    """Mirrors pipeline/2_activations.py's extract_activations_batch, single layer."""
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper(pm.tokenizer)

    chat_kwargs = {"enable_thinking": enable_thinking} if "qwen" in pm.model_name.lower() else {}

    all_activations = []
    for batch_start in range(0, len(conversations), batch_size):
        batch = conversations[batch_start:batch_start + batch_size]
        batch_activations, batch_metadata = extractor.batch_conversations(
            batch, layer=layer, max_length=max_length, **chat_kwargs
        )
        _, batch_spans, span_metadata = encoder.build_batch_turn_spans(batch, **chat_kwargs)
        conv_activations_list = span_mapper.map_spans(batch_activations, batch_spans, batch_metadata)

        for conv_acts in conv_activations_list:
            if conv_acts.numel() == 0:
                all_activations.append(None)
                continue
            if conv_acts.shape[0] >= 2:
                assistant_act = conv_acts[1::2]
                all_activations.append(assistant_act.mean(dim=0).cpu() if assistant_act.shape[0] > 0 else None)
            else:
                all_activations.append(None)

        del batch_activations
        if (batch_start // batch_size) % 5 == 0:
            torch.cuda.empty_cache()

    return all_activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--axis_path", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--enable_thinking", type=lambda x: x.lower() in ["true", "1", "yes"], default=True,
                         help="Whether the ORIGINAL responses were generated with thinking enabled "
                              "(persona_vectors' Qwen3-32B runs used the default, thinking-enabled mode). "
                              "This controls chat-template reconstruction only, not generation.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} responses from {args.input_csv}")

    conversations = [
        parse_prompt_to_conversation(row["prompt"], row["answer"])
        for _, row in df.iterrows()
    ]

    print(f"Loading {MODEL_NAME}...")
    pm = ProbingModel(MODEL_NAME)

    print(f"Extracting layer {TARGET_LAYER} activations for {len(conversations)} conversations...")
    activations = extract_activations_batch(
        pm, conversations, layer=TARGET_LAYER,
        batch_size=args.batch_size, enable_thinking=args.enable_thinking,
    )

    axis = load_axis(args.axis_path)
    print(f"Axis shape: {axis.shape}")

    projections = []
    for act in activations:
        if act is None:
            projections.append(float("nan"))
            continue
        # act shape: (1, hidden_size) since we only requested TARGET_LAYER (stored at index 0,
        # not index TARGET_LAYER). Squeeze to 1D so project() treats it as the already-extracted
        # layer and indexes only into the axis (which does have all 64 layers) at TARGET_LAYER.
        projections.append(project(act.squeeze(0), axis, layer=TARGET_LAYER))

    n_failed = sum(p != p for p in projections)  # NaN count
    if n_failed:
        print(f"WARNING: {n_failed}/{len(projections)} conversations failed activation extraction")

    df["assistant_axis_projection"] = projections
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")

    pm.close()


if __name__ == "__main__":
    main()
