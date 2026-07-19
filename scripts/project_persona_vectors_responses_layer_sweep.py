#!/usr/bin/env python3
"""
Fix 3 - Assistant Axis layer-fairness sweep. Same as
project_persona_vectors_responses.py, but extracts ALL even layers (0-64,
matching the persona-vector sweep's convention) from a single forward pass
per conversation, instead of only the model's published target_layer (32).
Writes one assistant_axis_projection_layer{N} column per layer, so the
per-condition/per-trait best-Axis-layer can be found locally afterward (no
GPU needed for that step) and reported alongside the fixed-layer number,
directly answering whether the fixed-layer comparison was an artifact.

batch_conversations already supports layer=<list of ints>, extracting every
requested layer from one forward pass (no extra GPU cost per layer beyond
the hooks) - this is what makes the sweep as cheap as the single-layer run.

Reads only from persona_vectors' directory; never writes there - all
outputs go to this repo's own results/ directory.

Usage:
    uv run scripts/project_persona_vectors_responses_layer_sweep.py \
        --input_csv /workspace/persona_vectors/eval_persona_eval/Qwen3-32B/evil_implicit_contextual_nothink.csv \
        --output_csv results/persona_vectors_projections_nothink_layersweep/evil_implicit_contextual_nothink_axis_sweep.csv \
        --axis_path /workspace/assistant-axis-cache/qwen-3-32b/assistant_axis.pt \
        --enable_thinking False
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
SWEEP_LAYERS = list(range(0, 64, 2))  # even layers 0-62 (32 values). NOTE: this differs from
# cal_projection.py's persona-vector convention (0-64, 33 values, matching HF hidden_states'
# extra embedding-output entry). The Assistant Axis extraction hooks raw nn.Module transformer
# layers via probing_model.get_layers(), which only has valid indices 0-63 for this 64-layer
# model - requesting layer 64 raises IndexError. Confirmed via the pre-computed axis tensor's
# own shape, torch.Size([64, 5120]) - only 64 valid layer rows.

TURN_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>|\Z)", re.DOTALL)


def parse_prompt_to_conversation(prompt: str, answer: str) -> list[dict]:
    turns = []
    for role, content in TURN_RE.findall(prompt):
        content = content.strip()
        if content:
            turns.append({"role": role, "content": content})
    turns.append({"role": "assistant", "content": answer})
    return turns


def extract_activations_batch(pm, conversations, layers, batch_size=16, max_length=4096, enable_thinking=False):
    """Extracts all requested layers per conversation in one forward pass each."""
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper(pm.tokenizer)

    chat_kwargs = {"enable_thinking": enable_thinking} if "qwen" in pm.model_name.lower() else {}

    all_activations = []  # per-conversation: tensor (num_layers, hidden_size) or None
    for batch_start in range(0, len(conversations), batch_size):
        batch = conversations[batch_start:batch_start + batch_size]
        batch_activations, batch_metadata = extractor.batch_conversations(
            batch, layer=layers, max_length=max_length, **chat_kwargs
        )
        _, batch_spans, span_metadata = encoder.build_batch_turn_spans(batch, **chat_kwargs)
        conv_activations_list = span_mapper.map_spans(batch_activations, batch_spans, batch_metadata)

        for conv_acts in conv_activations_list:
            # conv_acts shape: (num_turns, num_layers, hidden_size)
            if conv_acts.numel() == 0:
                all_activations.append(None)
                continue
            if conv_acts.shape[0] >= 2:
                assistant_act = conv_acts[1::2]  # (num_assistant_turns, num_layers, hidden_size)
                all_activations.append(
                    assistant_act.mean(dim=0).cpu() if assistant_act.shape[0] > 0 else None
                )
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
                         help="Whether the ORIGINAL responses were generated with thinking enabled. "
                              "Controls chat-template reconstruction only, not generation.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} responses from {args.input_csv}")

    conversations = [
        parse_prompt_to_conversation(row["prompt"], row["answer"])
        for _, row in df.iterrows()
    ]

    print(f"Loading {MODEL_NAME}...")
    pm = ProbingModel(MODEL_NAME)

    print(f"Extracting {len(SWEEP_LAYERS)} layers ({SWEEP_LAYERS[0]}-{SWEEP_LAYERS[-1]}) for {len(conversations)} conversations...")
    activations = extract_activations_batch(
        pm, conversations, layers=SWEEP_LAYERS,
        batch_size=args.batch_size, enable_thinking=args.enable_thinking,
    )

    axis = load_axis(args.axis_path)
    print(f"Axis shape: {axis.shape}")

    # act, when not None, has shape (len(SWEEP_LAYERS), hidden_size) - index i
    # corresponds to SWEEP_LAYERS[i], not the raw model layer number, since only
    # the requested layers were extracted (not the full 0-64 range as a dense array).
    for i, layer_num in enumerate(SWEEP_LAYERS):
        col = f"assistant_axis_projection_layer{layer_num}"
        projections = []
        for act in activations:
            if act is None:
                projections.append(float("nan"))
                continue
            # act[i] is already the extracted activation for SWEEP_LAYERS[i];
            # project() with a 1D input just dots it with axis[layer_num].
            projections.append(project(act[i], axis, layer=layer_num))
        df[col] = projections

    n_failed = sum(a is None for a in activations)
    if n_failed:
        print(f"WARNING: {n_failed}/{len(activations)} conversations failed activation extraction")

    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")

    pm.close()


if __name__ == "__main__":
    main()
