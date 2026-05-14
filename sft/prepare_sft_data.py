#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert ``data_pipeline`` success outputs into SFT-ready JSONL.

Each successful sample produced by the data pipeline lives in a directory
``<pipeline_root>/<key>/`` together with a ``result.json``. This script
walks every such directory, reconstructs the interleaved reasoning
trajectory (initial plan → intermediate images → reflections → final
image), and writes one trajectory per line in the requested format.

Run ``python prepare_sft_data.py --help`` for the full list of options.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm_src(src) -> List[str]:
    """Normalise ``src_path`` (str | list | None) into a list of paths."""
    if src is None:
        return []
    if isinstance(src, str):
        return [src]
    return list(src)


def _build_initial_plan(result: Dict) -> str:
    """Reconstruct the initial textual plan from a result.json record.

    Priority order:
        1. Multi-step plan if available (one bullet per atomic edit step).
        2. The edit prompt as a 1-line trivial plan otherwise.
    """
    steps = result.get("multi_step") or []
    if steps:
        lines = ["I will decompose this task into the following atomic edits:"]
        for i, step in enumerate(steps, 1):
            instr = step.get("step_instruction", "").strip()
            lines.append(f"{i}. {instr}")
        return "\n".join(lines)
    return f"I will perform the requested edit: {result.get('edit_prompt', '').strip()}"


def _build_reflection_text(reflect_round: Dict) -> str:
    """Render one reflection round into a short textual thought."""
    thinking = reflect_round.get("reflect_thinking", {}) or {}
    failure = thinking.get("failure_analysis", "").strip()
    issues = thinking.get("specific_issues", "").strip()
    plan = thinking.get("improvement_plan", "").strip()

    parts = []
    if failure:
        parts.append(f"Failure analysis: {failure}")
    if issues:
        parts.append(f"Specific issues: {issues}")
    if plan:
        parts.append(f"Improvement plan: {plan}")
    return " ".join(parts) if parts else "Let me try once more."


def _collect_trajectory(result: Dict, max_reflect: Optional[int]) -> Tuple[List[str], List[str], List[str]]:
    """Return three parallel lists describing the assistant turn:

    - ``thoughts[i]``  : textual ``<think>...</think>`` block at position i
    - ``images[i]``    : path to the image emitted after thought i
                          (empty string means "no image at this position")
    - all input images come from src_path (handled by caller).
    """
    thoughts: List[str] = []
    images: List[str] = []

    # 1. Initial plan + first generated image (direct edit).
    thoughts.append(_build_initial_plan(result))
    direct = result.get("direct_edit_image_path")
    images.append(direct or "")

    # 2. Reflection rounds.
    reflect_rounds = result.get("reflect") or []
    if max_reflect is not None:
        reflect_rounds = reflect_rounds[:max_reflect]
    for r in reflect_rounds:
        thoughts.append(_build_reflection_text(r))
        images.append(r.get("reflect_image_path") or "")

    # 3. Multi-step rounds (only those that were actually executed).
    for step in result.get("multi_step") or []:
        instr = step.get("step_instruction", "").strip()
        thoughts.append(f"Next sub-step: {instr}")
        images.append(step.get("step_image_path") or "")

    # 4. Final summary thought (anchored to final image if different).
    final = result.get("final_edited_image_path")
    if isinstance(final, str) and images and final and final != images[-1]:
        thoughts.append("This result satisfies all the requirements.")
        images.append(final)

    return thoughts, images, []


# ---------------------------------------------------------------------------
# Format emitters
# ---------------------------------------------------------------------------
def _emit_sharegpt(
    sample_key: str,
    user_text: str,
    user_images: List[str],
    thoughts: List[str],
    asst_images: List[str],
    image_token: str,
    think_open: str,
    think_close: str,
) -> Dict:
    user_value = (image_token + "\n") * len(user_images) + user_text
    parts: List[str] = []
    for thought, img in zip(thoughts, asst_images):
        parts.append(f"{think_open}{thought}{think_close}")
        if img:
            parts.append(image_token)
    asst_value = "".join(parts)
    return {
        "id": sample_key,
        "conversations": [
            {"from": "human", "value": user_value},
            {"from": "gpt",   "value": asst_value},
        ],
        "images": user_images + [p for p in asst_images if p],
    }


def _emit_messages(
    sample_key: str,
    user_text: str,
    user_images: List[str],
    thoughts: List[str],
    asst_images: List[str],
    image_token: str,
    think_open: str,
    think_close: str,
) -> Dict:
    user_content: List[Dict] = [{"type": "image", "image": p} for p in user_images]
    user_content.append({"type": "text", "text": user_text})

    asst_content: List[Dict] = []
    for thought, img in zip(thoughts, asst_images):
        asst_content.append({"type": "text", "text": f"{think_open}{thought}{think_close}"})
        if img:
            asst_content.append({"type": "image", "image": img})

    return {
        "id": sample_key,
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst_content},
        ],
    }


def _emit_chatml(
    sample_key: str,
    user_text: str,
    user_images: List[str],
    thoughts: List[str],
    asst_images: List[str],
    image_token: str,
    think_open: str,
    think_close: str,
) -> Dict:
    user_imgs = "".join(f"{image_token}{{{p}}}" for p in user_images)
    user_block = f"<|im_start|>user\n{user_imgs}{user_text}<|im_end|>"

    asst_parts: List[str] = []
    for thought, img in zip(thoughts, asst_images):
        asst_parts.append(f"{think_open}{thought}{think_close}")
        if img:
            asst_parts.append(f"{image_token}{{{img}}}")
    asst_block = "<|im_start|>assistant\n" + "".join(asst_parts) + "<|im_end|>"

    return {"id": sample_key, "text": user_block + asst_block}


EMITTERS = {
    "sharegpt": _emit_sharegpt,
    "messages": _emit_messages,
    "chatml":   _emit_chatml,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def convert_one(
    sample_dir: str,
    fmt: str,
    image_token: str,
    think_open: str,
    think_close: str,
    skip_no_reflect: bool,
    max_reflect: Optional[int],
) -> Optional[Dict]:
    result_path = os.path.join(sample_dir, "result.json")
    if not os.path.isfile(result_path):
        return None
    with open(result_path, "r") as f:
        result = json.load(f)

    if skip_no_reflect and not result.get("reflect") and not result.get("multi_step"):
        return None

    thoughts, asst_images, _ = _collect_trajectory(result, max_reflect)
    user_images = _norm_src(result.get("original_image_path"))
    user_text = result.get("edit_prompt", "")

    emit = EMITTERS[fmt]
    return emit(
        sample_key=result.get("task_id") or os.path.basename(sample_dir.rstrip("/")),
        user_text=user_text,
        user_images=user_images,
        thoughts=thoughts,
        asst_images=asst_images,
        image_token=image_token,
        think_open=think_open,
        think_close=think_close,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Convert data_pipeline outputs to SFT JSONL.")
    p.add_argument("--pipeline_root", required=True,
                   help="Root directory of pipeline 'success' outputs.")
    p.add_argument("--out_file", required=True, help="Output JSONL path.")
    p.add_argument("--format", default="sharegpt", choices=list(EMITTERS.keys()))
    p.add_argument("--image_token", default="<image>")
    p.add_argument("--think_open", default="<think>")
    p.add_argument("--think_close", default="</think>")
    p.add_argument("--skip_no_reflect", action="store_true",
                   help="Drop samples that have no reflection AND no multi-step plan.")
    p.add_argument("--max_reflect", type=int, default=None,
                   help="Maximum number of reflection rounds kept per sample.")
    args = p.parse_args()

    sample_dirs = [
        os.path.join(args.pipeline_root, d)
        for d in sorted(os.listdir(args.pipeline_root))
        if os.path.isdir(os.path.join(args.pipeline_root, d))
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)) or ".", exist_ok=True)
    written = 0
    skipped = 0
    with open(args.out_file, "w") as out_f:
        for sd in sample_dirs:
            try:
                record = convert_one(
                    sd,
                    fmt=args.format,
                    image_token=args.image_token,
                    think_open=args.think_open,
                    think_close=args.think_close,
                    skip_no_reflect=args.skip_no_reflect,
                    max_reflect=args.max_reflect,
                )
            except Exception as e:
                print(f"[warn] failed to convert {sd}: {e}")
                skipped += 1
                continue
            if record is None:
                skipped += 1
                continue
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} samples ({skipped} skipped) -> {args.out_file}")


if __name__ == "__main__":
    main()
