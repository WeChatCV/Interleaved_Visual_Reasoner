"""Reflection reward.

For each ``<think>`` block that occurs **after** the first image (i.e. a
reflection on a previously generated image), the judge decides whether the
reflection was a *useful* critique that led to a better next image.

Rollout schema expected::

    {
      "instruction": "...",
      "reflections": [
        {"prev_image": "/path/a.jpg", "text": "...", "next_image": "/path/b.jpg"},
        ...
      ]
    }

You can populate this list directly, or derive it from
``intermediate_steps`` + the model's textual reasoning. See
``rl/example_rollout.json`` for the canonical layout.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from . import prompts as P
from ._utils import parse_judge_json, rescale_1_5_to_unit


def reflection_reward(
    rollout: Dict,
    judge: Callable[[str, List[str]], str],
) -> Tuple[float, Dict]:
    instr = rollout.get("instruction", "")
    refls = rollout.get("reflections", []) or []

    if not refls:
        return 0.0, {"n_reflections": 0, "per_reflection": []}

    per_refl: List[Dict] = []
    raw_scores: List[float] = []

    for i, r in enumerate(refls):
        prev_img = r.get("prev_image")
        next_img = r.get("next_image")
        text = r.get("text", "")
        if prev_img is None or next_img is None or not text:
            per_refl.append({"index": i, "error": "incomplete reflection record"})
            raw_scores.append(1.0)
            continue

        prompt = P.PROMPT_REFLECTION_REWARD.format(
            reflection_text=text, instruction=instr
        )
        raw = judge(prompt, [prev_img, next_img])
        score, reasoning = parse_judge_json(raw, score_key="reflection_score", default=1.0)
        per_refl.append(
            {"index": i, "score_1_5": score, "reasoning": reasoning}
        )
        raw_scores.append(score)

    mean_1_5 = sum(raw_scores) / len(raw_scores)
    reward = rescale_1_5_to_unit(mean_1_5)
    return reward, {
        "n_reflections": len(refls),
        "mean_1_5": mean_1_5,
        "reward": reward,
        "per_reflection": per_refl,
    }
