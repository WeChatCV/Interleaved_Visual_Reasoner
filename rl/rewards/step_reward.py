"""Step reward.

Scores **each intermediate image** against the textual sub-instruction the
model emitted right before producing that image. This rewards informative
intermediate progress, encouraging the model to make every visual step
meaningful (the core motivation of the interleaved visual reasoner).

Rollout schema expected::

    {
      "instruction": "global request",
      "source_images": [...] or [],
      "intermediate_steps": [
        {"instruction": "step-1 sub-goal", "image": "/path/step1.jpg"},
        {"instruction": "step-2 sub-goal", "image": "/path/step2.jpg"},
        ...
      ],
      "final_image": "/path/last.jpg"           # optional; not used here
    }

Each step is scored 1-5 by a VLM. The aggregate reward is the mean of all
step scores rescaled to [0, 1]. The full per-step breakdown is returned in
``info`` so you can also use it for *dense* / *process* RL.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from . import prompts as P
from ._utils import parse_judge_json, rescale_1_5_to_unit, safe_list


def step_reward(
    rollout: Dict,
    judge: Callable[[str, List[str]], str],
) -> Tuple[float, Dict]:
    """Compute the mean step reward over all intermediate images.

    If the rollout has no ``intermediate_steps`` (single-shot generation),
    the reward is 0 and ``info["n_steps"] == 0``. You may decide in
    ``compose.py`` to skip the step term in that case (default behaviour).
    """
    global_instr = rollout.get("instruction", "")
    srcs = safe_list(rollout.get("source_images"))
    steps = rollout.get("intermediate_steps", []) or []

    if not steps:
        return 0.0, {"n_steps": 0, "per_step": []}

    per_step: List[Dict] = []
    raw_scores: List[float] = []

    # The "previous image" for step i is step i-1's image, or the source(s)
    # if i == 0.
    prev_images: List[str] = list(srcs)

    for i, step in enumerate(steps):
        sub_instr = step.get("instruction", "")
        cur_img = step.get("image")
        if cur_img is None:
            per_step.append({"index": i, "error": "no image"})
            raw_scores.append(1.0)
            continue

        images = list(prev_images) + [cur_img]
        prompt = P.PROMPT_STEP_REWARD.format(
            sub_instruction=sub_instr,
            global_instruction=global_instr,
        )
        raw = judge(prompt, images)
        score, reasoning = parse_judge_json(raw, score_key="step_score", default=1.0)
        per_step.append(
            {
                "index": i,
                "sub_instruction": sub_instr,
                "score_1_5": score,
                "reasoning": reasoning,
            }
        )
        raw_scores.append(score)
        prev_images = [cur_img]  # next step compares against this one

    mean_1_5 = sum(raw_scores) / len(raw_scores)
    reward = rescale_1_5_to_unit(mean_1_5)
    return reward, {
        "n_steps": len(steps),
        "mean_1_5": mean_1_5,
        "reward": reward,
        "per_step": per_step,
    }
