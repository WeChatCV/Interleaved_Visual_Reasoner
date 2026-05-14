"""Final-image reward.

Scores the **last** image of a rollout against the user instruction along four
axes inherited from the data-generation pipeline:

* consistency  - preservation of non-instruction content
* instruction  - faithfulness to the user request
* quality      - perceptual / structural quality
* knowledge    - physical / geometric plausibility

Each axis is judged by a VLM call and returns an integer in [1, 5]; the
final reward is the (weighted) mean rescaled to [0, 1].

Rollout schema (see ``rl/example_rollout.json`` for a full example)::

    {
      "instruction": "...",
      "source_images": ["/path/to/src.jpg", ...] or [],
      "final_image":   "/path/to/last.jpg",
      "explanation":   "..."           # optional, used by the knowledge axis
    }
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from . import prompts as P
from ._utils import parse_judge_json, rescale_1_5_to_unit, safe_list


DEFAULT_AXIS_WEIGHTS: Dict[str, float] = {
    "consistency": 0.25,
    "instruction": 0.40,
    "quality": 0.20,
    "knowledge": 0.15,
}


def _score_axis(
    judge: Callable[[str, List[str]], str],
    prompt: str,
    images: List[str],
    score_key: str,
) -> Tuple[float, str]:
    raw = judge(prompt, images)
    return parse_judge_json(raw, score_key=score_key, default=1.0)


def final_reward(
    rollout: Dict,
    judge: Callable[[str, List[str]], str],
    axis_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict]:
    """Compute the final-image reward.

    Returns ``(reward, info)`` where ``reward`` is in [0, 1] and ``info``
    contains the per-axis 1-5 scores and the judges' reasoning strings.
    """
    weights = axis_weights or DEFAULT_AXIS_WEIGHTS
    instr = rollout.get("instruction", "")
    explanation = rollout.get("explanation", "N/A")
    srcs = safe_list(rollout.get("source_images"))
    final = rollout.get("final_image")
    if final is None:
        return 0.0, {"error": "no final_image in rollout"}

    images = list(srcs) + [final]
    info: Dict[str, Dict] = {}

    c, c_r = _score_axis(
        judge, P.PROMPT_FINAL_CONSISTENCY.format(instruction=instr), images, "consistency_score"
    )
    i, i_r = _score_axis(
        judge, P.PROMPT_FINAL_INSTRUCTION.format(instruction=instr), images, "instruction_score"
    )
    q, q_r = _score_axis(
        judge, P.PROMPT_FINAL_QUALITY.format(instruction=instr), images, "quality_score"
    )
    k, k_r = _score_axis(
        judge,
        P.PROMPT_FINAL_KNOWLEDGE.format(instruction=instr, explanation=explanation),
        images,
        "knowledge_score",
    )

    info["consistency"] = {"score_1_5": c, "reasoning": c_r}
    info["instruction"] = {"score_1_5": i, "reasoning": i_r}
    info["quality"] = {"score_1_5": q, "reasoning": q_r}
    info["knowledge"] = {"score_1_5": k, "reasoning": k_r}

    total_w = sum(weights.values()) or 1.0
    weighted_1_5 = (
        weights["consistency"] * c
        + weights["instruction"] * i
        + weights["quality"] * q
        + weights["knowledge"] * k
    ) / total_w

    reward = rescale_1_5_to_unit(weighted_1_5)
    info["weighted_1_5"] = weighted_1_5
    info["reward"] = reward
    return reward, info
