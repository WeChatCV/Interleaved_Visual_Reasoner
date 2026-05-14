"""Format reward.

Pure-text, judge-free reward that encourages the model to follow the
interleaved output protocol::

    <think> reasoning </think>
    <image> ... </image>
    <think> reflection </think>
    <image> ... </image>
    ...

No VLM call is made. This makes the reward cheap and deterministic, which is
useful in early RL where you mainly want to stabilise the surface form.

Rollout schema::

    {"assistant_text": "<think>...</think><image>...</image>..."}

Scoring rubric (each item adds 0.25 of the unit reward):
1. At least one ``<think>...</think>`` block exists.
2. At least one ``<image>...</image>`` block exists.
3. All opening tags are properly closed (balanced).
4. The trajectory ends with an ``</image>`` (a generated image is the final
   answer), and every ``<image>`` is preceded by a ``</think>`` (every image
   has a justification).

Customise the tag names with ``think_tag`` / ``image_tag`` keyword args.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple


def _balanced(text: str, tag: str) -> bool:
    return text.count(f"<{tag}>") == text.count(f"</{tag}>")


def format_reward(
    rollout: Dict,
    judge=None,  # unused, kept for API symmetry
    think_tag: str = "think",
    image_tag: str = "image",
) -> Tuple[float, Dict]:
    text = rollout.get("assistant_text", "") or ""
    info: Dict[str, bool] = {}

    has_think = bool(re.search(rf"<{think_tag}>.*?</{think_tag}>", text, flags=re.DOTALL))
    has_image = bool(re.search(rf"<{image_tag}>.*?</{image_tag}>", text, flags=re.DOTALL))
    balanced = _balanced(text, think_tag) and _balanced(text, image_tag)

    # Every <image> should be immediately preceded by </think>, and the text
    # should end with </image> (allowing trailing whitespace).
    ends_with_image = bool(re.search(rf"</{image_tag}>\s*$", text))
    preceded = True
    for m in re.finditer(rf"<{image_tag}>", text):
        prefix = text[: m.start()]
        if not re.search(rf"</{think_tag}>\s*$", prefix):
            preceded = False
            break
    structure_ok = ends_with_image and preceded

    info["has_think"] = has_think
    info["has_image"] = has_image
    info["balanced"] = balanced
    info["structure_ok"] = structure_ok

    score = 0.25 * (int(has_think) + int(has_image) + int(balanced) + int(structure_ok))
    info["reward"] = score
    return score, info
