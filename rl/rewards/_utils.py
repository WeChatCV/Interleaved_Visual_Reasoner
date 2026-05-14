"""Shared helpers for reward functions.

A reward function in this package has the shape::

    def some_reward(rollout: dict, judge: Callable, **kwargs) -> Tuple[float, dict]

where ``judge`` is a user-supplied callable::

    judge(prompt: str, images: List[str]) -> str

that returns the **raw text** of a VLM response. We do not assume any specific
model or backend - you can wrap GPT-4o, Qwen-VL, InternVL, Gemini, or a local
endpoint. The reward returns a scalar in [0, 1] plus a breakdown dict for
logging.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


JsonDict = Dict[str, Any]


def rescale_1_5_to_unit(score: float) -> float:
    """Map a Likert score in [1, 5] to a reward in [0, 1]."""
    score = max(1.0, min(5.0, float(score)))
    return (score - 1.0) / 4.0


def parse_judge_json(text: str, score_key: str, default: float = 1.0) -> Tuple[float, str]:
    """Extract ``score_key`` and ``reasoning`` from a judge's response.

    The judge is *asked* to return strict JSON, but in practice VLMs often
    wrap it in code fences or prepend prose. We try, in order:
    1. ``json.loads(text)``
    2. The first ``{...}`` block found via regex.
    3. A line of the form ``"<score_key>": <int>``.

    Returns ``(score, reasoning)``. ``score`` is in [1, 5] (clamped); if the
    judge response is unparseable we fall back to ``default``.
    """
    reasoning = ""
    obj: Optional[JsonDict] = None

    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None

    if isinstance(obj, dict) and score_key in obj:
        try:
            score = float(obj[score_key])
        except Exception:
            score = default
        reasoning = str(obj.get("reasoning", ""))
        return max(1.0, min(5.0, score)), reasoning

    # Last-resort regex on the raw text.
    m = re.search(rf'"?{re.escape(score_key)}"?\s*[:=]\s*([1-5])', text)
    if m:
        return float(m.group(1)), text.strip()[:500]

    return float(default), text.strip()[:500]


def safe_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]
