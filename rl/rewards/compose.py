"""Compose the four reward components into a single scalar.

This is the function you plug into your RL trainer's reward callable. It
returns ``(total_reward, info)``. ``info`` contains every sub-score and the
judges' reasoning, which is invaluable for debugging GRPO/PPO runs.

Default weights are deliberately conservative; tune them for your task.
For pure T2I (no source image, no intermediate steps), the step and
reflection terms are skipped automatically and their weight is redistributed
to the remaining components.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from .final_reward import final_reward
from .step_reward import step_reward
from .format_reward import format_reward
from .reflection_reward import reflection_reward


DEFAULT_WEIGHTS: Dict[str, float] = {
    "final": 0.40,
    "step": 0.30,
    "format": 0.10,
    "reflection": 0.20,
}


def compute_total_reward(
    rollout: Dict,
    judge: Callable[[str, List[str]], str],
    weights: Optional[Dict[str, float]] = None,
    skip_empty: bool = True,
) -> Tuple[float, Dict]:
    """Run all four reward functions and return a weighted sum.

    Args:
        rollout: A dict matching ``rl/example_rollout.json``.
        judge:   Callable ``(prompt, images) -> str`` that wraps your VLM.
        weights: Optional override of ``DEFAULT_WEIGHTS``.
        skip_empty: If True, terms whose rollout fields are empty (e.g. no
            intermediate steps) are dropped and their weight is
            redistributed across the remaining terms.

    Returns:
        ``(total, info)`` where ``total`` is in [0, 1] and ``info`` contains:
            * ``components``: per-component reward and judge breakdown
            * ``effective_weights``: weights actually used after redistribution
    """
    w = dict(weights or DEFAULT_WEIGHTS)

    final_r, final_info = final_reward(rollout, judge)
    step_r, step_info = step_reward(rollout, judge)
    format_r, format_info = format_reward(rollout)
    refl_r, refl_info = reflection_reward(rollout, judge)

    components = {
        "final": {"reward": final_r, "info": final_info},
        "step": {"reward": step_r, "info": step_info},
        "format": {"reward": format_r, "info": format_info},
        "reflection": {"reward": refl_r, "info": refl_info},
    }

    if skip_empty:
        if step_info.get("n_steps", 0) == 0:
            w["step"] = 0.0
        if refl_info.get("n_reflections", 0) == 0:
            w["reflection"] = 0.0
        s = sum(w.values())
        if s > 0:
            w = {k: v / s for k, v in w.items()}
        else:
            # Degenerate: fall back to format only.
            w = {"final": 0.0, "step": 0.0, "format": 1.0, "reflection": 0.0}

    total = (
        w["final"] * final_r
        + w["step"] * step_r
        + w["format"] * format_r
        + w["reflection"] * refl_r
    )

    return total, {
        "total": total,
        "components": components,
        "effective_weights": w,
    }
