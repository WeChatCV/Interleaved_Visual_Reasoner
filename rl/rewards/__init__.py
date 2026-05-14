"""Framework-agnostic reward functions for the Interleaved Visual Reasoner.

The rewards consume **rollout dicts** (see ``rl/example_rollout.json``) and
return a scalar in :math:`[0, 1]`. They are designed to be dropped into any
RL framework: TRL, OpenRLHF, verl, or a hand-rolled loop. None of them
require gradients on the judge; they only need a callable ``judge`` that
sends prompts + images to a VLM and returns its raw text response.

Components
----------
* :func:`final_reward`      - final-image quality (multi-metric VLM judge).
* :func:`step_reward`       - per-intermediate-image quality.
* :func:`format_reward`     - interleaved-format compliance (string-level).
* :func:`reflection_reward` - reflection-text usefulness (VLM judge).
* :func:`compute_total_reward` - convex combination of the four above.
"""

from .final_reward import final_reward
from .step_reward import step_reward
from .format_reward import format_reward
from .reflection_reward import reflection_reward
from .compose import compute_total_reward, DEFAULT_WEIGHTS

__all__ = [
    "final_reward",
    "step_reward",
    "format_reward",
    "reflection_reward",
    "compute_total_reward",
    "DEFAULT_WEIGHTS",
]
