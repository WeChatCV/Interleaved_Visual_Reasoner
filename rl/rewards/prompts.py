"""Prompts used by the reward judges.

These prompts are derived from the data-pipeline evaluator
(`open_source/data_pipeline/image_scorer.py`) and re-targeted for online RL
scoring. They are intentionally kept in a single module so that you can
audit, edit, or translate them without touching the reward logic.

All prompts ask the judge to return a single JSON object with an integer
score in [1, 5] and a short reasoning string. The reward functions parse
this JSON and rescale to [0, 1].
"""

# ---------------------------------------------------------------------------
# Final-image rewards (multi-metric, four axes)
# ---------------------------------------------------------------------------

PROMPT_FINAL_CONSISTENCY = """\
You are a professional digital artist and vision-language evaluation specialist for X2i tasks.

You will be given:
1. **Input Image(s) (X)**: the reference source (single, multiple, or none).
2. **Instruction**: a directive describing the intended modification.
3. **Generated Image (Y)**: the final image to be evaluated.

## Objective
Evaluate the visual consistency between X and Y, focusing **exclusively** on
elements that are NOT specified for change in the instruction. Do not penalise
changes that are explicitly required.

> For GLOBAL / TRANSFORMATIVE edits (e.g. stylisation, colour grading), score
> the correct global execution rather than literal background preservation.

## Evaluation Scale (1-5)
- 5: All non-instruction elements are completely unchanged.
- 4: One very small, non-instruction detail differs.
- 3: One clear non-instruction element is changed.
- 2: Two or more non-instruction elements are noticeably altered.
- 1: Most or all major non-instruction details are different.

## Input
**Instruction**: {instruction}

## Output Format
Return ONLY a JSON object:
{{
  "consistency_score": X,
  "reasoning": "..."
}}
"""


PROMPT_FINAL_INSTRUCTION = """\
You are a professional image-editing evaluation specialist for X2i tasks.

You will be given:
1. **Input Image(s) (X)**: the reference source (single, multiple, or none).
2. **Instruction**: a directive describing the desired output.
3. **Generated Image (Y)**: the final image to be evaluated.

## Objective
Evaluate how faithfully Y fulfils the instruction.

Reason in four steps before scoring:
1. Detect change between X (if any) and Y.
2. Describe the ideal image if the instruction were perfectly executed.
3. Compare 1 and 2: was the correct subject/attribute modified or created?
4. Decide the score.

## Evaluation Scale (1-5)
- 5: Y precisely matches the instruction.
- 4: Core instruction met; minor detail off.
- 3: Main idea present; one major aspect ignored.
- 2: Most of the instruction is ignored.
- 1: Instruction not followed at all.

## Input
**Instruction**: {instruction}

## Output Format
Return ONLY a JSON object:
{{
  "instruction_score": X,
  "reasoning": "1. Detect Change 2. Expected Image 3. Match 4. Decision"
}}
"""


PROMPT_FINAL_QUALITY = """\
You are a strict visual-realism evaluator for X2i tasks.

You will be given:
1. **Input Image(s) (X)**.
2. **Instruction**.
3. **Generated Image (Y)**.

## Objective
Evaluate perceptual quality, structural integrity, and aesthetic harmony of Y.

Inspect:
* Structural coherence (no extra limbs, melted objects, garbled text).
* Lighting & colour harmony (consistent light source, no "pasted" look).
* Technical fidelity (no sticker effect, jagged edges, mismatched resolution).
* Compositional logic (sensible perspective, depth of field).

## Evaluation Scale (1-5)
- 5: Perfect realism / artistic execution.
- 4: Minor flaws that do not break immersion.
- 3: Visible "AI look" or minor structural distortions.
- 2: Significant distortions or major lighting contradictions.
- 1: Severe hallucinations or broken anatomy.

## Input
**Instruction**: {instruction}

## Output Format
Return ONLY a JSON object:
{{
  "quality_score": X,
  "reasoning": "..."
}}
"""


PROMPT_FINAL_KNOWLEDGE = """\
You are a strict visual-forensics expert for X2i tasks.

You will be given:
1. **Input Image(s) (X)**.
2. **Instruction**.
3. **Generated Image (Y)**.
4. **Explanation** (optional real-world facts that must hold).

## Objective
Inspect Y for **physical**, **geometric**, and **spatial** correctness:
* Geometry, scale, perspective, occlusion, gravity.
* Shadow presence and consistency with the scene's light source.
* Semantic consistency with the given `Explanation`.

## Evaluation Scale (1-5)
- 5: Perfect physical logic.
- 4: Minor logic flaw that doesn't defy gravity / perspective.
- 3: Obvious physics failure (wrong scale, floating).
- 2: Multiple physical conflicts.
- 1: Scene geometry completely ignored.

## Input
**Instruction**: {instruction}
**Explanation**: {explanation}

## Output Format
Return ONLY a JSON object:
{{
  "knowledge_score": X,
  "reasoning": "..."
}}
"""


# ---------------------------------------------------------------------------
# Step reward (per intermediate image)
# ---------------------------------------------------------------------------
#
# The step reward asks the judge to look at the textual sub-instruction emitted
# for one intermediate image and decide whether that intermediate image is a
# reasonable execution of that single atomic step. We deliberately ignore the
# global instruction here: the goal is to credit useful intermediate progress,
# not to require that every intermediate image already be the final answer.

PROMPT_STEP_REWARD = """\
You are a process-supervision judge for an interleaved visual reasoner.

A unified vision-language model is solving a complex image-editing or
text-to-image task by alternating between textual reasoning and intermediate
image generation. You will be given:

1. **Previous Image (P)**: the image the model produced at the previous step
   (or the source image, if this is the first step; or "None" for T2I).
2. **Sub-instruction (S)**: a short textual description of *what this single
   step is supposed to accomplish* (this is the textual thought the model
   emitted right before generating the current image).
3. **Current Image (C)**: the image the model produced at this step.
4. **Global Instruction (G)**: the original user request, provided for context
   only - **do not** require C to already satisfy G; only judge step S.

## Objective
Score how well image C executes the sub-instruction S **relative to** P.

Reason step by step:
1. **Step Goal**: What change does S require?
2. **Step Change**: What actually differs between P and C? (For T2I, what does
   C contain?)
3. **Goal Match**: Is the observed change a faithful execution of S?
   - Was the correct object/attribute modified or introduced?
   - Were unrelated elements preserved (P → C consistency outside the edit)?
4. **Plausibility**: Is C visually plausible (no severe artefacts, broken
   anatomy, sticker effect)?
5. **Decision**: Assign the score.

## Evaluation Scale (1-5)
- 5: C is a faithful, high-quality execution of S; unrelated content in P is
     preserved; no artefacts.
- 4: S is executed correctly; minor detail off OR a small unrelated change.
- 3: Partial execution of S, OR clear quality issues in C.
- 2: Most of S is missing or mis-executed; or major artefacts.
- 1: C is unrelated to S, or visually broken.

## Input
**Sub-instruction (S)**: {sub_instruction}
**Global Instruction (G)**: {global_instruction}

## Output Format
Return ONLY a JSON object:
{{
  "step_score": X,
  "reasoning": "1. Step Goal 2. Step Change 3. Goal Match 4. Plausibility 5. Decision"
}}
"""


# ---------------------------------------------------------------------------
# Reflection-text reward
# ---------------------------------------------------------------------------
#
# Given (previous image, instruction, reflection text, next image), the judge
# decides whether the reflection text was a *useful* critique that led to a
# better next image. This rewards informative reasoning, not just generation.

PROMPT_REFLECTION_REWARD = """\
You are a process-supervision judge for an interleaved visual reasoner.

You will be given:
1. **Previous Image (P)**: the model's previous attempt.
2. **Reflection Text (R)**: the model's textual critique of P, proposing
   how to fix it.
3. **Next Image (N)**: the model's next attempt produced after R.
4. **Instruction**: the original user request.

## Objective
Score how *useful* the reflection R was. A good reflection should:
* Correctly identify the concrete problems of P with respect to the instruction.
* Propose a specific, actionable fix.
* Result in an N that addresses the issues identified in R.

A bad reflection is vague, wrong about the issues, contradictory to the
instruction, or unrelated to the improvement actually achieved by N.

## Evaluation Scale (1-5)
- 5: R precisely identifies P's flaws, proposes a concrete fix, and N clearly
     implements that fix.
- 4: R is mostly correct; the fix is largely implemented in N.
- 3: R is partially correct; N implements only part of the proposed fix.
- 2: R is vague or off-target; N shows little correlation with R.
- 1: R is wrong, contradictory, or unrelated; N is not an improvement.

## Input
**Reflection (R)**:
{reflection_text}

**Instruction**: {instruction}

## Output Format
Return ONLY a JSON object:
{{
  "reflection_score": X,
  "reasoning": "..."
}}
"""
