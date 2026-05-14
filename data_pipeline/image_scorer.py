#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image editing scoring tool - uses GPT-4o API to evaluate image editing quality.

Supports 4 scoring metrics:
- consistency_score: visual consistency (whether non-edited regions remain unchanged)
- quality_score: image quality (structure, naturalness, absence of artifacts)
- instruction_score: instruction following (whether the edit matches the instruction)
- knowledge_score: knowledge accuracy (whether edits requiring domain knowledge are correct)

Usage example:
python image_scorer.py \
    --original_image original.jpg \
    --edited_image edited.jpg \
    --instruction "Change the apple to red" \
    --metrics consistency instruction_score quality_score
"""

import os
import sys
import json
import base64
import time
import re
import hmac
import hashlib
import argparse
from typing import List, Dict, Optional, Tuple, Union
from openai import OpenAI
from .qwen_placeholder import call_qwen_api
from . import config as _cfg

# ========== API configuration ==========
# Credentials are loaded from ``config.py`` (which reads environment variables).
# Set GPT_API_BASE_URL / GPT_APP_ID / GPT_APP_KEY / GPT_SOURCE / GPT_MODEL
# before running, or edit ``config.py`` directly.
SOURCE = _cfg.GPT_SOURCE
APPID = _cfg.GPT_APP_ID
APPKEY = _cfg.GPT_APP_KEY
BASE_URL = _cfg.GPT_API_BASE_URL
MODEL = _cfg.GPT_MODEL

# ========== Scoring prompt templates ==========

# PROMPT_CONSISTENCY = """
# You are a professional digital artist and image editing evaluation specialist.

# You will be given:
# 1. **Image A**: the original image.
# 2. **Image B**: an edited version of Image A.
# 3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.

# Your Objective:
# Your task is to **evaluate the visual consistency between the original and edited images, focusing exclusively on elements that are NOT specified for change in the instruction**. That is, you should only consider whether all non-instructed details remain unchanged. Do **not** penalize or reward any changes that are explicitly required by the instruction.

# - IMPORTANT: For GLOBAL / TRANSFORMATIVE edits(e.g., extraction, stylization, color grading), consistency is NOT about preserving the original background, composition, lighting, or shadows. Instead, it is about correctly accomplishing the global operation.

# ## Evaluation Scale (1 to 5):
# You will assign a **consistency_score** according to the following rules:
# - **5 Perfect Consistency**: All non-instruction elements are completely unchanged and visually identical.
# - **4 Minor Inconsistency**: Only one very small, non-instruction detail is different (e.g., a tiny accessory, a subtle shadow, or a minor background artifact).
# - **3 Noticeable Inconsistency**: One clear non-instruction element is changed (e.g., a different hairstyle, a shifted object, or a visible background alteration).
# - **2 Significant Inconsistency**: Two or more non-instruction elements have been noticeably altered.
# - **1 Severe Inconsistency**: Most or all major non-instruction details are different (e.g., changed identity, gender, or overall scene layout).

# ## Input
# **Image A**
# **Image B**
# **Editing Instruction**: {instruction}

# ## Output Format
# First, clearly explain your comparison process: list each major non-instruction element and state whether it is consistent (unchanged) or inconsistent (changed), with brief reasoning.
# Then, provide your evaluation in the following JSON format:
# {{
# "reasoning": **Compared to original image**, [list of non-instruction elements that changed or remained the same] **in the edited image**. 
# "consistency_score": X
# }}
# """

PROMPT_CONSISTENCY = """
You are a professional digital artist and vision-language evaluation specialist for X2i task.

You will be given:
1. **Input Image(s) (X)**: The original image(s). This could be a single image, multiple images (reference set), or nothing.
2. **Instruction**: A directive describing the desired output.
3. **Generated Image (Y)**: The resulting image to be evaluated.

Your Objective:
Evaluate how well the **identity, core attributes, and non-instructed elements** from the Input Image(s) (X) are preserved in the Generated Image (Y).

## Evaluation Logic by Input Type:
- **Case 1: Pure Text Input**: If there is no input image, the consistency is automatically perfect. 
- **Case 2: Single Image Input**: Evaluate whether all elements NOT mentioned in the instruction remain identical to the original image.
- **Case 3: Multiple Reference Images (Subject-driven)**: Evaluate whether the specific subject (e.g., a character, object, or pet) maintains its identity, unique features, and textures across the edit, even if the pose or environment changes as per the instruction.

## Evaluation Scale (1 to 5):
- **5 Perfect**: (Or Pure Text) Subject identity and all non-instructed details are perfectly preserved.
- **4 Minor Inconsistency**: The subject is clearly the same, but a tiny detail (e.g., a small pattern on clothing, a subtle eye color shift) is off.
- **3 Noticeable Inconsistency**: The subject is recognizable, but one major attribute changed (e.g., different hair length, a missing distinct scar, or a shifted background object in single-image mode).
- **2 Significant Inconsistency**: The subject looks like a "variant" rather than the same entity; multiple non-instructed features have changed.
- **1 Severe Inconsistency**: The subject identity is lost (e.g., different person, different breed of dog) or the scene layout is completely unrelated to the input.

## Input
**Input Image(s) (X)**
**Instruction**: {instruction}
**Generated Image Y**

## Output Format
First, identify the input type (Text-only, Single-Image, or Multi-Image). List the key elements/subject attributes that must remain consistent. State whether they are preserved or altered.

Then, provide your evaluation in the following JSON format:
{{
"reasoning": "Compared to original image, briefly explain what stayed the same and what changed in the generated image, and analyze if the consistency is acceptable.", 
"consistency_score": X
}}
"""

# PROMPT_QUALITY = """
# Your goal is to evaluate AI-edited images focusing on **Visual Harmonization and Generative Integrity**.

# You will be given:
# 1. **Image A**: the original image.
# 2. **Image B**: an edited version of Image A.
# 3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.

# ## Objective:
# Evaluate the perceptual quality and seamlessness of the edited image. You must determine if the edited image looks like a single, unified picture or a fragmented composite.

# ## Evaluation Criteria:
# 1. **Structural Coherence**: Are shapes and textures of the edited region accurate? (No extra limbs, melted objects, or garbled text).
# 2. **Lighting & Color Harmony**: Do highlights and shadows of the edited region match the global light source? Is the color grading consistent across the entire image? (Fail: An object looking "pasted" due to different lighting).
# 3. **Edge & Blending (The "Sticker Effect")**: Are the boundaries between edited objects natural? Check for "halos," jagged cutouts, or unrealistic sharpness at the edges.
# 4. **Resolution & Blur Match**: Does the level of grain/noise and focus (Depth of Field) of the edited region match across the original image? (Fail: A 4K sharp object in a blurry background, unless it's intended bokeh).

# ## Evaluation Scale (1 to 5):
# - **5 Excellent Quality**: Perfect integration. Lighting, resolution, and edges are indistinguishable from a real photo. No artifacts.
# - **4 Minor Issues**: Small flaws (e.g., slight color mismatch, a tiny bit too sharp at the edges) that don't break the illusion immediately.
# - **3 Noticeable Artifacts**: Clear "AI look." Noticeable blending issues, slight resolution mismatch, or minor structural distortions (e.g., slightly odd fingers).
# - **2 Structural/Blending Failure**: Significant "sticker effect," major lighting contradictions, or warped shapes that make it look like a poor Photoshop job.
# - **1 Severe Errors**: Major hallucinations, broken anatomy, or complete failure in image reconstruction.

# ## Input
# **Image A**
# **Image B**
# **Editing Instruction**: {instruction}

# ## Output Format:
# {{
# "reasoning": "Concise analysis of structure, lighting harmony, and edge blending.",
# "quality_score": X
# }}
# """

PROMPT_QUALITY = """
Your goal is to evaluate AI-generated images focusing on **Visual Realism and Generative Integrity** for X2i tasks.

You will be given:
1. **Input Image(s) (X)**: The reference source (Single, Multiple, or None).
2. **Instruction**: A directive describing the desired modification or creation.
3. **Generated Image (Y)**: The resulting image to be evaluated.

## Objective:
Evaluate the perceptual quality, structural integrity, and aesthetic harmony of the Generated Image (Y). You must determine if the image is a high-quality, physically plausible result or a flawed AI generation.

## Evaluation Criteria:
1. **Structural Coherence**: Are shapes, anatomy, and textures accurate? Check for "AI hallucinations" like extra limbs, melted objects, or garbled text.
2. **Lighting & Color Harmony**: Is the lighting consistent within the scene? Do shadows and highlights follow a logical light source? (Fail: Objects looking "pasted" or lighting that contradicts the environment).
3. **Technical Fidelity**: Check for "sticker effects," jagged edges, or unrealistic sharpness/blur. Does the image have consistent grain and resolution throughout?
4. **Compositional Logic**: Does the scene layout make sense? Are the perspective and Depth of Field (DoF) handled naturally? Does the level of grain/noise and focus (Depth of Field) of the edited region match across the original image? (Fail: A 4K sharp object in a blurry background, unless it's intended bokeh).

## Evaluation Scale (1 to 5):
- **5 Excellent Quality**: Perfect realism or artistic execution. No artifacts, flawless anatomy, and logical lighting.
- **4 Minor Issues**: Small flaws (e.g., a tiny texture artifact, slight lighting mismatch) that do not break the overall immersion.
- **3 Noticeable Artifacts**: Clear "AI look." Visible blending issues, minor structural distortions (e.g., slightly odd fingers/eyes), or resolution inconsistencies.
- **2 Structural Failure**: Significant distortions, major lighting contradictions, or warped shapes that make the image look poorly constructed.
- **1 Severe Errors**: Major hallucinations, broken anatomy, or complete failure in image rendering.

## Input
**Input Image(s) (X)**
**Instruction**: {instruction}
**Generated Image (Y)**

## Output Format:
{{
"reasoning": "Provide a concise analysis of generated image Y's structure, lighting harmony, and technical artifacts.",
"quality_score": X
}}
"""

# PROMPT_INSTRUCTION = """
# You are a professional digital artist and image editing evaluation specialist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules. 

# You will be given:
# 1. **Image A**: the original image.
# 2. **Image B**: an edited version of Image A.
# 3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.

# Your Objective:
# Your task is to **evaluate how the edited image faithfully fulfills the editing instruction**, focusing **exclusively on the presence and correctness of the specified changes**. 

# ## Reasoning:
# You must follow these reasoning steps before scoring:
# **1. Detect Difference**: What has visually changed between Image A and Image B? (e.g., size, shape, color, position)
# **2. Expected Visual Caption**: Write a factual description of how the edited image should look if the instruction were perfectly followed.
# **3. Instruction Match**: 
# Compare the observed differences in **1** to the expected change in **2**:
# - Was the correct object modified (not replaced)?
# - Was the requested attribute (e.g., size, color, position) modified as intended?
# - **For Size/Spatial changes**: Is the change clearly visible and in the correct direction while keeping all other attributes unchanged (refer to Special Guidelines)?
# - **For other attributes**: Is the modification accurate?
# **4. Decision**: Use the 1–5 scale to assign a final score.

# ## Evaluation Scale (1 to 5):
# You will assign an **instruction_score** with following rule:
# - **5 Perfect Compliance**: The edited image **precisely matches** the intended modification; all required changes are present and accurate. 
# - **4 Minor Omission**: The core change is made, but **minor detail** is missing or slightly incorrect. 
# - **3 Partial Compliance**: The main idea is present, but one or more required aspects are wrong or incomplete. 
# - **2 Major Omission**: Most of the required changes are missing or poorly implemented. 
# - **1 Non-Compliance**: The instruction is **not followed at all** or is **completely misinterpreted** 

# ## Input
# **Image A**
# **Image B**
# **Editing Instruction**: {instruction}

# ## Output Format
# Look at the input again, provide the evaluation score and the explanation in the following JSON format:
# {{
# "instruction_score": X,
# "reasoning": 1. Detect Difference 2. Expected Visual Caption 3. Instruction Match 4. Decision
# }}
# """

PROMPT_INSTRUCTION = """
You are a professional digital artist and image editing evaluation specialist for X2i tasks. 

You will be given:
1. **Input Image(s) (X)**: The reference source (Single, Multiple, or None).
2. **Instruction**: A directive describing the desired output.
3. **Generated Image (Y)**: The resulting image to be evaluated.

Your Objective:
Evaluate how faithfully the Generated Image (Y) fulfills the **Instruction**, focusing on whether the requested changes or additions were executed correctly.

## Reasoning Steps:
1. **Detect Change**: What has been added, modified, or created in Y compared to X? (If X is Text-only, evaluate Y directly against the text).
2. **Expected Visual Caption**: Describe the ideal result if the instruction were perfectly followed.
3. **Instruction Match**: 
- Was the correct subject/attribute modified or created?
- For **Spatial/Size** changes: Is the placement or scale correct relative to the instruction?
- For **Subject-driven** (Multi-image): Does the generated subject perform the action/state requested in the instruction?
4. **Decision**: Assign a score based on compliance.

## Evaluation Scale (1 to 5):
- **5 Perfect Compliance**: Y precisely matches the instruction; all required changes are present, accurate, and clearly visible.
- **4 Minor Omission**: The core instruction is met, but a minor detail or nuance of the prompt is missing or slightly off.
- **3 Partial Compliance**: The main idea is present, but at least one major aspect of the instruction is ignored or incorrect.
- **2 Major Omission**: Most of the instruction is ignored; only a small part of the request is reflected in the image.
- **1 Non-Compliance**: The instruction is not followed at all, is misinterpreted, or Y is completely unrelated to the prompt.

## Input
**Input Image(s) (X)**
**Instruction**: {instruction}
**Generated Image (Y)**

## Output Format:
{{
"instruction_score": X,
"reasoning": "1. Detect Change 2. Expected Visual Caption 3. Instruction Match 4. Decision"
}}
"""

# PROMPT_KNOWLEDGE = """
# You are a **Strict Visual Forensics Expert**. Your goal is to scrutinize AI-edited images for **Physical**, **Geometric**, and **Spatial** logic.

# **CORE ATTITUDE**: Focus on the "Law of Physics." Even if an image is visually pretty (High Quality), it might be physically impossible (Low Knowledge).

# You will be given:
# 1. **Image A**: the original image.
# 2. **Image B**: an edited version of Image A.
# 3. **Editing Instruction**: a directive describing the intended modification.
# 4. **Explanation** (optional): Additional context about the knowledge required.

# ## The 3-Step Forensic Inspection Protocol

# **Phase 1: Geometry, Scale & Depth (Priority)**
# - **Relative Scale**: Relative to same-depth reference anchors in the original image, is the edited object scaled plausibly for its perceived depth?
# - **Occlusion & Intersection**: Does it correctly sit *behind* foreground objects, or does it "clip" through solid matter?

# **Phase 2: Shadow & Grounding Logic**
# - **Shadow Presence**: If the object is on the ground and there is directional light, is there a cast shadow? Note that if there are no shadows present in the background, the generation of shadows is not required.
# - **Contact Shadows (AO)**: If shadow generation is required, is the shadow in the edited region consistent with the background shadows in both direction and intensity?
# - **Gravity**: Does the placement respect the physical world (not floating unless specified)?

# **Phase 3: Semantic Consistency**
# - Does the result align with the provided `Explanation`?

# ## Evaluation Scale (Strict):
# - **5 (Flawless)**: Perfect physical logic. Scale, perspective, and shadow placement are scientifically accurate.
# - **4 (Minor Logic Flaw)**: Small scale error or slightly misplaced shadow that doesn't defy gravity.
# - **3 (Obvious Physics Failure)**: Floating objects (where shadows are needed), wrong scale (e.g., dog bigger than a car), or perspective mismatch.
# - **2 (Major Logical Conflict)**: Multiple failures (e.g., object clipping through walls AND floating).
# - **1 (Nonsense)**: Content completely ignores the scene's geometry or the editing instruction.

# ## Input
# **Image A**
# **Image B**
# **Editing Instruction**: {instruction}
# **Explanation**: {explanation}

# ## Output Format:
# {{
# "knowledge_score": X,
# "reasoning": "Forensic evidence regarding Scale, Perspective, and Shadow Logic."
# }}
# """

PROMPT_KNOWLEDGE = """
You are a **Strict Visual Forensics Expert** for X2i tasks. Your goal is to scrutinize the Generated Image (Y) for **Physical**, **Geometric**, and **Spatial** logic, especially relative to any grounding provided by the Input Image(s) (X).

**CORE ATTITUDE**: Focus on the "Law of Physics." Even if an image is visually appealing (High Quality), it might be physically impossible (Low Knowledge).

You will be given:
1. **Input Image(s) (X)**: The reference source (Single, Multiple, or None).
2. **Instruction**: A directive describing the intended modification or creation.
3. **Generated Image (Y)**: The resulting image to be evaluated.
4. **Explanation** (optional): Additional context about the knowledge required.

## The 3-Step Forensic Inspection Protocol

**Phase 1: Geometry, Scale & Depth (Priority)**
- **Perspective**: Is the perspective of the generated content (new subject, edit area) consistent with the background or existing scene in X?
- **Relative Scale**: Is the size of the generated object plausible for its perceived distance/depth within the scene?
- **Occlusion & Intersection**: Does the generated content correctly sit *behind* foreground objects, or does it "clip" through solid matter (e.g., a hand passing through a mug)?

**Phase 2: Shadow & Grounding Logic**
- **Shadow Presence**: If the scene (or the instruction) dictates directional light, is a cast shadow present for any grounded objects?
- **Shadow Consistency**: Is the generated shadow consistent with the light source and intensity of the scene in X?
- **Gravity**: Does the placement respect the physical world (i.e., not floating or defying stable placement unless explicitly specified)?

**Phase 3: Semantic Consistency**
- Does the result align with the provided `Explanation`? (e.g., If the explanation says "cars drive on the road," is the car on the road?)

## Evaluation Scale (Strict):
- **5 (Flawless)**: Perfect physical logic. Scale, perspective, and shadow placement are scientifically accurate and harmonize seamlessly with the input X (if applicable).
- **4 (Minor Logic Flaw)**: Small scale error or slightly misplaced shadow that doesn't fundamentally defy gravity or perspective.
- **3 (Obvious Physics Failure)**: Floating objects (where shadows are needed), significantly wrong scale (e.g., a person the size of a thumbnail in the foreground), or clear perspective mismatch.
- **2 (Major Logical Conflict)**: Multiple failures (e.g., object clipping through walls AND floating, or completely distorted scene geometry).
- **1 (Nonsense)**: The generated content completely ignores the scene's established physical rules or geometric layout.

## Input
**Input Image(s) (X)**
**Instruction**: {instruction}
**Generated Image (Y)**
**Explanation**: {explanation}

## Output Format:
{{
"knowledge_score": X,
"reasoning": "Provide forensic evidence regarding Scale, Perspective, and Shadow Logic in the Generated Image Y."
}}
"""



# ========== Utility functions ==========

def _calc_authorization(source: str, appkey: str) -> Tuple[str, int]:
    """Calculate API authentication signature."""
    timestamp = int(time.time())
    sign_str = f"x-timestamp: {timestamp}\nx-source: {source}"
    sign = hmac.new(appkey.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha256).digest()
    return sign.hex(), timestamp


def _get_auth_headers() -> Dict[str, str]:
    """Build API authentication headers."""
    auth, ts = _calc_authorization(SOURCE, APPKEY)
    return {
        "X-AppID": APPID,
        "X-Source": SOURCE,
        "X-Timestamp": str(ts),
        "X-Authorization": auth,
    }


def encode_image_to_base64(image_path: Union[str, List[str]]) -> Optional[Union[str, List[str]]]:
    if image_path is None:
        return None
    """Encode an image file to a base64 string."""
    if isinstance(image_path, list):
        encoded_list = []
        for path in image_path:
            encoded = encode_image_to_base64(path)
            if encoded is None:
                return None
            encoded_list.append(encoded)
        return encoded_list

    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def extract_score_and_reason(response: str, score_key: str, reason_fields: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """Extract score and reasoning from a GPT response."""
    # Attempt JSON parsing
    for reason_field in reason_fields:
        try:
            # Search for a JSON block
            pattern = r"\{[^{}]*" + re.escape(score_key) + r"[^{}]*\}"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                score = data.get(score_key)
                reason = data.get(reason_field)
                if score is not None:
                    return int(score), reason
        except Exception:
            continue
    
    # Fall back to regex
    patterns = [
        rf"{score_key}\s*[:：]?\s*([1-5])",
        r"([1-5])\s*/\s*5",
        r"([1-5])\s+out\s+of\s+5",
        r"\b([1-5])\b",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return int(match.group(1)), None
    
    return None, None


def call_gpt_api(
    prompt: str,
    original_base64: Optional[Union[str, List[str]]] = None,
    edited_base64: Optional[str] = None,
    max_retries: int = 0
) -> str:
    """
    Call the GPT API for evaluation.
    
    Args:
        prompt: Evaluation prompt text.
        original_base64: Base64-encoded original image(s); can be a single string or a list.
        edited_base64: Base64-encoded edited image.
        max_retries: Maximum number of retry attempts.
    
    Returns:
        API response content.
    """
    # Build the message
    message = {"role": "user", "content": [{"type": "text", "text": prompt}]}
    
    if original_base64:
        if isinstance(original_base64, list):
            for i, b64 in enumerate(original_base64):
                message["content"].extend([
                    {"type": "text", "text": f"This is Input Image {i+1}:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ])
        else:
            message["content"].extend([
                {"type": "text", "text": "This is the Input Image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_base64}"}}
            ])
    
    if edited_base64:
        message["content"].extend([
            {"type": "text", "text": "This is the Generated Image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{edited_base64}"}}
        ])
    
    # Initialize client
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL)
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            extra_headers = _get_auth_headers()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[message],
                stream=False,
                max_tokens=1000,
                extra_headers=extra_headers
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                
    print("API call failed after all retries, trying Qwen API as fallback")
    
    # Handle list for Qwen fallback
    qwen_original_base64 = original_base64
    # if isinstance(original_base64, list) and original_base64:
    #     qwen_original_base64 = original_base64[0]
        
    response = call_qwen_api(prompt, qwen_original_base64, edited_base64)
    
    return response


# ========== Scoring functions ==========

def evaluate_consistency(
    original_image: Union[str, List[str]],
    edited_image: str,
    instruction: str
) -> Dict:
    """Evaluate visual consistency."""
    print("Evaluating consistency...")
    
    original_b64 = encode_image_to_base64(original_image)
    edited_b64 = encode_image_to_base64(edited_image)
    
    # if not original_b64 or not edited_b64:
    #     return {"consistency_score": None, "consistency_reasoning": "Failed to encode images"}
    
    prompt = PROMPT_CONSISTENCY.format(instruction=instruction)
    response = call_gpt_api(prompt, original_b64, edited_b64)
    
    score, reason = extract_score_and_reason(
        response,
        score_key="consistency_score",
        reason_fields=["reasoning", "reason"]
    )
    
    return {
        "consistency_score": score,
        "consistency_reasoning": reason or response
    }


def evaluate_quality(original_image: Union[str, List[str]],
    edited_image: str,
    instruction: str
) -> Dict:
    """Evaluate image quality."""
    print("Evaluating quality...")
    
    original_b64 = encode_image_to_base64(original_image)
    edited_b64 = encode_image_to_base64(edited_image)
    
    # if not original_b64 or not edited_b64:
    #     return {"quality_score": None, "quality_reasoning": "Failed to encode image"}
    
    prompt = PROMPT_QUALITY.format(instruction=instruction)
    response = call_gpt_api(prompt, original_b64, edited_b64)
    
    score, reason = extract_score_and_reason(
        response,
        score_key="quality_score",
        reason_fields=["reasoning", "reason"]
    )
    
    return {
        "quality_score": score,
        "quality_reasoning": reason or response
    }


def evaluate_instruction(
    original_image: Union[str, List[str]],
    edited_image: str,
    instruction: str
) -> Dict:
    """Evaluate instruction following."""
    print("Evaluating instruction following...")
    
    original_b64 = encode_image_to_base64(original_image)
    edited_b64 = encode_image_to_base64(edited_image)
    
    # if not original_b64 or not edited_b64:
    #     return {"instruction_score": None, "instruction_reasoning": "Failed to encode images"}
    
    prompt = PROMPT_INSTRUCTION.format(instruction=instruction)
    response = call_gpt_api(prompt, original_b64, edited_b64)
    
    score, reason = extract_score_and_reason(
        response,
        score_key="instruction_score",
        reason_fields=["reasoning", "reason"]
    )
    
    return {
        "instruction_score": score,
        "instruction_reasoning": reason or response
    }


def evaluate_knowledge(
    original_image: Union[str, List[str]],
    edited_image: str,
    instruction: str,
    explanation: str = ""
) -> Dict:
    """Evaluate knowledge accuracy."""
    print("Evaluating knowledge accuracy...")
    
    original_b64 = encode_image_to_base64(original_image)
    edited_b64 = encode_image_to_base64(edited_image)
    
    # if not original_b64 or not edited_b64:
    #     return {"knowledge_score": None, "knowledge_reasoning": "Failed to encode images"}
    
    prompt = PROMPT_KNOWLEDGE.format(instruction=instruction, explanation=explanation)
    response = call_gpt_api(prompt, original_b64, edited_b64)
    
    score, reason = extract_score_and_reason(
        response,
        score_key="knowledge_score",
        reason_fields=["reasoning", "reason"]
    )
    
    return {
        "knowledge_score": score,
        "knowledge_reasoning": reason or response
    }


def evaluate_images(
    original_image: Union[str, List[str]],
    edited_image: str,
    instruction: str,
    explanation: str = "",
    metrics: List[str] = None
) -> Dict:
    """
    Comprehensive image editing evaluation.
    
    Args:
        original_image: Path to the original image; can be a single path or a list of paths.
        edited_image: Path to the edited image.
        instruction: Editing instruction.
        explanation: Additional context (used for knowledge evaluation).
        metrics: List of metrics to evaluate; valid values:
                 ["consistency", "quality", "instruction", "knowledge"]
                 If None, all metrics are evaluated.
    
    Returns:
        Dictionary containing all evaluation scores.
    """
    if metrics is None:
        metrics = ["consistency", "quality", "instruction", "knowledge"]
    
    results = {}
    
    if "consistency" in metrics:
        results.update(evaluate_consistency(original_image, edited_image, instruction))
    
    if "quality" in metrics:
        results.update(evaluate_quality(original_image, edited_image, instruction))
    
    if "instruction" in metrics:
        results.update(evaluate_instruction(original_image, edited_image, instruction))
    
    if "knowledge" in metrics:
        results.update(evaluate_knowledge(original_image, edited_image, instruction, explanation))
    
    return results


def print_results(results: Dict):
    """Print evaluation results."""
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    score_names = {
        "consistency_score": "Visual Consistency",
        "quality_score": "Image Quality",
        "instruction_score": "Instruction Following",
        "knowledge_score": "Knowledge Accuracy"
    }
    
    for score_key, display_name in score_names.items():
        if score_key in results:
            score = results[score_key]
            reason_key = score_key.replace("_score", "_reasoning")
            reason = results.get(reason_key, "")
            
            print(f"\n{display_name}: {score}/5")
            if reason:
                print(f"Reason: {reason[:200]}..." if len(reason) > 200 else f"Reason: {reason}")
    
    print("="*60)


# ========== Main entry point ==========

def main():
    parser = argparse.ArgumentParser(
        description="Image editing scoring tool - uses GPT-4o API to evaluate image editing quality."
    )
    
    parser.add_argument(
        "--original_image",
        type=str,
        required=True,
        help="Path to the original image."
    )
    
    parser.add_argument(
        "--edited_image",
        type=str,
        required=True,
        help="Path to the edited image."
    )
    
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Editing instruction."
    )
    
    parser.add_argument(
        "--explanation",
        type=str,
        default="",
        help="Additional context (used for knowledge evaluation)."
    )
    
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=["consistency", "quality", "instruction", "knowledge"],
        default=None,
        help="Metrics to evaluate; defaults to all metrics."
    )
    
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Save results to a JSON file."
    )
    
    args = parser.parse_args()
    
    # Check that files exist
    if not os.path.exists(args.original_image):
        print(f"Error: Original image not found: {args.original_image}")
        sys.exit(1)
    
    if not os.path.exists(args.edited_image):
        print(f"Error: Edited image not found: {args.edited_image}")
        sys.exit(1)
    
    # Run evaluation
    print(f"Original image: {args.original_image}")
    print(f"Edited image: {args.edited_image}")
    print(f"Instruction: {args.instruction}")
    if args.explanation:
        print(f"Explanation: {args.explanation}")
    print(f"Metrics: {args.metrics or 'all'}")
    print()
    
    start_time = time.time()
    
    results = evaluate_images(
        original_image=args.original_image,
        edited_image=args.edited_image,
        instruction=args.instruction,
        explanation=args.explanation,
        metrics=args.metrics
    )
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print_results(results)
    print(f"\nTotal elapsed time: {elapsed_time:.2f}s")
    
    # Save to JSON
    if args.output_json:
        output_data = {
            "original_image": args.original_image,
            "edited_image": args.edited_image,
            "instruction": args.instruction,
            "explanation": args.explanation,
            "metrics": args.metrics or ["consistency", "quality", "instruction", "knowledge"],
            "results": results,
            "elapsed_time": elapsed_time
        }
        
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {args.output_json}")


# ========== Test example ==========

def test_example():
    """Test example: evaluate a pair of images."""
    print("="*60)
    print("Test Example")
    print("="*60)
    
    # Example configuration
    original_image = "/path/to/your/original_image.png"
    edited_image = "/path/to/your/edited_image.png"
    instruction = "Add a brown bear standing several feet away from the dining table, fully visible and upright on all four legs. The bear is clearly separated from the table. Its fur is thick and textured, blending shades of brown with lighter tones around its muzzle and chest, matching the realistic style of the scene."
    explanation = ""
    
    # Evaluate a subset of metrics
    metrics = ["consistency", "instruction", "quality", "knowledge"]
    
    print(f"\nOriginal image: {original_image}")
    print(f"Edited image: {edited_image}")
    print(f"Instruction: {instruction}")
    print(f"Metrics: {metrics}\n")
    
    # Run evaluation
    results = evaluate_images(
        original_image=original_image,
        edited_image=edited_image,
        instruction=instruction,
        explanation=explanation,
        metrics=metrics
    )
    
    # Print results
    # print_results(results)
    print(json.dumps(results, ensure_ascii=False))
    
    return results


if __name__ == "__main__":
    # If command-line arguments are provided, run the main program
    if len(sys.argv) > 1:
        main()
    else:
        # Otherwise run the test example
        print("No command-line arguments provided; running the test example...\n")
        print("Usage:")
        print("python image_scorer.py \\")
        print("    --original_image test_images/original.jpg \\")
        print("    --edited_image test_images/edited.jpg \\")
        print("    --instruction 'Change the apple to red' \\")
        print("    --metrics consistency instruction quality")
        print("\nTo run the test, make sure test_images/original.jpg and test_images/edited.jpg exist.")
        print("\nPress Enter to continue with the test, or Ctrl+C to exit...")
        
        try:
            input()
            test_example()
        except KeyboardInterrupt:
            print("\n\nTest cancelled.")
            sys.exit(0)
