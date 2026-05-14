"""
Multi-step editing instruction generation module.

This module provides functionality to decompose complex image editing instructions
into a sequence of atomic, executable editing steps using Gemini API.
"""

import os
import re
import json
import base64
import time
import hmac
import hashlib
from typing import List, Optional, Tuple, Dict
import requests
from io import BytesIO
from PIL import Image
from .qwen_placeholder import call_qwen_api
from . import config as _cfg

# ========== ChatGatewayClient Class ==========
class ChatGatewayClient:
    """Chat gateway client (supports text-to-text and image+text-to-text)."""
    
    def __init__(self, api_base_url: str, app_id: str, app_secret: str, source: str = "python-client"):
        """
        Initialize the client.
        
        Args:
            api_base_url: API base URL (user-defined gateway), provided by config.py.
            app_id: Application ID.
            app_secret: Application secret (used to generate the signature).
            source: Source identifier.
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.app_id = app_id
        self.app_secret = app_secret
        self.source = source
        self.endpoint = f"{self.api_base_url}/chat_completions"
    
    def _generate_auth(self, timestamp: int) -> str:
        """
        Generate authentication signature (HMAC-SHA256).
        
        Args:
            timestamp: Unix timestamp.
            
        Returns:
            Authentication signature string.
        """
        sign_str = f"x-timestamp: {timestamp}\nx-source: {self.source}"
        sign = hmac.new(
            self.app_secret.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return sign.hex()
    
    def _image_to_base64(self, image_input) -> str:
        """
        Convert an image file or PIL Image object to base64 format.
        
        Args:
            image_input: Image file path (str) or PIL.Image.Image object.
            
        Returns:
            Base64-encoded image data string (with data URI scheme).
        """
        if isinstance(image_input, (str, os.PathLike)):
            with open(image_input, "rb") as f:
                image_data = f.read()
            image = Image.open(image_input)
            mime_type = Image.MIME.get(image.format, "image/jpeg")
        elif isinstance(image_input, Image.Image):
            buffer = BytesIO()
            image_format = image_input.format if image_input.format else "JPEG"
            image_input.save(buffer, format=image_format)
            image_data = buffer.getvalue()
            mime_type = Image.MIME.get(image_format, "image/jpeg")
        else:
            raise TypeError(
                f"image_input must be a file path (str) or PIL.Image.Image object, "
                f"but received {type(image_input)}"
            )
        
        base64_data = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime_type};base64,{base64_data}"
    
    def _build_content(self, text: str = None, images: List = None, detail: str = "auto"):
        """
        Build the content field.
        
        Args:
            text: Text content.
            images: List of images (URLs, file paths, or PIL.Image.Image objects).
            detail: Image detail level; one of "low", "high", or "auto".
            
        Returns:
            Content list.
        """
        content = []
        
        # Add images
        if images:
            for image in images:
                if isinstance(image, str) and image.startswith(('http://', 'https://')):
                    # URL format
                    image_url = image
                else:
                    # File path or PIL Image – convert to base64
                    image_url = self._image_to_base64(image)
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": detail
                    }
                })
        
        # Add text (placed after images)
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        return content
    
    def chat(
        self,
        prompt: str,
        images: Optional[List] = None,
        model: str = "gemini-3-pro-image-preview",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        cid: Optional[str] = None,
        detail: str = "auto",
        stream: bool = False
    ) -> dict:
        """
        Send a chat request (supports text-to-text and image+text-to-text).
        
        Args:
            prompt: User input text.
            images: Optional list of images; supports:
                   - URL strings
                   - Local file paths
                   - PIL.Image.Image objects
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.
            cid: Session ID (optional; recommended to fill in for traceability).
            detail: Image detail level; one of "low", "high", or "auto".
            stream: Whether to use streaming output.
            
        Returns:
            API response data.
        """
        # Generate timestamp and authentication
        timestamp = int(time.time())
        auth = self._generate_auth(timestamp)
        
        # Build request headers
        headers = {
            "Content-Type": "application/json",
            "X-AppID": self.app_id,
            "X-Source": self.source,
            "X-Timestamp": str(timestamp),
            "X-Authorization": auth
        }
        
        # Build content
        content = self._build_content(text=prompt, images=images, detail=detail)
        
        # Build request body
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        if cid:
            payload["cid"] = cid
        
        # Send request
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=150,
            stream=stream
        )
        
        # Check response
        response.raise_for_status()
        result = response.json()
        
        if result.get('code') != 0:
            raise Exception(f"API returned an error: {result.get('msg', 'Unknown error')}")
        
        return result
    
    def extract_response_text(self, result: dict) -> str:
        """
        Extract text content from an API response.
        
        Args:
            result: API response data.
            
        Returns:
            Generated text content.
        """
        try:
            # Chat gateway response format: top-level 'response' field
            if 'response' in result:
                return result['response']
            # Or extract from 'detail'
            elif 'detail' in result:
                return result['detail']['choices'][0]['message']['content']
            # Or standard OpenAI format
            elif 'choices' in result:
                return result['choices'][0]['message']['content']
            else:
                raise KeyError("Cannot find response text field")
        except (KeyError, IndexError) as e:
            raise Exception(f"Cannot extract text from response: {e}, response keys: {list(result.keys())}")


# ========== Prompt Template ==========
PROMPT_GENERATE_EDITING_STEPS = """
# Role
You are an expert **Image Editing Planning Agent**. Your goal is to analyze a user's high-level editing instruction for an input image and decompose it into a precise sequence of atomic, executable editing prompts.

# Objective
Complex image editing tasks often fail when attempted in a single step. Your job is to break down the "User Instruction" into a logical chain of 2-5 sub-prompts. These sub-prompts will be executed sequentially by a generative editing model to achieve the final result.

# Guidelines for Planning
1.  **Atomic Steps:** Each step should focus on changing one specific visual aspect.
2.  **Logical Order (Local -> Global):**
    * **Priority 1: Local Structure & Content.** specific object modifications (e.g., "change clothes," "add glasses," "fix hair") must happen *first*. This anchors the subject's identity before the environment changes.
    * **Priority 2: Global Atmosphere & Style.** Broad changes (e.g., "change time of day," "apply oil painting style," "change lighting") should happen *last*. These act as a "unifying filter" over the modified content.
    * **Dependency:** Ensure logical cause-and-effect (e.g., "add a candle" must happen before "light the candle").
3.  **Visual Reasoning:** Do not just split the sentence grammatically. Think about *how* an image generator works. If the user says "Make the apple rotten," plan it visually: "add mold spots (Local)" -> "change color to brown (Local)" -> "adjust lighting to be gloomy (Global)".
4.  **Preservation:** Implicitly maintain the identity of the parts that shouldn't change.
5.  **Step Count:** Aim for 2-3 steps. Achieve the edit in as few steps as possible.
6.  **The "Move" Logic** If the user asks to move an object, decompose it into removing the object from the original position first, then adding it to the new position.
7.  **Subject Reference Update** Update the terminology in later steps to match changes made in earlier steps. If Step 1 turns a 'cat' into a 'tiger', Step 2 must refer to it as 'the tiger', not 'the cat'.
8.  **Atomic Interaction** Keep tight physical interactions combined. 'A man holding a sword' is better generated in one specific step or by explicitly targeting the interaction area, rather than generating a man and a sword separately.
9.  **Clean Slate Strategy** If adding an object to a cluttered area, consider an implicit step to 'clear or empty' that specific surface first to ensure clean generation.
10. Each step MUST state that all unrelated visual regions remain unchanged.

# Input Data:
1. **Image**: The source image.
2. **Editing Instruction**: "{}"
3. **Explanation** (optional): "{}"

# Output Format
Return **only** a JSON list of strings, where each string is a prompt for a single step. Do not include markdown code blocks or explanations outside the JSON.

# Few-Shot Examples

**Example 1 (Logic: Local Shape First -> Surface Material)**
* **User Instruction:** "Turn the wooden chair into a futuristic gaming chair."
* **Reasoning:** Change the physical structure first (Local), then apply the material (Surface), then lights.
* **Output:**
    [
        "Reshape the chair to have a high back and ergonomic racing style",
        "Change the material of the chair to sleek black metal and carbon fiber",
        "Add neon blue LED light strips to the edges of the chair"
    ]

**Example 2 (Logic: Subject Detail -> Global Style)**
* **User Instruction:** "Turn this photo of a woman into a 1920s vintage sepia portrait."
* **Reasoning:** If we apply sepia first, we might lose facial details. We must change the fashion/hair (Local) first, then apply the photo style (Global).
* **Output:**
    [
        "Change the woman's clothes to a 1920s flapper dress with beads",
        "Change the woman's hairstyle to short finger waves",
        "Apply a sepia tone filter with film grain and vignette to the whole image"
    ]

**Example 3 (Logic: Specific Features -> Global Atmosphere)**
* **User Instruction:** "Make the cute teddy bear look like a horror movie villain."
* **Reasoning:** Modify the bear's specific scary features first so they are clearly defined, then darken the mood.
* **Output:**
    [
        "Change the teddy bear's eyes to glowing red and angry",
        "Add stitching scars, tears, and grime to the teddy bear's fur",
        "Add a sharp, rusty knife in the teddy bear's hand",
        "Change the overall lighting to be dark, dramatic, and coming from below"
    ]

**Example 4 (Logic: Sequential Addition)**
* **User Instruction:** "Put a birthday cake on the table and have a dog eating it."
* **Reasoning:** Need the cake first to establish the scene, then the interaction.
* **Output:**
    [
        "Add a chocolate birthday cake with lit candles on the table",
        "Add a golden retriever standing on hind legs reaching for the cake",
        "Add frosting smears on the dog's nose"
    ]

"""


# ========== Helper Functions ==========

def validate_editing_steps(steps: List[str]) -> bool:
    """
    Validate that the editing steps conform to the expected format.
    
    Args:
        steps: List of editing step strings.
        
    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(steps, list):
        return False
    
    if len(steps) < 1 or len(steps) > 5:
        return False
    
    for step in steps:
        if not isinstance(step, str) or len(step.strip()) == 0:
            return False
        if len(step) > 500:  # Maximum length per step
            return False
    
    return True


# ========== Main Function ==========

def generate_editing_steps(
    image_path: str,
    instruction: str,
    explanation: str = "",
    api_base_url: str = None,
    app_id: str = None,
    app_secret: str = None,
    source: str = None,
    model: str = None,
    max_retries: int = 0
) -> Optional[List[str]]:
    """
    Generate a multi-step editing instruction sequence.
    
    This function uses the Gemini API to decompose a complex editing instruction
    into a logical sequence of 2–5 atomic steps.
    
    Args:
        image_path: Path to the source image.
        instruction: High-level editing instruction.
        explanation: Optional instruction clarification or context.
        api_base_url: API base URL.
        app_id: Application ID.
        app_secret: Application secret.
        source: Source identifier.
        model: Model name (default: gemini-3-pro-preview).
        max_retries: Maximum number of retry attempts (default: 3).
        
    Returns:
        List of editing step strings, or None on failure.
        
    Example:
        >>> steps = generate_editing_steps(
        ...     "path/to/image.jpg",
        ...     "Turn the wooden chair into a futuristic gaming chair"
        ... )
        >>> print(steps)
        [
            "Reshape the chair to have a high back and ergonomic racing style",
            "Change the material of the chair to sleek black metal and carbon fiber",
            "Add neon blue LED light strips to the edges of the chair"
        ]
    """
    # Resolve credentials (CLI/explicit args take precedence over config.py)
    api_base_url = api_base_url or _cfg.CHAT_API_BASE_URL
    app_id = app_id or _cfg.CHAT_APP_ID
    app_secret = app_secret or _cfg.CHAT_APP_SECRET
    source = source or _cfg.CHAT_SOURCE
    model = model or _cfg.CHAT_MODEL

    # Initialize client
    chat_client = ChatGatewayClient(
        api_base_url=api_base_url,
        app_id=app_id,
        app_secret=app_secret,
        source=source
    )
    
    # Format the prompt
    formatted_prompt = PROMPT_GENERATE_EDITING_STEPS.format(instruction, explanation)
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            print(f"🔄 Attempting to generate editing steps (attempt {attempt + 1}/{max_retries})...")
            
            # Call API
            result = chat_client.chat(
                prompt=formatted_prompt,
                images=[image_path],
                model=model
            )
            
            # Extract response text
            response_text = chat_client.extract_response_text(result)
            
            # Strip possible markdown code-fence markers
            cleaned_text = response_text.strip()
            
            # Remove leading ```json or ```
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
            
            # Remove trailing ```
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            
            # Strip whitespace again
            cleaned_text = cleaned_text.strip()
            
            # Parse JSON
            try:
                steps = json.loads(cleaned_text)
                
                # Validate the response
                if validate_editing_steps(steps):
                    print(f"✅ Successfully generated {len(steps)} editing steps.")
                    return steps
                else:
                    print(f"⚠️  Invalid editing step format (attempt {attempt + 1}/{max_retries}).")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential back-off
                    continue
                    
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"   First 100 chars of response: {cleaned_text[:100]}...")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
                
        except Exception as e:
            print(f"⚠️  API call error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
    
    print(f"❌ Failed to generate editing steps after {max_retries} attempts. Trying Qwen fallback API.")
    try:
        response_text = call_qwen_api(formatted_prompt, image_path)
        
        # Strip possible markdown code-fence markers
        cleaned_text = response_text.strip()
        
        # Remove leading ```json or ```
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
        
        # Remove trailing ```
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        
        # Strip whitespace again
        cleaned_text = cleaned_text.strip()
        
        # Parse JSON
        try:
            parsed_json = json.loads(cleaned_text)
            
            # Handle two possible formats:
            # 1. Direct array: ["step1", "step2", ...]
            # 2. Nested format: {"steps": ["step1", "step2", ...]}
            if isinstance(parsed_json, list):
                steps = parsed_json
            elif isinstance(parsed_json, dict) and 'steps' in parsed_json:
                steps = parsed_json['steps']
            else:
                print(f"⚠️  Unrecognized JSON format: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else type(parsed_json)}")
                return None
            
            # Validate the response
            if validate_editing_steps(steps):
                print(f"✅ Successfully generated {len(steps)} editing steps (via Qwen fallback API).")
                return steps
            else:
                print(f"⚠️  Invalid editing step format.")
                return None
                
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print(f"   First 100 chars of response: {cleaned_text[:100]}...")
            return None
            
    except Exception as e:
        print(f"⚠️  Qwen API call error: {e}")
        return None


# ========== Test Function ==========

def test_generate_editing_steps():
    """Test the generate_editing_steps function."""
    # Test cases
    test_cases = [
        {
            "instruction": "Transform the rusted metal door handle and backplate assembly into solid, polished gold that features a large, oval-shaped sapphire embedded in the backplate’s flat surface below the handle, from which a soft blue glow emanates to illuminate the surrounding gold material.",
            "explanation": ""
        }
    ]
    
    # Provide a valid test image path
    test_image_path = "/path/to/your/test_image.jpg"
    
    if not os.path.exists(test_image_path):
        print("⚠️  Please provide a valid test image path.")
        return
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test case {i+1}")
        print(f"{'='*60}")
        print(f"Instruction: {test_case['instruction']}")
        if test_case['explanation']:
            print(f"Explanation: {test_case['explanation']}")
        
        steps = generate_editing_steps(
            test_image_path,
            test_case['instruction'],
            test_case['explanation']
        )
        
        if steps:
            print("\nGenerated steps:")
            for j, step in enumerate(steps, 1):
                print(f"  {j}. {step}")
        else:
            print("\n❌ Step generation failed.")


if __name__ == "__main__":
    test_generate_editing_steps()

