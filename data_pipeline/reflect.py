# -*- coding: utf-8 -*-

import json
from typing import Optional, Dict, List

import base64
import hashlib
import hmac
import os
import time
from io import BytesIO
from typing import List, Optional

import requests
import PIL
from PIL import Image
from .qwen_placeholder import call_qwen_api
from . import config as _cfg

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
        model: str = "gpt-4o-mini",  # ✅ Use a GPT model for chat
        temperature: float = 0.2,
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
            model: Model name, e.g. "gpt-4o-mini", "gpt-4o", "gpt-4.1".
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



PROMPT_REFLECTION = """
You are an expert Image Editing Auditor. Your goal is to diagnose editing failures and generate an optimized instruction to fix them.

### Input Data
1. **Original Image**: The source image.
2. **Edited Image**: The failed attempt.
3. **Editing Instruction**: {editing_instruction}
4. **Evaluation Scores & Evidence**:
   - **Consistency ({consistency_score}/5)**: Reasoning: {consistency_reasoning}
   - **Instruction ({instruction_score}/5)**: Reasoning: {instruction_reasoning}
   - **Quality ({quality_score}/5)**: Reasoning: {quality_reasoning}
   - **Knowledge ({knowledge_score}/5)**: Reasoning: {knowledge_reasoning}

### Task 1: Failure Analysis
Diagnose the root cause by synthesizing the scores and reasoning. Focus on:
- **Targeting Error**: Edited wrong object or missed the target.
- **Over-editing**: Changed background/identity that should be static.
- **Under-editing**: Ignored parts of the prompt.
- **Visual Artifacts**: Poor blending, "sticker effect," poorly harmonized, or low-res textures (Quality issues).
- **Logic Flaws**: Defied physics, bad scale, or broken shadows (Knowledge issues).

### Task 2: Optimized Instruction
Write a revised prompt for the next attempt using these strategies:
- **Visual Anchors**: Use descriptors (e.g., "the red cup on the right") to fix targeting.
- **Strict Constraints**: Use negative constraints (e.g., "preserve the background exactly").
- **Physical Clarity**: Specify lighting, shadow or scale to fix logic flaws.
- **Simplification**: Break down complex or "bleeding" concepts.

## Output Format
Return ONLY a raw JSON object with the following structure:
{{
"failure_analysis": "Detailed diagnosis citing specific failure types (e.g., 'Grounding Failure: The model modified the table instead of the chair...') and explaining why the scores are low.",
"improvement_plan": "A concrete, optimized text prompt/instruction to be used for the next attempt. Incorporate negative constraints and visual descriptors where necessary."
}}

**IMPORTANT**: Return ONLY the raw JSON object. Do NOT use markdown code blocks.
"""

PROMPT_RE_EDITING = """
You are a precision-oriented AI Image Editor. Your goal is to rectify a failed image editing attempt by following a specialized Improvement Plan.

### 1. Input Context
- **Original Image**: The starting source image.
- **Failed Attempt**: The previous edited version that was rejected.
- **Original Instruction**: {editing_instruction}

### 2. Forensic Feedback (The Diagnosis)
- **Failure Analysis**: {failure_analysis}
- **Improvement Plan (The Strategy)**: {improvement_plan}

### 3. Your Task: High-Fidelity Rectification
You must generate a new version of the editing image that fulfills the original intent while strictly fixing the issues identified in the analysis.

## Output Requirement:
Produce the corrected editing image based on above messages.
"""

def validate_reflection_response(response_text: str) -> tuple[bool, Optional[Dict]]:
    """
    Validate that the API's reflection result conforms to the expected format.
    
    Args:
        response_text: Text returned by the API.
        
    Returns:
        (is_valid, dict_data_or_None)
    """
    try:
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
        
        # Attempt JSON parsing
        data = json.loads(cleaned_text)
        
        # Check required fields (only two fields needed)
        required_fields = ['failure_analysis', 'improvement_plan']
        for field in required_fields:
            if field not in data:
                print(f"⚠️  Missing required field: {field}")
                return False, None
        
        # Check field types
        if not isinstance(data['failure_analysis'], str):
            print(f"⚠️  failure_analysis must be a string.")
            return False, None
        
        if not isinstance(data['improvement_plan'], str):
            print(f"⚠️  improvement_plan must be a string.")
            return False, None
        
        # Check that fields are not empty
        if not data['failure_analysis'].strip():
            print(f"⚠️  failure_analysis cannot be empty.")
            return False, None
        
        if not data['improvement_plan'].strip():
            print(f"⚠️  improvement_plan cannot be empty.")
            return False, None
        
        # Optional quality check: minimum content length
        if len(data['failure_analysis']) < 20:
            print(f"⚠️  failure_analysis is too short (fewer than 20 characters).")
            return False, None
        
        if len(data['improvement_plan']) < 20:
            print(f"⚠️  improvement_plan is too short (fewer than 20 characters).")
            return False, None
        
        return True, data
    
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parse error: {e}")
        print(f"   First 100 chars of problematic text: {cleaned_text[:100] if 'cleaned_text' in locals() else response_text[:100]}")
        return False, None
    except Exception as e:
        print(f"⚠️  Validation exception: {e}")
        return False, None


def reflect_on_editing_failure(
    original_image_path: str,
    edited_image_path: str,
    instruction: str,
    evaluation_dict: Dict,
    api_base_url: str = None,
    app_id: str = None,
    app_secret: str = None,
    source: str = None,
    model: str = None,
    max_retries: int = 0
) -> Optional[Dict]:
    """
    Perform reflection analysis on an editing failure and suggest improvements.
    
    Args:
        original_image_path: Path to the original image.
        edited_image_path: Path to the edited image.
        instruction: Original editing instruction.
        evaluation_dict: Evaluation result dict, should contain consistency_score and instruction_score.
        api_base_url: API base URL.
        app_id: Application ID.
        app_secret: Application secret.
        source: Source identifier.
        model: Model name.
        max_retries: Maximum number of retry attempts (default 3).
        
    Returns:
        Reflection result dict containing failure_analysis, specific_issues, improvement_plan,
        or None on failure.
    """
    # Resolve credentials (explicit args take precedence over config.py)
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
    
    # Extract all keys ending with _score from the evaluation dict
    # score_dict = {"consistency_score": None, "instruction_score": None, "quality_score": None, "knowledge_score": None}
    # for key, value in evaluation_dict.items():
    #     if key.endswith('_score'):
    #         score_dict[key] = value

    score_dict = evaluation_dict.copy()
    
    # Format the prompt (using dict unpacking)
    score_dict['editing_instruction'] = instruction
    formatted_prompt = PROMPT_REFLECTION.format(**score_dict)
    
    # Add the original instruction to the prompt
    # full_prompt = f"{formatted_prompt}\n\n**Original Instruction**: {instruction}"
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            # print(f"Attempting reflection analysis (attempt {attempt + 1}/{max_retries})...")
            
            # Call API
            result = chat_client.chat(
                prompt=formatted_prompt,
                images=[original_image_path, edited_image_path],
                model=model,
                temperature=0.2,
                detail="low"
            )
            
            # Extract response text
            response_text = chat_client.extract_response_text(result)
            # print(f"Received response (length: {len(response_text)} chars)")
            
            # Validate response format
            is_valid, reflection_data = validate_reflection_response(response_text)
            
            if is_valid and reflection_data is not None:
                return reflection_data
            else:
                print(f"❌ Attempt {attempt + 1}: invalid response format.")
                if attempt < max_retries - 1:
                    print(f"   Retrying...")
        
        except Exception as e:
            print(f"❌ Attempt {attempt + 1}: exception occurred - {str(e)}")
            if attempt < max_retries - 1:
                print(f"   Retrying...")

    print(f"❌ Reflection analysis failed: all {max_retries} attempts exhausted. Trying Qwen API.")
    try:
        response_text = call_qwen_api(formatted_prompt,original_image_path,edited_image_path)
        is_valid, reflection_data = validate_reflection_response(response_text)
        if is_valid and reflection_data is not None:
            return reflection_data
        else:
            print(f"❌ Qwen API reflection response has invalid format.")
    except Exception as e:
        print(f"❌ Qwen API reflection exception - {str(e)}")

    return None


# ==================== Test code ====================

if __name__ == "__main__":
    # Test example
    test_evaluation = {
      "consistency_score": 5,
      "consistency_reasoning": "Compared to original image, all non-instruction elements including the room architecture, lighting, furniture, plants, and background scenery remained completely unchanged in the edited image. The two images appear to be identical, meaning there is perfect consistency in all areas not targeted by the instruction.",
      "instruction_score": 1,
      "instruction_reasoning": "1. Detect Difference: There are no visible differences between Image A and Image B. The images appear to be identical. \n2. Expected Visual Caption: The two desks in the center of the room should be significantly smaller, scaled down to approximately 50% of their original size.\n3. Instruction Match: The instruction to reduce the size of the tables was not followed at all. The tables remain exactly the same size.\n4. Decision: Since the instruction was completely ignored and no changes were made, the score is 1.",
      "quality_score": 4,
      "quality_reasoning": "The image displays excellent structural coherence, with accurate perspective in the room geometry, windows, and skylights. The lighting is consistent, with shadows on the floor matching the light sources. However, there are minor imperfections preventing a perfect score: the legs of the desks and chairs are unnaturally thin and lack structural weight, and the stack of books in the bottom right corner appears slightly blurry and low-resolution compared to the sharp details elsewhere.",
    }
    
    # Note: replace with actual image paths
    original_img = "/path/to/your/original_image.png"
    edited_img = "/path/to/your/edited_image.png"
    instruction = "Reduce the size of the tables by half."
    
    print("=" * 80)
    print("Testing the reflection feature")
    print("=" * 80)
    print(f"Original image: {original_img}")
    print(f"Edited image: {edited_img}")
    print(f"Instruction: {instruction}")
    print(f"Evaluation scores: {test_evaluation}")
    print("=" * 80)
    print()
    
    # Call the reflection function
    result = reflect_on_editing_failure(
        original_image_path=original_img,
        edited_image_path=edited_img,
        instruction=instruction,
        evaluation_dict=test_evaluation
    )
    
    if result:
        print()
        print("=" * 80)
        print("Reflection result:")
        print("=" * 80)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=" * 80)
    else:
        print()
        print("=" * 80)
        print("Reflection failed.")
        print("=" * 80)
