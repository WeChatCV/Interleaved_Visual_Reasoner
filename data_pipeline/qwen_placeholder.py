#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen API calling module - for image evaluation.

Provides a Qwen API calling function that offers the same functionality
as call_gpt_api in image_scorer.py.
"""

import os
import base64
import time
from typing import Optional, Union, List
from openai import OpenAI


def call_qwen_api(
    prompt: str,
    original_base64: Optional[Union[str, List[str]]] = None,
    edited_base64: Optional[str] = None,
    max_retries: int = 3,
    api_base_url: str = None,
    api_key: str = None,
    model: str = None
) -> str:
    """
    Call the Qwen API for image evaluation.
    
    Args:
        prompt: Evaluation prompt text.
        original_base64: Base64-encoded original image(s) or image file path(s);
                         can be a single string or a list of strings.
        edited_base64: Base64-encoded edited image or its file path.
        max_retries: Maximum number of retry attempts.
        api_base_url: API base URL (defaults to QWEN_API_BASE_URL env var).
        api_key: API key (defaults to QWEN_API_KEY env var, falls back to "EMPTY").
        model: Model name (defaults to QWEN_MODEL env var; auto-detected if None).
    
    Returns:
        API response content.
    """
    # Helper: convert an image path or base64 string to base64
    def ensure_base64(image_input: Optional[str]) -> Optional[str]:
        """
        Ensure the input is in base64 format.
        
        Args:
            image_input: A base64 string or an image file path.
            
        Returns:
            Base64-encoded string, or None if the input is None.
        """
        if image_input is None:
            return None
        
        # Check if it's a file path (exists and is a file).
        # Guard against very long strings causing issues with os.path.isfile.
        if len(image_input) < 4096 and os.path.isfile(image_input):
            try:
                with open(image_input, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                print(f"⚠️ Failed to read image file {image_input}: {e}")
                return None
        else:
            # Assume it is already a base64 string
            return image_input
    
    # Convert image inputs to a base64 list
    original_images_b64 = []
    if original_base64:
        if isinstance(original_base64, list):
            for img in original_base64:
                b64 = ensure_base64(img)
                if b64:
                    original_images_b64.append(b64)
        else:
            b64 = ensure_base64(original_base64)
            if b64:
                original_images_b64.append(b64)

    edited_base64 = ensure_base64(edited_base64)
    
    # Read endpoint and credentials from environment variables (no defaults).
    api_base_url = api_base_url or os.environ.get("QWEN_API_BASE_URL", "")
    api_key = api_key or os.environ.get("QWEN_API_KEY", "EMPTY")

    if not api_base_url:
        print("⚠️ QWEN_API_BASE_URL is not configured; skipping Qwen fallback.")
        return ""

    # Initialize the OpenAI-compatible client for Qwen
    client = OpenAI(
        api_key=api_key,
        base_url=api_base_url,
    )
    
    # Auto-detect model name if not provided
    detected_model = None
    if not model and not os.environ.get("QWEN_MODEL"):
        try:
            print(f"Connecting to API: {api_base_url} ...")
            models = client.models.list()
            if models.data:
                detected_model = models.data[0].id
                print(f"✓ Connected successfully! Auto-detected model: {detected_model}")
        except Exception as e:
            print(f"⚠️ Failed to connect to {api_base_url}: {e}")

    model = model or os.environ.get("QWEN_MODEL", detected_model or "qwen-vl")
    
    # Build message content
    message_content = [{"type": "text", "text": prompt}]
    
    if original_images_b64:
        if len(original_images_b64) == 1:
            message_content.extend([
                {"type": "text", "text": "This is the Input Image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{original_images_b64[0]}"}
                }
            ])
        else:
            for i, b64 in enumerate(original_images_b64):
                message_content.extend([
                    {"type": "text", "text": f"This is Input Image {i+1}:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    }
                ])
    
    if edited_base64:
        message_content.extend([
            {"type": "text", "text": "This is the Generated Image:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{edited_base64}"}
            }
        ])
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                max_tokens=4096,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    print("API call failed after all retries")
    return ""


# Test function
if __name__ == "__main__":
    # Simple test
    test_prompt = "Describe what you see in this image."
    
    # Test with no image input
    print("Testing text-only prompt...")
    response = call_qwen_api(test_prompt)
    print(f"Response: {response}\n")
    
    # If a test image is available, test with an image
    # test_image_path = "/path/to/your/test_image.jpg"
    # if os.path.exists(test_image_path):
    #     print(f"Testing with image: {test_image_path}")
    #     with open(test_image_path, "rb") as f:
    #         image_base64 = base64.b64encode(f.read()).decode('utf-8')
        
    #     response = call_qwen_api(
    #         prompt="Describe this image in detail.",
    #         original_base64=image_base64
    #     )
    #     print(f"Response: {response[:200]}...")
    # else:
    #     print(f"Test image not found: {test_image_path}")
