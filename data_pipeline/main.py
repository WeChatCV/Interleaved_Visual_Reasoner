#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image-editing / T2I data pipeline entry point.

Pipeline stages (run sequentially per sample):
    1. Direct edit  (Gemini image edit / T2I)
    2. Multi-metric evaluation
    3. Reflection-based re-edit (up to N attempts)
    4. Multi-step decomposition fallback

The grounding stage has been intentionally removed for the open-source release.
Supports text-to-image samples where ``src_path`` is ``None``.
"""

import base64
import hashlib
import hmac
import json
import os
import shutil
import time
from io import BytesIO
from typing import List, Optional

import PIL
import requests
from PIL import Image

from . import config as _cfg
from .image_scorer import evaluate_images
from .multi_step import generate_editing_steps
from .reflect import PROMPT_RE_EDITING, reflect_on_editing_failure


error_file = "log_error.txt"


class GeminiImageGenerator:
    """Gemini image-generation client (via the chat gateway API)."""

    def __init__(self, api_base_url: str, app_id: str, app_secret: str,
                 source: str = "python-client"):
        self.api_base_url = api_base_url.rstrip("/")
        self.app_id = app_id
        self.app_secret = app_secret
        self.source = source
        self.endpoint = f"{self.api_base_url}/txt2img/gemini/generate"

    def _generate_auth(self, timestamp: int) -> str:
        sign_str = f"x-timestamp: {timestamp}\nx-source: {self.source}"
        sign = hmac.new(
            self.app_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return sign.hex()

    def _image_to_base64(self, image_input) -> str:
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
                f"image_input must be a path or PIL.Image, got {type(image_input)}"
            )
        b64 = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def generate(self, prompt, images=None, model="gemini-3-pro-image-preview",
                 cid=None, image_format="url"):
        timestamp = int(time.time())
        auth = self._generate_auth(timestamp)
        headers = {
            "Content-Type": "application/json",
            "X-AppID": self.app_id,
            "X-Source": self.source,
            "X-Timestamp": str(timestamp),
            "X-Authorization": auth,
        }
        image_urls = []
        if images is not None:
            for image in images:
                if image is None:
                    continue
                if isinstance(image, str) and image.startswith(("http://", "https://")):
                    image_urls.append(image)
                elif isinstance(image, PIL.Image.Image) or isinstance(image, (str, os.PathLike)):
                    image_urls.append(self._image_to_base64(image))
                else:
                    with open(error_file, "a") as ef:
                        ef.write(f"Unsupported image type: {type(image)}\n")
                    raise TypeError("Unsupported image type")
        payload = {
            "prompt": prompt,
            "model": model,
            "image_urls": image_urls,
            "image_format": image_format,
        }
        if cid:
            payload["cid"] = cid
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            raise Exception(f"API error: {result.get('msg', 'Unknown error')}")
        return result

    def decode_base64_image(self, base64_str, save_path=None):
        if base64_str.startswith("data:image"):
            base64_data = base64_str.split(",", 1)[1]
        else:
            base64_data = base64_str
        image_content = base64.b64decode(base64_data)
        if save_path:
            with open(save_path, "wb") as f:
                f.write(image_content)
        return image_content


def generate_image(generator, src_img_path, instruction, retry=3):
    for attempt in range(retry):
        print(f"  Attempt {attempt + 1}/{retry}...")
        try:
            if isinstance(src_img_path, str):
                image_list = [src_img_path]
            else:
                image_list = src_img_path
            result = generator.generate(
                prompt=instruction,
                images=image_list,
                model=config.gemini_model,
                image_format="base64",
            )
            if (result.get("code") == 0 and "data" in result
                    and "urls" in result["data"] and len(result["data"]["urls"]) > 0):
                return result["data"]["urls"][0]
            print(f"Generation failed: {result}")
        except Exception as e:
            print(f"    Error during generation: {e}")
    print(f"  Failed to generate image after {retry} attempts.")
    return None


def evaluate_image_pair(original_image, edited_image, instruction,
                        explanation="", metrics=None):
    metrics = metrics or []
    scores = evaluate_images(
        original_image=original_image,
        edited_image=edited_image,
        instruction=instruction,
        explanation=explanation,
        metrics=metrics,
    )
    for metric in metrics:
        score_key = f"{metric}_score"
        reason_key = f"{metric}_reasoning"
        if score_key not in scores or reason_key not in scores:
            return None
        if scores[score_key] is None or scores[reason_key] is None:
            return None
    return scores


def edit_and_save(generator, src_img_path, instruction, output_path):
    response = generate_image(generator, src_img_path, instruction)
    if response is not None:
        generator.decode_base64_image(response, save_path=output_path)
        return output_path
    return None


def if_edit_end(score_dict, thresholds):
    for metric, threshold in thresholds.items():
        score = score_dict.get(metric)
        if score is None or score < threshold:
            return False
    return True


def _dump_and_move_to_fail(return_json, out_dir, task_id, fail_root):
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(return_json, f, indent=4, ensure_ascii=False)
    fail_dir = os.path.join(fail_root, task_id)
    os.makedirs(fail_dir, exist_ok=True)
    shutil.move(out_dir, fail_dir)


class Config:
    def __init__(self):
        self.use_explanation = True
        self.out_put_root_dir = "./outputs/success"
        self.out_put_fail_root_dir = "./outputs/fail"
        self.if_direct_edit = True
        self.direct_edit_evaluation_metrics = ["consistency", "instruction", "quality", "knowledge"]
        self.direct_edit_evaluation_thresholds = {
            "consistency_score": 3, "instruction_score": 5,
            "quality_score": 3, "knowledge_score": 3,
        }
        self.reflection_attempt = 3
        self.reflection_evaluation_metrics = ["consistency", "instruction", "quality", "knowledge"]
        self.reflection_evaluation_thresholds = {
            "consistency_score": 3, "instruction_score": 5,
            "quality_score": 3, "knowledge_score": 3,
        }
        self.use_multi_step = True
        self.multi_step_attempt = 3
        self.multi_step_evaluation_metrics = ["consistency", "instruction", "quality", "knowledge"]
        self.multi_step_evaluation_thresholds = {
            "consistency_score": 3, "instruction_score": 5,
            "quality_score": 4, "knowledge_score": 3,
        }
        self.gemini_api_base_url = _cfg.GEMINI_IMG_API_BASE_URL
        self.gemini_app_id = _cfg.GEMINI_IMG_APP_ID
        self.gemini_app_secret = _cfg.GEMINI_IMG_APP_SECRET
        self.gemini_source = _cfg.GEMINI_IMG_SOURCE
        self.gemini_model = _cfg.GEMINI_IMG_MODEL


config = Config()


def single_date_processing(data_dict):
    generator = GeminiImageGenerator(
        api_base_url=config.gemini_api_base_url,
        app_id=config.gemini_app_id,
        app_secret=config.gemini_app_secret,
        source=config.gemini_source,
    )

    src_img_path = data_dict["src_path"]
    edit_explanation = data_dict.get("explanation", "") if config.use_explanation else ""
    instruction = data_dict["edit_prompt"]
    task_id = f"{data_dict['key']}"

    out_dir = os.path.join(config.out_put_root_dir, task_id)
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(src_img_path, list):
        for i, p in enumerate(src_img_path):
            shutil.copy(p, os.path.join(out_dir, f"original_image_{i}.png"))
    elif src_img_path is not None:
        shutil.copy(src_img_path, os.path.join(out_dir, "original_image.png"))

    return_json = {
        "task_id": task_id,
        "edit_key": data_dict.get("edit_key"),
        "original_image_path": src_img_path,
        "edit_prompt": instruction,
        "direct_edit_image_path": None,
        "evaluation": {},
        "reflect": [],
        "multi_step": [],
        "final_edited_image_path": None,
        "failure_reason": None,
        "original_success": False,
    }

    use_exist_edit_result = False
    if data_dict.get("edit_path") is not None:
        direct_path = os.path.join(out_dir, "step1_direct_edit_result.png")
        shutil.copy(data_dict["edit_path"], direct_path)
        use_exist_edit_result = True
    else:
        direct_path = edit_and_save(
            generator, src_img_path, instruction,
            os.path.join(out_dir, "step1_direct_edit_result.png"),
        )
        if direct_path is None:
            return_json["failure_reason"] = "Direct edit failed"
            _dump_and_move_to_fail(return_json, out_dir, task_id, config.out_put_fail_root_dir)
            return None

    return_json["direct_edit_image_path"] = direct_path

    direct_scores = evaluate_image_pair(
        original_image=src_img_path,
        edited_image=direct_path,
        instruction=instruction,
        explanation=edit_explanation,
        metrics=config.direct_edit_evaluation_metrics,
    )
    if direct_scores is None:
        return_json["failure_reason"] = "Direct edit evaluation failed"
        _dump_and_move_to_fail(return_json, out_dir, task_id, config.out_put_fail_root_dir)
        return None
    return_json["evaluation"] = direct_scores

    if if_edit_end(direct_scores, config.direct_edit_evaluation_thresholds):
        return_json["final_edited_image_path"] = direct_path
        if use_exist_edit_result:
            return_json["original_success"] = True
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(return_json, f, indent=4, ensure_ascii=False)
        return return_json

    reflect_path = direct_path
    reflect_scores = direct_scores
    for i in range(config.reflection_attempt):
        round_dict = {}
        reflect_dict = reflect_on_editing_failure(
            original_image_path=src_img_path,
            edited_image_path=reflect_path,
            instruction=instruction,
            evaluation_dict=reflect_scores,
        )
        if reflect_dict is None:
            return_json["failure_reason"] = "Reflection generation failed"
            _dump_and_move_to_fail(return_json, out_dir, task_id, config.out_put_fail_root_dir)
            return None

        round_dict["reflect_thinking"] = reflect_dict.copy()
        reflect_dict["editing_instruction"] = instruction
        reflect_instruction = PROMPT_RE_EDITING.format(**reflect_dict)

        if isinstance(src_img_path, list):
            to_reflect_images = src_img_path + [reflect_path]
        elif src_img_path is not None:
            to_reflect_images = [src_img_path, reflect_path]
        else:
            to_reflect_images = [reflect_path]

        reflect_path = edit_and_save(
            generator, to_reflect_images, reflect_instruction,
            os.path.join(out_dir, f"step2_reflect_edit_result_{i}.png"),
        )
        if reflect_path is None:
            return_json["failure_reason"] = "Reflection edit failed"
            _dump_and_move_to_fail(return_json, out_dir, task_id, config.out_put_fail_root_dir)
            return None

        reflect_scores = evaluate_image_pair(
            original_image=src_img_path,
            edited_image=reflect_path,
            instruction=instruction,
            explanation=edit_explanation,
            metrics=config.reflection_evaluation_metrics,
        )
        if reflect_scores is None:
            return_json["failure_reason"] = "Reflection evaluation failed"
            _dump_and_move_to_fail(return_json, out_dir, task_id, config.out_put_fail_root_dir)
            return None

        round_dict["reflect_image_path"] = reflect_path
        round_dict["evaluation"] = reflect_scores
        return_json["reflect"].append(round_dict)

        if if_edit_end(reflect_scores, config.reflection_evaluation_thresholds):
            return_json["final_edited_image_path"] = reflect_path
            with open(os.path.join(out_dir, "result.json"), "w") as f:
                json.dump(return_json, f, indent=4, ensure_ascii=False)
            return return_json

    if config.use_multi_step:
        editing_steps = generate_editing_steps(
            image_path=src_img_path, instruction=instruction,
        )
        if editing_steps is None:
            return_json["failure_reason"] = "Multi-step generation failed"
            _dump_and_move_to_fail(return_json, out_dir, task_id, config.out_put_fail_root_dir)
            return None

        current = src_img_path
        for step_idx, step_instr in enumerate(editing_steps):
            ok = False
            for _ in range(config.multi_step_attempt):
                ms_path = edit_and_save(
                    generator, current, step_instr,
                    os.path.join(out_dir, f"step4_multi_step_edit_result_{step_idx}.png"),
                )
                if ms_path is None:
                    continue
                ms_scores = evaluate_image_pair(
                    original_image=current, edited_image=ms_path,
                    instruction=step_instr, explanation="",
                    metrics=config.multi_step_evaluation_metrics,
                )
                if ms_scores is None:
                    continue
                step_dict = {
                    "step_instruction": step_instr,
                    "step_image_path": ms_path,
                    "evaluation": ms_scores,
                }
                if if_edit_end(ms_scores, config.multi_step_evaluation_thresholds):
                    return_json["multi_step"].append(step_dict)
                    current = ms_path
                    if isinstance(src_img_path, list) and len(src_img_path) > 1:
                        current = src_img_path + [ms_path]
                    ok = True
                    break
            if not ok:
                return_json["failure_reason"] = "Multi-step editing failed"
                _dump_and_move_to_fail(return_json, out_dir, task_id, config.out_put_fail_root_dir)
                return None

        return_json["final_edited_image_path"] = current
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(return_json, f, indent=4, ensure_ascii=False)
        return return_json

    return_json["failure_reason"] = "All editing strategies exhausted"
    _dump_and_move_to_fail(return_json, out_dir, task_id, config.out_put_fail_root_dir)
    return None


def run(input_json_path):
    with open(input_json_path, "r") as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} samples from {input_json_path}")
    for idx, item in enumerate(data_list):
        print(f"\n=== [{idx + 1}/{len(data_list)}] key={item.get('key')} ===")
        try:
            single_date_processing(item)
        except Exception as e:
            print(f"Sample failed with exception: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Image editing / T2I data pipeline")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--success_dir", type=str, default=None)
    parser.add_argument("--fail_dir", type=str, default=None)
    args = parser.parse_args()
    if args.success_dir:
        config.out_put_root_dir = args.success_dir
    if args.fail_dir:
        config.out_put_fail_root_dir = args.fail_dir
    os.makedirs(config.out_put_root_dir, exist_ok=True)
    os.makedirs(config.out_put_fail_root_dir, exist_ok=True)
    run(args.input_json)
