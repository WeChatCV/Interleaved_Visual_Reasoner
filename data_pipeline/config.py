#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Centralized API credentials.

All values are read from environment variables and default to empty strings,
so the user must explicitly configure them before running the pipeline.

You can either:

1. Export environment variables before launching the pipeline, e.g.::

       export GEMINI_IMG_API_BASE_URL="https://your.gateway.example/api"
       export GEMINI_IMG_APP_ID="your-app-id"
       export GEMINI_IMG_APP_SECRET="your-app-secret"

2. Or edit the default strings below directly.

The pipeline calls three (potentially distinct) services:

* ``GEMINI_IMG_*``   - image generation / editing endpoint
                       (HMAC-SHA256 authenticated, the chat gateway-style gateway).
* ``CHAT_*``         - text/chat endpoint that drives reflection and
                       multi-step decomposition (same gateway, different
                       endpoint path).
* ``GPT_*``          - OpenAI-compatible vision-LLM endpoint used by the
                       image scorer (e.g. GPT-4o behind a gateway).
* ``QWEN_*``         - optional OpenAI-compatible Qwen-VL endpoint used as
                       a fallback evaluator.
"""

import os


# ---------------------------------------------------------------------------
# Gemini image generation / editing endpoint
# ---------------------------------------------------------------------------
GEMINI_IMG_API_BASE_URL = os.environ.get("GEMINI_IMG_API_BASE_URL", "")
GEMINI_IMG_APP_ID = os.environ.get("GEMINI_IMG_APP_ID", "")
GEMINI_IMG_APP_SECRET = os.environ.get("GEMINI_IMG_APP_SECRET", "")
GEMINI_IMG_SOURCE = os.environ.get("GEMINI_IMG_SOURCE", "python-client")
GEMINI_IMG_MODEL = os.environ.get("GEMINI_IMG_MODEL", "gemini-3-pro-image-preview")


# ---------------------------------------------------------------------------
# Gemini text / chat endpoint (reflection, multi-step planning)
# ---------------------------------------------------------------------------
CHAT_API_BASE_URL = os.environ.get("CHAT_API_BASE_URL", "")
CHAT_APP_ID = os.environ.get("CHAT_APP_ID", "")
CHAT_APP_SECRET = os.environ.get("CHAT_APP_SECRET", "")
CHAT_SOURCE = os.environ.get("CHAT_SOURCE", "python-client")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemini-3-pro-preview")


# ---------------------------------------------------------------------------
# Vision LLM used as the primary image scorer (e.g. GPT-4o behind a gateway)
# ---------------------------------------------------------------------------
GPT_API_BASE_URL = os.environ.get("GPT_API_BASE_URL", "")
GPT_APP_ID = os.environ.get("GPT_APP_ID", "")
GPT_APP_KEY = os.environ.get("GPT_APP_KEY", "")
GPT_SOURCE = os.environ.get("GPT_SOURCE", "python-client")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# Qwen-VL fallback scorer (OpenAI-compatible server, e.g. vLLM)
# ---------------------------------------------------------------------------
QWEN_API_BASE_URL = os.environ.get("QWEN_API_BASE_URL", "")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "EMPTY")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "")
