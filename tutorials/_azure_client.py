"""Shared Azure OpenAI client used by all tutorials."""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AZURE_KEY      = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
MODEL_NAME     = os.environ.get("AZURE_MODEL_NAME", "gpt-5.2-chat")


def get_client() -> OpenAI:
    """Return a configured OpenAI client pointing at Azure."""
    return OpenAI(api_key=AZURE_KEY, base_url=AZURE_ENDPOINT)


# gpt-5.2-chat uses internal reasoning tokens before visible output.
# Always use high max_completion_tokens or responses will be empty.
DEFAULT_MAX_TOKENS = 4000


async def chat(system: str, user: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Single async-compatible chat call to Azure OpenAI."""
    import asyncio
    loop = asyncio.get_event_loop()
    client = get_client()

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})

    def _call():
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=msgs,
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return await loop.run_in_executor(None, _call)


def chat_sync(system: str, user: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Synchronous chat call — used by CrewAI, AutoGen, Agno."""
    client = get_client()
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=msgs,
        max_completion_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def llm_config_dict() -> dict:
    """AutoGen-style llm_config using Azure endpoint."""
    return {
        "config_list": [{
            "model":    MODEL_NAME,
            "api_key":  AZURE_KEY,
            "base_url": AZURE_ENDPOINT,
            "api_type": "openai",
        }],
        "max_completion_tokens": DEFAULT_MAX_TOKENS,
    }
