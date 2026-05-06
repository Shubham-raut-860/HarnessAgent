"""Hermes / Qwen XML function-calling provider for HarnessAgent.

Targets NousResearch Hermes-2-Pro and Qwen models that emit tool calls as
``<tool_call>`` XML tags rather than OpenAI-native function-calling JSON.

Wire-protocol format emitted by the model::

    <tool_call>
    {"name": "tool_name", "arguments": {"key": "value"}}
    </tool_call>

Tool results are injected back as::

    <tool_response>
    {"name": "tool_name", "content": "<result>"}
    </tool_response>

The provider talks to any OpenAI-compatible backend (SGLang, vLLM, Ollama)
via the standard ``/v1/chat/completions`` endpoint.  Native tool-calling is
intentionally disabled; all tool schema injection and call parsing happens
in this layer.
"""

from __future__ import annotations

import json
import logging
import re
import uuid as _uuid
from collections.abc import AsyncIterator
from typing import Any

from harness.core.context import LLMResponse, ToolCall
from harness.core.errors import FailureClass, LLMError
from harness.llm.local import ModelCapabilities, OpenAICompatProvider

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)
_TOOL_RESPONSE_TAG = "<tool_response>\n{body}\n</tool_response>"

_SYSTEM_TOOL_PREAMBLE = (
    "You have access to the following tools. "
    "To call a tool, respond ONLY with one or more <tool_call> blocks "
    "containing a JSON object with \"name\" and \"arguments\" keys. "
    "Do not add any other text when calling tools.\n\n"
    "Available tools (JSON schema):\n"
    "{tool_json}"
)


def format_tool_result(name: str, content: str) -> str:
    """Build a ``<tool_response>`` message to append to the conversation."""
    body = json.dumps({"name": name, "content": content})
    return _TOOL_RESPONSE_TAG.format(body=body)


class HermesXMLProvider(OpenAICompatProvider):
    """OpenAI-compat provider that speaks the Hermes / Qwen XML tool-call dialect.

    Inherits all transport, retry, and streaming logic from
    :class:`~harness.llm.local.OpenAICompatProvider`.  Two behaviours are
    overridden:

    1. **System-prompt injection** — tool schemas are serialised as JSON and
       embedded in the system prompt instead of the OpenAI ``tools`` parameter.
    2. **Response parsing** — ``<tool_call>`` XML blocks are extracted from the
       model's text reply and converted to :class:`~harness.core.context.ToolCall`
       objects.

    Args:
        base_url:        Base URL of the SGLang / vLLM / Ollama server.
        model:           Model identifier (e.g. ``"NousResearch/Hermes-2-Pro-Llama-3-8B"``).
        api_key:         API key for the backend (default ``"not-required"``).
        context_window:  Token context window for this model (default 8 192).
        timeout:         HTTP request timeout in seconds (default 120).
    """

    provider_name: str = "hermes_xml"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-required",
        context_window: int = 8_192,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            capabilities=ModelCapabilities(
                supports_tool_calling=False,  # use XML injection, not native
                supports_system_prompt=True,
                context_window=context_window,
            ),
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Tool schema injection
    # ------------------------------------------------------------------

    def _inject_tools_into_system(
        self,
        system: str | None,
        tools: list[dict[str, Any]],
    ) -> str:
        """Embed tool schemas as a Hermes-format JSON list in the system prompt."""
        schemas = [
            {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or t.get("input_schema", {}),
            }
            for t in tools
        ]
        preamble = _SYSTEM_TOOL_PREAMBLE.format(
            tool_json=json.dumps(schemas, indent=2)
        )
        if system:
            return f"{system}\n\n{preamble}"
        return preamble

    # ------------------------------------------------------------------
    # Tool call parsing
    # ------------------------------------------------------------------

    def _parse_tool_calls_from_text(self, text: str) -> list[ToolCall]:
        """Extract all ``<tool_call>`` blocks from the model's reply."""
        tool_calls: list[ToolCall] = []
        for match in _TOOL_CALL_RE.finditer(text):
            raw = match.group(1).strip()
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("HermesXMLProvider: could not parse tool_call JSON: %r", raw[:120])
                continue

            name = obj.get("name") or obj.get("tool", "")
            args = obj.get("arguments") or obj.get("args") or {}
            if not name:
                continue
            if not isinstance(args, dict):
                args = {}

            tool_calls.append(
                ToolCall(
                    id=_uuid.uuid4().hex[:8],
                    name=str(name),
                    args=args,
                )
            )
        return tool_calls

    # ------------------------------------------------------------------
    # complete() override — strip tool blocks from content when calls found
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a completion and parse Hermes XML tool calls from the reply.

        Tool schemas are injected into the system prompt before the request.
        ``<tool_call>`` blocks in the response are parsed into
        :class:`~harness.core.context.ToolCall` objects and stripped from the
        plain-text ``content``.
        """
        response = await super().complete(
            messages,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            **kwargs,
        )

        if tools:
            extracted = self._parse_tool_calls_from_text(response.content)
            if extracted:
                clean_content = _TOOL_CALL_RE.sub("", response.content).strip()
                return LLMResponse(
                    content=clean_content,
                    tool_calls=extracted,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    model=response.model,
                    provider=self.provider_name,
                    cached=response.cached,
                )

        return LLMResponse(
            content=response.content,
            tool_calls=response.tool_calls,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=response.model,
            provider=self.provider_name,
            cached=response.cached,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream raw text deltas (tool call tags are streamed verbatim)."""
        async for chunk in super().stream(messages, **kwargs):
            yield chunk
