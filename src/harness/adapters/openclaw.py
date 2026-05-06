"""OpenClaw adapter — runs an OpenClaw RL rollout inside the HarnessAgent lifecycle.

OpenClaw is a multi-turn RL training framework that pairs a Qwen / Hermes model
(via SGLang) with a Docker-based execution environment.  Each *episode* is a
conversation loop:

    observation → LLM generates action (Qwen XML tool call) → Docker env steps
    → new observation → … → terminal reward

This adapter wraps a **single rollout episode** (inference / evaluation mode,
not gradient updates) so the harness can observe, cost-track, and safety-check
every step.  RL training itself happens outside the harness.

Usage::

    from harness.adapters.openclaw import OpenClawAdapter

    adapter = OpenClawAdapter(
        env=my_openclaw_env,          # OpenClaw DockerEnv or gym-compatible env
        provider=hermes_xml_provider, # HermesXMLProvider or any harness provider
        system_prompt="You are a coding agent…",
        max_turns=20,
    )
    adapter.attach_harness(safety_pipeline=pipe, cost_tracker=tracker)

    async for event in adapter.run_with_harness(ctx, input={"task": "Fix the bug"}):
        print(event)

    result = await adapter.get_result()
    print("reward:", result.metadata["total_reward"])

The adapter also supports MCP tool injection via ``attach_mcp()``.  MCP tools
are merged into the Qwen XML tool list that the provider receives on each turn.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator

from harness.adapters.base import FrameworkAdapter, FrameworkResult

if TYPE_CHECKING:
    from harness.core.context import AgentContext, StepEvent
    from harness.observability.event_bus import EventBus

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OpenClawAdapter(FrameworkAdapter):
    """Wraps an OpenClaw rollout episode inside the HarnessAgent lifecycle.

    The adapter drives a multi-turn loop:

    1. Reset the environment to get the initial observation.
    2. On each turn, call the LLM provider with the current conversation history
       and any available tools (framework tools + MCP-injected tools).
    3. If the LLM emits a tool call, execute it via the environment ``step()``
       and append the result to the conversation.
    4. If the LLM emits plain text (no tool call), treat it as the final answer
       and terminate the episode.
    5. The episode also ends when the environment signals ``done=True`` or when
       ``max_turns`` is exhausted.

    A :class:`~harness.adapters.base.FrameworkResult` is produced after the
    episode with ``total_reward``, ``turns``, and the full trajectory in
    ``metadata``.

    Args:
        env:            OpenClaw environment.  Must expose async ``reset()``,
                        ``step(action)``, and ``close()`` methods.  ``step()``
                        must return ``(observation, reward, done, info)`` where
                        ``observation`` is a ``str`` or ``dict`` with a
                        ``"content"`` key.
        provider:       Any harness LLM provider (``complete()`` interface).
                        Use :class:`~harness.llm.hermes.HermesXMLProvider` for
                        Qwen / Hermes models with Qwen XML tool calls.
        tools:          Static tool definitions to include every turn (in addition
                        to any MCP-injected tools).  Each entry is a dict with
                        ``"name"``, ``"description"``, and ``"input_schema"``.
        system_prompt:  System prompt for the LLM (overrides env default if set).
        max_turns:      Maximum conversation turns before the episode is cut off.
        max_tokens:     Token budget per LLM call.
        event_bus:      Optional event bus for real-time step publishing.
    """

    framework_name = "openclaw"

    def __init__(
        self,
        env: Any,
        provider: Any,
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        max_turns: int = 20,
        max_tokens: int = 2_048,
        event_bus: "EventBus | None" = None,
    ) -> None:
        super().__init__()
        self._env = env
        self._provider = provider
        self._static_tools: list[dict[str, Any]] = list(tools or [])
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._event_bus = event_bus

        # Per-episode state (reset in run())
        self._trajectory: list[dict[str, Any]] = []
        self._total_reward: float = 0.0
        self._turns: int = 0
        self._final_output: str = ""
        self._done: bool = False

    # ------------------------------------------------------------------
    # FrameworkAdapter implementation
    # ------------------------------------------------------------------

    async def run(
        self,
        ctx: "AgentContext",
        input: dict,  # noqa: A002
    ) -> AsyncIterator["StepEvent"]:
        """Execute one OpenClaw rollout episode and stream StepEvents.

        ``input`` must contain either a ``"task"`` key (plain text) or a
        ``"messages"`` key (pre-built message list for the first turn).

        Yields one :class:`~harness.core.context.StepEvent` per LLM call and
        one per tool execution.
        """
        from harness.core.context import LLMResponse, StepEvent, ToolCall, ToolResult

        # --- Reset per-episode state ---
        self._trajectory = []
        self._total_reward = 0.0
        self._turns = 0
        self._final_output = ""
        self._done = False

        # --- Resolve all tools (static + MCP-injected) ---
        mcp_tools = await self._resolve_mcp_tools()
        all_tools = self._static_tools + [
            {k: v for k, v in t.items() if k != "_mcp_client"}
            for t in mcp_tools
        ]
        # Keep a quick lookup from name → mcp_client for execution
        mcp_tool_map: dict[str, Any] = {
            t["name"]: t["_mcp_client"]
            for t in mcp_tools
            if "_mcp_client" in t
        }

        # --- Reset environment ---
        try:
            obs = await _ensure_async(self._env.reset())
        except Exception as exc:
            error_event = StepEvent(
                run_id=ctx.run_id,
                step=0,
                event_type="failed",
                payload={"framework": self.framework_name, "error": str(exc)},
                timestamp=_utcnow(),
            )
            await self._publish(error_event)
            yield error_event
            return

        # --- Build initial conversation ---
        task_text = input.get("task") or input.get("message") or ""
        messages: list[dict[str, Any]] = list(input.get("messages") or [])
        if not messages and task_text:
            messages = [{"role": "user", "content": task_text}]

        # Append initial environment observation if env returned one
        obs_text = _obs_to_str(obs)
        if obs_text and obs_text not in (task_text, ""):
            messages.append({"role": "user", "content": obs_text})

        system = self._system_prompt or _env_system_prompt(self._env)

        # --- Main rollout loop ---
        for turn in range(self._max_turns):
            self._turns = turn + 1

            if not ctx.is_budget_ok():
                budget_event = StepEvent(
                    run_id=ctx.run_id,
                    step=turn,
                    event_type="budget_exceeded",
                    payload={
                        "framework": self.framework_name,
                        "turn": turn,
                        "max_steps": ctx.max_steps,
                    },
                    timestamp=_utcnow(),
                )
                await self._publish(budget_event)
                yield budget_event
                break

            ctx.tick()

            # --- LLM call ---
            try:
                llm_response: LLMResponse = await self._provider.complete(
                    messages,
                    max_tokens=self._max_tokens,
                    system=system,
                    tools=all_tools or None,
                )
            except Exception as exc:
                logger.warning("OpenClaw turn %d LLM error: %s", turn, exc)
                error_event = StepEvent(
                    run_id=ctx.run_id,
                    step=turn,
                    event_type="failed",
                    payload={"framework": self.framework_name, "error": str(exc), "turn": turn},
                    timestamp=_utcnow(),
                )
                await self._publish(error_event)
                yield error_event
                break

            # Record token usage
            ctx.token_count += llm_response.input_tokens + llm_response.output_tokens

            self._trajectory.append({
                "turn": turn,
                "role": "assistant",
                "content": llm_response.content,
                "tool_calls": [
                    {"name": tc.name, "args": tc.args} for tc in llm_response.tool_calls
                ],
                "input_tokens": llm_response.input_tokens,
                "output_tokens": llm_response.output_tokens,
            })

            llm_event = StepEvent(
                run_id=ctx.run_id,
                step=turn,
                event_type="llm_called",
                payload={
                    "framework": self.framework_name,
                    "turn": turn,
                    "content": llm_response.content[:300],
                    "tool_calls": len(llm_response.tool_calls),
                    "input_tokens": llm_response.input_tokens,
                    "output_tokens": llm_response.output_tokens,
                },
                timestamp=_utcnow(),
            )
            await self._publish(llm_event)
            yield llm_event

            # Append assistant turn to history
            messages.append({"role": "assistant", "content": llm_response.content})

            # --- No tool calls → final answer ---
            if not llm_response.tool_calls:
                self._final_output = llm_response.content
                self._done = True

                done_event = StepEvent(
                    run_id=ctx.run_id,
                    step=turn,
                    event_type="completed",
                    payload={
                        "framework": self.framework_name,
                        "output": self._final_output[:300],
                        "total_reward": self._total_reward,
                        "turns": self._turns,
                    },
                    timestamp=_utcnow(),
                )
                await self._publish(done_event)
                yield done_event
                break

            # --- Execute each tool call ---
            for tc in llm_response.tool_calls:
                tool_result_content, step_reward, step_done = await self._execute_tool(
                    tc, mcp_tool_map
                )
                self._total_reward += step_reward

                tool_event = StepEvent(
                    run_id=ctx.run_id,
                    step=turn,
                    event_type="tool_called",
                    payload={
                        "framework": self.framework_name,
                        "turn": turn,
                        "tool": tc.name,
                        "args": tc.args,
                        "result": tool_result_content[:300],
                        "reward": step_reward,
                        "done": step_done,
                    },
                    timestamp=_utcnow(),
                )
                await self._publish(tool_event)
                yield tool_event

                self._trajectory.append({
                    "turn": turn,
                    "role": "tool",
                    "tool": tc.name,
                    "result": tool_result_content,
                    "reward": step_reward,
                    "done": step_done,
                })

                # Append tool result to conversation
                messages.append({
                    "role": "user",
                    "content": _format_tool_result(tc.name, tool_result_content),
                })

                if step_done:
                    self._done = True
                    self._final_output = tool_result_content
                    break

            if self._done:
                break

        else:
            # max_turns exhausted without terminal
            logger.info(
                "OpenClaw episode reached max_turns=%d without terminating (reward=%.3f)",
                self._max_turns,
                self._total_reward,
            )
            if not self._final_output and messages:
                self._final_output = messages[-1].get("content", "")

    async def get_result(self) -> FrameworkResult:
        """Return the rollout result after ``run()`` completes."""
        return FrameworkResult(
            framework=self.framework_name,
            output=self._final_output,
            steps=self._turns,
            metadata={
                "total_reward": self._total_reward,
                "turns": self._turns,
                "done": self._done,
                "trajectory": self._trajectory,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        tc: Any,
        mcp_tool_map: dict[str, Any],
    ) -> tuple[str, float, bool]:
        """Execute one tool call and return (result_str, reward, done)."""
        # MCP tool: delegate to the MCP client
        if tc.name in mcp_tool_map:
            client = mcp_tool_map[tc.name]
            try:
                raw = await client.call_tool(tc.name, tc.args)
                result_str = str(raw)
            except Exception as exc:
                logger.warning("MCP tool %r raised: %s", tc.name, exc)
                result_str = f"Tool error: {exc}"
            return result_str, 0.0, False

        # Environment tool: pass the full XML-format action to env.step()
        action = _format_tool_call_action(tc.name, tc.args)
        try:
            step_out = await _ensure_async(self._env.step(action))
        except Exception as exc:
            logger.warning("OpenClaw env.step raised for tool %r: %s", tc.name, exc)
            return f"Environment error: {exc}", 0.0, False

        # Normalise the step output
        if isinstance(step_out, tuple):
            obs, reward, done = step_out[0], step_out[1], step_out[2]
        else:
            obs, reward, done = step_out, 0.0, False

        return _obs_to_str(obs), float(reward), bool(done)

    async def _publish(self, event: "StepEvent") -> None:
        if self._event_bus is None:
            return
        try:
            await self._event_bus.publish(event.run_id, event)
        except Exception as exc:
            logger.debug("OpenClawAdapter: event_bus.publish failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ensure_async(obj: Any) -> Any:
    """Await coroutines; return sync values directly."""
    import asyncio
    if asyncio.iscoroutine(obj):
        return await obj
    return obj


def _obs_to_str(obs: Any) -> str:
    """Convert an environment observation to a plain string."""
    if obs is None:
        return ""
    if isinstance(obs, str):
        return obs
    if isinstance(obs, dict):
        return obs.get("content") or obs.get("observation") or str(obs)
    return str(obs)


def _env_system_prompt(env: Any) -> str | None:
    """Return the environment's default system prompt if it exposes one."""
    return getattr(env, "system_prompt", None) or getattr(env, "task_description", None)


def _format_tool_call_action(name: str, args: dict) -> str:
    """Serialise a tool call in Qwen XML format for env.step()."""
    import json
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': args})}\n</tool_call>"


def _format_tool_result(name: str, content: str) -> str:
    """Wrap a tool result in Qwen XML format to append to the conversation."""
    import json
    body = json.dumps({"name": name, "content": content})
    return f"<tool_response>\n{body}\n</tool_response>"
