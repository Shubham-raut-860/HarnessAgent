"""AutoGen adapter — runs a ConversableAgent conversation inside the harness lifecycle."""

from __future__ import annotations

import asyncio
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


class AutoGenAdapter(FrameworkAdapter):
    """Wraps an AutoGen ``ConversableAgent`` or ``GroupChat`` inside the harness.

    AutoGen is synchronous, so the adapter runs ``initiate_chat`` in a thread
    pool via :func:`asyncio.get_running_loop().run_in_executor`.  Messages are
    captured by monkey-patching ``recipient.receive`` before the chat starts and
    restoring it afterward in a ``finally`` block.

    After ``run()`` completes, each captured message is yielded as a
    :class:`~harness.core.context.StepEvent` with ``event_type="message"``.

    Args:
        initiator_agent:            The agent that calls ``initiate_chat``.
        recipient_agent_or_groupchat: The agent (or GroupChat proxy) that
                                      receives the initial message.
        max_turns:                  Maximum conversation turns (default 10).
        event_bus:                  Optional event bus for real-time publishing.
    """

    framework_name = "autogen"

    def __init__(
        self,
        initiator_agent: Any,
        recipient_agent_or_groupchat: Any,
        max_turns: int = 10,
        event_bus: "EventBus | None" = None,
    ) -> None:
        self._initiator = initiator_agent
        self._recipient = recipient_agent_or_groupchat
        self._max_turns = max_turns
        self._event_bus = event_bus
        self._messages: list[dict[str, str]] = []
        self._chat_result: Any = None

    # ------------------------------------------------------------------
    # FrameworkAdapter implementation
    # ------------------------------------------------------------------

    async def run(
        self,
        ctx: "AgentContext",
        input: dict,  # noqa: A002
    ) -> AsyncIterator["StepEvent"]:
        """Run the AutoGen conversation and yield one StepEvent per message.

        The ``input`` dict is probed for the task string in this order:
        ``"task"``, ``"message"``, then ``str(input)`` as a fallback.

        Raises:
            ImportError: If ``pyautogen`` / ``autogen`` is not installed.
        """
        try:
            import autogen  # noqa: F401 — import check only
        except ImportError as exc:
            raise ImportError(
                "AutoGen is not installed.  Install it with:  pip install pyautogen"
            ) from exc

        task: str = input.get("task") or input.get("message") or str(input)

        self._messages = []
        step_counter: list[int] = [0]  # mutable container for closure capture

        # --- Monkey-patch recipient.receive to capture messages in order ---
        original_receive = self._recipient.receive

        def _capturing_receive(
            message: Any,
            sender: Any,
            request_reply: bool | None = None,
            silent: bool = False,
        ) -> Any:
            step_counter[0] += 1
            sender_name: str = getattr(sender, "name", str(sender))
            self._messages.append(
                {"role": sender_name, "content": str(message)}
            )
            return original_receive(message, sender, request_reply, silent)

        self._recipient.receive = _capturing_receive

        try:
            loop = asyncio.get_running_loop()
            self._chat_result = await loop.run_in_executor(
                None,
                lambda: self._initiator.initiate_chat(
                    self._recipient,
                    message=task,
                    max_turns=self._max_turns,
                ),
            )
        finally:
            # Always restore the original method.
            self._recipient.receive = original_receive

        # --- Yield StepEvents for all captured messages ---
        for i, msg in enumerate(self._messages):
            from harness.core.context import StepEvent

            if not ctx.is_budget_ok():
                budget_event = StepEvent(
                    run_id=ctx.run_id,
                    step=i,
                    event_type="budget_exceeded",
                    payload={
                        "framework": self.framework_name,
                        "step_count": ctx.step_count,
                        "max_steps": ctx.max_steps,
                    },
                    timestamp=_utcnow(),
                )
                await self._publish(budget_event)
                yield budget_event
                return

            event = StepEvent(
                run_id=ctx.run_id,
                step=i,
                event_type="message",
                payload={
                    "framework": self.framework_name,
                    "role": msg["role"],
                    "preview": msg["content"][:200],
                },
                timestamp=_utcnow(),
            )
            await self._publish(event)
            yield event

    async def get_result(self) -> FrameworkResult:
        """Return the :class:`FrameworkResult` built from the captured messages.

        The last captured message is used as the final output text.

        Raises:
            RuntimeError: If ``run()`` has not been called yet.
        """
        if self._chat_result is None and not self._messages:
            raise RuntimeError(
                "AutoGenAdapter.get_result() called before run() completed."
            )

        output = self._messages[-1]["content"] if self._messages else ""
        return FrameworkResult(
            framework=self.framework_name,
            output=output,
            steps=len(self._messages),
            metadata={"messages": list(self._messages)},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _publish(self, event: "StepEvent") -> None:
        """Fire-and-forget publish to the optional EventBus."""
        if self._event_bus is None:
            return
        try:
            await self._event_bus.publish(event.run_id, event)
        except Exception as exc:  # pragma: no cover
            logger.debug("AutoGenAdapter: event_bus.publish failed: %s", exc)
