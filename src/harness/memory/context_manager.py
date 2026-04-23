"""Context window manager: prevents context overflow in long agent loops."""

from __future__ import annotations

import logging
from typing import Any

from harness.memory.embedder import estimate_tokens
from harness.memory.schemas import ConversationMessage, ContextWindow

logger = logging.getLogger(__name__)


class ContextWindowManager:
    """
    Manages the active context window sent to the LLM.

    Responsibilities:
    - Track token budget across system prompt, retrieved context, and messages.
    - Implement a sliding window that keeps the most recent messages.
    - Optionally summarise dropped messages via an LLM call.
    """

    def __init__(
        self,
        max_tokens: int,
        summarizer_provider: Any = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._summarizer = summarizer_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fit(
        self,
        messages: list[ConversationMessage],
        system_tokens: int = 0,
        retrieved_tokens: int = 0,
        reserve_output: int = 2000,
    ) -> ContextWindow:
        """
        Fit ``messages`` into the available token budget.

        The available budget is:
            max_tokens - system_tokens - retrieved_tokens - reserve_output

        System messages are always kept.  The most recent non-system messages
        are retained until the budget is consumed.  If any messages are
        dropped and a summarizer is configured, the dropped messages are
        compressed into a single summary message.
        """
        available = self._max_tokens - system_tokens - retrieved_tokens - reserve_output
        if available <= 0:
            logger.warning(
                "Context window fully consumed by system/retrieved tokens "
                "(available=%d); returning empty message list.",
                available,
            )
            return ContextWindow(
                messages=[],
                total_tokens=0,
                truncated=bool(messages),
                summary=None,
            )

        # Separate system messages (always kept) from conversational messages
        system_msgs = [m for m in messages if m.role == "system"]
        conv_msgs = [m for m in messages if m.role != "system"]

        system_budget_used = sum(self._msg_tokens(m) for m in system_msgs)
        remaining = available - system_budget_used

        # Slide from the end (most recent) towards the front
        kept: list[ConversationMessage] = []
        tokens_used = 0

        for msg in reversed(conv_msgs):
            t = self._msg_tokens(msg)
            if tokens_used + t <= remaining:
                kept.insert(0, msg)
                tokens_used += t
            else:
                # No more room
                break

        dropped = conv_msgs[: len(conv_msgs) - len(kept)]
        truncated = bool(dropped)
        summary: str | None = None

        if dropped and self._summarizer is not None:
            try:
                summary = await self.summarize(dropped)
                summary_msg = ConversationMessage(
                    role="system",
                    content=f"[Summary of earlier conversation]\n{summary}",
                    tokens=self._estimate_tokens(summary),
                )
                # Only include summary if it fits
                summary_tokens = self._msg_tokens(summary_msg)
                if tokens_used + summary_tokens <= remaining:
                    kept.insert(0, summary_msg)
                    tokens_used += summary_tokens
            except Exception as exc:
                logger.warning("Summarizer failed, continuing without summary: %s", exc)

        final_messages = system_msgs + kept
        total_tokens = system_budget_used + tokens_used

        return ContextWindow(
            messages=final_messages,
            total_tokens=total_tokens,
            truncated=truncated,
            summary=summary,
        )

    async def summarize(self, messages: list[ConversationMessage]) -> str:
        """
        Compress a list of messages into a brief summary string.

        When an LLM summarizer is available it is used; otherwise a rule-based
        fallback extracts one user/assistant pair every 5 messages.
        """
        if self._summarizer is not None:
            try:
                formatted = "\n".join(
                    f"{m.role.upper()}: {m.content[:500]}" for m in messages
                )
                prompt_messages = [
                    {
                        "role": "user",
                        "content": (
                            "Summarise the following conversation in 3-5 sentences, "
                            "preserving key facts, decisions, and context:\n\n"
                            + formatted
                        ),
                    }
                ]
                response = await self._summarizer.complete(
                    messages=prompt_messages,
                    max_tokens=400,
                )
                return response.content.strip()
            except Exception as exc:
                logger.warning("LLM summarizer call failed: %s", exc)

        # Fallback: extract representative pairs
        pairs: list[str] = []
        for i in range(0, len(messages), 5):
            chunk = messages[i : i + 5]
            for m in chunk:
                if m.role in ("user", "assistant"):
                    snippet = m.content[:200].replace("\n", " ")
                    pairs.append(f"{m.role}: {snippet}")
                    break  # one per chunk

        if not pairs:
            return "(prior conversation omitted)"
        return "Prior context: " + " | ".join(pairs)

    def count_tokens(self, messages: list[ConversationMessage]) -> int:
        """Total token count for a list of messages."""
        return sum(self._msg_tokens(m) for m in messages)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _msg_tokens(self, msg: ConversationMessage) -> int:
        if msg.tokens > 0:
            return msg.tokens
        return self._estimate_tokens(msg.content)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return estimate_tokens(text)
