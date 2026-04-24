"""Base adapter contracts for framework integration with HarnessAgent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from harness.core.context import AgentContext, StepEvent


@dataclass
class FrameworkResult:
    """Normalised result produced by any framework adapter after a run completes.

    Attributes:
        framework: Canonical framework name — "langgraph" | "autogen" | "crewai".
        output:    Final text output extracted from the framework's state.
        steps:     Total number of observable steps (nodes / messages / tasks).
        metadata:  Framework-specific raw data (final state, message list, etc.).
    """

    framework: str
    output: str
    steps: int
    metadata: dict = field(default_factory=dict)


class FrameworkAdapter(ABC):
    """Wraps an external framework's execution inside the harness lifecycle.

    Concrete subclasses implement ``run()`` and ``get_result()``.  The harness
    calls them in sequence::

        async for event in adapter.run(ctx, input):
            ...
        result = await adapter.get_result()

    Both methods must be safe to call from an async context.  ``run()`` is
    responsible for emitting :class:`~harness.core.context.StepEvent` objects so
    that the rest of the harness (tracer, event-bus, MLflow) can observe the
    framework without any special-casing.
    """

    # Subclasses set this as a class attribute.
    framework_name: str = "unknown"

    @abstractmethod
    async def run(
        self,
        ctx: "AgentContext",
        input: dict,  # noqa: A002  (shadows built-in intentionally)
    ) -> AsyncIterator["StepEvent"]:
        """Execute the framework and yield a :class:`StepEvent` per observable step.

        Args:
            ctx:   The harness :class:`~harness.core.context.AgentContext` for
                   this run.  Adapters MUST call ``ctx.tick()`` and check
                   ``ctx.is_budget_ok()`` to respect budget limits.
            input: Framework-specific input dictionary.  Common keys:
                   ``"task"``, ``"message"``, ``"messages"``.

        Yields:
            One :class:`~harness.core.context.StepEvent` per observable step.
            Budget-exceeded events use ``event_type="budget_exceeded"``.
        """
        # This is an abstract method; the ``yield`` makes Python treat it as an
        # async generator, which is required for ``AsyncIterator`` compatibility.
        # Concrete implementations must also be async generators (or return one).
        raise NotImplementedError  # pragma: no cover
        # Satisfy the async-generator protocol for type checkers.
        yield  # type: ignore[misc]

    @abstractmethod
    async def get_result(self) -> FrameworkResult:
        """Return the final :class:`FrameworkResult` after ``run()`` has completed.

        Raises:
            RuntimeError: If called before ``run()`` has finished.
        """
        raise NotImplementedError  # pragma: no cover
