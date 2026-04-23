"""Framework adapters for Codex Harness.

Each adapter wraps an external multi-agent framework so it can execute inside
the harness lifecycle and receive observability, memory, safety, and cost
tracking for free.

Supported frameworks
--------------------
* **LangGraph** — :class:`LangGraphAdapter` wraps a ``StateGraph`` or compiled
  graph and streams ``astream()`` output as :class:`StepEvent` objects.
* **AutoGen** — :class:`AutoGenAdapter` wraps a ``ConversableAgent`` pair (or
  ``GroupChat``) and emits one event per conversation message.
* **CrewAI** — :class:`CrewAIAdapter` wraps a ``Crew`` and emits one event per
  agent step captured via ``step_callback``.

Typical usage::

    from harness.adapters import LangGraphAdapter, FrameworkResult

    adapter = LangGraphAdapter(graph=my_compiled_graph)
    async for event in adapter.run(ctx, input={"messages": [...]}):
        print(event.event_type, event.payload)

    result: FrameworkResult = await adapter.get_result()
    print(result.output)

Framework packages (``langgraph``, ``pyautogen`` / ``autogen``, ``crewai``) are
**optional** runtime dependencies.  An :exc:`ImportError` with a helpful install
message is raised only when the adapter is actually used.
"""

from harness.adapters.autogen import AutoGenAdapter
from harness.adapters.base import FrameworkAdapter, FrameworkResult
from harness.adapters.crewai import CrewAIAdapter
from harness.adapters.langgraph import LangGraphAdapter

__all__ = [
    "FrameworkResult",
    "FrameworkAdapter",
    "LangGraphAdapter",
    "AutoGenAdapter",
    "CrewAIAdapter",
]
