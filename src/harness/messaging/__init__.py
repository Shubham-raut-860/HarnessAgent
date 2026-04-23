"""Harness messaging module — inter-agent communication via Redis Streams."""

from harness.messaging.bus import AgentMessageBus
from harness.messaging.patterns import MessagePatterns
from harness.messaging.schema import AgentMessage

__all__ = [
    "AgentMessage",
    "AgentMessageBus",
    "MessagePatterns",
]
