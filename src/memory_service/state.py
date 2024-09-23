"""Define the shared values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed.shared_value import SharedValue
from typing_extensions import Annotated


@dataclass(kw_only=True)
class State:
    """Main graph state."""

    messages: Annotated[list[AnyMessage], add_messages]
    """The messages in the conversation."""


@dataclass(kw_only=True)
class PatchNodeState(State):
    """Extractor state."""

    function_name: str
    user_states: Annotated[dict[str, dict[str, Any]], SharedValue.on("user_id")]


@dataclass(kw_only=True)
class SemanticNodeState(State):
    """Extractor state."""

    function_name: str
    user_states: Annotated[dict[str, dict[str, Any]], SharedValue.on("user_id")]


__all__ = [
    "State",
    "PatchNodeState",
    "SemanticNodeState",
]
