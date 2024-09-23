"""Define the shared values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import AnyMessage
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import add_messages
from typing_extensions import Annotated


@dataclass(kw_only=True)
class State:
    """Main graph state."""

    messages: Annotated[List[AnyMessage], add_messages]
    """The messages in the conversation."""
    eager: bool


@dataclass(kw_only=True)
class SingleExtractorState(State):
    """Extractor state."""

    function_name: str
    responses: list[BaseModel]
    user_state: Optional[Dict[str, Any]]


__all__ = [
    "State",
    "SingleExtractorState",
]
