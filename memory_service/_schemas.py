from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import add_messages
from typing_extensions import Annotated, Literal, TypedDict


class FunctionSchema(TypedDict):
    name: str
    """Name of the function."""
    description: str
    """A description of the function."""
    parameters: dict
    """The JSON Schema for the memory."""


class MemoryConfig(TypedDict, total=False):
    function: FunctionSchema
    """The function to use for the memory assistant."""
    system_prompt: Optional[str]
    """The system prompt to use for the memory assistant."""
    update_mode: Literal["patch", "insert"]
    """Whether to continuously patch the memory, or treat each new

    generation as a new memory.

    Patching is useful for maintaining a structured profile or core list
    of memories. Inserting is useful for maintaining all interactions and
    not losing any information.

    For patched memories, you can GET the current state at any given time.
    For inserted memories, you can query the full history of interactions.
    """


class GraphConfig(TypedDict, total=False):
    delay: float
    """The delay in seconds to wait before considering a conversation complete.
    
    Default is 60 seconds.
    """
    model: str
    """The model to use for generating memories.
     
    Defaults to Fireworks's "accounts/fireworks/models/firefunction-v2"
    """
    schemas: dict[str, MemoryConfig]
    """The schemas for the memory assistant."""
    thread_id: str
    """The thread ID of the conversation."""
    user_id: str
    """The ID of the user to remember in the conversation."""


class State(TypedDict):
    messages: Annotated[list, add_messages]
    """The messages in the conversation."""
    eager: bool


class SingleExtractorState(State):
    function_name: str
    responses: list[BaseModel]
    user_state: Optional[Dict[str, Any]]


__all__ = [
    "State",
    "GraphConfig",
    "SingleExtractorState",
    "FunctionSchema",
    "MemoryConfig",
]
