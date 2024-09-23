import os
from dataclasses import dataclass, field, fields
from typing import Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig


class FunctionSchema(TypedDict):
    name: str
    """Name of the function."""
    description: str
    """A description of the function."""
    parameters: dict
    """The JSON Schema for the memory."""


@dataclass(kw_only=True)
class MemoryConfig:
    function: FunctionSchema
    """The function to use for the memory assistant."""
    system_prompt: Optional[str] = ""
    """The system prompt to use for the memory assistant."""
    update_mode: Literal["patch", "insert"] = field(default="patch")
    """Whether to continuously patch the memory, or treat each new

    generation as a new memory.

    Patching is useful for maintaining a structured profile or core list
    of memories. Inserting is useful for maintaining all interactions and
    not losing any information.

    For patched memories, you can GET the current state at any given time.
    For inserted memories, you can query the full history of interactions.
    """


@dataclass(kw_only=True)
class Configuration:
    delay: float = 60  # seconds
    """The delay in seconds to wait before considering a conversation complete.
    
    Default is 60 seconds.
    """
    model: str = "gpt-4o"
    """The model to use for generating memories. """
    schemas: dict = field(default_factory=dict)
    """The schemas for the memory assistant."""
    thread_id: str
    """The thread ID of the conversation."""
    user_id: str = "default"
    """The ID of the user to remember in the conversation."""

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig):
        configurable = config["configurable"]
        values = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        values["schemas"] = {
            k: MemoryConfig(**v) for k, v in (values["schemas"] or {}).items()
        }
        return cls(**{k: v for k, v in values.items() if v})
