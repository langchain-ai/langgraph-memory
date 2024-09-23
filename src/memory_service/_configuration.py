import os
from dataclasses import dataclass, field, fields
from typing import Literal, Optional

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class FunctionSchema:
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

    def __post_init__(self):
        if isinstance(self.function, dict):
            self.function = FunctionSchema(**self.function)


@dataclass(kv_only=True)
class Configuration:
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""
    pinecone_namespace: str = "ns1"
    model: str = "accounts/fireworks/models/firefunction-v2"
    delay: float = 60  # seconds
    """The delay in seconds to wait before considering a conversation complete.
    
    Default is 60 seconds.
    """
    model: str
    """The model to use for generating memories.
     
    Defaults to Fireworks's "accounts/fireworks/models/firefunction-v2"
    """
    schemas: dict[str, MemoryConfig] = field(default_factory=dict)
    """The schemas for the memory assistant."""
    thread_id: str
    """The thread ID of the conversation."""
    user_id: str
    """The ID of the user to remember in the conversation."""

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig):
        configurable = config["configurable"]
        values = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        values["schemas"] = {k: MemoryConfig(**v) for k, v in values["schemas"].items()}
        return cls(**values)
