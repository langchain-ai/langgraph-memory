from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import langsmith
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)
from langchain_fireworks import FireworksEmbeddings
from pinecone import Pinecone

from memory_service import _schemas as schemas
from memory_service import _settings as settings

_DEFAULT_DELAY = 60  # seconds


def get_index():
    pc = Pinecone(api_key=settings.SETTINGS.pinecone_api_key)
    return pc.Index(settings.SETTINGS.pinecone_index_name)


@langsmith.traceable
def ensure_memory_config(config: dict) -> schemas.MemoryConfig:
    """Merge the user-provided config with default values."""
    return {
        **config,
        **schemas.MemoryConfig(
            function=config.get("function", {}),
            system_prompt=config.get("system_prompt"),
            update_mode=config.get("update_mode", "patch"),
        ),
    }


@langsmith.traceable
def ensure_configurable(config: dict) -> schemas.GraphConfig:
    """Merge the user-provided config with default values."""
    function_schemas = config.get("schemas") or {}
    return {
        **config,
        **schemas.GraphConfig(
            delay=config.get("delay", _DEFAULT_DELAY),
            model=config.get("model", settings.SETTINGS.model),
            schemas={k: ensure_memory_config(v) for k, v in function_schemas.items()},
            thread_id=config["thread_id"],
            user_id=config["user_id"],
        ),
    }


def prepare_messages(
    messages: Sequence[AnyMessage], system_prompt: str
) -> list[AnyMessage]:
    """Merge message runs and add instructions before and after to stay on task."""
    sys = SystemMessage(
        content=system_prompt
        + """

<memory-system>Reflect on following interaction. Use the provided tools to \
 retain any necessary memories about the user.</memory-system>
"""
    )
    m = HumanMessage(
        content="## End of conversation\n\n"
        "<memory-system>Reflect on the interaction above."
        " What memories ought to be retained or updated?</memory-system>",
    )
    return merge_message_runs([sys] + list(messages) + [m])


@lru_cache
def get_embeddings():
    return FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")


__all__ = ["ensure_configurable", "prepare_messages"]
