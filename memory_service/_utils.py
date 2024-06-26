from __future__ import annotations

from typing import Sequence

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)
from pinecone import Pinecone

from memory_service import _schemas as schemas
from memory_service import _settings as settings

_DEFAULT_DELAY = 60  # seconds


def get_index():
    pc = Pinecone(api_key=settings.SETTINGS.pinecone_api_key)
    return pc.Index(settings.SETTINGS.pinecone_index_name)


def ensure_configurable(config: dict) -> schemas.GraphConfig:
    """Merge the user-provided config with default values."""
    return {
        **config,
        **schemas.GraphConfig(
            delay=config.get("delay", _DEFAULT_DELAY),
            model=config.get("model", "accounts/fireworks/models/firefunction-v2"),
            schemas=config.get("schemas", {}),
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


__all__ = ["ensure_configurable", "prepare_messages"]
