from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)
from langchain_openai import OpenAIEmbeddings


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
def get_embeddings() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


__all__ = ["prepare_messages"]
