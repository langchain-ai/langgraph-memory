import json
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.pydantic_v1 import BaseModel, Field
from langsmith import expect, get_current_run_tree, test

from memory_service._constants import PATCH_PATH
from memory_service.graph import memgraph
from memory_service.state import GraphConfig, MemoryConfig


# To test the patch-based memory
class CoreMemories(BaseModel):
    """Core memories about the user."""

    memories: List[str]


@pytest.fixture
def core_memory_func() -> MemoryConfig:
    return {
        "function": {
            "name": "core_memories",
            "description": "A list of core memories about the user.",
            "parameters": CoreMemories.schema(),
        },
        "system_prompt": "You may add or remove memories that are core to the"
        " user's identity or that will help you better interact with the user.",
        "update_mode": "patch",
    }


@test(output_keys=["num_mems_expected"])
@pytest.mark.parametrize(
    "messages, existing, num_mems_expected",
    [
        ([], {}, 0),
        ([("user", "When I was young, I had a dog named spot")], {}, 1),
        (
            [("user", "When I was young, I had a dog named spot.")],
            {"memories": ["I am afraid of spiders."]},
            2,
        ),
    ],
)
async def test_patch_memory(
    core_memory_func: MemoryConfig,
    messages: List[str],
    num_mems_expected: int,
    existing: dict,
):
    # patch memory_service.graph.index with a mock
    user_id = "4fddb3ef-fcc9-4ef7-91b6-89e4a3efd112"
    thread_id = "e1d0b7f7-0a8b-4c5f-8c4b-8a6c9f6e5c7a"
    function_name = "CoreMemories"
    with patch("memory_service._utils.get_index") as get_index:
        index = MagicMock()
        get_index.return_value = index
        # No existing memories
        if existing:
            path = PATCH_PATH.format(
                user_id=user_id,
                function_name=function_name,
            )
            index.fetch.return_value = {
                "vectors": {path: {"metadata": {"content": existing}}}
            }
        else:
            index.fetch.return_value = {}

        # When the memories are patched
        await memgraph.ainvoke(
            {
                "messages": messages,
            },
            {
                "configurable": GraphConfig(
                    delay=0.1,
                    user_id=user_id,
                    thread_id=thread_id,
                    schemas={function_name: core_memory_func},
                ),
            },
        )
        if num_mems_expected:
            # Check if index.upsert was called
            index.upsert.assert_called_once()
            # Get named call args
            vectors = index.upsert.call_args.kwargs["vectors"]
            rt = get_current_run_tree()
            rt.outputs = {"upserted": [v["metadata"]["content"] for v in vectors]}
            assert len(vectors) == 1
            # Check if the memory was added
            mem = vectors[0]["metadata"]["content"]
            memories = json.loads(mem)["memories"]
            expect(len(memories)).to_equal(num_mems_expected)


# To test the insertion memory
class MemorableEvent(BaseModel):
    """A memorable event."""

    description: str
    participants: List[str] = Field(
        description="Names of participants in the event and their relationship to the user."
    )


@pytest.fixture
def memorable_event_func() -> MemoryConfig:
    return {
        "function": {
            "name": "memorable_event",
            "description": "Any event, observation, insight, or other detail that you may"
            " want to recall in later interactions with the user.",
            "parameters": MemorableEvent.schema(),
        },
        "system_prompt": "Extract all events that are memorable and relevant to the user."
        " using parallel tool calling. If nothing of interest occured in the diologue, simply reply 'None'.",
        "update_mode": "insert",
    }


@test(output_keys=["num_events_expected"])
@pytest.mark.parametrize(
    "messages, num_events_expected",
    [
        ([], 0),
        (
            [
                ("user", "I went to the beach with my friends."),
                ("assistant", "That sounds like a fun day."),
            ],
            1,
        ),
        (
            [
                ("user", "I went to the beach with my friends."),
                ("assistant", "That sounds like a fun day."),
                ("user", "I also went to the park with my family - I like the park."),
            ],
            2,
        ),
    ],
)
async def test_insert_memory(
    memorable_event_func: MemoryConfig,
    messages: List[str],
    num_events_expected: int,
):
    # patch memory_service.graph.index with a mock
    user_id = "4fddb3ef-fcc9-4ef7-91b6-89e4a3efd112"
    thread_id = "e1d0b7f7-0a8b-4c5f-8c4b-8a6c9f6e5c7a"
    function_name = "MemorableEvent"
    with patch("memory_service._utils.get_index") as get_index:
        index = MagicMock()
        get_index.return_value = index
        index.fetch.return_value = {}
        # When the events are inserted
        await memgraph.ainvoke(
            {
                "messages": messages,
            },
            {
                "configurable": GraphConfig(
                    delay=0.1,
                    user_id=user_id,
                    thread_id=thread_id,
                    schemas={function_name: memorable_event_func},
                ),
            },
        )
        if num_events_expected:
            # Check if index.upsert was called
            index.upsert.assert_called_once()
            # Get named call args
            vectors = index.upsert.call_args.kwargs["vectors"]
            assert len(vectors) == num_events_expected
