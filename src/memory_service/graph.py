"""Graphs that extract memories on a schedule."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from trustcall import create_extractor
from typing_extensions import Literal

from memory_service import _constants as constants
from memory_service import _schemas as schemas
from memory_service import _configuration as settings
from memory_service import _utils as utils

logger = logging.getLogger("memory")
# Handle patch memory, where we update a single document in the database.
# If the document doesn't exist, the LLM will generate a new one.
# Otherwise, it will generate JSON patches to update the existing document.


async def fetch_patched_state(
    state: schemas.SingleExtractorState, config: RunnableConfig
) -> dict:
    """Fetch the user's state from the database.

    This is a placeholder function. You should replace this with a function
    that fetches the user's state from the database.
    """
    configurable = utils.ensure_configurable(config["configurable"])
    path = constants.PATCH_PATH.format(
        user_id=configurable["user_id"], function_name=state["function_name"]
    )
    # TODO: does pinecone have an async api in their SDK...?
    response = utils.get_index().fetch(
        ids=[path], namespace=settings.SETTINGS.pinecone_namespace
    )
    if vectors := response.get("vectors"):
        document = vectors[path]
        payload = document["metadata"][constants.PAYLOAD_KEY]
        return {"user_state": payload}
    return {"user_state": None}


async def extract_patch_memories(
    state: schemas.SingleExtractorState, config: RunnableConfig
) -> dict:
    """Extract the user's state from the conversation."""
    configurable = utils.ensure_configurable(config["configurable"])
    schemas = configurable["schemas"]
    memory_config = schemas[state["function_name"]]
    llm = init_chat_model(model=configurable["model"])
    messages = utils.prepare_messages(
        state["messages"], memory_config.get("system_prompt") or ""
    )
    extractor = create_extractor(
        llm,
        tools=[memory_config["function"]],
        tool_choice=memory_config["function"]["name"],
    )
    inputs = {
        "messages": messages,
    }
    if existing := state["user_state"]:
        inputs["existing"] = {memory_config["function"]["name"]: existing}
    result = await extractor.ainvoke(inputs, config)
    return {"responses": result["responses"]}


async def upsert_patched_state(
    state: schemas.SingleExtractorState, config: RunnableConfig
) -> dict:
    """Upsert the user's state to the database."""
    configurable = utils.ensure_configurable(config["configurable"])
    path = constants.PATCH_PATH.format(
        user_id=configurable["user_id"], function_name=state["function_name"]
    )
    serialized = state["responses"][0].model_dump_json()
    embeddings = utils.get_embeddings()
    vector = await embeddings.aembed_query(serialized)
    utils.get_index().upsert(
        vectors=[
            {
                "id": path,
                "values": vector,
                "metadata": {
                    constants.PAYLOAD_KEY: serialized,
                    constants.PATH_KEY: path,
                    constants.TIMESTAMP_KEY: datetime.now(tz=timezone.utc),
                    "user_id": configurable["user_id"],
                },
            }
        ],
        namespace=settings.SETTINGS.pinecone_namespace,
    )
    return {"user_state": {}}


patch_builder = StateGraph(schemas.SingleExtractorState, schemas.GraphConfig)
patch_builder.add_node(fetch_patched_state)
patch_builder.add_node(extract_patch_memories)
patch_builder.add_node(upsert_patched_state)
patch_builder.add_edge(START, "fetch_patched_state")
patch_builder.add_edge("fetch_patched_state", "extract_patch_memories")


def should_commit_patch(
    state: schemas.SingleExtractorState, config: RunnableConfig
) -> Literal["upsert_patched_state", "__end__"]:
    """Whether there are things extracted to commit to the DB."""
    return "upsert_patched_state" if state["responses"] else END


patch_builder.add_conditional_edges("extract_patch_memories", should_commit_patch)
patch_graph = patch_builder.compile()

# Handle semantic memory, where we insert each memory event
# as a new document in the database.


async def extract_semantic_memories(
    state: schemas.SingleExtractorState, config: RunnableConfig
) -> dict:
    """Extract embeddable "events"."""
    configurable = utils.ensure_configurable(config["configurable"])
    llm = init_chat_model(model=configurable["model"])
    memory_config = configurable["schemas"][state["function_name"]]
    messages = utils.prepare_messages(
        state["messages"], memory_config.get("system_prompt") or ""
    )

    extractor = create_extractor(llm, tools=[memory_config["function"]])
    # We don't have an "existing" value here since we are continuously inserting
    # new memories.
    result = await extractor.ainvoke({"messages": messages})
    return {"responses": result["responses"]}


async def insert_memories(
    state: schemas.SingleExtractorState, config: RunnableConfig
) -> dict:
    """Insert the user's state to the database."""
    configurable = utils.ensure_configurable(config["configurable"])
    embeddings = utils.get_embeddings()
    serialized = [r.model_dump_json() for r in state["responses"]]
    # You could alternatively do multi-vector lookup based on the schema.
    vectors = await embeddings.aembed_documents(serialized)
    current_time = datetime.now(tz=timezone.utc)
    paths = [
        constants.INSERT_PATH.format(
            user_id=configurable["user_id"],
            function_name=state["function_name"],
            event_id=str(uuid.uuid4()),
        )
        for _ in range(len(vectors))
    ]
    documents = [
        {
            "id": path,
            "values": vector,
            "metadata": {
                constants.PAYLOAD_KEY: serialized,
                constants.PATH_KEY: path,
                constants.TIMESTAMP_KEY: current_time,
                "user_id": configurable["user_id"],
            },
        }
        for path, vector, serialized in zip(paths, vectors, serialized)
    ]
    utils.get_index().upsert(
        vectors=documents,
        namespace=settings.SETTINGS.pinecone_namespace,
    )
    return {"user_state": {}}


semantic_builder = StateGraph(schemas.SingleExtractorState, schemas.GraphConfig)
# Lots of quality improvements can be made here, such as:
# - Fetch similar memories and prompt model to combine or extrapolate
# - Adding advanced indexing by the memory schema (like importance, relevance, etc.)
semantic_builder.add_node(extract_semantic_memories)
semantic_builder.add_node(insert_memories)
semantic_builder.add_edge(START, "extract_semantic_memories")


def should_insert(
    state: schemas.SingleExtractorState, config: RunnableConfig
) -> Literal["insert_memories", "__end__"]:
    """Whether there are things extracted to commit to the DB."""
    return "insert_memories" if state["responses"] else END


semantic_builder.add_conditional_edges("extract_semantic_memories", should_insert)
semantic_graph = semantic_builder.compile()


# This graph is public facing. It receives conversations and distibutes them to the
# memory types as needed.


async def schedule(state: schemas.State, config: RunnableConfig) -> dict:
    """Delay the start of processing to simulate run scheduling.

    We only really need to process a conversation after it is completed.
    In general, we don't know when a conversation is completed, so we will
    delay the processing of the conversation for a set amount of time.

    This is configurable at the assistant and run level, and to bypass this,
    you can set `eager` to True in the run inputs.

    If a new message comes in before the delay is up, the run can be cancelled
    and a new one scheduled.
    """
    if state.get("eager", False):
        return {"messages": state["messages"]}
    configurable = utils.ensure_configurable(config["configurable"])
    if configurable["delay"]:
        await asyncio.sleep(configurable["delay"])
    return {"messages": []}


# Create the graph + all nodes
builder = StateGraph(schemas.State, schemas.GraphConfig)
builder.add_node(schedule)
builder.add_node("handle_patch_memory", patch_graph)
builder.add_node("handle_semantic_memory", semantic_graph)

# Add edges
builder.add_edge(START, "schedule")


def scatter_schemas(state: schemas.State, config: RunnableConfig) -> list[Send]:
    """Route the schemas for the memory assistant.

    These will be executed in parallel.
    """
    configuration = utils.ensure_configurable(config["configurable"])
    sends = []
    for k, v in configuration["schemas"].items():
        update_mode = v["update_mode"]
        match update_mode:
            case "patch":
                target = "handle_patch_memory"
            case "insert":
                target = "handle_semantic_memory"
            case _:
                raise ValueError(f"Unknown update mode: {update_mode}")
        sends.append(Send(target, {**state, "function_name": k}))
    return sends


builder.add_conditional_edges("schedule", scatter_schemas)

memgraph = builder.compile()


__all__ = ["memgraph"]
