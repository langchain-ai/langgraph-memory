"""Graphs that extract memories on a schedule."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_nomic.embeddings import NomicEmbeddings
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from trustcall import create_extractor
from typing_extensions import Literal

from memory_service import _constants as constants
from memory_service import _schemas as schemas
from memory_service import _settings as settings
from memory_service import _utils as utils

logger = logging.getLogger("memory")


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
    existing = state["user_state"]
    result = await extractor.ainvoke(
        {
            "messages": messages,
            "existing": {memory_config["function"]["name"]: existing},
        }
    )
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
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    vector = await embeddings.aembed_query(serialized)
    utils.get_index().upsert(
        vectors=[
            {
                "id": path,
                "values": vector,
                "metadata": {
                    constants.PAYLOAD_KEY: json.loads(serialized),
                    constants.PATH_KEY: path,
                    constants.TIMESTAMP_KEY: datetime.now(tz=timezone.utc),
                },
            }
        ],
        namespace=settings.SETTINGS.pinecone_namespace,
    )
    return {"user_state": {}}


async def extract_insertion_memories(
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
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
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
                constants.PAYLOAD_KEY: json.loads(serialized),
                constants.PATH_KEY: path,
                constants.TIMESTAMP_KEY: current_time,
            },
        }
        for path, vector, serialized in zip(paths, vectors, serialized)
    ]
    utils.get_index().upsert(
        vectors=documents,
        namespace=settings.SETTINGS.pinecone_namespace,
    )
    return {"user_state": {}}


def route_inbound(
    state: schemas.SingleExtractorState, config: RunnableConfig
) -> Literal["fetch_patched_state", "extract_insertion_memories"]:
    configurable = utils.ensure_configurable(config["configurable"])
    update_mode = configurable["schemas"][state["function_name"]]["update_mode"]
    match update_mode:
        case "patch":
            return "fetch_patched_state"
        case "insert":
            return "extract_insertion_memories"
        case _:
            raise ValueError(f"Unknown update mode: {update_mode}")


extraction_builder = StateGraph(schemas.SingleExtractorState, schemas.GraphConfig)
extraction_builder.add_node(fetch_patched_state)
extraction_builder.add_node(extract_patch_memories)
extraction_builder.add_node(upsert_patched_state)
extraction_builder.add_node(extract_insertion_memories)
extraction_builder.add_node(insert_memories)

extraction_builder.add_conditional_edges(START, route_inbound)
extraction_builder.add_edge("fetch_patched_state", "extract_patch_memories")


def should_commit(
    state: schemas.SingleExtractorState, config: RunnableConfig
) -> Literal["insert_memories", "upsert_patched_state", "__end__"]:
    """Whether there are things extracted to commit to the DB."""
    if not state["responses"]:
        return END
    configurable = utils.ensure_configurable(config["configurable"])
    function = configurable["schemas"][state["function_name"]]
    commit_node_name = (
        "insert_memories"
        if function["update_mode"] == "insert"
        else "upsert_patched_state"
    )
    return commit_node_name


extraction_builder.add_conditional_edges("extract_patch_memories", should_commit)
extraction_builder.add_conditional_edges("extract_insertion_memories", should_commit)
extraction_graph = extraction_builder.compile()


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
builder.add_node("extract", extraction_graph)

# Add edges
builder.add_edge(START, "schedule")


def scatter_schemas(state: schemas.State, config: RunnableConfig) -> list[Send]:
    """Route the schemas for the memory assistant.

    These will be executed in parallel.
    """
    configuration = utils.ensure_configurable(config["configurable"])
    return [
        Send("extract", {**state, "function_name": k}) for k in configuration["schemas"]
    ]


builder.add_conditional_edges("schedule", scatter_schemas)

memgraph = builder.compile()


__all__ = ["memgraph"]
