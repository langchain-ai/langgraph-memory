"""Graphs that extract memories on a schedule."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import asdict
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import StateGraph
from langgraph.store.store import Store
from trustcall import create_extractor

from memory_service import _configuration as configuration
from memory_service import _utils as utils
from memory_service import state as schemas

logger = logging.getLogger("memory")


async def _extract_memory(
    extractor,
    messages: list,
    memory_config: configuration.MemoryConfig,
    existing: Optional[dict],
    config: RunnableConfig,
):
    prepared_messages = utils.prepare_messages(messages, memory_config.system_prompt)
    inputs = {"messages": prepared_messages, "existing": existing}
    result = await extractor.ainvoke(inputs, config)
    return result["responses"][0].model_dump(mode="json")


async def handle_patch_memory(
    state: schemas.ProcessorState, config: RunnableConfig, *, store: Store
) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    configurable = configuration.Configuration.from_runnable_config(config)
    namespace = ("user_states", configurable.user_id, state.function_name)
    doc_id = uuid.uuid5(uuid.NAMESPACE_URL, "/".join(namespace))
    existing_item = await store.aget(namespace, str(doc_id))
    existing = {existing_item.id: existing_item.value} if existing_item else None
    memory_config = configurable.schemas[state.function_name]
    extractor = create_extractor(
        init_chat_model(model=configurable.model),
        tools=[memory_config.function],
        tool_choice=memory_config.function["name"],
    )

    result = await _extract_memory(
        extractor, state.messages, memory_config, existing, config
    )
    await store.aput(namespace, doc_id, result)
    return {"messages": []}


async def handle_insertion_memory(
    state: schemas.ProcessorState, config: RunnableConfig, *, store: Store
) -> dict:
    """Upsert memory events."""
    configurable = configuration.Configuration.from_runnable_config(config)
    namespace = ("events", configurable.user_id, state.function_name)
    existing_items = await store.asearch(
        namespace, filter=None, query=None, weights=None, limit=5
    )
    memory_config: configuration.MemoryConfig = configurable.schemas[
        state.function_name
    ]
    extractor = create_extractor(
        init_chat_model(model=configurable.model),
        tools=[memory_config.function],
        tool_choice="any",
        enable_inserts=True,
    )
    extracted = await extractor.ainvoke(
        {
            "messages": utils.prepare_messages(
                state.messages, memory_config.system_prompt
            ),
            "existing": (
                [
                    (existing_item.id, state.function_name, existing_item.value)
                    for existing_item in existing_items
                ]
                if existing_items
                else None
            ),
        },
        config,
    )
    await asyncio.gather(
        *(
            store.aput(
                namespace,
                rmeta.get("json_doc_id", str(uuid.uuid4())),
                r.model_dump(mode="json"),
            )
            for r, rmeta in zip(extracted["responses"], extracted["response_metadata"])
        )
    )
    return {"messages": []}


# Create the graph + all nodes
builder = StateGraph(schemas.State, config_schema=configuration.Configuration)

builder.add_node(handle_patch_memory, input=schemas.ProcessorState)
builder.add_node(handle_insertion_memory, input=schemas.ProcessorState)


def scatter_schemas(state: schemas.State, config: RunnableConfig) -> list[Send]:
    """Route the schemas for the memory assistant.

    These will be executed in parallel.
    """
    configurable = configuration.Configuration.from_runnable_config(config)
    sends = []
    current_state = asdict(state)
    for k, v in configurable.schemas.items():
        update_mode = v.update_mode
        match update_mode:
            case "patch":
                target = "handle_patch_memory"
            case "insert":
                target = "handle_insertion_memory"
            case _:
                raise ValueError(f"Unknown update mode: {update_mode}")

        sends.append(
            Send(
                target,
                schemas.ProcessorState(**{**current_state, "function_name": k}),
            )
        )
    return sends


builder.add_conditional_edges(
    "__start__", scatter_schemas, ["handle_patch_memory", "handle_insertion_memory"]
)

memgraph = builder.compile()


__all__ = ["memgraph"]
