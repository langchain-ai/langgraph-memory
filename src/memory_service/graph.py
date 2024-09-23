"""Graphs that extract memories on a schedule."""

from __future__ import annotations

import logging
from dataclasses import asdict

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import StateGraph
from trustcall import create_extractor

from memory_service import _configuration as configuration
from memory_service import _utils as utils
from memory_service import state as schemas

logger = logging.getLogger("memory")
# Handle patch memory, where we update a single document in the database.
# If the document doesn't exist, the LLM will generate a new one.
# Otherwise, it will generate JSON patches to update the existing document.


async def handle_patch_memory(
    state: schemas.PatchNodeState, config: RunnableConfig
) -> dict:
    """Extract the user's state from the conversation."""
    configurable = configuration.Configuration.from_runnable_config(config)
    existing = state.user_states.get(state.function_name)
    memory_config = configurable.schemas[state.function_name]
    llm = init_chat_model(model=configurable.model)
    messages = utils.prepare_messages(state.messages, memory_config.system_prompt)
    extractor = create_extractor(
        llm,
        tools=[memory_config.function],
        tool_choice=memory_config.function["name"],
    )
    inputs = {"messages": messages, "existing": existing}
    result = await extractor.ainvoke(inputs, config)
    serialized = result["responses"][0].model_dump()
    # Update the memories for this schema type, (langgraph manages user namespacing)
    return {"user_states": {state.function_name: serialized}}


# async def extract_semantic_memories(
#     state: schemas.PatchNodeState, config: RunnableConfig
# ) -> dict:
#     """Extract embeddable "events"."""
#     configurable = configuration.Configuration.from_runnable_config(config)
#     llm = init_chat_model(model=configurable.model)
#     memory_config = configurable.schemas[state.function_name]
#     messages = utils.prepare_messages(state.messages, memory_config.system_prompt)

#     extractor = create_extractor(llm, tools=[memory_config.function])
#     # We don't have an "existing" value here since we are continuously inserting
#     # new memories.
#     result = await extractor.ainvoke({"messages": messages})
#     return {"responses": result["responses"]}


# async def extract_semantic_memories(
#     state: schemas.PatchNodeState, config: RunnableConfig
# ) -> dict:
#     """Extract the user's state from the conversation."""
#     configurable = configuration.Configuration.from_runnable_config(config)
#     existing = state.user_states.get(state.function_name)
#     memory_config = configurable.schemas[state.function_name]
#     llm = init_chat_model(model=configurable.model)
#     messages = utils.prepare_messages(state.messages, memory_config.system_prompt)
#     extractor = create_extractor(
#         llm,
#         tools=[memory_config.function],
#         tool_choice=memory_config.function["name"],
#     )
#     inputs = {"messages": messages, "existing": existing}
#     result = await extractor.ainvoke(inputs, config)
#     serialized = result["responses"][0].model_dump_json()
#     # Update the memories for this schema type, (langgraph manages user namespacing)
#     return {"user_states": {state.function_name: serialized}}


# Create the graph + all nodes
builder = StateGraph(schemas.State, config_schema=configuration.Configuration)


builder.add_node(handle_patch_memory, input=schemas.PatchNodeState)
builder.add_node("handle_semantic_memory", lambda x: x)


def scatter_schemas(state: schemas.State, config: RunnableConfig) -> list[Send]:
    """Route the schemas for the memory assistant.

    These will be executed in parallel.
    """
    configurable = configuration.Configuration.from_runnable_config(config)
    sends = []
    current_state = asdict(state)
    print(f"CURRENT STATE, {current_state}", flush=True)
    for k, v in configurable.schemas.items():
        update_mode = v.update_mode
        match update_mode:
            case "patch":
                target = "handle_patch_memory"
            # case "insert":
            #     target = "handle_semantic_memory"
            case _:
                raise ValueError(f"Unknown update mode: {update_mode}")

        sends.append(
            Send(
                target,
                schemas.PatchNodeState(
                    **{"user_states": {}, **current_state, "function_name": k}
                ),
            )
        )
    return sends


builder.add_conditional_edges("__start__", scatter_schemas)

memgraph = builder.compile()


__all__ = ["memgraph"]
