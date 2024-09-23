"""Example chatbot that incorporates user memories."""

import os
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, List

import langsmith
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, add_messages
from langgraph.managed.shared_value import SharedValue
from langgraph_sdk import get_client
from typing_extensions import Annotated


@dataclass
class ChatState:
    """The state of the chatbot."""

    messages: Annotated[List[AnyMessage], add_messages]
    user_states: Annotated[dict[str, dict[str, Any]], SharedValue.on("user_id")]


@dataclass(kw_only=True)
class ChatConfigurable:
    """The configurable fields for the chatbot."""

    user_id: str
    thread_id: str
    model: str = "gpt-4o"
    delay_seconds: int = 60  # For debouncing memory creation
    mem_assistant_id: str = "memory"  # Default to just the graph ID
    memory_service_url: str | None = None

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig):
        """Load configuration."""
        configurable = config["configurable"]
        values = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and friendly chatbot. Get to know the user!"
            " Ask questions! Be spontaneous!"
            "{user_info}\n\nSystem Time: {time}",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(
    time=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
)


@langsmith.traceable
def format_memories(memories: dict[str : dict[str, Any]]) -> str:
    """Format the user's memories."""
    if not memories:
        return ""
    # Note Bene: You can format better than this....
    memories = "\n".join(str(m) for m in memories.values())
    return f"""

## Memories

You have noted the following memorable events from previous interactions with the user.
<memories>
{memories}
</memories>
"""


async def bot(state: ChatState, config: RunnableConfig) -> ChatState:
    """Prompt the bot to resopnd to the user, incorporating memories (if provided)."""
    configurable = ChatConfigurable.from_runnable_config(config)
    model = init_chat_model(configurable.model)
    chain = PROMPT | model
    memories = format_memories(state.user_states)
    m = await chain.ainvoke(
        {
            "messages": state.messages,
            "user_info": memories,
        },
        config,
    )

    langgraph_client = get_client(url=configurable.memory_service_url)
    thread_id = config["configurable"]["thread_id"]
    await langgraph_client.runs.create(
        thread_id,
        assistant_id=configurable.mem_assistant_id,
        input={
            "messages": state.messages,  # the service dedupes messages
        },
        config={
            "configurable": {
                "user_id": configurable.user_id,
            },
        },
        multitask_strategy="rollback",
        after_seconds=configurable.delay_seconds,
    )
    return {"messages": [m]}


builder = StateGraph(ChatState, config_schema=ChatConfigurable)
builder.add_node(bot)

builder.add_edge("__start__", "bot")

chat_graph = builder.compile()
