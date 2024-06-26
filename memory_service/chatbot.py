"""Example chatbot that incorporates user memories."""

import os
from datetime import datetime, timezone
from typing import List

import langsmith
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_nomic.embeddings import NomicEmbeddings
from langgraph.graph import START, StateGraph, add_messages
from langgraph_sdk import get_client
from typing_extensions import Annotated, TypedDict

from memory_service import _constants as constants
from memory_service import _settings as settings
from memory_service import _utils as utils


class ChatState(TypedDict):
    """The state of the chatbot."""

    messages: Annotated[List[AnyMessage], add_messages]
    user_memories: List[dict]


class ChatConfigurable(TypedDict):
    """The configurable fields for the chatbot."""

    user_id: str
    thread_id: str
    memory_service_url: str = ""


def _ensure_configurable(config: RunnableConfig) -> ChatConfigurable:
    """Ensure the configuration is valid."""
    return ChatConfigurable(
        user_id=config["configurable"]["user_id"],
        thread_id=config["configurable"]["thread_id"],
        memory_service_url=config["configurable"].get(
            "memory_service_url", os.environ.get("MEMORY_SERVICE_URL", "")
        ),
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and friendly chatbot.{user_info}\n\nSystem Time: {time}",
        )
    ]
).partial(
    time=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
)


@langsmith.traceable
def format_query(messages: List[AnyMessage]) -> str:
    """Format the query for the user's memories."""
    # This is quite naive :)
    return " ".join([str(m.content) for m in messages if m.type == "human"][-5:])


async def query_memories(state: ChatState, config: RunnableConfig) -> ChatState:
    """Query the user's memories."""
    configurable: ChatConfigurable = config["configurable"]
    user_id = configurable["user_id"]
    index = utils.get_index()
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

    query = format_query(state["messages"])
    vec = await embeddings.embed_query(query)
    # You can also filter by memory type, etc. here.
    response = index.query(
        vector=vec,
        filter={"user": {"$eq": user_id}},
        include_metadata=True,
        top_k=10,
        namespace=settings.SETTINGS.pinecone_namespace,
    )
    memories = []
    if matches := response.get("matches"):
        memories = [m["metadata"]["memory"][constants.PAYLOAD_KEY] for m in matches]
    return {
        "user_memories": memories,
    }


@langsmith.traceable
def format_memories(memories: List[dict]) -> str:
    """Format the user's memories."""
    if not memories:
        return ""
    # Note Bene: You can format better than this....
    memories = "\n".join(str(m) for m in memories)
    return f"""

## Memories

You have noted the following memorable events from previous interactions with the user.
<memories>
{memories}
</memories>
"""


async def bot(state: ChatState, config: RunnableConfig) -> ChatState:
    """Prompt the bot to resopnd to the user, incorporating memories (if provided)."""
    model = init_chat_model("claude-3-5-sonnet-20240620")
    chain = PROMPT | model
    memories = format_memories(state["user_memories"])
    m = await chain.ainvoke(
        {
            "messages": state["messages"],
            "user_info": memories,
        },
        config,
    )

    return {
        "messages": [m],
    }


async def post_messages(state: ChatState, config: RunnableConfig) -> ChatState:
    """Query the user's memories."""
    configurable = _ensure_configurable(config)
    langgraph_client = get_client(url=configurable["memory_service_url"])
    thread_id = config["configurable"]["thread_id"]
    try:
        thread = await langgraph_client.threads.create(thread_id=thread_id)
    except Exception:
        thread = await langgraph_client.threads.get(thread_id=thread_id)

    await langgraph_client.runs.create(
        thread["thread_id"],
        input={
            "messages": state["messages"],  # the service dedupes messages
        },
        config={
            "system_prompt": "You are a helpful and friendly chatbot.",
            "configurable": {
                "user_id": thread["user_id"],
                "thread_id": thread["thread_id"],
                "schemas": {
                    "system_prompt": "Extract any memorable events from the user's"
                    " messages that you would like to remember.",
                    "MemorableEvent": {
                        "function": {
                            "name": "memorable_event",
                            "description": "",
                            "parameters": {
                                "name": "memorable_event",
                                "description": "Any event, observation, insight, or "
                                "other detail that you may want to recall in "
                                "later interactions with the user.",
                                "parameters": {
                                    "description": "Any event, observation, insight, or"
                                    " other detail that you may want to recall in"
                                    " later interactions with the user.",
                                    "properties": {
                                        "description": {
                                            "title": "Description",
                                            "type": "string",
                                        },
                                        "participants": {
                                            "description": "Names of participants in"
                                            " the event and their relationship to the "
                                            "user.",
                                            "items": {"type": "string"},
                                            "title": "Participants",
                                            "type": "array",
                                        },
                                    },
                                    "required": ["description", "participants"],
                                    "title": "memorable_event",
                                    "type": "object",
                                },
                            },
                        }
                    },
                },
            },
        },
        multitask_strategy="interrupt",
    )

    return {
        "messages": state["messages"],
        "user_memories": [],
    }


builder = StateGraph(ChatState, ChatConfigurable)
builder.add_node(query_memories)
builder.add_node(bot)
builder.add_node(post_messages)
builder.add_edge(START, "query_memories")
builder.add_edge("query_memories", "bot")
builder.add_edge("bot", "post_messages")

chat_graph = builder.compile()
