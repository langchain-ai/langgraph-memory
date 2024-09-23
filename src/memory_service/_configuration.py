import os
from dataclasses import dataclass, fields

from langchain_core.runnables import RunnableConfig


@dataclass(kv_only=True)
class Settings:
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""
    pinecone_namespace: str = "ns1"
    model: str = "accounts/fireworks/models/firefunction-v2"

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig):
        configurable = config.get("configurable") or {}
        _env_env_defaults = {
            f.name: os.environ.get(f.name.upper(), "") for f in fields(cls) if f.init
        }
        return cls(
            **{**_env_env_defaults, **{k: v for k, v in configurable.items() if k in _env_env_defaults}}
        )
