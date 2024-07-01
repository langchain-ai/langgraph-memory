# LangGraph Memory Service

This repo provides a simple example of memory service you can build and deploy using LanGraph.

Inspired by papers like [MemGPT](https://memgpt.ai/) and distilled from our own works on long-term memory, the graph
extracts memories from chat interactions and persists them to a database. This information can later be read or queried semantically
to provide personalized context when your bot is responding to a particular user.

The memory graph handles thread process deduplication and supports continuous updates to a single "memory schema" as well as "event-based" memories that can be queried semantically.

![Memory Diagram](./img/memory_graph.png)

#### Project Structure

```bash
├── langgraph.json # LangGraph Cloud Configuration
├── memory_service
│   ├── __init__.py
│   └── graph.py # Define the memory service
├── poetry.lock
├── pyproject.toml # Project dependencies
└── tests # Add testing + evaluation logic
    └── evals
        └── test_memories.py
```

## Quickstart

This quick start will get your memory service deployed on [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/). Once created, you can interact with it from any API.

#### Prerequisites

This example defaults to using Pinecone for its memory database, and `nomic-ai/nomic-embed-text-v1.5` as the text encoder (hosted on Fireworks).

1. [Create an index](https://docs.pinecone.io/reference/api/control-plane/create_index) with a dimension size of `768`. Note down your Pinecone API key, index name, and namespac for the next step.
2. [Create an API Key](https://fireworks.ai/api-keys) to use for the LLM & embeddings models served on Fireworks.

#### Deploy to LangGraph Cloud

**Note:** (_Closed Beta_) LangGraph Cloud is a managed service for deploying and hosting LangGraph applications. It is currently (as of 26 June, 2024) in closed beta. If you are interested in applying for access, please fill out [this form](https://www.langchain.com/langgraph-cloud-beta).

To deploy this example on LangGraph, fork the [repo](https://github.com/langchain-ai/langgraph-memory).

Next, navigate to the 🚀 deployments tab on [LangSmith](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/).

**If you have not deployed to LangGraph Cloud before:** there will be a button that shows up saying `Import from GitHub`. You’ll need to follow that flow to connect LangGraph Cloud to GitHub.

Once you have set up your GitHub connection, select **+New Deployment**. Fill out the required information, including:

1. Your GitHub username (or organization) and the name of the repo you just forked.
2. You can leave the defaults for the config file (`langgraph.config`) and branch (`main`)
3. Environment variables (see below)

The default required environment variables can be found in [.env.example](.env.example) and are copied below:

```bash
# .env
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...
PINECONE_NAMESPACE=...
FIREWORKS_API_KEY=...

# You can add other keys as appropriate, depending on
# the services you are using.
```

You can fill these out locally, copy the .env file contents, and paste them in the first `Name` argument.

Assuming you've followed the steps above, in just a couple of minutes, you should have a working memory service deployed!

Now let's try it out.

## How to connect to the memory service

Check out the [example notebook](./example.ipynb) to show how to connect your chat bot (in this case a second graph) to your new memory service.

This chat bot reads from the same memory DB as your memory service to easily query from "recall memory".

Connecting to this type of memory service typically follows an interaction pattern similar to the one outlined below:

![Interaction Pattern](./img/memory_interactions.png)

A typical user-facing application you'd build to connect with this service would have 3 or more nodes. The first node queries the DB for useful memories. The second node, which contains the LLM, generates the response. The third node posts the new  messages to the service.

The service waits for a pre-determined interval before it considers the thread "complete". If the user queries a second time within that interval, the memory run is [rolled-back](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/rollback_concurrent/?h=roll) to avoid duplicate processing of a thread.


## How to evaluate

Memory management can be challenging to get right. To make sure your schemas suit your applications' needs, we recommend starting from an evaluation set,
adding to it over time as you find and address common errors in your service.

We have provided a few example evaluation cases in [the test file here](./tests/evals/test_memories.py). As you can see, the metrics themselves don't have to be terribly complicated,
especially not at the outset.

We use [LangSmith's @test decorator](https://docs.smith.langchain.com/how_to_guides/evaluation/unit_testing#write-a-test) to sync all the evalutions to LangSmith so you can better optimize your system and identify the root cause of any issues that may arise.
