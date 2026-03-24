# Vercel AI SDK LangGraph Python Adapter

A small Python adapter that converts LangGraph event streams into the Vercel AI SDK
Data Stream Protocol (v1). This lets Vercel AI SDK frontend hooks (like `useChat`
and `useAssistant`) consume a LangGraph graph without changes to your agent logic.

## Features
- Converts LangGraph streaming output to Vercel Data Stream Protocol v1
- Supports tool call events, tool outputs, files, and sources
- Optional custom data field streaming
- Works with any LangGraph graph that uses `MessagesState`

## Requirements
- Python 3.11+
- `langgraph`, `langchain`, `fastapi`, `httpx`

## Installation

### Install from GitHub (pip)
```bash
pip install git+https://github.com/Nadil-Dulnidu/vercel-ai-sdk-langraph-python-adapter.git
```

### Install from GitHub (uv)
```bash
uv add git+https://github.com/Nadil-Dulnidu/vercel-ai-sdk-langraph-python-adapter.git
```

### Local editable install
```bash
pip install -e .
```

## Quick Start

```python
from vercel_ai_sdk_langraph_python_adapter import stream_langgraph_to_vercel

async for event in stream_langgraph_to_vercel(graph, initial_state, config):
    yield event
```

## HTTP Headers

When returning the stream from an HTTP endpoint, add these headers:
- `Content-Type: text/event-stream`
- `x-vercel-ai-ui-message-stream: v1`
- `x-vercel-ai-protocol: data`

A helper is provided in `http_headers.py`:

```python
from fastapi.responses import StreamingResponse
from vercel_ai_sdk_langraph_python_adapter.http_headers import patch_vercel_headers

response = StreamingResponse(stream_generator(), media_type="text/event-stream")
response = patch_vercel_headers(response)
```

## Notes
- The adapter expects graphs that follow LangGraph's `MessagesState` pattern.
- Tool calls should use the LangChain tool call format: `{id, name, args}`.
- Human messages are not streamed (they already exist on the frontend).

## License

MIT
