# vercel-ai-sdk-langraph-python-adapter

A small Python adapter that converts LangGraph event streams into the Vercel AI SDK
Data Stream Protocol (v1) so the frontend hooks (like `useChat` / `useAssistant`)
can consume your LangGraph graph without changes.

## Install (local)

```bash
pip install -e .
```

## Build a package

```bash
python -m pip install build
python -m build
```

This will create a `dist/` folder with the wheel and sdist.

## Quick usage

```python
from vercel_ai_sdk_langraph_python_adapter import stream_langgraph_to_vercel

async for event in stream_langgraph_to_vercel(graph, initial_state, config):
    yield event
```
