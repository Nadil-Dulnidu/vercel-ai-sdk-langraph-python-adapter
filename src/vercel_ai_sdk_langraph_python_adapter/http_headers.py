"""
HTTP header utilities for Vercel AI SDK compatibility.

This module provides utilities for adding required HTTP headers
to streaming responses for compatibility with Vercel AI SDK.
"""

from fastapi.responses import StreamingResponse


def patch_vercel_headers(response: StreamingResponse) -> StreamingResponse:
    """
    Add required headers for Vercel AI SDK compatibility.

    Adds headers required by the Vercel Data Stream Protocol to ensure
    proper streaming behavior with the Vercel AI SDK frontend hooks
    (useChat, useAssistant, etc.).

    Args:
        response: FastAPI StreamingResponse to patch

    Returns:
        StreamingResponse with added Vercel-compatible headers

    Headers added:
        - x-vercel-ai-ui-message-stream: Protocol version identifier (v1)
        - x-vercel-ai-protocol: Protocol type (data)
        - Cache-Control: Prevents caching of streaming responses
        - Connection: Keeps connection alive for streaming
        - X-Accel-Buffering: Disables nginx buffering for immediate streaming
    """
    response.headers["x-vercel-ai-ui-message-stream"] = "v1"
    response.headers["x-vercel-ai-protocol"] = "data"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"

    return response
