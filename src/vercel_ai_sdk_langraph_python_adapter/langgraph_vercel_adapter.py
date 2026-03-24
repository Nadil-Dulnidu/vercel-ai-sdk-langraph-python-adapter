"""
Pluggable adapter for converting LangGraph event streams to Vercel Data Stream Protocol.

This module provides a clean separation between LangGraph's agentic logic and
Vercel's streaming protocol, allowing any LangGraph graph to work with Vercel AI SDK
frontend hooks (useChat, useAssistant) without modifying core agent logic.

Standard Contract for LangGraph Graphs:
    1. State must extend MessagesState (contains 'messages' field)
    2. Nodes should return:
       - AIMessage for conversational responses
       - ToolMessage for tool execution results (with tool_call_id)
    3. Messages should have non-empty content for visibility (optional for tool-only calls)
    4. HumanMessage types are automatically skipped (already in frontend)
    5. Tool calls should follow LangChain format: {id, name, args}
    6. For human-in-the-loop, use interrupt() - adapter propagates as finish with finishReason
    7. Custom state fields are preserved and can be configured to stream (via custom_data_fields)
    8. No other requirements - the adapter handles protocol translation!

Usage:
    # Basic usage (works with any compliant LangGraph graph)
    adapter = LangGraphToVercelAdapter()

    # With custom data fields streaming (e.g., for travel planner)
    adapter = LangGraphToVercelAdapter(
        custom_data_fields=["requirements", "itinerary", "bookings"]
    )

    async for sse_event in adapter.stream(
        graph=your_graph,
        initial_state=initial_state,
        config=config
    ):
        yield sse_event
"""

import json
import uuid
from .config.logging import get_logger 
from typing import AsyncIterator, Dict, Any, Optional, Callable
from datetime import datetime

from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

from .message_extractors import default_message_extractor

logger = get_logger(__name__)


class LangGraphToVercelAdapter:
    """
    Converts LangGraph event streams to Vercel Data Stream Protocol (SSE format).

    This adapter provides a pluggable streaming layer that works with any
    LangGraph graph following the minimal standard contract.

    Protocol Compliance:
        The adapter implements the Vercel Data Stream Protocol v1. When using this
        adapter, ensure the HTTP response includes the following header:
            Content-Type: text/event-stream
            x-vercel-ai-ui-message-stream: v1

        This header signals to Vercel AI SDK frontend hooks (useChat, useAssistant) that
        the response uses the v1 data stream protocol.
    """

    def __init__(
        self,
        message_extractor: Optional[Callable[[Dict[str, Any]], str]] = None,
        include_reasoning: bool = False,
        chunk_size: int = 50,
        custom_data_fields: Optional[list[str]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            message_extractor: Custom function to extract text from state.
                             Defaults to extracting from messages[-1].content
            include_reasoning: Whether to include reasoning in the stream
                             (for models that support chain-of-thought)
            chunk_size: Number of characters per text-delta chunk (default: 50).
                       Larger values reduce overhead, smaller values feel more "real-time".
            custom_data_fields: List of state field names to stream as custom data events.
                              E.g., ["requirements", "itinerary"] will emit data-requirements
                              and data-itinerary events. Optional - if None, no custom data
                              is streamed.
        """
        self.message_extractor = message_extractor or default_message_extractor
        self.include_reasoning = include_reasoning
        self.chunk_size = chunk_size
        self.custom_data_fields = custom_data_fields or []
        self.current_message_id: Optional[str] = None

    def _format_sse_event(self, data: Dict[str, Any]) -> str:
        """
        Format a dictionary as a Server-Sent Event.

        Args:
            data: Dictionary to send as SSE event

        Returns:
            Formatted SSE string

        Example:
            >>> adapter._format_sse_event({"type": "text-delta", "delta": "Hello"})
            'data: {"type":"text-delta","delta":"Hello"}\\n\\n'
        """
        return f"data: {json.dumps(data)}\n\n"

    def _create_message_id(self) -> str:
        """Generate a unique message ID for Vercel protocol."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"msg_{timestamp}_{unique_id}"

    async def _stream_text_chunked(
        self,
        content: str,
        message_id: str,
    ) -> AsyncIterator[str]:
        """
        Stream text content in chunks as per Vercel protocol.

        Emits text-delta events in configurable chunk sizes to provide
        a more authentic streaming experience than sending all content at once.

        Args:
            content: The text content to stream
            message_id: The message ID for this stream

        Yields:
            text-delta SSE events with chunk content
        """
        # Stream content in chunks
        for i in range(0, len(content), self.chunk_size):
            chunk = content[i : i + self.chunk_size]
            yield self._format_sse_event(
                {
                    "type": "text-delta",
                    "id": message_id,
                    "delta": chunk,
                }
            )

    async def _stream_reasoning_chunked(
        self,
        reasoning: str,
        reasoning_id: str,
    ) -> AsyncIterator[str]:
        """
        Stream reasoning content in chunks as per Vercel protocol.

        Emits reasoning-delta events for models that support chain-of-thought reasoning
        (e.g., OpenAI o1, Claude with extended thinking).

        Args:
            reasoning: The reasoning content to stream
            reasoning_id: The reasoning ID for this stream

        Yields:
            reasoning-delta SSE events with chunk content
        """
        # Stream reasoning in chunks
        for i in range(0, len(reasoning), self.chunk_size):
            chunk = reasoning[i : i + self.chunk_size]
            yield self._format_sse_event(
                {
                    "type": "reasoning-delta",
                    "id": reasoning_id,
                    "delta": chunk,
                }
            )

    def _extract_reasoning(self, message: BaseMessage) -> Optional[str]:
        """
        Extract reasoning/thinking content from a message.

        Looks for reasoning in these locations (in order of preference):
        1. LangChain's think_content attribute (Claude extended thinking)
        2. Message metadata with 'thinking' key
        3. Response metadata with 'reasoning' or 'thinking' fields

        Args:
            message: The AI message to extract reasoning from

        Returns:
            Reasoning content if found, None otherwise
        """
        # Check for think_content attribute (Claude extended thinking)
        if hasattr(message, "think_content") and message.think_content:
            return str(message.think_content)

        # Check message metadata
        if hasattr(message, "metadata") and isinstance(message.metadata, dict):
            if "thinking" in message.metadata:
                return str(message.metadata["thinking"])
            if "reasoning" in message.metadata:
                return str(message.metadata["reasoning"])

        # Check response metadata
        if hasattr(message, "response_metadata") and isinstance(
            message.response_metadata, dict
        ):
            if "reasoning" in message.response_metadata:
                return str(message.response_metadata["reasoning"])
            if "thinking" in message.response_metadata:
                return str(message.response_metadata["thinking"])

        return None

    async def _stream_tool_calls(self, message: BaseMessage) -> AsyncIterator[str]:
        """
        Stream tool calls from an AI message.

        Detects tool_calls in AIMessage and streams them as tool-input-available
        events per Vercel protocol. Also looks for corresponding ToolMessage
        responses in subsequent messages.

        Args:
            message: The AI message that contains tool calls

        Yields:
            tool-input-available and tool-output-available SSE events
        """
        # Check for tool_calls attribute (LangChain AIMessage with function_call)
        if not hasattr(message, "tool_calls"):
            return

        tool_calls = message.tool_calls
        if not tool_calls:
            return

        # Stream each tool call
        for tool_call in tool_calls:
            # tool_call is typically a dict or ToolCall object with:
            # - id: unique tool call ID
            # - name: tool/function name
            # - args: tool arguments (dict)
            tool_call_id = None
            tool_name = None
            tool_input = None

            # Handle dict-based tool calls
            if isinstance(tool_call, dict):
                tool_call_id = tool_call.get("id", uuid.uuid4().hex[:8])
                tool_name = tool_call.get("name", "unknown")
                tool_input = tool_call.get("args", {})
            # Handle ToolCall objects from LangChain
            elif hasattr(tool_call, "id") and hasattr(tool_call, "name"):
                tool_call_id = tool_call.id
                tool_name = tool_call.name
                tool_input = (
                    tool_call.args
                    if hasattr(tool_call, "args")
                    else getattr(tool_call, "input", {})
                )
            else:
                continue

            # Stream tool call available event
            if tool_call_id and tool_name:
                yield self._format_sse_event(
                    {
                        "type": "tool-input-available",
                        "toolCallId": tool_call_id,
                        "toolName": tool_name,
                        "input": tool_input if isinstance(tool_input, dict) else {},
                    }
                )

    def _extract_tool_outputs(self, messages: list) -> Dict[str, Any]:
        """
        Extract tool outputs from ToolMessage objects in message history.

        Looks for ToolMessage objects that contain outputs from tool execution
        and maps them to their corresponding tool call IDs.

        Args:
            messages: List of messages from the state

        Returns:
            Dict mapping tool_call_id -> tool output content
        """
        tool_outputs = {}
        for message in messages:
            if isinstance(message, ToolMessage):
                # ToolMessage has tool_call_id and content attributes
                if hasattr(message, "tool_call_id"):
                    tool_call_id = message.tool_call_id
                    output = (
                        message.content if hasattr(message, "content") else str(message)
                    )
                    tool_outputs[tool_call_id] = output
        return tool_outputs

    async def _stream_files(self, message: BaseMessage) -> AsyncIterator[str]:
        """
        Stream file references from a message.

        Detects file attachments or references in message metadata/response_metadata
        and streams them as file events per Vercel protocol.

        Args:
            message: The message to extract files from

        Yields:
            file SSE events
        """
        files = []

        # Check response metadata for files
        if hasattr(message, "response_metadata") and isinstance(
            message.response_metadata, dict
        ):
            if "files" in message.response_metadata:
                files_data = message.response_metadata["files"]
                if isinstance(files_data, list):
                    files.extend(files_data)
                else:
                    files.append(files_data)

            if "attachments" in message.response_metadata:
                attachments_data = message.response_metadata["attachments"]
                if isinstance(attachments_data, list):
                    files.extend(attachments_data)
                else:
                    files.append(attachments_data)

        # Check metadata for files
        if hasattr(message, "metadata") and isinstance(message.metadata, dict):
            if "files" in message.metadata:
                files_data = message.metadata["files"]
                if isinstance(files_data, list):
                    files.extend(files_data)
                else:
                    files.append(files_data)

        # Stream each file
        for file_ref in files:
            if isinstance(file_ref, dict):
                # Handle dict-based file references
                if "url" in file_ref:
                    yield self._format_sse_event(
                        {
                            "type": "file",
                            "url": file_ref["url"],
                            "mediaType": file_ref.get(
                                "mediaType", "application/octet-stream"
                            ),
                        }
                    )
            elif isinstance(file_ref, str):
                # Assume it's a URL or file path
                yield self._format_sse_event(
                    {
                        "type": "file",
                        "url": file_ref,
                        "mediaType": "application/octet-stream",
                    }
                )

    async def _stream_sources(self, message: BaseMessage) -> AsyncIterator[str]:
        """
        Stream source references from a message.

        Detects URLs and document references in message metadata/response_metadata
        and streams them as source-url or source-document events per Vercel protocol.

        Args:
            message: The message to extract sources from

        Yields:
            source-url and source-document SSE events
        """
        sources = []

        # Check response metadata for sources
        if hasattr(message, "response_metadata") and isinstance(
            message.response_metadata, dict
        ):
            # Look for common source fields
            if "sources" in message.response_metadata:
                sources_data = message.response_metadata["sources"]
                if isinstance(sources_data, list):
                    sources.extend(sources_data)
                else:
                    sources.append(sources_data)

            if "documents" in message.response_metadata:
                docs_data = message.response_metadata["documents"]
                if isinstance(docs_data, list):
                    sources.extend(docs_data)
                else:
                    sources.append(docs_data)

            if "citations" in message.response_metadata:
                citations_data = message.response_metadata["citations"]
                if isinstance(citations_data, list):
                    sources.extend(citations_data)
                else:
                    sources.append(citations_data)

        # Check metadata for sources
        if hasattr(message, "metadata") and isinstance(message.metadata, dict):
            if "sources" in message.metadata:
                sources_data = message.metadata["sources"]
                if isinstance(sources_data, list):
                    sources.extend(sources_data)
                else:
                    sources.append(sources_data)

        # Stream each source
        for idx, source in enumerate(sources):
            source_id = f"src_{uuid.uuid4().hex[:8]}"

            if isinstance(source, dict):
                # Handle dict-based sources
                if "url" in source and source["url"]:
                    yield self._format_sse_event(
                        {
                            "type": "source-url",
                            "sourceId": source_id,
                            "url": source["url"],
                        }
                    )
                elif "title" in source or "content" in source:
                    # Document source
                    yield self._format_sse_event(
                        {
                            "type": "source-document",
                            "sourceId": source_id,
                            "mediaType": source.get("mediaType", "text/plain"),
                            "title": source.get("title", f"Source {idx}"),
                        }
                    )
            elif isinstance(source, str):
                # Try to detect if it's a URL
                if source.startswith(("http://", "https://", "www.")):
                    yield self._format_sse_event(
                        {
                            "type": "source-url",
                            "sourceId": source_id,
                            "url": source,
                        }
                    )

    async def stream(
        self,
        graph: StateGraph,
        initial_state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """
        Stream LangGraph execution as Vercel Data Stream Protocol events.

        This is the main entry point for the adapter. It transforms LangGraph's
        state updates into Vercel-compatible SSE events.

        Args:
            graph: The compiled LangGraph graph to execute
            initial_state: Initial state dictionary for the graph
            config: Configuration dict (must include thread_id in configurable)

        Yields:
            SSE-formatted strings ready to send to the frontend

        Note on HTTP Headers:
            When using this in an HTTP response, ensure you set these headers:
            - Content-Type: text/event-stream
            - x-vercel-ai-ui-message-stream: v1
            These headers are required for proper Vercel AI SDK integration.

        Example:
            async for event in adapter.stream(my_graph, initial_state, config):
                # event is like: 'data: {"type":"text-delta","delta":"Hello"}\\n\\n'
                response.write(event)
        """
        # Stream the graph execution
        logger.info(f"[ADAPTER] Starting stream with config: {config}")
        logger.info(f"[ADAPTER] Initial state type: {type(initial_state)}")

        try:
            chunk_count = 0
            message_id = self._create_message_id()
            async for chunk in graph.astream(
                initial_state,
                config,
                stream_mode="values",
            ):
                chunk_count += 1
                logger.info(
                    f"[ADAPTER] Received chunk #{chunk_count}: {list(chunk.keys())}"
                )

                # chunk is the state dict itself
                async for sse_event in self._handle_node_update(chunk, message_id):
                    logger.info(f"[ADAPTER] Yielding SSE event: {sse_event[:100]}...")
                    yield sse_event

            logger.info(f"[ADAPTER] Stream completed. Total chunks: {chunk_count}")

            # Send finish event after successful completion
            yield self._format_sse_event(
                {
                    "type": "finish",
                }
            )

            # Terminate stream with [DONE]
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"[ADAPTER] Error during streaming: {e}", exc_info=True)
            # Send error event
            yield self._format_sse_event(
                {
                    "type": "error",
                    "errorText": str(e),
                }
            )
            return

    async def _handle_node_update(
        self, chunk: Dict[str, Any], message_id: str | None = None
    ) -> AsyncIterator[str]:
        """
        Process state from astream(stream_mode="values").

        Args:
            chunk: The state dict itself (not wrapped in node names)
                  Format: {'messages': [...], 'requirements': ..., 'itinerary': ..., 'bookings': ...}

        Yields:
            SSE-formatted event strings
        """
        # With stream_mode="values", chunk IS the state dict
        state = chunk
        logger.info(f"[STATE] Processing state with keys: {list(state.keys())}")

        # Check for interrupt first
        if "__interrupt__" in state:
            logger.info(f"[STATE] Interrupt detected")
            async for sse_event in self._handle_interrupt(state, message_id):
                yield sse_event
            return  # Stop processing after interrupt

        # Extract and stream messages
        if "messages" in state:
            messages = state["messages"]
            logger.info(f"[STATE] Found {len(messages) if messages else 0} messages")

            if messages:
                # Mark the start of a step (LLM reasoning/response generation)
                yield self._format_sse_event(
                    {
                        "type": "start-step",
                    }
                )
                # Get the last message (most recent addition)
                last_message = messages[-1]
                logger.info(f"[STATE] Last message type: {type(last_message)}")

                # Handle different message types
                # AIMessage: AI responses
                # ToolMessage: Tool execution results
                # Skip HumanMessage (already in frontend)
                should_stream = isinstance(last_message, (AIMessage, ToolMessage))

                if not should_stream:
                    message_type = type(last_message).__name__
                    logger.info(
                        f"[STATE] Skipping {message_type} message (only stream AI and Tool messages)"
                    )
                    return  # Don't stream user messages

                # Special handling for ToolMessage: emit tool-output-available
                if isinstance(last_message, ToolMessage):
                    tool_call_id = getattr(last_message, "tool_call_id", None)
                    if tool_call_id:
                        content = None
                        if isinstance(last_message, BaseMessage):
                            content = last_message.content
                        elif isinstance(last_message, dict):
                            content = last_message.get("content", "")
                        else:
                            content = str(last_message)

                        # Try to parse as JSON if it looks like JSON
                        tool_output = content
                        if content and content.strip():
                            try:
                                tool_output = json.loads(content)
                            except (json.JSONDecodeError, ValueError):
                                # Not valid JSON, use as string
                                tool_output = content

                        logger.info(
                            f"[STATE] Emitting tool-output-available for tool_call_id: {tool_call_id}"
                        )

                        yield self._format_sse_event(
                            {
                                "type": "tool-output-available",
                                "toolCallId": tool_call_id,
                                "output": tool_output,
                            }
                        )

                        # Mark the end of the step
                        yield self._format_sse_event(
                            {
                                "type": "finish-step",
                            }
                        )
                        return  # Don't continue with regular text streaming

                # Extract content from message
                content = None
                if isinstance(last_message, BaseMessage):
                    content = last_message.content
                elif isinstance(last_message, dict):
                    content = last_message.get("content", "")
                else:
                    content = str(last_message)

                logger.info(
                    f"[STATE] Extracted content length: {len(content) if content else 0}"
                )
                if content:
                    logger.info(f"[STATE] Content preview: {content[:100]}")

                # Create unique message ID for this message
                # message_id = self._create_message_id()
                logger.info(f"[STATE] Streaming message with ID: {message_id}")

                # Always send message start event (per Vercel protocol)
                # This happens regardless of whether there's text content
                yield self._format_sse_event(
                    {
                        "type": "start",
                        "messageId": message_id,
                    }
                )

                # Stream reasoning if available and enabled (AIMessage only)
                if self.include_reasoning and isinstance(last_message, AIMessage):
                    reasoning = self._extract_reasoning(last_message)
                    if reasoning and reasoning.strip():
                        reasoning_id = self._create_message_id()
                        logger.info(
                            f"[STATE] Streaming reasoning with ID: {reasoning_id}"
                        )

                        # Send reasoning-start event
                        yield self._format_sse_event(
                            {
                                "type": "reasoning-start",
                                "id": reasoning_id,
                            }
                        )

                        # Stream reasoning in chunks
                        async for chunk_event in self._stream_reasoning_chunked(
                            reasoning, reasoning_id
                        ):
                            yield chunk_event

                        # Send reasoning-end event
                        yield self._format_sse_event(
                            {
                                "type": "reasoning-end",
                                "id": reasoning_id,
                            }
                        )

                # Stream tool calls if present (AIMessage only)
                if isinstance(last_message, AIMessage):
                    async for tool_event in self._stream_tool_calls(last_message):
                        yield tool_event

                # Stream file and source references (AIMessage and ToolMessage)
                async for file_event in self._stream_files(last_message):
                    yield file_event

                async for source_event in self._stream_sources(last_message):
                    yield source_event

                # Stream text content only if available
                if content and content.strip():
                    # Send text-start event
                    yield self._format_sse_event(
                        {
                            "type": "text-start",
                            "id": message_id,
                        }
                    )

                    # Stream content in chunks
                    async for chunk_event in self._stream_text_chunked(
                        content, message_id
                    ):
                        yield chunk_event

                    # Send text-end event
                    yield self._format_sse_event(
                        {
                            "type": "text-end",
                            "id": message_id,
                        }
                    )
                else:
                    if content:
                        logger.warning(f"[STATE] Content is empty or whitespace only")
                    else:
                        logger.info(
                            f"[STATE] No text content (may have reasoning/tools/files only)"
                        )

                # Mark the end of the step
                yield self._format_sse_event(
                    {
                        "type": "finish-step",
                    }
                )
            else:
                logger.warning(f"[STATE] Messages array is empty")
        else:
            logger.warning(f"[STATE] No 'messages' key in state")

        # Stream custom data fields if configured
        # This allows graph-specific data to be sent alongside messages
        for field in self.custom_data_fields:
            if field in state and state[field]:
                yield self._format_sse_event(
                    {
                        "type": f"data-{field}",
                        "data": state[field],
                    }
                )

    async def _handle_interrupt(
        self, state_update: Dict[str, Any], message_id: str | None = None
    ) -> AsyncIterator[str]:
        """
        Handle graph interruption (human-in-the-loop).

        When a LangGraph node calls interrupt(message), LangGraph stores an Interrupt
        object in state["__interrupt__"] as a list. We extract the message and stream
        it as text-delta events, then send a finish event with interrupt reason.

        Args:
            state_update: The state update containing interrupt information

        Yields:
            Text events with interrupt message, then finish event
        """
        interrupt_list = state_update.get("__interrupt__", [])
        logger.info(
            f"[INTERRUPT] Interrupt list type: {type(interrupt_list)}, length: {len(interrupt_list) if isinstance(interrupt_list, list) else 'N/A'}"
        )

        # Extract the interrupt message from the Interrupt object
        # Format: [Interrupt(value="message")]
        interrupt_message = ""
        if interrupt_list:
            interrupt_obj = interrupt_list[0]
            logger.info(f"[INTERRUPT] Interrupt object type: {type(interrupt_obj)}")

            # Interrupt objects have a .value attribute
            if hasattr(interrupt_obj, "value"):
                interrupt_message = str(interrupt_obj.value)
                logger.info(
                    f"[INTERRUPT] Extracted message from .value: {interrupt_message}"
                )
            else:
                # Fallback if structure is different
                interrupt_message = str(interrupt_obj)
                logger.info(
                    f"[INTERRUPT] Using string representation: {interrupt_message}"
                )

        # Stream the interrupt message as text events (so frontend displays it)
        if interrupt_message:
            # message_id = self._create_message_id()
            logger.info(
                f"[INTERRUPT] Streaming interrupt message with ID: {message_id}"
            )

            # Send message start event (per Vercel protocol)
            yield self._format_sse_event(
                {
                    "type": "start",
                    "messageId": message_id,
                }
            )

            # Send text-start event
            yield self._format_sse_event(
                {
                    "type": "text-start",
                    "id": message_id,
                }
            )

            # Stream interrupt message in chunks
            async for chunk_event in self._stream_text_chunked(
                interrupt_message, message_id
            ):
                yield chunk_event

            # Send text-end event
            yield self._format_sse_event(
                {
                    "type": "text-end",
                    "id": message_id,
                }
            )

        # Send finish event with interrupt reason
        logger.info(f"[INTERRUPT] Sending finish event with interrupt reason")
        yield self._format_sse_event(
            {
                "type": "finish",
                "finishReason": "interrupt",
            }
        )

    async def stream_with_final_state(
        self,
        graph: StateGraph,
        initial_state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> tuple[AsyncIterator[str], Dict[str, Any]]:
        """
        Stream execution and return final state.

        This is useful when you need both the stream for the frontend
        and the final state for logging/processing.

        Args:
            graph: The compiled LangGraph graph
            initial_state: Initial state dictionary
            config: Configuration dict

        Returns:
            Tuple of (event iterator, final state dict)
        """
        final_state = None

        async def _stream_and_capture():
            nonlocal final_state
            async for event in self.stream(graph, initial_state, config):
                yield event

            # Capture final state after streaming completes
            final_state = await graph.aget_state(config)

        return _stream_and_capture(), final_state


# Convenience function for common use case
async def stream_langgraph_to_vercel(
    graph: StateGraph,
    initial_state: Dict[str, Any],
    config: Dict[str, Any],
    message_extractor: Optional[Callable] = None,
    custom_data_fields: Optional[list[str]] = None,
) -> AsyncIterator[str]:
    """
    Convenience function to stream a LangGraph graph to Vercel protocol.

    Args:
        graph: The compiled LangGraph graph to execute
        initial_state: Initial state dictionary for the graph
        config: Configuration dict (must include thread_id)
        message_extractor: Optional custom message extractor
        custom_data_fields: Optional list of state field names to stream as custom data events.
                          E.g., ["requirements", "itinerary"] will emit data-requirements
                          and data-itinerary events.

    Yields:
        SSE-formatted event strings

    Example:
        async for event in stream_langgraph_to_vercel(
            my_graph, state, config,
            custom_data_fields=["requirements", "itinerary"]
        ):
            yield event
    """
    adapter = LangGraphToVercelAdapter(
        message_extractor=message_extractor,
        custom_data_fields=custom_data_fields,
    )
    async for event in adapter.stream(graph, initial_state, config):
        yield event
