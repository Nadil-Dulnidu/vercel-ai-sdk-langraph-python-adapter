"""
Message extraction strategies for LangGraph to Vercel adapter.

These extractors provide flexible ways to extract conversational text from
LangGraph state, allowing developers to customize how messages are pulled
from their graph without modifying the adapter.
"""

import json
from typing import Dict, Any, Optional, Callable
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage


def default_message_extractor(state: Dict[str, Any]) -> str:
    """
    Default extractor: Gets the last message content from the messages array.

    This works for any LangGraph graph that extends MessagesState and
    returns AIMessage objects in its nodes.

    Args:
        state: The LangGraph state dictionary

    Returns:
        The content of the last message, or empty string if none exists

    Example:
        >>> state = {"messages": [AIMessage(content="Hello world")]}
        >>> default_message_extractor(state)
        'Hello world'
    """
    messages = state.get("messages", [])
    if not messages:
        return ""

    last_message = messages[-1]

    # Handle both LangChain message objects and dict representations
    if isinstance(last_message, BaseMessage):
        return last_message.content or ""
    elif isinstance(last_message, dict):
        return last_message.get("content", "")

    return str(last_message)


def structured_data_extractor(field_name: str = "result") -> Callable:
    """
    Factory for creating extractors that pull from specific state fields.

    Useful when your graph stores results in custom fields instead of messages.

    Args:
        field_name: The name of the state field to extract

    Returns:
        An extractor function that pulls from the specified field

    Example:
        >>> extractor = structured_data_extractor("bookings")
        >>> state = {"bookings": {"flight": "ABC123"}}
        >>> extractor(state)
        '{"flight": "ABC123"}'
    """
    def extractor(state: Dict[str, Any]) -> str:
        data = state.get(field_name)
        if data is None:
            return ""

        # If it's already a string, return it
        if isinstance(data, str):
            return data

        # Otherwise, serialize to JSON
        return json.dumps(data, indent=2)

    return extractor


def multi_field_extractor(fields: list[str], separator: str = "\n\n") -> Callable:
    """
    Factory for creating extractors that combine multiple state fields.

    Useful when you want to display multiple pieces of state data as a message.

    Args:
        fields: List of field names to extract
        separator: String to join the fields with

    Returns:
        An extractor function that combines the specified fields

    Example:
        >>> extractor = multi_field_extractor(["requirements", "itinerary"])
        >>> state = {
        ...     "requirements": {"origin": "Tokyo"},
        ...     "itinerary": {"days": 5}
        ... }
        >>> result = extractor(state)
    """
    def extractor(state: Dict[str, Any]) -> str:
        parts = []
        for field in fields:
            value = state.get(field)
            if value is not None:
                if isinstance(value, str):
                    parts.append(value)
                else:
                    parts.append(json.dumps(value, indent=2))

        return separator.join(parts)

    return extractor


def summary_field_extractor(state: Dict[str, Any]) -> str:
    """
    Extractor for graphs that explicitly provide a 'summary' field.

    Falls back to default message extraction if no summary exists.

    Args:
        state: The LangGraph state dictionary

    Returns:
        The summary field content, or the last message content

    Example:
        >>> state = {
        ...     "summary": "Trip booked successfully",
        ...     "messages": [AIMessage(content="Booking...")]
        ... }
        >>> summary_field_extractor(state)
        'Trip booked successfully'
    """
    if "summary" in state and state["summary"]:
        return str(state["summary"])

    # Fallback to default message extraction
    return default_message_extractor(state)


class MessageExtractorChain:
    """
    Chains multiple extractors together, trying each until one returns content.

    Useful for graceful fallback strategies.

    Example:
        >>> chain = MessageExtractorChain([
        ...     summary_field_extractor,
        ...     default_message_extractor,
        ...     structured_data_extractor("results")
        ... ])
        >>> state = {"results": {"status": "done"}}
        >>> chain.extract(state)
        '{"status": "done"}'
    """

    def __init__(self, extractors: list[Callable]):
        """
        Args:
            extractors: List of extractor functions to try in order
        """
        self.extractors = extractors

    def extract(self, state: Dict[str, Any]) -> str:
        """
        Try each extractor until one returns non-empty content.

        Args:
            state: The LangGraph state dictionary

        Returns:
            The first non-empty extracted content, or empty string
        """
        for extractor in self.extractors:
            result = extractor(state)
            if result and result.strip():
                return result

        return ""


# Pre-configured extractor chains for common patterns
DEFAULT_CHAIN = MessageExtractorChain([
    default_message_extractor,
])

SUMMARY_FIRST_CHAIN = MessageExtractorChain([
    summary_field_extractor,
    default_message_extractor,
])
