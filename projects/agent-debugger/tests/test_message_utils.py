"""Unit tests for agent_debugger.message_utils."""

from __future__ import annotations

from agent_debugger.message_utils import extract_chat_text


def test_extract_from_dict_with_text() -> None:
    assert extract_chat_text({"text": "hello"}) == "hello"


def test_extract_from_dict_with_text_and_extra_fields() -> None:
    assert extract_chat_text({"text": "hello", "recommendations": [{"id": 1}]}) == "hello"


def test_extract_from_dict_without_text_returns_none() -> None:
    assert extract_chat_text({"foo": "bar"}) is None


def test_extract_from_dict_with_non_string_text_returns_none() -> None:
    assert extract_chat_text({"text": 42}) is None
    assert extract_chat_text({"text": None}) is None


def test_extract_from_dict_with_empty_text_returns_none() -> None:
    """Empty text returns None so callers fall back to event.text instead of
    silently dropping a payload that may have useful metadata in other fields."""
    assert extract_chat_text({"text": ""}) is None
    assert extract_chat_text({"text": "", "recommendations": [{"id": 1}]}) is None
    assert extract_chat_text('{"text": ""}') is None


def test_extract_from_json_string_with_text() -> None:
    assert extract_chat_text('{"text": "hi"}') == "hi"


def test_extract_from_json_string_with_leading_whitespace() -> None:
    assert extract_chat_text('  {"text": "hi"}') == "hi"


def test_extract_from_json_string_without_text_key_returns_none() -> None:
    assert extract_chat_text('{"foo": "bar"}') is None


def test_extract_from_malformed_json_returns_none() -> None:
    assert extract_chat_text("{not valid json") is None


def test_extract_from_python_repr_returns_none() -> None:
    """Python repr (single quotes) is not valid JSON; should return None.

    The original structured content should be passed instead of repr-stringifying.
    """
    assert extract_chat_text("{'text': 'hi'}") is None


def test_extract_from_plain_string_returns_none() -> None:
    assert extract_chat_text("just a regular response") is None


def test_extract_from_empty_string_returns_none() -> None:
    assert extract_chat_text("") is None


def test_extract_from_none_returns_none() -> None:
    assert extract_chat_text(None) is None


def test_extract_from_list_returns_none() -> None:
    """Lists are handled upstream by content_to_text; extract_chat_text returns None."""
    assert extract_chat_text([{"text": "hi"}]) is None


def test_extract_from_json_string_that_parses_to_non_dict_returns_none() -> None:
    """JSON arrays or scalars should not match the dict-with-text shape."""
    assert extract_chat_text('["hi"]') is None
    # Note: starts with "{" guard means this is filtered before parsing.
    assert extract_chat_text("42") is None
