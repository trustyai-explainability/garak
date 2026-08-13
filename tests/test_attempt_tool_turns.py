# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool-calling metadata support in the Conversation/Turn/Message model."""

from garak.attempt import Conversation, Turn, roles, tool_message_keys


def test_tool_role_allowed():
    assert "tool" in roles


def test_turn_from_dict_routes_tool_calls_to_notes():
    turn = Turn.from_dict(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f"}}],
        }
    )
    assert turn.role == "assistant"
    assert turn.content.text is None
    assert turn.content.notes["tool_calls"][0]["id"] == "c1"


def test_turn_from_dict_routes_tool_result_keys():
    turn = Turn.from_dict(
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "f",
            "content": "result",
        }
    )
    assert turn.role == "tool"
    assert turn.content.text == "result"
    assert turn.content.notes["tool_call_id"] == "c1"
    assert turn.content.notes["name"] == "f"


def test_conversation_from_openai_round_trip():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "f", "content": "out"},
    ]
    conv = Conversation.from_openai(messages)
    assert [t.role for t in conv.turns] == ["user", "assistant", "tool"]
    assert conv.turns[1].content.notes["tool_calls"][0]["id"] == "c1"
    assert conv.turns[2].content.notes["tool_call_id"] == "c1"


def test_tool_message_keys_constant():
    assert set(tool_message_keys) == {"tool_calls", "tool_call_id", "name"}
