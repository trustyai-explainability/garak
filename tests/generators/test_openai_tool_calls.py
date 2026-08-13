# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool-calling support in the OpenAI chat generator."""

import os
import types

import pytest

from garak.attempt import Conversation, Message, Turn
from garak.generators.openai import OpenAICompatible, OpenAIGenerator


@pytest.fixture
def set_fake_env(request) -> None:
    stored_env = os.getenv(OpenAIGenerator.ENV_VAR, None)

    def restore_env():
        if stored_env is not None:
            os.environ[OpenAIGenerator.ENV_VAR] = stored_env
        else:
            del os.environ[OpenAIGenerator.ENV_VAR]

    os.environ[OpenAIGenerator.ENV_VAR] = os.path.abspath(__file__)
    request.addfinalizer(restore_env)


TOOL_CONVERSATION = [
    {"role": "user", "content": "summarise my email"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_email", "arguments": "{}"},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "read_email",
        "content": "malicious payload",
    },
]


def test_conversation_to_list_serializes_tool_turns():
    conv = Conversation.from_openai(TOOL_CONVERSATION)
    out = OpenAICompatible._conversation_to_list(conv)

    assert out[0] == {"role": "user", "content": "summarise my email"}

    assert out[1]["role"] == "assistant"
    assert out[1]["content"] is None
    assert out[1]["tool_calls"][0]["id"] == "call_1"
    assert out[1]["tool_calls"][0]["function"]["name"] == "read_email"

    assert out[2]["role"] == "tool"
    assert out[2]["tool_call_id"] == "call_1"
    assert out[2]["name"] == "read_email"
    assert out[2]["content"] == "malicious payload"


class _FakeToolCall:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def _fake_response(content, tool_calls=None):
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


@pytest.mark.usefixtures("set_fake_env")
def test_call_model_captures_tool_calls_and_forwards_tools(mocker):
    gen = OpenAIGenerator(name="gpt-4o")

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        payload = {
            "id": "call_send",
            "type": "function",
            "function": {"name": "send_email", "arguments": "{}"},
        }
        return _fake_response("done", tool_calls=[_FakeToolCall(payload)])

    mocker.patch.object(gen.generator, "create", side_effect=fake_create)

    tools = [{"type": "function", "function": {"name": "send_email"}}]
    conv = Conversation.from_openai(
        TOOL_CONVERSATION, notes={"tools": tools, "tool_choice": "auto"}
    )

    outputs = gen._call_model(conv)

    # per-conversation tools + tool_choice forwarded to the request
    assert captured["tools"] == tools
    assert captured["tool_choice"] == "auto"

    # response tool calls captured into Message.notes
    assert len(outputs) == 1
    assert outputs[0].text == "done"
    assert outputs[0].notes["tool_calls"][0]["function"]["name"] == "send_email"


@pytest.mark.usefixtures("set_fake_env")
def test_call_model_without_tools_omits_tool_keys(mocker):
    gen = OpenAIGenerator(name="gpt-4o")

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("hello", tool_calls=None)

    mocker.patch.object(gen.generator, "create", side_effect=fake_create)

    conv = Conversation([Turn(role="user", content=Message("hi"))])
    outputs = gen._call_model(conv)

    assert "tools" not in captured
    assert outputs[0].text == "hello"
    assert "tool_calls" not in outputs[0].notes
