# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the indirect prompt injection probe."""

import json

import pytest

import garak._plugins
from garak.attempt import Conversation
from garak.probes.injection import IndirectInjection


def test_probe_loads_shipped_conversations():
    probe = garak._plugins.load_plugin("probes.injection.IndirectInjection")
    assert isinstance(probe, IndirectInjection)
    assert len(probe.prompts) >= 1
    assert probe.primary_detector == "injection_judge.InjectionJudge"

    conv = probe.prompts[0]
    assert isinstance(conv, Conversation)
    # notes carry tool schema + judge criteria for downstream consumers
    assert conv.notes.get("judge_description")
    # first shipped conversation exercises tool turns
    assert "tool" in {turn.role for turn in conv.turns}
    assert conv.notes.get("tools")


def test_probe_reads_custom_source(tmp_path):
    source = tmp_path / "convos.json"
    entry = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
        "tools": [{"type": "function", "function": {"name": "noop"}}],
        "judge_description": "attack succeeds if the model leaks the secret",
    }
    source.write_text(json.dumps([entry]), encoding="utf-8")

    config_root = {
        "probes": {
            "injection": {
                "IndirectInjection": {"conversation_source": str(source)},
            }
        }
    }
    probe = IndirectInjection(config_root=config_root)

    assert len(probe.prompts) == 1
    conv = probe.prompts[0]
    assert [t.role for t in conv.turns] == ["user", "assistant"]
    assert (
        conv.notes["judge_description"]
        == "attack succeeds if the model leaks the secret"
    )
    assert conv.notes["tools"][0]["function"]["name"] == "noop"


def test_probe_skips_malformed_entries(tmp_path):
    source = tmp_path / "convos.json"
    entries = [
        {"messages": [{"role": "user", "content": "ok"}]},
        {"no_messages": True},  # malformed -> skipped
    ]
    source.write_text(json.dumps(entries), encoding="utf-8")

    config_root = {
        "probes": {
            "injection": {
                "IndirectInjection": {"conversation_source": str(source)},
            }
        }
    }
    probe = IndirectInjection(config_root=config_root)
    assert len(probe.prompts) == 1


def test_probe_reads_jsonl(tmp_path):
    source = tmp_path / "convos.jsonl"
    lines = [
        json.dumps({"messages": [{"role": "user", "content": "a"}]}),
        json.dumps({"messages": [{"role": "user", "content": "b"}]}),
    ]
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    config_root = {
        "probes": {
            "injection": {
                "IndirectInjection": {"conversation_source": str(source)},
            }
        }
    }
    probe = IndirectInjection(config_root=config_root)
    assert len(probe.prompts) == 2
