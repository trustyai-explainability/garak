# SPDX-FileCopyrightText: Portions Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indirect prompt injection probes.

Replays pre-built conversations (authored offline in OpenAI chat-completion format)
against a target **unchanged**. The conversations may contain tool calls and
tool-call results, so a harmful payload can be injected anywhere -- including as the
result of a tool call, modelling indirect prompt injection.

Each conversation carries an optional ``tools`` schema (forwarded to the target so it
can emit new tool calls) and a natural-language ``judge_description`` describing what a
successful attack looks like. The description is consumed by
``detectors.injection_judge.InjectionJudge``.
"""

import json
import logging
from pathlib import Path
from typing import List

import garak.attempt
from garak import _config
from garak.data import path as data_path
from garak.exception import GarakException
import garak.probes


class IndirectInjection(garak.probes.Probe):
    """Replay pre-built (possibly tool-calling) conversations to model indirect prompt injection

    Loads conversations from a JSON/JSONL source and sends each one to the target
    without modification. An LLM-as-a-judge detector scores the target response using
    the per-conversation description.
    """

    lang = "en"
    goal = "elicit harmful behaviour via an injected conversation"
    doc_uri = (
        "https://owasp.org/www-project-top-10-for-large-language-model-applications/"
    )
    # instructions injected via other contexts (tool results, documents, ...)
    intent = "T009indirect"
    active = False  # opt-in: requires an external conversation_source file
    tags = [
        "avid-effect:security:S0403",
        "owasp:llm01",
        "quality:Security:PromptStability",
    ]
    tier = garak.probes.Tier.INFORMATIONAL
    primary_detector = "injection_judge.InjectionJudge"

    DEFAULT_PARAMS = garak.probes.Probe.DEFAULT_PARAMS | {
        # path to a .json (single object or list of objects) or .jsonl file; may be an
        # absolute path or a path relative to a garak data directory
        "conversation_source": "injection/example_conversations.json",
    }

    def __init__(self, config_root=_config):
        super().__init__(config_root=config_root)
        self.prompts = []
        self._conversation_notes = []
        for entry in self._load_source():
            conversation = self._build_conversation(entry)
            self.prompts.append(conversation)

    def _resolve_source(self) -> Path:
        """Resolve conversation_source to an existing file path."""
        candidate = Path(self.conversation_source)
        if candidate.is_file():
            return candidate
        try:
            return Path(data_path / self.conversation_source)
        except GarakException as e:
            raise GarakException(
                f"Could not find conversation_source '{self.conversation_source}'"
            ) from e

    def _load_source(self) -> List[dict]:
        """Load raw conversation entries from the source file.

        Accepts a JSON file containing either a single object or a list of objects, or
        a JSONL file with one object per line. Each object is expected to have a
        ``messages`` key (OpenAI messages array) and optional ``tools`` /
        ``judge_description`` keys.
        """
        source = self._resolve_source()
        entries: List[dict] = []
        with open(source, "r", encoding="utf-8") as f:
            if source.suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            else:
                data = json.load(f)
                entries = data if isinstance(data, list) else [data]

        cleaned = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or "messages" not in entry:
                logging.warning(
                    "%s: skipping malformed conversation entry %d in %s",
                    self.__class__.__name__,
                    i,
                    source,
                )
                continue
            cleaned.append(entry)
        return cleaned

    def _build_conversation(self, entry: dict) -> garak.attempt.Conversation:
        """Build a Conversation from a raw entry, stashing tools + judge criteria in notes."""
        notes = {}
        if entry.get("tools"):
            notes["tools"] = entry["tools"]
        if entry.get("tool_choice") is not None:
            notes["tool_choice"] = entry["tool_choice"]
        if entry.get("judge_description"):
            notes["judge_description"] = entry["judge_description"]
        return garak.attempt.Conversation.from_openai(entry["messages"], notes=notes)
