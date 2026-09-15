# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import threading
import time
from unittest.mock import Mock

import pytest

from garak import _config, _plugins, evaluators
from garak.attempt import ATTEMPT_STARTED, Attempt, Conversation, Message, Turn
from garak.harnesses.earlystop import (
    EarlyStopHarness,
    _detect_attempt,
    _is_rejected,
)
from garak.intents import TextStub
from garak.probes.base import IntentProbe


def _attempt(stub, output="response"):
    attempt = Attempt(
        probe_classname="test.TestProbe",
        intent=stub.intent,
        notes={"source_stub": stub},
        goal=stub.content,
        status=ATTEMPT_STARTED,
        prompt=Conversation([Turn("user", Message(stub.content))]),
    )
    if output is not None:
        attempt.outputs = [Message(output)]
    return attempt


@pytest.fixture
def harness_env(tmp_path):
    _config.load_base_config()
    _config.transient.intent_spec = "T999test"
    _config.run.serve_detectorless_intents = True
    report_file = (tmp_path / "report.jsonl").open("w+", encoding="utf-8")
    _config.transient.reportfile = report_file
    _config.transient.report_filename = str(tmp_path / "report.jsonl")
    _config.plugins.detector_spec = "always.Fail"
    harness = EarlyStopHarness()
    evaluator = evaluators.ThresholdEvaluator()
    yield harness, evaluator, report_file
    report_file.close()


def test_intent_probe_does_not_attach_earlystop_metadata(harness_env):
    probe = IntentProbe.__new__(IntentProbe)
    probe.prompt_intents = ["T999test"]
    probe.prompt_stubs = [TextStub("T999test", "source")]
    attempt = Attempt(prompt=Conversation([Turn("user", Message("source"))]), notes={})

    probe._attempt_prestore_hook(attempt, 0)

    assert "source_stub" not in attempt.notes


def test_empty_detector_list_fails_closed(harness_env):
    _, evaluator, _ = harness_env
    attempt = _attempt(TextStub("T999test", "original request"))

    assert _detect_attempt(attempt, [], evaluator), "No detector must reject an attempt"


def test_all_detectors_run_before_consensus(harness_env):
    _, evaluator, _ = harness_env
    first = Mock()
    first.detectorname = "test.First"
    first.detect.return_value = [0.0]
    second = Mock()
    second.detectorname = "test.Second"
    second.detect.return_value = [1.0]
    attempt = _attempt(TextStub("T999test", "original request"))

    rejected = _detect_attempt(attempt, [first, second], evaluator)

    assert rejected, "One safe detector result must reject the attempt"
    assert first.detect.called and second.detect.called, "Every detector must run"
    assert set(attempt.detector_results) == {"test.First", "test.Second"}


def test_original_prompt_exists_before_detector(harness_env):
    _, evaluator, _ = harness_env
    seen = []
    detector = Mock()
    detector.detectorname = "test.MockDetector"

    def detect(attempt):
        seen.append(attempt.notes.get("original_prompt"))
        return [0.0]

    detector.detect.side_effect = detect
    attempt = _attempt(TextStub("T999test", "original request"))
    attempt.prompt.turns[0].content.text = "wrapped request"

    assert _is_rejected(attempt, [detector], evaluator)
    assert seen == ["original request"]


def test_stub_linkage_uses_source_stub_value(harness_env):
    harness, evaluator, _ = harness_env
    detector = _plugins.load_plugin("detectors.always.Fail", break_on_fail=False)
    stub = TextStub("T999test", "same request")
    baseline = _attempt(stub, output=None)
    attacked = _attempt(TextStub("T999test", "same request"))

    accepted, rejected = harness._update_attempt_status(
        [attacked], [], [baseline], [detector], evaluator
    )

    assert len(accepted) == 1, "Equal source stubs must link attack results"
    assert not rejected, "A successful linked attack must remove the stub"


def test_string_source_stub_fails_closed(harness_env):
    harness, evaluator, _ = harness_env
    detector = _plugins.load_plugin("detectors.always.Fail", break_on_fail=False)
    stub = TextStub("T999test", "same request")
    baseline = _attempt(stub, output=None)
    attacked = _attempt(stub)
    attacked.notes["source_stub"] = "same request"

    accepted, rejected = harness._update_attempt_status(
        [attacked], [], [baseline], [detector], evaluator
    )

    assert not accepted, "Raw source-stub strings must not create a false match"
    assert len(rejected) == 1, "Unlinked attacks must leave the stub rejected"


def test_parallel_detector_calls_overlap(harness_env):
    harness, evaluator, _ = harness_env
    state = {"active": 0, "maximum": 0}
    state_lock = threading.Lock()

    class ConcurrentDetector:
        detectorname = "test.Concurrent"

        def detect(self, attempt):
            with state_lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            time.sleep(0.02)
            with state_lock:
                state["active"] -= 1
            return [0.0]

    stubs = [TextStub("T999test", f"request {index}") for index in range(4)]
    baselines = [_attempt(stub, output=None) for stub in stubs]
    attacks = [_attempt(TextStub(stub.intent, stub.content)) for stub in stubs]
    _config.system.parallel_attempts = 4

    harness._update_attempt_status(
        attacks, [], baselines, [ConcurrentDetector()], evaluator
    )

    assert state["maximum"] > 1, "Parallel detector requests must overlap"


def test_parallel_status_matches_sequential(harness_env):
    harness, evaluator, _ = harness_env
    detector = _plugins.load_plugin("detectors.always.Fail", break_on_fail=False)
    stubs = [TextStub("T999test", f"request {index}") for index in range(4)]
    baselines = [_attempt(stub, output=None) for stub in stubs]
    attacks = [_attempt(TextStub(stub.intent, stub.content)) for stub in stubs]
    _config.system.parallel_attempts = 4

    accepted, rejected = harness._update_attempt_status(
        attacks, [], baselines, [detector], evaluator
    )

    assert len(accepted) == len(
        stubs
    ), "Parallel status must accept every safe detector hit"
    assert not rejected, "Parallel status must match sequential status"


def test_rejected_consensus_does_not_write_hitlog(harness_env):
    harness, evaluator, report_file = harness_env
    first = Mock()
    first.detectorname = "test.First"
    first.detect.return_value = [1.0]
    second = Mock()
    second.detectorname = "test.Second"
    second.detect.return_value = [0.0]
    attempt = _attempt(TextStub("T999test", "original request"))

    harness._classify_attempts(
        [attempt],
        {"T999test": [first, second]},
        evaluator,
    )
    report_file.flush()

    hitlog_path = report_file.name.replace("report.jsonl", "report.hitlog.jsonl")
    assert not Path(hitlog_path).exists(), "Rejected consensus must not create a hitlog"


def test_run_writes_one_summary_per_source_stub(harness_env):
    harness, evaluator, report_file = harness_env
    generator = _plugins.load_plugin("generators.test.Repeat")

    harness.run(generator, [], ["detectors.always.Fail"], evaluator)
    report_file.flush()
    report_file.seek(0)
    entries = [json.loads(line) for line in report_file]

    summaries = [
        entry for entry in entries if entry["entry_type"] == "harness_stub_summary"
    ]
    attempts = [entry for entry in entries if entry["entry_type"] == "attempt"]
    aggregate = [entry for entry in entries if entry["entry_type"] == "harness_summary"]
    assert len(summaries) == 6, "Each text source stub needs one final summary"
    assert len(attempts) == 6, "Each baseline attempt must appear once in the report"
    assert len(aggregate) == 1, "The harness needs one aggregate summary"
    assert aggregate[0]["accepted_stubs"] == 6
    assert aggregate[0]["attack_success_rate"] == 1.0

    hitlog_path = report_file.name.replace("report.jsonl", "report.hitlog.jsonl")
    assert Path(hitlog_path).exists(), "Accepted baseline outputs need hitlog records"
    assert len(Path(hitlog_path).read_text(encoding="utf-8").splitlines()) == 6


def test_attack_probe_plugin_cache_entry(harness_env):
    harness, evaluator, report_file = harness_env
    generator = _plugins.load_plugin("generators.test.Repeat")

    harness.run(
        generator,
        ["probes.multilingual.TranslationIntent"],
        ["detectors.always.Pass"],
        evaluator,
    )
    report_file.flush()
    report_file.seek(0)
    entries = [json.loads(line) for line in report_file]
    cached = [entry for entry in entries if entry["entry_type"] == "plugin_cache"]
    cached_plugins = {
        name for entry in cached for name in entry["plugin_cache"].get("probes", {})
    }

    assert "probes.multilingual.TranslationIntent" in cached_plugins
