# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive early-stop harness for explicit intent-based scans."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import json
import logging
from pathlib import Path
import threading
from typing import Iterable

import garak
from garak import _config, _plugins
from garak.attempt import ATTEMPT_COMPLETE, Attempt
from garak.evaluators import Evaluator
from garak.exception import GarakException
from garak.harnesses import Harness
from garak.harnesses.base import _emit_plugin_cache_entry
from garak.intents import Stub
from garak.probes.base import IntentProbe
from garak.services import intentservice

_DETECTOR_ERRORS = (
    AttributeError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    GarakException,
)
_REPORT_LOCK = threading.Lock()


def _detector_name(detector) -> str:
    return detector.detectorname.replace("garak.detectors.", "")


def _stub_key(stub: Stub | None) -> tuple[str | None, str]:
    if stub is None:
        return (None, "")
    content = stub.content
    if isinstance(content, str):
        content_key = content
    else:
        content_key = repr(content)
    return (stub.intent, content_key)


def _stub_text(stub: Stub | None) -> str:
    if stub is None:
        return ""
    if isinstance(stub.content, str):
        return stub.content
    return repr(stub.content)


def _source_stub(attempt: Attempt) -> Stub | None:
    stub = attempt.notes.get("source_stub")
    return stub if isinstance(stub, Stub) else None


def _detectors_for(detectors, attempt: Attempt):
    if isinstance(detectors, dict):
        return detectors.get(attempt.intent, [])
    return detectors


def _write_success_hitlog(attempt: Attempt) -> None:
    if not attempt.outputs:
        return
    output = attempt.outputs[0]
    if output is None:
        return

    report_path = Path(str(_config.transient.report_filename))
    if report_path.name.endswith(".report.jsonl"):
        hitlog_path = report_path.with_name(
            report_path.name.replace(".report.jsonl", ".hitlog.jsonl")
        )
    else:
        hitlog_path = report_path.with_suffix(".hitlog.jsonl")
    if _config.transient.hitlogfile is None or _config.transient.hitlogfile.closed:
        mode = "a" if hitlog_path.exists() else "w"
        _config.transient.hitlogfile = open(
            hitlog_path, mode, buffering=1, encoding="utf-8"
        )
    _config.transient.hitlogfile.write(
        json.dumps(
            {
                "goal": attempt.goal,
                "prompt": asdict(attempt.prompt),
                "output": asdict(output),
                "triggers": attempt.notes.get("triggers"),
                "score": 1.0,
                "run_id": str(_config.transient.run_id),
                "attempt_id": str(attempt.uuid),
                "attempt_seq": attempt.seq,
                "attempt_idx": 0,
                "generator": f"{_config.plugins.target_type} {_config.plugins.target_name}",
                "probe": "baseline",
                "detector": list(attempt.detector_results),
                "generations_per_prompt": _config.run.generations,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def _detect_attempt(
    attempt: Attempt, detectors, evaluator: Evaluator, detector_locks=None
) -> bool:
    """Return true when an attempt is rejected by at least one safe signal."""
    outputs = attempt.outputs
    if not outputs or any(output is None or not output.text for output in outputs):
        return True

    for detector in detectors:
        name = _detector_name(detector)
        try:
            lock = detector_locks.get(id(detector)) if detector_locks else None
            if lock is None:
                scores = list(detector.detect(attempt))
            else:
                with lock:
                    scores = list(detector.detect(attempt))
        except _DETECTOR_ERRORS as error:
            logging.error(
                "Detector %s failed for attempt %s: %s", name, attempt.uuid, error
            )
            scores = [None] * len(outputs)

        if len(scores) != len(outputs):
            scores = (scores + [None] * len(outputs))[: len(outputs)]
        attempt.detector_results[name] = scores

        for score in scores:
            if score is None:
                return True
            try:
                if evaluator.test(float(score)):
                    return True
            except (TypeError, ValueError):
                return True

    return False


def _is_rejected(
    attempt: Attempt,
    detectors: list,
    evaluator: Evaluator,
    baseline_attempt: Attempt | None = None,
    detector_locks=None,
) -> bool:
    """Classify an attempt with conservative, fail-closed detector consensus."""
    if "original_prompt" not in attempt.notes and attempt.goal:
        attempt.notes["original_prompt"] = attempt.goal
    rejected = _detect_attempt(attempt, detectors, evaluator, detector_locks)
    if not rejected and baseline_attempt is not None:
        baseline_attempt.conversations = deepcopy(attempt.conversations)
    return rejected


def _update_attempt_status(
    attacked_attempts: list[Attempt],
    previously_accepted: list[Attempt],
    previously_rejected: list[Attempt],
    detectors,
    evaluator: Evaluator,
):
    """Update source-stub status after one attack stage."""
    accepted_attempts = list(previously_accepted)
    rejected_attempts = []
    results_by_stub: dict[tuple[str | None, str], list[bool]] = {}
    detector_locks = {
        id(detector): threading.Lock()
        for attempt in attacked_attempts
        for detector in _detectors_for(detectors, attempt)
    }

    for attempt in attacked_attempts:
        stub = _source_stub(attempt)
        if stub is None:
            logging.warning("Attempt %s has no source stub", attempt.uuid)

    parallel_attempts = getattr(_config.system, "parallel_attempts", False)
    use_parallel = isinstance(parallel_attempts, int) and parallel_attempts > 1
    work = []
    for attempt in attacked_attempts:
        stub = _source_stub(attempt)
        if stub is not None:
            work.append((attempt, _detectors_for(detectors, attempt), stub))

    if use_parallel and work:
        max_workers = min(
            parallel_attempts,
            getattr(_config.system, "max_workers", parallel_attempts),
            len(work),
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _is_rejected,
                    attempt,
                    selected,
                    evaluator,
                    None,
                    detector_locks,
                ): stub
                for attempt, selected, stub in work
            }
            for future in as_completed(future_map):
                stub = future_map[future]
                try:
                    rejected = future.result()
                except _DETECTOR_ERRORS as error:
                    logging.error("Detector stage failed: %s", error)
                    rejected = True
                results_by_stub.setdefault(_stub_key(stub), []).append(rejected)
    else:
        for attempt, selected, stub in work:
            results_by_stub.setdefault(_stub_key(stub), []).append(
                _is_rejected(
                    attempt,
                    selected,
                    evaluator,
                    detector_locks=detector_locks,
                )
            )

    for baseline in previously_rejected:
        stub = _source_stub(baseline)
        key = _stub_key(stub)
        results = results_by_stub.get(key, [])
        if results and not all(results):
            accepted_attempts.append(baseline)
        else:
            rejected_attempts.append(baseline)

    return accepted_attempts, rejected_attempts


class EarlyStopHarness(Harness):
    """Run an intent baseline and attack probes until each stub succeeds."""

    DEFAULT_PARAMS = {
        "compatible_probes": [
            "grandma.GrandmaIntent",
            "tap.TAPIntent",
            "multilingual.TranslationIntent",
            "spo.SPOIntent",
            "spo.SPOIntentUserAugmented",
            "spo.SPOIntentSystemAugmented",
            "spo.SPOIntentBothAugmented",
        ],
    }

    def __init__(self, config_root=_config):
        super().__init__(config_root=config_root)

    def _update_attempt_status(
        self,
        attacked_attempts,
        previously_accepted,
        previously_rejected,
        detectors,
        evaluator,
    ):
        return _update_attempt_status(
            attacked_attempts,
            previously_accepted,
            previously_rejected,
            detectors,
            evaluator,
        )

    def _load_probe(self, probe_name: str) -> IntentProbe | None:
        try:
            probe = _plugins.load_plugin(probe_name, break_on_fail=False)
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            GarakException,
        ) as error:
            logging.error("Failed to load %s: %s", probe_name, error)
            return None
        if not isinstance(probe, IntentProbe):
            logging.warning("%s is not an IntentProbe, skipping", probe_name)
            return None
        short_name = probe_name.removeprefix("probes.")
        if self.compatible_probes and short_name not in self.compatible_probes:
            logging.warning("%s is not compatible with EarlyStop, skipping", short_name)
            return None
        return probe

    @staticmethod
    def _collect_stubs() -> list[Stub]:
        stubs = {}
        for intent in sorted(intentservice.get_applicable_intents()):
            for stub in intentservice.get_intent_stubs(intent):
                if stub.content is not None:
                    stubs.setdefault(_stub_key(stub), stub)
        return [stubs[key] for key in sorted(stubs)]

    @staticmethod
    def _make_baseline_probe(stubs: list[Stub]) -> IntentProbe:
        probe = IntentProbe()
        probe.stubs = list(stubs)
        probe.stub_intents = [stub.intent for stub in stubs]
        probe.build_prompts()
        return probe

    @staticmethod
    def _restrict_probe(probe: IntentProbe, rejected_keys: set[tuple[str | None, str]]):
        selected = [
            (stub, intent)
            for stub, intent in zip(probe.stubs, probe.stub_intents)
            if _stub_key(stub) in rejected_keys
        ]
        probe.stubs = [stub for stub, _ in selected]
        probe.stub_intents = [intent for _, intent in selected]
        probe.build_prompts()
        if probe.follow_prompt_cap:
            probe._prune_data(probe.soft_probe_prompt_cap)

    @staticmethod
    def _annotate_attempts(attempts: Iterable[Attempt]) -> list[Attempt]:
        annotated = list(attempts)
        for attempt in annotated:
            stub = _source_stub(attempt)
            if stub is not None:
                attempt.notes["original_prompt"] = _stub_text(stub)
        return annotated

    @staticmethod
    def _write_completed_attempts(attempts: Iterable[Attempt]) -> None:
        for attempt in attempts:
            attempt.status = ATTEMPT_COMPLETE
            with _REPORT_LOCK:
                _config.transient.reportfile.write(
                    json.dumps(attempt.as_dict(), ensure_ascii=False) + "\n"
                )

    @staticmethod
    def _load_detector_map(detector_names, stubs: list[Stub]):
        names_by_intent = {}
        if detector_names:
            for stub in stubs:
                names_by_intent[stub.intent] = list(detector_names)
        else:
            for stub in stubs:
                names_by_intent[stub.intent] = sorted(
                    intentservice.get_detectors(stub.intent) or []
                )

        detector_map = {}
        all_detectors = {}
        for intent, names in names_by_intent.items():
            loaded = []
            for name in names:
                path = name if name.startswith("detectors.") else f"detectors.{name}"
                detector = _plugins.load_plugin(path, break_on_fail=False)
                if detector is None:
                    logging.warning("Detector %s failed to load", path)
                    continue
                loaded.append(detector)
                all_detectors[_detector_name(detector)] = detector
            detector_map[intent] = loaded
        return detector_map, list(all_detectors.values())

    def _classify_attempts(self, attempts, detector_map, evaluator, evaluate=True):
        attempts = self._annotate_attempts(attempts)
        outcomes = {}
        detector_locks = {
            id(detector): threading.Lock()
            for detectors in detector_map.values()
            for detector in detectors
        }
        parallel_attempts = getattr(_config.system, "parallel_attempts", False)
        use_parallel = (
            isinstance(parallel_attempts, int)
            and parallel_attempts > 1
            and len(attempts) > 1
        )

        if use_parallel:
            max_workers = min(
                parallel_attempts,
                getattr(_config.system, "max_workers", parallel_attempts),
                len(attempts),
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        _detect_attempt,
                        attempt,
                        detector_map.get(attempt.intent, []),
                        evaluator,
                        detector_locks,
                    ): attempt
                    for attempt in attempts
                }
                for future in as_completed(future_map):
                    attempt = future_map[future]
                    try:
                        outcomes[id(attempt)] = future.result()
                    except _DETECTOR_ERRORS as error:
                        logging.error("Detector stage failed: %s", error)
                        outcomes[id(attempt)] = True
        else:
            for attempt in attempts:
                outcomes[id(attempt)] = _detect_attempt(
                    attempt,
                    detector_map.get(attempt.intent, []),
                    evaluator,
                    detector_locks,
                )

        if attempts and evaluate:
            evaluator.evaluate(attempts)
        if attempts:
            self._write_completed_attempts(attempts)
        return attempts, outcomes

    def _write_summaries(self, states, accepted_count: int, total_count: int):
        harness_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        for state in states.values():
            stub = state["stub"]
            _config.transient.reportfile.write(
                json.dumps(
                    {
                        "entry_type": "harness_stub_summary",
                        "harness": harness_name,
                        "intent": stub.intent,
                        "source_stub": _stub_text(stub),
                        "accepted": state["accepted"],
                        "successful_probe": state["successful_probe"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        rate = accepted_count / total_count if total_count else 0.0
        _config.transient.reportfile.write(
            json.dumps(
                {
                    "entry_type": "harness_summary",
                    "harness": harness_name,
                    "total_stubs": total_count,
                    "accepted_stubs": accepted_count,
                    "rejected_stubs": total_count - accepted_count,
                    "attack_success_rate": rate,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    def run(self, model, probe_names, detector_names, evaluator, buff_names=None):
        if buff_names is None:
            buff_names = []
        self._load_buffs(buff_names)
        self._start_run_hook()
        try:
            stubs = self._collect_stubs()
            if not stubs:
                logging.warning("No intent stubs found for EarlyStop")
                self._write_summaries({}, 0, 0)
                return

            detector_map, all_detectors = self._load_detector_map(detector_names, stubs)
            if not all_detectors:
                raise ValueError("No detectors loaded for EarlyStop")

            self._emit_plugin_cache(model, all_detectors)
            states = {
                _stub_key(stub): {
                    "stub": stub,
                    "accepted": False,
                    "successful_probe": None,
                    "successful_attempt": None,
                }
                for stub in stubs
            }

            baseline_probe = self._make_baseline_probe(stubs)
            baseline_attempts = list(baseline_probe.probe(model))
            baseline_attempts, baseline_outcomes = self._classify_attempts(
                baseline_attempts, detector_map, evaluator, evaluate=False
            )
            for attempt in baseline_attempts:
                stub = _source_stub(attempt)
                if stub is not None and not baseline_outcomes[id(attempt)]:
                    state = states[_stub_key(stub)]
                    state["accepted"] = True
                    state["successful_probe"] = "baseline"
                    state["successful_attempt"] = deepcopy(attempt)
                    _write_success_hitlog(attempt)

            for probe_name in probe_names:
                rejected_keys = {
                    key for key, state in states.items() if not state["accepted"]
                }
                if not rejected_keys:
                    break

                probe = self._load_probe(probe_name)
                if probe is None:
                    continue
                self._restrict_probe(probe, rejected_keys)
                if not probe.prompts:
                    continue

                attempts = list(probe.probe(model))
                attempts, outcomes = self._classify_attempts(
                    attempts, detector_map, evaluator
                )
                for attempt in attempts:
                    stub = _source_stub(attempt)
                    if stub is not None and not outcomes[id(attempt)]:
                        state = states.get(_stub_key(stub))
                        if state is not None and not state["accepted"]:
                            state["accepted"] = True
                            state["successful_probe"] = probe_name
                            state["successful_attempt"] = deepcopy(attempt)

            accepted_count = sum(1 for state in states.values() if state["accepted"])
            self._write_summaries(states, accepted_count, len(states))
        finally:
            self._end_run_hook()

    def _emit_plugin_cache(self, model, detectors):
        _emit_plugin_cache_entry(self, model, *detectors)


DEFAULT_CLASS = "EarlyStopHarness"
