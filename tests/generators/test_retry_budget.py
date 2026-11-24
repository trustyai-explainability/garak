"""The retry budget stops a generator from retrying a failing request forever.

Generators retry transient failures (rate limits, 5xx, timeouts) with
``@backoff.on_exception``, which has no limit of its own. ``retry_budget`` gives
up after the generator's ``max_retry_seconds`` and aborts the run.
"""

import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from garak import _config
from garak.attempt import Conversation, Message, Turn
from garak.exception import GarakException, GeneratorBackoffTrigger
from garak.generators.base import Generator, retry_budget
from garak.generators.rest import RestGenerator

URI = "http://model.test/v1/chat"


class _HTTPError(Exception):
    def __init__(self, status_code, text):
        super().__init__(text)
        self.status_code = status_code


def _details(generator, elapsed, wait=1.0, tries=3, exception=None):
    return {
        "target": None,
        "args": (generator,),
        "kwargs": {},
        "tries": tries,
        "elapsed": elapsed,
        "wait": wait,
        "exception": exception,
    }


def _generator(budget, uri="https://user:secret@model.test:8443/v1?key=abc"):
    return SimpleNamespace(
        max_retry_seconds=budget, fullname="OpenAICompatible:judge-llm", uri=uri
    )


def test_default_budget_is_inherited_by_generators():
    assert Generator.DEFAULT_PARAMS["max_retry_seconds"] == 1800
    assert RestGenerator.DEFAULT_PARAMS["max_retry_seconds"] == 1800


def test_under_budget_keeps_retrying():
    retry_budget(_details(_generator(60), elapsed=30, wait=20))


def test_budget_reached_by_the_next_wait_gives_up():
    cause = _HTTPError(503, "Service Unavailable")
    try:
        raise GeneratorBackoffTrigger("transient") from cause
    except GeneratorBackoffTrigger as e:
        trigger = e
    with pytest.raises(GarakException) as excinfo:
        retry_budget(
            _details(_generator(60), elapsed=45, wait=20, tries=9, exception=trigger)
        )
    msg = str(excinfo.value)
    assert "OpenAICompatible:judge-llm" in msg
    assert "kept failing for 45s (9 tries" in msg
    assert "HTTP 503 (_HTTPError: Service Unavailable)" in msg
    assert "retrying again would exceed max_retry_seconds=60" in msg
    # the endpoint is named without credentials or query string
    assert "at https://model.test:8443/v1:" in msg
    assert "secret" not in msg and "key=abc" not in msg
    # raised "from None" so it pickles across multiprocessing.Pool boundaries
    assert excinfo.value.__suppress_context__


@pytest.mark.parametrize("budget", [0, None, True, "1800", -5])
def test_budget_disabled_or_invalid_never_gives_up(budget):
    retry_budget(_details(_generator(budget), elapsed=10**6))


def test_objects_without_a_budget_are_ignored():
    retry_budget(_details(object(), elapsed=10**6))
    retry_budget({"args": (), "elapsed": 10**6, "wait": 1})


def test_every_generator_retry_has_the_budget():
    """A new @backoff.on_exception in a generator must come with on_backoff=retry_budget."""
    missing = []
    for path in sorted(Path(_config.transient.package_dir / "generators").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"@backoff\.on_exception\(", source):
            depth, pos = 1, match.end()
            while depth:
                depth += {"(": 1, ")": -1}.get(source[pos], 0)
                pos += 1
            if "on_backoff=retry_budget" not in source[match.end() : pos]:
                line = source.count("\n", 0, match.start()) + 1
                missing.append(f"{path.name}:{line}")
    assert not missing, f"backoff without retry_budget: {missing}"


# --------------------------------------------------------------------------- REST end to end


@pytest.fixture
def rest_generator():
    def _make(budget):
        _config.plugins.generators["rest"] = {
            "RestGenerator": {
                "name": "rest-test",
                "uri": URI,
                "max_retry_seconds": budget,
            }
        }
        return RestGenerator()

    return _make


def _conv():
    return Conversation([Turn("user", Message("hello"))])


def test_rest_gives_up_on_a_target_that_stays_down(rest_generator, requests_mock):
    requests_mock.post(URI, status_code=503, reason="Service Unavailable")
    generator = rest_generator(1)
    started = time.monotonic()
    with pytest.raises(GarakException, match="would exceed max_retry_seconds=1"):
        generator._call_model(_conv())
    assert time.monotonic() - started < 10
    assert requests_mock.call_count >= 2, "the request should have been retried first"


def test_rest_recovers_within_the_budget(rest_generator, requests_mock):
    requests_mock.post(
        URI,
        [
            {"status_code": 503},
            {"status_code": 503},
            {"status_code": 200, "text": "ok"},
        ],
    )
    assert rest_generator(60)._call_model(_conv()) == [Message("ok")]
    assert requests_mock.call_count == 3


def test_rest_budget_can_be_disabled(rest_generator, requests_mock):
    requests_mock.post(
        URI,
        [{"status_code": 503}] * 3 + [{"status_code": 200, "text": "ok"}],
    )
    generator = rest_generator(0)
    assert generator.max_retry_seconds == 0
    assert generator._call_model(_conv()) == [Message("ok")]
