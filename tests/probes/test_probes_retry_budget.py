"""A generator giving up (retry budget) aborts the run promptly.

Two things could defeat it: a probe that catches the GarakException, and a
multiprocessing pool that waits for every queued attempt before re-raising.
"""

import time

import pytest

import garak._plugins
from garak import _config
from garak.attempt import Message
from garak.exception import GarakException
from garak.generators.base import Generator

SLOW_SECONDS = 2


class GivesUpOnFirstPrompt(Generator):
    """Fails like retry_budget on the prompt "fail"; every other prompt is slow."""

    generator_family_name = "RetryBudgetTest"
    parallel_capable = True
    supports_multiple_generations = False

    def __init__(self, name="retry-budget-test", config_root=_config):
        super().__init__(name, config_root=config_root)

    def _call_model(self, prompt, generations_this_call=1):
        if prompt.last_message().text == "fail":
            raise GarakException(
                "Giving up on retry-budget-test: max_retry_seconds reached"
            )
        time.sleep(SLOW_SECONDS)
        return [Message("ok")]


def test_parallel_attempts_stop_when_a_generator_gives_up():
    """close() + join() would wait for every queued attempt (6 x 2 s over 2 workers)."""
    _config.load_base_config()
    probe = garak._plugins.load_plugin("probes.test.Test", config_root=_config)
    probe.prompts = ["fail"] + [f"slow {i}" for i in range(6)]
    probe.parallel_attempts = 2
    probe.max_workers = 2
    generator = GivesUpOnFirstPrompt()

    started = time.monotonic()
    with pytest.raises(GarakException, match="max_retry_seconds reached"):
        probe.probe(generator)
    elapsed = time.monotonic() - started
    assert elapsed < SLOW_SECONDS * 2, f"took {elapsed:.1f}s to abort"


@pytest.mark.parametrize("probe_name", ["probes.tap.TAP", "probes.tap.PAIR"])
def test_tap_and_pair_let_a_run_abort(probe_name):
    _config.load_base_config()
    probe = garak._plugins.load_plugin(probe_name, config_root=_config)

    def gives_up(*args, **kwargs):
        raise GarakException("Giving up on judge: max_retry_seconds reached")

    probe.run_tap = gives_up
    with pytest.raises(GarakException, match="max_retry_seconds reached"):
        probe.probe(GivesUpOnFirstPrompt())


@pytest.mark.parametrize("probe_name", ["probes.tap.TAP", "probes.tap.PAIR"])
def test_tap_and_pair_still_tolerate_other_errors(probe_name):
    """Other failures keep the existing behaviour: logged, and the probe yields no attempts."""
    _config.load_base_config()
    probe = garak._plugins.load_plugin(probe_name, config_root=_config)

    def fails(*args, **kwargs):
        raise ValueError("attacker returned unparseable JSON")

    probe.run_tap = fails
    assert not probe.probe(GivesUpOnFirstPrompt())
