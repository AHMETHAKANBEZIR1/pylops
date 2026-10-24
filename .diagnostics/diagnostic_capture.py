"""Record one existing test's RNG inputs without altering its result or tolerances."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

TARGET = "test_radon.py::test_Radon3D[float32-par1]"
_trace_complete = False


def _output():
    path = Path(os.environ["DIAG_OUTPUT_DIR"])
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if item.nodeid.endswith(TARGET):
        state = np.random.get_state()
        np.savez(
            _output() / "rng-before-radon.npz",
            algorithm=state[0],
            keys=state[1],
            position=state[2],
            has_gauss=state[3],
            cached_gaussian=state[4],
        )
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if item.nodeid.endswith(TARGET) and report.when == "call":
        data = {
            "nodeid": item.nodeid,
            "outcome": report.outcome,
            "duration": report.duration,
            "failure": str(report.longrepr) if report.failed else None,
        }
        (_output() / "radon-result.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )


def pytest_sessionfinish(session, exitstatus):
    data = {
        "exitstatus": int(exitstatus),
        "collected": session.testscollected,
        "failed": session.testsfailed,
    }
    (_output() / "session-result.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _trace_state(nodeid, phase):
    state = np.random.get_state()
    data = {
        "nodeid": nodeid,
        "phase": phase,
        "keys_sha256": hashlib.sha256(state[1].tobytes()).hexdigest(),
        "position": state[2],
        "has_gauss": state[3],
        "cached_gaussian": state[4],
    }
    with (_output() / "rng-trace.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data) + "\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _trace_complete
    record = not _trace_complete
    if record:
        _trace_state(item.nodeid, "before_protocol")
    yield
    if record:
        _trace_state(item.nodeid, "after_protocol")
    if item.nodeid.endswith(TARGET):
        _trace_complete = True
