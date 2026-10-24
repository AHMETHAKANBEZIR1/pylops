"""Record one existing test's RNG inputs without altering its result or tolerances."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

TARGET = "test_radon.py::test_Radon3D[float32-par1]"


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
