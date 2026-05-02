"""Tests for env-readable timeout and temperature in DGM pro_harness and llm."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# pro_harness: CROSS_RUNNER_AGENT_TIMEOUT_SEC / DGM_SWEBENCH_AGENT_TIMEOUT_SEC
# ---------------------------------------------------------------------------

def test_default_agent_timeout_is_32400(monkeypatch):
    """Without any override env var, legacy default of 32400 is used."""
    monkeypatch.delenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("DGM_SWEBENCH_AGENT_TIMEOUT_SEC", raising=False)

    import importlib
    import sys
    mod_name = "swe_bench.pro_harness"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    module = importlib.import_module(mod_name)
    assert module.DEFAULT_AGENT_TIMEOUT_SEC == 32400


def test_cross_runner_env_overrides_default(monkeypatch):
    """CROSS_RUNNER_AGENT_TIMEOUT_SEC takes priority over legacy default."""
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "3600")
    monkeypatch.delenv("DGM_SWEBENCH_AGENT_TIMEOUT_SEC", raising=False)

    import importlib
    import sys
    mod_name = "swe_bench.pro_harness"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    module = importlib.import_module(mod_name)
    assert module.DEFAULT_AGENT_TIMEOUT_SEC == 3600


def test_dgm_specific_env_overrides_default(monkeypatch):
    """DGM_SWEBENCH_AGENT_TIMEOUT_SEC overrides when cross-runner is absent."""
    monkeypatch.delenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("DGM_SWEBENCH_AGENT_TIMEOUT_SEC", "1800")

    import importlib
    import sys
    mod_name = "swe_bench.pro_harness"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    module = importlib.import_module(mod_name)
    assert module.DEFAULT_AGENT_TIMEOUT_SEC == 1800


def test_cross_runner_wins_over_dgm_specific(monkeypatch):
    """CROSS_RUNNER_AGENT_TIMEOUT_SEC takes priority over DGM_SWEBENCH_AGENT_TIMEOUT_SEC."""
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "3600")
    monkeypatch.setenv("DGM_SWEBENCH_AGENT_TIMEOUT_SEC", "1800")

    import importlib
    import sys
    mod_name = "swe_bench.pro_harness"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    module = importlib.import_module(mod_name)
    assert module.DEFAULT_AGENT_TIMEOUT_SEC == 3600


def test_cross_runner_zero_falls_through_to_dgm_specific(monkeypatch):
    """CROSS_RUNNER_AGENT_TIMEOUT_SEC=0 is treated as 'unset' and falls through."""
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "0")
    monkeypatch.setenv("DGM_SWEBENCH_AGENT_TIMEOUT_SEC", "1800")

    import importlib
    import sys
    mod_name = "swe_bench.pro_harness"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    module = importlib.import_module(mod_name)
    assert module.DEFAULT_AGENT_TIMEOUT_SEC == 1800


# ---------------------------------------------------------------------------
# llm.py: _resolve_dgm_temperature
# ---------------------------------------------------------------------------

def test_temperature_default_is_0_7(monkeypatch):
    """Without DGM_TEMPERATURE set, legacy default 0.7 is returned."""
    monkeypatch.delenv("DGM_TEMPERATURE", raising=False)
    from llm import _resolve_dgm_temperature
    assert _resolve_dgm_temperature() == pytest.approx(0.7)


def test_temperature_reads_env_zero(monkeypatch):
    """DGM_TEMPERATURE=0.0 returns 0.0."""
    monkeypatch.setenv("DGM_TEMPERATURE", "0.0")
    from llm import _resolve_dgm_temperature
    assert _resolve_dgm_temperature() == pytest.approx(0.0)


def test_temperature_reads_env_nonzero(monkeypatch):
    """DGM_TEMPERATURE=0.5 returns 0.5."""
    monkeypatch.setenv("DGM_TEMPERATURE", "0.5")
    from llm import _resolve_dgm_temperature
    assert _resolve_dgm_temperature() == pytest.approx(0.5)


def test_temperature_invalid_env_falls_back(monkeypatch):
    """Non-float DGM_TEMPERATURE falls back to 0.7."""
    monkeypatch.setenv("DGM_TEMPERATURE", "not_a_float")
    from llm import _resolve_dgm_temperature
    assert _resolve_dgm_temperature() == pytest.approx(0.7)


def test_temperature_out_of_range_falls_back(monkeypatch):
    """DGM_TEMPERATURE outside [0.0, 2.0] falls back to 0.7."""
    monkeypatch.setenv("DGM_TEMPERATURE", "3.0")
    from llm import _resolve_dgm_temperature
    assert _resolve_dgm_temperature() == pytest.approx(0.7)

    monkeypatch.setenv("DGM_TEMPERATURE", "-0.1")
    assert _resolve_dgm_temperature() == pytest.approx(0.7)


def test_temperature_boundary_values(monkeypatch):
    """0.0 and 2.0 are valid boundary values."""
    from llm import _resolve_dgm_temperature

    monkeypatch.setenv("DGM_TEMPERATURE", "0.0")
    assert _resolve_dgm_temperature() == pytest.approx(0.0)

    monkeypatch.setenv("DGM_TEMPERATURE", "2.0")
    assert _resolve_dgm_temperature() == pytest.approx(2.0)
