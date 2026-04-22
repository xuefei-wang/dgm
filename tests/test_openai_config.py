import os
import subprocess
import sys
from types import SimpleNamespace

from llm import MAX_OUTPUT_TOKENS, openai_reasoning_config
from llm_withtools import get_response_withtools


class _FakeResponses:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(usage=None)


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


def test_default_openai_model_uses_gpt54_mini():
    env = os.environ.copy()
    env.pop("DGM_OPENAI_MODEL", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from llm_withtools import OPENAI_MODEL; print(OPENAI_MODEL)",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.stdout.strip() == "gpt-5.4-mini"


def test_openai_model_env_override():
    env = os.environ.copy()
    env["DGM_OPENAI_MODEL"] = "o3-mini-2025-01-31"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from llm_withtools import OPENAI_MODEL; print(OPENAI_MODEL)",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.stdout.strip() == "o3-mini-2025-01-31"


def test_reasoning_config_only_applies_to_reasoning_models(monkeypatch):
    monkeypatch.delenv("DGM_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("REASONING_EFFORT", raising=False)

    assert openai_reasoning_config("gpt-5.4-mini") == {"effort": "medium"}
    assert openai_reasoning_config("o3-mini-2025-01-31") == {"effort": "medium"}
    assert openai_reasoning_config("gpt-4.1-2025-04-14") is None


def test_reasoning_effort_env_precedence(monkeypatch):
    monkeypatch.setenv("REASONING_EFFORT", "low")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
    monkeypatch.setenv("DGM_REASONING_EFFORT", "medium")

    assert openai_reasoning_config("gpt-5.4-mini") == {"effort": "medium"}

    monkeypatch.delenv("DGM_REASONING_EFFORT")
    assert openai_reasoning_config("gpt-5.4-mini") == {"effort": "high"}

    monkeypatch.delenv("OPENAI_REASONING_EFFORT")
    assert openai_reasoning_config("gpt-5.4-mini") == {"effort": "low"}


def test_responses_tool_call_kwargs_include_budget_and_reasoning(monkeypatch):
    monkeypatch.delenv("DGM_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("REASONING_EFFORT", raising=False)
    client = _FakeClient()

    get_response_withtools(
        client=client,
        model="gpt-5.4-mini",
        messages=[],
        tools=[],
        tool_choice="auto",
        logging=lambda _message: None,
    )

    assert client.responses.kwargs["model"] == "gpt-5.4-mini"
    assert client.responses.kwargs["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert client.responses.kwargs["reasoning"] == {"effort": "medium"}
