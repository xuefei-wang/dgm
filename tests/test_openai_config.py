from types import SimpleNamespace

from llm import openai_reasoning_config
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


def test_reasoning_config_only_applies_to_reasoning_models(monkeypatch):
    monkeypatch.setenv("DGM_REASONING_EFFORT", "medium")

    assert openai_reasoning_config("gpt-5.4-mini") == {"effort": "medium"}
    assert openai_reasoning_config("o3-mini-2025-01-31") == {"effort": "medium"}
    assert openai_reasoning_config("gpt-4.1-2025-04-14") is None


def test_responses_tool_call_kwargs_include_budget_and_reasoning(monkeypatch):
    monkeypatch.setenv("DGM_REASONING_EFFORT", "medium")
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
    assert client.responses.kwargs["reasoning"] == {"effort": "medium"}
