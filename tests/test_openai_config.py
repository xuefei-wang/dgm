from types import SimpleNamespace

import llm_withtools
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
    assert openai_reasoning_config("o3") == {"effort": "medium"}
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


def test_openai_tool_followup_preserves_full_response_output(monkeypatch):
    reasoning_item = SimpleNamespace(type="reasoning", id="rs_123")
    tool_call = SimpleNamespace(
        type="function_call",
        call_id="call_123",
        name="dummy_tool",
        arguments="{}",
    )
    trailing_item = SimpleNamespace(type="message", content=[])
    first_response = SimpleNamespace(
        output=[reasoning_item, tool_call, trailing_item],
        output_text="",
        usage=None,
    )
    final_response = SimpleNamespace(output=[], output_text="done", usage=None)

    class _RecordingResponses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return first_response if len(self.calls) == 1 else final_response

    class _RecordingClient:
        def __init__(self):
            self.responses = _RecordingResponses()

    client = _RecordingClient()
    monkeypatch.setattr(llm_withtools, "create_client", lambda _model: (client, "gpt-5.4-mini"))
    monkeypatch.setattr(
        llm_withtools,
        "load_all_tools",
        lambda logging=None: [
            {
                "info": {
                    "name": "dummy_tool",
                    "description": "Dummy tool",
                    "input_schema": {"type": "object", "properties": {}, "required": []},
                },
                "function": lambda: "tool output",
            }
        ],
    )

    llm_withtools.chat_with_agent_openai(
        "use a tool",
        model="gpt-5.4-mini",
        logging=lambda _message: None,
    )

    followup_input = client.responses.calls[1]["input"]
    assert reasoning_item in followup_input
    assert tool_call in followup_input
    assert trailing_item in followup_input
    assert followup_input[-1] == {
        "type": "function_call_output",
        "call_id": "call_123",
        "output": "tool output",
    }
