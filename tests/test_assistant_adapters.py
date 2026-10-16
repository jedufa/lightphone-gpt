from __future__ import annotations

from lightphone_gpt.adapters.assistant import MockAssistant
from lightphone_gpt.adapters.errors import AssistantAdapterError
from lightphone_gpt.contracts import TransientDependencyError
from lightphone_gpt.models import AssistantResponse, PromptMessage


def test_mock_assistant_is_deterministic() -> None:
    adapter = MockAssistant(
        [AssistantResponse("one", "mock-1"), "two"],
        default_text="fallback",
    )
    prompt = [PromptMessage("user", "hello")]

    assert adapter.respond(prompt) == AssistantResponse("one", "mock-1")
    assert adapter.respond(prompt) == AssistantResponse("two")
    assert adapter.respond(prompt) == AssistantResponse("fallback")
    assert len(adapter.request_digests) == 3
    assert "hello" not in repr(adapter.request_digests)


def test_assistant_operational_errors_are_retryable() -> None:
    assert isinstance(AssistantAdapterError("temporary"), TransientDependencyError)
