from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from lightphone_gpt.adapters.chatgpt_plan import ChatGPTPlanAssistant
from lightphone_gpt.adapters.errors import (
    ChatGPTAccountNotEligibleError,
    ChatGPTAmbiguousOutcomeError,
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTPlanUsageLimitError,
    ChatGPTTransientError,
)
from lightphone_gpt.models import AssistantResponse, PromptMessage


class FakeTokens:
    def __init__(self, value: str = "oauth-access-token") -> None:
        self.value = value
        self.calls: list[bool] = []

    def get_access_token(self, *, force_refresh: bool = False) -> str:
        self.calls.append(force_refresh)
        return self.value


class FakeStream:
    def __init__(
        self,
        events: list[object],
        *,
        failure: Exception | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.events = events
        self.failure = failure
        self.fail_after = fail_after
        self.closed = False

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def __iter__(self):
        for index, event in enumerate(self.events):
            if self.failure is not None and self.fail_after == index:
                raise self.failure
            yield event
        if self.failure is not None and self.fail_after == len(self.events):
            raise self.failure

    def close(self) -> None:
        self.closed = True


class FakeResponses:
    def __init__(self, stream: FakeStream, failure: Exception | None = None) -> None:
        self.stream = stream
        self.failure = failure
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> FakeStream:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return self.stream


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


@dataclass
class FakeSdkError(Exception):
    status_code: int
    code: str | None = None
    body: object | None = None


def assistant_for(
    stream: FakeStream,
    *,
    response_failure: Exception | None = None,
    max_stream_chars: int = 24_000,
) -> tuple[ChatGPTPlanAssistant, FakeTokens, FakeResponses, list[tuple[str, float]]]:
    tokens = FakeTokens()
    responses = FakeResponses(stream, response_failure)
    factory_calls: list[tuple[str, float]] = []

    def factory(token: str, timeout: float) -> FakeClient:
        factory_calls.append((token, timeout))
        return FakeClient(responses)

    return (
        ChatGPTPlanAssistant(
            tokens=tokens,
            model="gpt-test",
            client_factory=factory,
            timeout_seconds=12.0,
            max_stream_chars=max_stream_chars,
        ),
        tokens,
        responses,
        factory_calls,
    )


def test_streamed_response_uses_preview_contract_and_completes() -> None:
    stream = FakeStream(
        [
            {"type": "response.output_text.delta", "delta": "Short "},
            {"type": "response.output_text.delta", "delta": "answer."},
            {"type": "response.completed", "response": {"id": "resp_123"}},
        ]
    )
    assistant, tokens, responses, factory_calls = assistant_for(stream)

    result = assistant.respond(
        [
            PromptMessage("system", "Answer plainly."),
            PromptMessage("user", "Earlier question"),
            PromptMessage("assistant", "Earlier answer"),
            PromptMessage("user", "Current question"),
        ]
    )

    assert result == AssistantResponse("Short answer.", "resp_123")
    assert tokens.calls == [False]
    assert factory_calls == [("oauth-access-token", 12.0)]
    request = responses.requests[0]
    assert request["model"] == "gpt-test"
    assert request["store"] is False
    assert request["stream"] is True
    assert request["input"] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Current question"},
    ]
    assert "Answer plainly." in str(request["instructions"])
    assert "400-600 characters" in str(request["instructions"])
    forbidden = {
        "max_output_tokens",
        "previous_response_id",
        "temperature",
        "tools",
        "metadata",
        "user",
        "conversation",
    }
    assert forbidden.isdisjoint(request)
    assert stream.closed


def test_partial_stream_is_ambiguous_and_never_returns_partial_text() -> None:
    private_partial = "partial private answer"
    stream = FakeStream(
        [{"type": "response.output_text.delta", "delta": private_partial}],
        failure=OSError("connection reset"),
        fail_after=1,
    )
    assistant, *_ = assistant_for(stream)

    with pytest.raises(ChatGPTAmbiguousOutcomeError) as captured:
        assistant.respond([PromptMessage("user", "question")])
    assert private_partial not in str(captured.value)


def test_post_event_http_failure_is_ambiguous() -> None:
    stream = FakeStream(
        [{"type": "response.output_text.delta", "delta": "partial"}],
        failure=FakeSdkError(503),
        fail_after=1,
    )
    assistant, *_ = assistant_for(stream)

    with pytest.raises(ChatGPTAmbiguousOutcomeError):
        assistant.respond([PromptMessage("user", "question")])


def test_eof_without_completed_is_ambiguous() -> None:
    assistant, *_ = assistant_for(
        FakeStream([{"type": "response.output_text.delta", "delta": "partial"}])
    )
    with pytest.raises(ChatGPTAmbiguousOutcomeError):
        assistant.respond([PromptMessage("user", "question")])


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("subscription_sharing_user_not_eligible", ChatGPTAccountNotEligibleError),
        ("subscription_sharing_usage_limit_exceeded", ChatGPTPlanUsageLimitError),
        ("subscription_sharing_invalid_user", ChatGPTAuthorizationRequiredError),
        ("subscription_sharing_unsupported_capability", ChatGPTConfigurationError),
        ("subscription_sharing_usage_unavailable", ChatGPTTransientError),
    ],
)
def test_failed_terminal_event_maps_exact_code(
    code: str, expected: type[Exception]
) -> None:
    assistant, *_ = assistant_for(
        FakeStream(
            [
                {
                    "type": "response.failed",
                    "response": {"error": {"code": code}},
                }
            ]
        )
    )
    with pytest.raises(expected):
        assistant.respond([PromptMessage("user", "question")])


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (FakeSdkError(401), ChatGPTAuthorizationRequiredError),
        (FakeSdkError(403), ChatGPTConfigurationError),
        (FakeSdkError(429), ChatGPTTransientError),
        (FakeSdkError(503), ChatGPTTransientError),
        (
            FakeSdkError(
                429,
                body={
                    "error": {
                        "code": "subscription_sharing_usage_limit_exceeded"
                    }
                },
            ),
            ChatGPTPlanUsageLimitError,
        ),
    ],
)
def test_pre_stream_transport_error_mapping(
    error: Exception, expected: type[Exception]
) -> None:
    assistant, *_ = assistant_for(FakeStream([]), response_failure=error)
    with pytest.raises(expected):
        assistant.respond([PromptMessage("user", "question")])


def test_local_stream_character_cap_closes_and_fails_without_output() -> None:
    stream = FakeStream(
        [
            {"type": "response.output_text.delta", "delta": "1234"},
            {"type": "response.output_text.delta", "delta": "5678"},
            {"type": "response.completed", "response": {"id": "too-large"}},
        ]
    )
    assistant, *_ = assistant_for(stream, max_stream_chars=7)

    with pytest.raises(ChatGPTConfigurationError, match="safety limit"):
        assistant.respond([PromptMessage("user", "question")])
    assert stream.closed


def test_completed_without_text_is_not_success() -> None:
    assistant, *_ = assistant_for(
        FakeStream([{"type": "response.completed", "response": {"id": "empty"}}])
    )
    with pytest.raises(ChatGPTAmbiguousOutcomeError, match="without returning text"):
        assistant.respond([PromptMessage("user", "question")])


def test_completed_refusal_is_returned_without_retry() -> None:
    stream = FakeStream(
        [
            {
                "type": "response.refusal.delta",
                "delta": "I cannot help with that request.",
            },
            {"type": "response.completed", "response": {"id": "refusal"}},
        ]
    )
    assistant, tokens, *_ = assistant_for(stream)

    assert assistant.respond([PromptMessage("user", "question")]) == (
        AssistantResponse("I cannot help with that request.", "refusal")
    )
    assert tokens.calls == [False]


def test_refusal_done_fallback_is_not_discarded() -> None:
    stream = FakeStream(
        [
            {
                "type": "response.refusal.done",
                "refusal": "I cannot help with that request.",
            },
            {"type": "response.completed", "response": {"id": "refusal"}},
        ]
    )
    assistant, *_ = assistant_for(stream)

    assert assistant.respond([PromptMessage("user", "question")]) == (
        AssistantResponse("I cannot help with that request.", "refusal")
    )
