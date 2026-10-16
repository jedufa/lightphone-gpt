"""ChatGPT-plan assistant using supported OAuth and streamed Responses."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any, Protocol

from lightphone_gpt.chatgpt_oauth import AccessTokenProvider, RESOURCE
from lightphone_gpt.contracts import PermanentDependencyError
from lightphone_gpt.models import AssistantResponse, PromptMessage

from lightphone_gpt.chatgpt_errors import (
    ChatGPTAccountNotEligibleError,
    ChatGPTAmbiguousOutcomeError,
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTPlanUsageLimitError,
    ChatGPTTransientError,
)


DEFAULT_LOCAL_STREAM_CHAR_LIMIT = 24_000
PHONE_BREVITY_INSTRUCTION = (
    "Keep the complete answer concise for a small e-ink phone; aim for roughly "
    "400-600 characters unless the user explicitly asks for more detail."
)


class ResponsesEndpoint(Protocol):
    def create(self, **request: object) -> AbstractContextManager[Any]: ...


class ResponsesClient(Protocol):
    responses: ResponsesEndpoint


ClientFactory = Callable[[str, float], ResponsesClient]


def _default_client_factory(
    access_token: str, timeout_seconds: float
) -> ResponsesClient:
    try:
        from openai import OpenAI
    except ImportError:
        raise ChatGPTConfigurationError(
            "Install the chatgpt optional dependency to enable ChatGPT"
        ) from None
    # The SDK parameter is named api_key, but this value is the OAuth bearer
    # token granted by Sign in with ChatGPT. No API key is accepted or stored.
    return OpenAI(
        api_key=access_token,
        base_url=RESOURCE,
        max_retries=0,
        timeout=timeout_seconds,
    )


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _nested(value: object, *names: str) -> object | None:
    current: object | None = value
    for name in names:
        if current is None:
            return None
        current = _field(current, name)
    return current


_ACCOUNT_INELIGIBLE_CODES = {"subscription_sharing_user_not_eligible"}
_PLAN_LIMIT_CODES = {"subscription_sharing_usage_limit_exceeded"}
_AUTHORIZATION_CODES = {
    "subscription_sharing_invalid_user",
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
}
_CONFIGURATION_CODES = {
    "subscription_sharing_unsupported_capability",
    "subscription_sharing_route_not_supported",
    "chatpass_v2_scope_not_authorized",
    "chatpass_v2_invalid_authorization_context",
    "invalid_client",
}
_TRANSIENT_CODES = {
    "subscription_sharing_usage_unavailable",
    "subscription_sharing_user_unavailable",
    "rate_limit_exceeded",
    "server_overloaded",
}


def _mapped_error(
    *, code: str | None, status_code: int | None, ambiguous: bool = False
) -> Exception:
    if code in _ACCOUNT_INELIGIBLE_CODES:
        return ChatGPTAccountNotEligibleError(
            "The selected ChatGPT account cannot use plan sharing"
        )
    if code in _PLAN_LIMIT_CODES:
        return ChatGPTPlanUsageLimitError(
            "ChatGPT plan usage is paused at its configured limit"
        )
    if code in _AUTHORIZATION_CODES or status_code == 401:
        return ChatGPTAuthorizationRequiredError(
            "ChatGPT authorization must be renewed"
        )
    if code in _CONFIGURATION_CODES:
        return ChatGPTConfigurationError(
            "The ChatGPT plan request is not supported"
        )
    if code in _TRANSIENT_CODES:
        return ChatGPTTransientError(
            "ChatGPT plan inference is temporarily unavailable"
        )
    if status_code == 403:
        return ChatGPTConfigurationError(
            "ChatGPT plan inference was denied by account or regional policy"
        )
    if status_code in {400, 404}:
        return ChatGPTConfigurationError(
            "The ChatGPT plan request was rejected"
        )
    if status_code == 429 or (status_code is not None and status_code >= 500):
        return ChatGPTTransientError(
            "ChatGPT plan inference is temporarily unavailable"
        )
    if ambiguous:
        return ChatGPTAmbiguousOutcomeError(
            "ChatGPT response stream ended without a complete answer"
        )
    return ChatGPTTransientError("ChatGPT plan inference failed")


def _exception_details(error: BaseException) -> tuple[str | None, int | None]:
    status = getattr(error, "status_code", None)
    status_code = status if isinstance(status, int) else None
    code = getattr(error, "code", None)
    if not isinstance(code, str):
        code = None
    body = getattr(error, "body", None)
    if isinstance(body, Mapping):
        nested_error = body.get("error")
        if isinstance(nested_error, Mapping) and isinstance(
            nested_error.get("code"), str
        ):
            code = nested_error["code"]
        elif isinstance(body.get("code"), str):
            code = body["code"]
    return code, status_code


class ChatGPTPlanAssistant:
    """Complete one text-only turn using the user's authorized ChatGPT plan.

    Conversation history remains local and is supplied in full. The adapter
    deliberately does not expose tools and treats only ``response.completed``
    as success. Partial streamed text is never returned to the durable worker.
    """

    def __init__(
        self,
        *,
        tokens: AccessTokenProvider,
        model: str,
        client_factory: ClientFactory = _default_client_factory,
        timeout_seconds: float = 60.0,
        max_stream_chars: int = DEFAULT_LOCAL_STREAM_CHAR_LIMIT,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_stream_chars < 1:
            raise ValueError("max_stream_chars must be positive")
        self._tokens = tokens
        self._model = model.strip()
        self._client_factory = client_factory
        self._timeout_seconds = timeout_seconds
        self._max_stream_chars = max_stream_chars

    def respond(self, messages: Sequence[PromptMessage]) -> AssistantResponse:
        request = self._request(messages)
        access_token = self._tokens.get_access_token()
        saw_stream_event = False
        completed = False
        response_id: str | None = None
        parts: list[str] = []
        character_count = 0
        saw_refusal_delta = False

        try:
            client = self._client_factory(access_token, self._timeout_seconds)
            with client.responses.create(**request) as stream:
                for event in stream:
                    saw_stream_event = True
                    event_type = _field(event, "type")
                    if event_type in {
                        "response.output_text.delta",
                        "response.refusal.delta",
                    }:
                        delta = _field(event, "delta")
                        if isinstance(delta, str) and delta:
                            if event_type == "response.refusal.delta":
                                saw_refusal_delta = True
                            character_count += len(delta)
                            if character_count > self._max_stream_chars:
                                close = getattr(stream, "close", None)
                                if callable(close):
                                    close()
                                raise ChatGPTConfigurationError(
                                    "ChatGPT response exceeded the local safety limit"
                                )
                            parts.append(delta)
                    elif (
                        event_type == "response.refusal.done"
                        and not saw_refusal_delta
                    ):
                        refusal = _field(event, "refusal")
                        if isinstance(refusal, str) and refusal:
                            character_count += len(refusal)
                            if character_count > self._max_stream_chars:
                                close = getattr(stream, "close", None)
                                if callable(close):
                                    close()
                                raise ChatGPTConfigurationError(
                                    "ChatGPT response exceeded the local safety limit"
                                )
                            parts.append(refusal)
                    elif event_type == "response.completed":
                        completed = True
                        raw_id = _nested(event, "response", "id")
                        if isinstance(raw_id, str) and raw_id:
                            response_id = raw_id
                        break
                    elif event_type == "response.failed":
                        code = _nested(event, "response", "error", "code")
                        if code is None:
                            code = _nested(event, "error", "code")
                        raise _mapped_error(
                            code=code if isinstance(code, str) else None,
                            status_code=None,
                            ambiguous=bool(parts),
                        )
                    elif event_type in {"response.incomplete", "error"}:
                        code = _nested(event, "response", "error", "code")
                        if code is None:
                            code = _nested(event, "error", "code")
                        if code is None:
                            code = _field(event, "code")
                        if isinstance(code, str) and (
                            code in _ACCOUNT_INELIGIBLE_CODES
                            or code in _PLAN_LIMIT_CODES
                            or code in _AUTHORIZATION_CODES
                            or code in _CONFIGURATION_CODES
                            or code in _TRANSIENT_CODES
                        ):
                            raise _mapped_error(
                                code=code,
                                status_code=None,
                                ambiguous=bool(parts),
                            )
                        raise ChatGPTAmbiguousOutcomeError(
                            "ChatGPT response stream ended without a complete answer"
                        )
        except (
            ChatGPTAccountNotEligibleError,
            ChatGPTAmbiguousOutcomeError,
            ChatGPTAuthorizationRequiredError,
            ChatGPTConfigurationError,
            ChatGPTPlanUsageLimitError,
            ChatGPTTransientError,
        ):
            raise
        except Exception as error:
            if saw_stream_event or parts:
                raise ChatGPTAmbiguousOutcomeError(
                    "ChatGPT response stream ended without a complete answer"
                ) from None
            code, status_code = _exception_details(error)
            raise _mapped_error(
                code=code,
                status_code=status_code,
            ) from None

        if not completed:
            raise ChatGPTAmbiguousOutcomeError(
                "ChatGPT response stream ended without a complete answer"
            )
        text = "".join(parts).strip()
        if not text:
            # The remote response is known to have completed. Repeating it as
            # an ordinary transient could compute a second answer, so route it
            # through the explicit ambiguity gate instead.
            raise ChatGPTAmbiguousOutcomeError(
                "ChatGPT completed without returning text"
            )
        return AssistantResponse(text=text, response_id=response_id)

    def _request(self, messages: Sequence[PromptMessage]) -> dict[str, object]:
        if not messages:
            raise PermanentDependencyError("assistant prompt must not be empty")
        instructions: list[str] = []
        inputs: list[dict[str, str]] = []
        for message in messages:
            content = message.content.strip()
            if not content:
                raise PermanentDependencyError(
                    "assistant prompt messages must not be blank"
                )
            if message.role == "system":
                instructions.append(content)
            else:
                inputs.append({"role": message.role, "content": content})
        if not inputs:
            raise PermanentDependencyError(
                "assistant prompt must contain a user or assistant message"
            )
        instructions.append(PHONE_BREVITY_INSTRUCTION)
        return {
            "model": self._model,
            "input": inputs,
            "instructions": "\n\n".join(instructions),
            "store": False,
            "stream": True,
        }
