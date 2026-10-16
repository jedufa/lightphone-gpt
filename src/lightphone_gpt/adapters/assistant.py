"""Assistant boundary implementations used by the bridge."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable, Sequence
from threading import Lock

from lightphone_gpt.models import AssistantResponse, PromptMessage


class MockAssistant:
    """Deterministic in-memory assistant used by simulation and unit tests."""

    def __init__(
        self,
        responses: Iterable[str | AssistantResponse] = (),
        *,
        default_text: str = "Simulated assistant response.",
    ) -> None:
        if not default_text.strip():
            raise ValueError("default_text must not be blank")
        self._responses = deque(
            response
            if isinstance(response, AssistantResponse)
            else AssistantResponse(str(response))
            for response in responses
        )
        self._default = AssistantResponse(default_text)
        self._request_digests: list[tuple[tuple[str, str], ...]] = []
        self._lock = Lock()

    def respond(self, messages: Sequence[PromptMessage]) -> AssistantResponse:
        digest = tuple(
            (
                message.role,
                hashlib.sha256(message.content.encode("utf-8")).hexdigest(),
            )
            for message in messages
        )
        with self._lock:
            self._request_digests.append(digest)
            return self._responses.popleft() if self._responses else self._default

    @property
    def request_digests(self) -> tuple[tuple[tuple[str, str], ...], ...]:
        with self._lock:
            return tuple(self._request_digests)
