"""Specific, sanitized errors raised by boundary adapters."""

from lightphone_gpt.contracts import (
    PermanentDependencyError,
    TransientDependencyError,
)
from lightphone_gpt.chatgpt_errors import (
    ChatGPTAccountNotEligibleError,
    ChatGPTAmbiguousOutcomeError,
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTPlanUsageLimitError,
    ChatGPTTransientError,
)


class AssistantAdapterError(TransientDependencyError):
    """The assistant request failed or returned an unusable response."""


class TranscriptionAdapterError(TransientDependencyError):
    """The local transcription process failed."""


class LightAdapterError(TransientDependencyError):
    """The Light provider failed without exposing credentials or note content."""


class LightWriteDisabledError(PermanentDependencyError):
    """A Light mutation was rejected by the configured write policy."""


__all__ = [
    "AssistantAdapterError",
    "ChatGPTAccountNotEligibleError",
    "ChatGPTAmbiguousOutcomeError",
    "ChatGPTAuthorizationRequiredError",
    "ChatGPTConfigurationError",
    "ChatGPTPlanUsageLimitError",
    "ChatGPTTransientError",
    "LightAdapterError",
    "LightWriteDisabledError",
    "TranscriptionAdapterError",
]
