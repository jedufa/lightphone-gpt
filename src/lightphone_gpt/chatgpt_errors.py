"""Typed, content-free failures for ChatGPT authorization and inference."""

from .contracts import PermanentDependencyError, TransientDependencyError


class ChatGPTTransientError(TransientDependencyError):
    """ChatGPT plan inference or token renewal can be retried later."""


class ChatGPTAmbiguousOutcomeError(ChatGPTTransientError):
    """A streamed request may have run, but no complete answer was received."""


class ChatGPTCredentialAmbiguousError(PermanentDependencyError):
    """Refresh-token rotation may have occurred and requires manual review."""


class ChatGPTAuthorizationRequiredError(PermanentDependencyError):
    """The saved ChatGPT authorization needs explicit user renewal."""


class ChatGPTConfigurationError(PermanentDependencyError):
    """The ChatGPT integration or request contract is invalid."""


class ChatGPTAccountNotEligibleError(PermanentDependencyError):
    """The selected ChatGPT account or workspace cannot use plan sharing."""


class ChatGPTPlanUsageLimitError(PermanentDependencyError):
    """ChatGPT plan inference is paused until the user changes its limit."""
