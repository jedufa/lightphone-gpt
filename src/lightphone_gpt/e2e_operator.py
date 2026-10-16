"""Isolated, fail-closed Light-to-ChatGPT-to-Light proof operation.

The operator deliberately has no polling loop and no durable conversation
state.  A caller binds one exact Light note identity, kind, and content digest;
the operator reads only that note, routes one ``Assistant`` question, asks the
provided assistant once, and creates one new bridge-owned text note.  It never
updates or deletes a note.

Source text, transcripts, answers, response identifiers, Light note
identifiers, titles, and content digests are intentionally absent from the
result and error payloads.  The only place the question and answer are written
is the newly created Light output note.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import hmac
import re
from typing import Protocol
import uuid

from .contracts import Assistant, LightProvider, Transcriber
from .conversation import DEFAULT_SYSTEM_PROMPT
from .chatgpt_errors import (
    ChatGPTAccountNotEligibleError,
    ChatGPTAmbiguousOutcomeError,
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTCredentialAmbiguousError,
    ChatGPTPlanUsageLimitError,
    ChatGPTTransientError,
)
from .models import (
    AssistantResponse,
    MutationOutcome,
    MutationReceipt,
    NoteKind,
    PromptMessage,
    RemoteNote,
    RouteKind,
)
from .routing import route_spoken_transcript, route_transcript


DEFAULT_OUTPUT_TITLE_PREFIX = "LIGHTPHONE GPT TEST E2E"
_NOTE_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,256}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_OPERATION_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class _RichLightProvider(LightProvider, Protocol):
    """Atomic metadata-and-payload helpers exposed by the real adapter."""

    def read_text_with_metadata(
        self, note_id: str
    ) -> tuple[RemoteNote, bytes]: ...

    def download_audio_with_metadata(
        self, note_id: str
    ) -> tuple[RemoteNote, bytes]: ...


@dataclass(frozen=True, slots=True)
class E2EInputBinding:
    """Private exact-source binding supplied by the one-shot caller."""

    note_id: str
    expected_kind: NoteKind
    expected_sha256: str

    def __post_init__(self) -> None:
        if _NOTE_ID_RE.fullmatch(self.note_id) is None:
            raise ValueError("note_id is invalid")
        if not isinstance(self.expected_kind, NoteKind):
            raise ValueError("expected_kind must be a NoteKind")
        if _SHA256_RE.fullmatch(self.expected_sha256) is None:
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class E2EProofLimits:
    """Local bounds applied before each subsequent dependency call."""

    max_audio_bytes: int = 32 * 1024 * 1024
    max_text_bytes: int = 64 * 1024
    max_transcript_chars: int = 16_000
    max_transcript_bytes: int = 64 * 1024
    max_answer_chars: int = 4_000
    max_answer_bytes: int = 32 * 1024
    max_output_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        values = (
            self.max_audio_bytes,
            self.max_text_bytes,
            self.max_transcript_chars,
            self.max_transcript_bytes,
            self.max_answer_chars,
            self.max_answer_bytes,
            self.max_output_bytes,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("all proof limits must be positive integers")


@dataclass(frozen=True, slots=True)
class E2EProofResult:
    """Content-free success receipt safe for ordinary operator output."""

    operation_id: str
    source_kind: NoteKind
    source_bytes: int
    transcription_used: bool
    transcript_characters: int
    request_characters: int
    answer_characters: int
    output_bytes: int
    response_id_present: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": True,
            "status": "verified",
            "operation_id": self.operation_id,
            "source_kind": self.source_kind.value,
            "source_bytes": self.source_bytes,
            "transcription_used": self.transcription_used,
            "transcript_characters": self.transcript_characters,
            "request_characters": self.request_characters,
            "answer_characters": self.answer_characters,
            "output_bytes": self.output_bytes,
            "response_id_present": self.response_id_present,
            "route_kind": RouteKind.QUESTION.value,
            "create_calls": 1,
            "read_back_verified": True,
        }


class E2EProofError(RuntimeError):
    """Stable, content-free error for the isolated proof command."""

    def __init__(
        self,
        category: str,
        *,
        operation_id: str,
        mutation_attempted: bool = False,
    ) -> None:
        super().__init__(category)
        self.category = category
        self.operation_id = operation_id
        self.mutation_attempted = mutation_attempted

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": False,
            "status": "failed_closed",
            "operation_id": self.operation_id,
            "category": self.category,
            "mutation_attempted": self.mutation_attempted,
        }


def _default_operation_id() -> str:
    return uuid.uuid4().hex


def _valid_title_prefix(value: str) -> bool:
    return (
        bool(value)
        and value == value.strip()
        and len(value) <= 80
        and _CONTROL_RE.search(value) is None
    )


def _source_payload(
    provider: LightProvider,
    binding: E2EInputBinding,
    *,
    operation_id: str,
) -> tuple[RemoteNote, bytes]:
    """Read payload and metadata together when the provider supports it."""

    try:
        if binding.expected_kind is NoteKind.AUDIO:
            rich_read = getattr(provider, "download_audio_with_metadata", None)
            if callable(rich_read):
                note, payload = rich_read(binding.note_id)
            else:
                before = provider.get_note(binding.note_id)
                payload = provider.download_audio(binding.note_id)
                after = provider.get_note(binding.note_id)
                if before != after or after is None:
                    raise E2EProofError(
                        "source_changed_during_read", operation_id=operation_id
                    )
                note = after
        else:
            rich_read = getattr(provider, "read_text_with_metadata", None)
            if callable(rich_read):
                note, payload = rich_read(binding.note_id)
            else:
                before = provider.get_note(binding.note_id)
                payload = provider.read_note_content(binding.note_id)
                after = provider.get_note(binding.note_id)
                if before != after or after is None:
                    raise E2EProofError(
                        "source_changed_during_read", operation_id=operation_id
                    )
                note = after
    except E2EProofError:
        raise
    except Exception:
        raise E2EProofError(
            "source_read_failed", operation_id=operation_id
        ) from None

    if not isinstance(note, RemoteNote) or not isinstance(
        payload, (bytes, bytearray, memoryview)
    ):
        raise E2EProofError(
            "source_contract_invalid", operation_id=operation_id
        )
    return note, bytes(payload)


def _validate_source(
    *,
    note: RemoteNote,
    payload: bytes,
    binding: E2EInputBinding,
    limits: E2EProofLimits,
    operation_id: str,
) -> None:
    if note.note_id != binding.note_id:
        raise E2EProofError("source_identity_mismatch", operation_id=operation_id)
    if note.kind is not binding.expected_kind:
        raise E2EProofError("source_kind_mismatch", operation_id=operation_id)
    maximum = (
        limits.max_audio_bytes
        if binding.expected_kind is NoteKind.AUDIO
        else limits.max_text_bytes
    )
    if not payload or len(payload) > maximum:
        raise E2EProofError("source_size_rejected", operation_id=operation_id)
    observed = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(observed, binding.expected_sha256):
        raise E2EProofError("source_digest_mismatch", operation_id=operation_id)


def _transcript(
    *,
    note: RemoteNote,
    payload: bytes,
    transcriber: Transcriber,
    limits: E2EProofLimits,
    operation_id: str,
) -> tuple[str, bool]:
    if note.kind is NoteKind.AUDIO:
        try:
            value = transcriber.transcribe(payload, mime_type=note.mime_type)
        except Exception:
            raise E2EProofError(
                "transcription_failed", operation_id=operation_id
            ) from None
        used = True
    else:
        try:
            value = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise E2EProofError(
                "text_is_not_utf8", operation_id=operation_id
            ) from None
        used = False

    if not isinstance(value, str):
        raise E2EProofError(
            "transcript_contract_invalid", operation_id=operation_id
        )
    value = value.strip()
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise E2EProofError(
            "transcript_encoding_invalid", operation_id=operation_id
        ) from None
    if (
        not value
        or len(value) > limits.max_transcript_chars
        or len(encoded) > limits.max_transcript_bytes
    ):
        raise E2EProofError("transcript_size_rejected", operation_id=operation_id)
    return value, used


def _render_output(
    *,
    request_text: str,
    answer: str,
    completed_at: datetime,
) -> str:
    timestamp = completed_at
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("now must return a timezone-aware datetime")
    timestamp = timestamp.astimezone(UTC)
    return (
        f"YOU SAID\n{request_text}\n\n"
        f"ANSWER\n{answer}\n\n"
        f"Completed {timestamp:%Y-%m-%d %H:%M} UTC"
    )


def _assistant_error_category(error: Exception) -> str:
    """Map the complete typed ChatGPT taxonomy without exposing details."""

    if isinstance(error, ChatGPTCredentialAmbiguousError):
        return "chatgpt_credential_ambiguous"
    if isinstance(error, ChatGPTAuthorizationRequiredError):
        return "chatgpt_authorization_required"
    if isinstance(error, ChatGPTAccountNotEligibleError):
        return "chatgpt_account_not_eligible"
    if isinstance(error, ChatGPTPlanUsageLimitError):
        return "chatgpt_plan_usage_limit"
    if isinstance(error, ChatGPTConfigurationError):
        return "chatgpt_configuration_error"
    if isinstance(error, ChatGPTAmbiguousOutcomeError):
        return "chatgpt_response_ambiguous"
    if isinstance(error, ChatGPTTransientError):
        return "chatgpt_transient"
    return "assistant_failed"


def run_e2e_proof(
    *,
    provider: LightProvider,
    transcriber: Transcriber,
    assistant: Assistant,
    binding: E2EInputBinding,
    limits: E2EProofLimits = E2EProofLimits(),
    output_title_prefix: str = DEFAULT_OUTPUT_TITLE_PREFIX,
    operation_id_factory: Callable[[], str] = _default_operation_id,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> E2EProofResult:
    """Run one exact-bound proof and create at most one Light output note.

    ``assistant`` is expected to be a :class:`ChatGPTPlanAssistant` in the real
    operator.  The structural protocol keeps this function deterministic in
    tests.  If a create call has any uncertain result, the call fails closed
    and never attempts a second create.
    """

    operation_id = operation_id_factory()
    if not isinstance(operation_id, str) or _OPERATION_ID_RE.fullmatch(
        operation_id
    ) is None:
        raise ValueError("operation_id_factory returned an invalid identifier")
    if not _valid_title_prefix(output_title_prefix):
        raise ValueError("output_title_prefix is invalid")

    note, payload = _source_payload(
        provider, binding, operation_id=operation_id
    )
    _validate_source(
        note=note,
        payload=payload,
        binding=binding,
        limits=limits,
        operation_id=operation_id,
    )
    transcript, transcription_used = _transcript(
        note=note,
        payload=payload,
        transcriber=transcriber,
        limits=limits,
        operation_id=operation_id,
    )

    route = (
        route_spoken_transcript(transcript)
        if note.kind is NoteKind.AUDIO
        else route_transcript(transcript)
    )
    if route.kind is RouteKind.IGNORED:
        raise E2EProofError("assistant_prefix_required", operation_id=operation_id)
    if route.kind is not RouteKind.QUESTION or route.text is None:
        raise E2EProofError(
            "command_not_supported_in_one_shot", operation_id=operation_id
        )
    request_text = route.text.strip()
    if not request_text:
        raise E2EProofError("question_is_empty", operation_id=operation_id)

    try:
        response = assistant.respond(
            (
                PromptMessage("system", DEFAULT_SYSTEM_PROMPT),
                PromptMessage("user", request_text),
            )
        )
    except Exception as error:
        raise E2EProofError(
            _assistant_error_category(error), operation_id=operation_id
        ) from None
    if not isinstance(response, AssistantResponse):
        raise E2EProofError(
            "assistant_contract_invalid", operation_id=operation_id
        )
    if not isinstance(response.response_id, str) or not response.response_id.strip():
        raise E2EProofError(
            "assistant_response_identity_missing", operation_id=operation_id
        )
    answer = response.text.strip() if isinstance(response.text, str) else ""
    try:
        answer_bytes = answer.encode("utf-8")
    except UnicodeEncodeError:
        raise E2EProofError(
            "answer_encoding_invalid", operation_id=operation_id
        ) from None
    if (
        not answer
        or len(answer) > limits.max_answer_chars
        or len(answer_bytes) > limits.max_answer_bytes
    ):
        raise E2EProofError("answer_size_rejected", operation_id=operation_id)

    try:
        output = _render_output(
            request_text=request_text,
            answer=answer,
            completed_at=now(),
        )
        output_bytes = output.encode("utf-8")
    except Exception:
        raise E2EProofError(
            "output_render_failed", operation_id=operation_id
        ) from None
    if len(output_bytes) > limits.max_output_bytes:
        raise E2EProofError("output_size_rejected", operation_id=operation_id)

    title = f"{output_title_prefix} {operation_id}"
    desired_sha256 = hashlib.sha256(output_bytes).hexdigest()
    try:
        receipt = provider.create_text_note(title=title, content=output)
    except Exception:
        raise E2EProofError(
            "publication_outcome_unknown",
            operation_id=operation_id,
            mutation_attempted=True,
        ) from None

    if not isinstance(receipt, MutationReceipt):
        raise E2EProofError(
            "publication_outcome_unknown",
            operation_id=operation_id,
            mutation_attempted=True,
        )
    if receipt.outcome is MutationOutcome.UNKNOWN:
        raise E2EProofError(
            "publication_outcome_unknown",
            operation_id=operation_id,
            mutation_attempted=True,
        )
    if receipt.outcome is not MutationOutcome.VERIFIED or receipt.note is None:
        raise E2EProofError(
            "publication_conflict",
            operation_id=operation_id,
            mutation_attempted=True,
        )
    created = receipt.note
    if (
        _NOTE_ID_RE.fullmatch(created.note_id) is None
        or created.kind is not NoteKind.TEXT
        or created.title != title
        or receipt.observed_content_sha256 != desired_sha256
    ):
        raise E2EProofError(
            "publication_receipt_mismatch",
            operation_id=operation_id,
            mutation_attempted=True,
        )

    # Independently read the exact returned identity.  Title matching or list
    # position is never used to select the output, so duplicate titles cannot
    # redirect verification to another note.
    try:
        observed_note = provider.get_note(created.note_id)
        observed_content = provider.read_note_content(created.note_id)
        observed_after = provider.get_note(created.note_id)
    except Exception:
        raise E2EProofError(
            "publication_readback_unknown",
            operation_id=operation_id,
            mutation_attempted=True,
        ) from None
    if (
        observed_note is None
        or observed_after is None
        or observed_note.note_id != created.note_id
        or observed_after.note_id != created.note_id
        or observed_note.kind is not NoteKind.TEXT
        or observed_after.kind is not NoteKind.TEXT
        or observed_note.title != title
        or observed_after.title != title
        or not isinstance(observed_content, (bytes, bytearray, memoryview))
        or not hmac.compare_digest(bytes(observed_content), output_bytes)
    ):
        raise E2EProofError(
            "publication_readback_mismatch",
            operation_id=operation_id,
            mutation_attempted=True,
        )

    return E2EProofResult(
        operation_id=operation_id,
        source_kind=binding.expected_kind,
        source_bytes=len(payload),
        transcription_used=transcription_used,
        transcript_characters=len(transcript),
        request_characters=len(request_text),
        answer_characters=len(answer),
        output_bytes=len(output_bytes),
        response_id_present=True,
    )
