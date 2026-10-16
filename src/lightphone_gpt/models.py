"""Small, dependency-free value types shared by the bridge and its adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal


class NoteKind(StrEnum):
    AUDIO = "audio"
    TEXT = "text"


class JobState(StrEnum):
    IGNORED = "ignored"
    QUEUED = "queued"
    TRANSCRIBED = "transcribed"
    ANSWERED = "answered"
    PUBLISHED = "published"
    FAILED = "failed"


class RouteKind(StrEnum):
    IGNORED = "ignored"
    QUESTION = "question"
    CONTINUE = "continue"
    NEW_CONVERSATION = "new_conversation"
    RESET = "reset"


class MutationOutcome(StrEnum):
    """Result of a remote mutation after the adapter's verification pass."""

    VERIFIED = "verified"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


class SourceCleanupState(StrEnum):
    """Durable lifecycle for deleting one accepted source note."""

    PENDING = "pending"
    ATTEMPTED = "attempted"
    CONFLICT = "conflict"
    CONFIRMED = "confirmed"


class EnqueueOutcome(StrEnum):
    ENQUEUED = "enqueued"
    IGNORED = "ignored"
    DUPLICATE = "duplicate"
    INTEGRITY_ERROR = "integrity_error"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class RemoteNote:
    """A normalized note returned by every Light provider implementation."""

    note_id: str
    kind: NoteKind
    title: str
    updated_at: datetime | None = None
    mime_type: str | None = None


@dataclass(frozen=True, slots=True)
class MutationReceipt:
    outcome: MutationOutcome
    note: RemoteNote | None = None
    observed_content_sha256: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class DeletionReceipt:
    """Result of one exact source-note deletion operation.

    ``deletion_attempted`` separates a safe preflight failure from an
    ambiguous result after the remote DELETE boundary was crossed.
    """

    outcome: MutationOutcome
    deletion_attempted: bool
    note: RemoteNote | None = None
    observed_content_sha256: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PromptMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    text: str
    response_id: str | None = None


@dataclass(frozen=True, slots=True)
class RouteResult:
    kind: RouteKind
    text: str | None = None


@dataclass(frozen=True, slots=True)
class RetentionReport:
    conversations_redacted: int = 0
    jobs_deleted: int = 0


@dataclass(frozen=True, slots=True)
class JobRecord:
    note_id: str
    queue_seq: int
    state: JobState
    remote_updated_at: datetime | None
    mime_type: str | None
    source_kind: NoteKind
    audio_sha256: str | None
    audio: bytes | None
    route_kind: RouteKind | None
    transcript: str | None
    answer: str | None
    answer_sections: tuple[str, ...]
    next_section_index: int
    delivered_sections: int
    source_job_note_id: str | None
    source_section_index: int | None
    publication_content: str | None
    publication_sha256: str | None
    publication_title_sha256: str | None
    publication_output_mode: Literal["rolling", "per_turn"] | None
    conversation_id: int | None
    response_id: str | None
    output_note_id: str | None
    accepted_output_epoch: int
    accepted_output_note_id: str | None
    failure_stage: str | None
    resume_state: JobState | None
    retryable: bool
    manual_retry_allowed: bool
    next_retry_at: datetime | None
    last_error: str | None
    transcription_attempts: int
    model_attempts: int
    publication_attempts: int
    discovered_at: datetime
    updated_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class SourceCleanupRecord:
    source_note_id: str
    source_kind: NoteKind
    source_sha256: str
    state: SourceCleanupState
    delete_attempts: int
    precheck_failures: int
    readback_attempts: int
    missing_observations: int
    next_retry_at: datetime | None
    scheduled_at: datetime
    attempted_at: datetime | None
    confirmed_at: datetime | None
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class JobRecoveryCandidate:
    """Content-free identity and state for one failed processing stage."""

    source_note_id: str
    failure_stage: str
    attempts: int
    retry_allowed: bool
    failed_at: datetime | None


@dataclass(frozen=True, slots=True)
class JobRecoveryResult:
    """Durable receipt for one explicitly retried terminal job."""

    audit_id: int
    source_note_id: str
    failure_stage: Literal["transcription", "model"]
    prior_attempts: int
    resumed_state: JobState
    retried_at: datetime


@dataclass(frozen=True, slots=True)
class IgnoredAudioRecoveryCandidate:
    """Exact content-free identity for one ignored audio request."""

    source_note_id: str
    retained_sha256: str
    queue_seq: int
    transcription_attempts: int
    model_attempts: int
    publication_attempts: int
    ignored_at: datetime


@dataclass(frozen=True, slots=True)
class IgnoredAudioRecoveryResult:
    """Content-free receipt for one atomic ignored-audio restage."""

    source_note_id: str
    retained_sha256: str
    queue_seq: int
    audit_id: int
    prior_transcription_attempts: int
    accepted_output_epoch: int
    accepted_output_note_id: str | None
    already_restaged: bool
    restaged_at: datetime


@dataclass(frozen=True, slots=True)
class OutputSnapshot:
    epoch: int
    note_id: str | None
    content_sha256: str | None


@dataclass(frozen=True, slots=True)
class CreateRecoveryCandidate:
    """Content-free identifiers required for reviewed create recovery."""

    source_note_id: str
    publication_sha256: str
    publication_title_sha256: str
    publication_attempts: int
    output_epoch: int
    output_mode: Literal["rolling", "per_turn"]
    provisional_output_note_id: str | None
    provisional_content_sha256: str | None
    gated_at: datetime


@dataclass(frozen=True, slots=True)
class UpdateRecoveryCandidate:
    """Content-free identifiers required for uncertain update recovery."""

    source_note_id: str
    publication_sha256: str
    publication_title_sha256: str
    publication_attempts: int
    output_epoch: int
    output_note_id: str
    previous_content_sha256: str
    readback_attempts: int
    gated_at: datetime


@dataclass(frozen=True, slots=True)
class AppliedCreateVerification:
    """Content-free evidence from a stable, read-only Light observation."""

    method: Literal["light_read_only_v1"]
    output_note_id: str
    observed_content_sha256: str
    observed_title_sha256: str
    observed_kind: NoteKind
    remote_updated_at: datetime
    started_at: datetime
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class CreateRecoveryResult:
    """Durable receipt for one operator-resolved ambiguous create."""

    audit_id: int
    decision: Literal["applied", "not_applied"]
    source_note_id: str
    publication_sha256: str
    output_note_id: str | None
    output_epoch: int
    output_mode: Literal["rolling", "per_turn"]
    decided_at: datetime


@dataclass(frozen=True, slots=True)
class UpdateRecoveryResult:
    """Durable receipt for one reviewed uncertain-update action."""

    audit_id: int
    action: Literal["applied", "resume_readback"]
    source_note_id: str
    publication_sha256: str
    output_note_id: str
    output_epoch: int
    prior_readback_attempts: int
    decided_at: datetime
