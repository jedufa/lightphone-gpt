"""Explicit server-side retention; it never deletes Light Cloud notes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .models import RetentionReport
from .store import Store


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    closed_conversation_days: int = 30
    metadata_days: int = 90

    def __post_init__(self) -> None:
        if self.closed_conversation_days < 0 or self.metadata_days < 0:
            raise ValueError("retention durations cannot be negative")


def apply_retention(
    store: Store,
    *,
    now: datetime,
    policy: RetentionPolicy = RetentionPolicy(),
) -> RetentionReport:
    conversations = store.redact_closed_conversations(
        closed_before=now - timedelta(days=policy.closed_conversation_days),
        now=now,
    )
    jobs = store.delete_expired_metadata(
        completed_before=now - timedelta(days=policy.metadata_days)
    )
    return RetentionReport(conversations_redacted=conversations, jobs_deleted=jobs)
