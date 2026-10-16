"""Stable adapter contracts used by both simulated and real integrations."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .models import (
    AssistantResponse,
    DeletionReceipt,
    MutationReceipt,
    NoteKind,
    PromptMessage,
    RemoteNote,
)


class DependencyError(RuntimeError):
    """Base class for a safely reportable adapter failure."""


class TransientDependencyError(DependencyError):
    """An operation may be retried without changing its meaning."""


class PermanentDependencyError(DependencyError):
    """An operation cannot succeed without configuration or user action."""


class IntegrityError(PermanentDependencyError):
    """A remote identity unexpectedly refers to different content."""


@runtime_checkable
class LightProvider(Protocol):
    """The sole Light boundary; simulators and real clients share this API."""

    def list_notes(self) -> Sequence[RemoteNote]: ...

    def get_note(self, note_id: str) -> RemoteNote | None: ...

    def read_note_content(self, note_id: str) -> bytes: ...

    def download_audio(self, note_id: str) -> bytes: ...

    def create_text_note(self, *, title: str, content: str) -> MutationReceipt: ...

    def update_text_note(
        self,
        note_id: str,
        *,
        title: str,
        content: str,
        expected_title: str,
        expected_content_sha256: str,
    ) -> MutationReceipt: ...

    def delete_source_note(
        self,
        note_id: str,
        *,
        expected_kind: NoteKind,
        expected_content_sha256: str,
    ) -> DeletionReceipt: ...


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, audio: bytes, *, mime_type: str | None = None) -> str: ...


@runtime_checkable
class Assistant(Protocol):
    def respond(self, messages: Sequence[PromptMessage]) -> AssistantResponse: ...
