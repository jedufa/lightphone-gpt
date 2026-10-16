"""Durable, file-backed simulation of the Light Notes provider contract."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from lightphone_gpt.contracts import PermanentDependencyError
from lightphone_gpt.models import (
    DeletionReceipt,
    MutationOutcome,
    MutationReceipt,
    NoteKind,
    RemoteNote,
)

from .faults import FaultScript, SimulatedProviderError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SimulatedLightProvider:
    """SQLite-backed Light provider with deterministic failure points.

    Provider mutations commit before an ``after`` fault is triggered. Those
    faults therefore return ``UNKNOWN`` receipts and model a lost response
    after the cloud may already have accepted the write.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        faults: FaultScript | None = None,
        clock: Callable[[], datetime] | None = None,
        list_duplicates: Mapping[str, int] | None = None,
    ) -> None:
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._faults = faults or FaultScript()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._duplicates = dict(list_duplicates or {})
        self._lock = RLock()
        self._initialize()

    def list_notes(self) -> Sequence[RemoteNote]:
        operation = "list_notes"
        self._faults.before(operation)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT note_id, kind, title, updated_at, mime_type
                FROM notes WHERE deleted = 0
                ORDER BY sequence_number ASC
                """
            ).fetchall()
        notes = [self._remote_note(row) for row in rows]
        for note in tuple(notes):
            extra_copies = self._duplicates.get(note.note_id, 0)
            if extra_copies > 0:
                notes.extend([note] * extra_copies)
        self._faults.after(operation)
        return notes

    def get_note(self, note_id: str) -> RemoteNote | None:
        operation = "get_note"
        self._faults.before(operation)
        with self._lock, self._connect() as connection:
            row = self._fetch_row(connection, note_id)
        note = self._remote_note(row) if row is not None else None
        self._faults.after(operation)
        return note

    def read_note_content(self, note_id: str) -> bytes:
        operation = "read_note_content"
        self._faults.before(operation)
        with self._lock, self._connect() as connection:
            row = self._fetch_row(connection, note_id)
            if row is None:
                raise PermanentDependencyError("note does not exist")
            content = bytes(row["content"])
        self._faults.after(operation)
        return content

    def download_audio(self, note_id: str) -> bytes:
        operation = "download_audio"
        self._faults.before(operation)
        with self._lock, self._connect() as connection:
            row = self._fetch_row(connection, note_id)
            if row is None:
                raise PermanentDependencyError("audio note does not exist")
            if row["kind"] != NoteKind.AUDIO.value:
                raise PermanentDependencyError("requested note is not audio")
            content = bytes(row["content"])
        self._faults.after(operation)
        return content

    def create_text_note(self, *, title: str, content: str) -> MutationReceipt:
        operation = "create_text_note"
        self._faults.before(operation)
        note_id: str | None = None
        try:
            with self._lock, self._connect() as connection:
                sequence = self._next_sequence(connection)
                note_id = f"sim-{sequence:08d}"
                connection.execute(
                    """
                    INSERT INTO notes (
                        note_id, sequence_number, kind, title, updated_at,
                        mime_type, content, deleted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        note_id,
                        sequence,
                        NoteKind.TEXT.value,
                        title,
                        self._timestamp(),
                        "text/plain; charset=utf-8",
                        content.encode("utf-8"),
                    ),
                )
                connection.commit()
            self._faults.after(operation)
        except SimulatedProviderError:
            return MutationReceipt(
                MutationOutcome.UNKNOWN,
                note=self._get_note_without_fault(note_id) if note_id else None,
                detail="create_outcome_unknown",
            )

        return self._verify_created(note_id, title=title, content=content)

    def update_text_note(
        self,
        note_id: str,
        *,
        title: str,
        content: str,
        expected_title: str,
        expected_content_sha256: str,
    ) -> MutationReceipt:
        self._validate_digest(expected_content_sha256)
        existing = self.get_note(note_id)
        if existing is None:
            return MutationReceipt(
                MutationOutcome.CONFLICT,
                detail="note_missing",
            )
        if existing.kind is not NoteKind.TEXT:
            return MutationReceipt(
                MutationOutcome.CONFLICT,
                note=existing,
                detail="note_is_not_text",
            )
        if existing.title != expected_title:
            return MutationReceipt(
                MutationOutcome.CONFLICT,
                note=existing,
                detail="title_changed",
            )

        existing_content = self.read_note_content(note_id)
        observed = hashlib.sha256(existing_content).hexdigest()
        if observed != expected_content_sha256:
            return MutationReceipt(
                MutationOutcome.CONFLICT,
                note=existing,
                observed_content_sha256=observed,
                detail="content_changed",
            )

        operation = "update_text_note"
        self._faults.before(operation)
        try:
            with self._lock, self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE notes
                    SET title = ?, content = ?, updated_at = ?
                    WHERE note_id = ? AND deleted = 0 AND kind = ?
                      AND title = ? AND content = ?
                    """,
                    (
                        title,
                        content.encode("utf-8"),
                        self._timestamp(),
                        note_id,
                        NoteKind.TEXT.value,
                        expected_title,
                        existing_content,
                    ),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    return MutationReceipt(
                        MutationOutcome.CONFLICT,
                        detail="note_changed_during_update",
                    )
                connection.commit()
            self._faults.after(operation)
        except SimulatedProviderError:
            return MutationReceipt(
                MutationOutcome.UNKNOWN,
                note=self._get_note_without_fault(note_id),
                detail="update_outcome_unknown",
            )

        return self._verify_updated(note_id, title=title, content=content)

    def delete_note(self, note_id: str) -> None:
        """Simulate a user/cloud deletion; this is outside the core protocol."""

        operation = "delete_note"
        self._faults.before(operation)
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE notes SET deleted = 1, updated_at = ? WHERE note_id = ?",
                (self._timestamp(), note_id),
            )
            connection.commit()
        self._faults.after(operation)

    def delete_source_note(
        self,
        note_id: str,
        *,
        expected_kind: NoteKind,
        expected_content_sha256: str,
    ) -> DeletionReceipt:
        """Apply the production source-delete contract with fault boundaries."""

        self._validate_digest(expected_content_sha256)
        with self._lock, self._connect() as connection:
            row = self._fetch_row(connection, note_id)
            if row is None:
                return DeletionReceipt(
                    MutationOutcome.VERIFIED,
                    deletion_attempted=False,
                    detail="source_already_missing",
                )
            note = self._remote_note(row)
            if note.kind is not expected_kind:
                return DeletionReceipt(
                    MutationOutcome.CONFLICT,
                    deletion_attempted=False,
                    note=note,
                    detail="source_kind_changed",
                )
            observed = hashlib.sha256(bytes(row["content"])).hexdigest()
            if observed != expected_content_sha256:
                return DeletionReceipt(
                    MutationOutcome.CONFLICT,
                    deletion_attempted=False,
                    note=note,
                    observed_content_sha256=observed,
                    detail="source_content_changed",
                )

        operation = "delete_source_note"
        try:
            self._faults.before(operation)
        except SimulatedProviderError:
            return DeletionReceipt(
                MutationOutcome.UNKNOWN,
                deletion_attempted=False,
                detail="source_delete_precheck_unavailable",
            )
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    "UPDATE notes SET deleted = 1, updated_at = ? "
                    "WHERE note_id = ? AND deleted = 0",
                    (self._timestamp(), note_id),
                )
                connection.commit()
            self._faults.after(operation)
        except SimulatedProviderError:
            return DeletionReceipt(
                MutationOutcome.UNKNOWN,
                deletion_attempted=True,
                detail="source_delete_outcome_unknown",
            )
        return DeletionReceipt(
            MutationOutcome.VERIFIED,
            deletion_attempted=True,
            detail="source_delete_verified",
        )

    def seed_audio_note(
        self,
        content: bytes,
        *,
        title: str = "Voice note",
        mime_type: str = "audio/mp4",
        note_id: str | None = None,
    ) -> RemoteNote:
        """Model a recording arriving from a phone or delayed cloud sync."""

        if not content:
            raise ValueError("audio content must not be empty")
        with self._lock, self._connect() as connection:
            sequence = self._next_sequence(connection)
            assigned_id = note_id or f"sim-{sequence:08d}"
            connection.execute(
                """
                INSERT INTO notes (
                    note_id, sequence_number, kind, title, updated_at,
                    mime_type, content, deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    assigned_id,
                    sequence,
                    NoteKind.AUDIO.value,
                    title,
                    self._timestamp(),
                    mime_type,
                    bytes(content),
                ),
            )
            connection.commit()
        note = self._get_note_without_fault(assigned_id)
        assert note is not None
        return note

    def seed_text_note(
        self,
        content: str,
        *,
        title: str = "Text note",
        note_id: str | None = None,
    ) -> RemoteNote:
        """Model a handset-authored text note arriving through Light Cloud."""

        if not content:
            raise ValueError("text content must not be empty")
        with self._lock, self._connect() as connection:
            sequence = self._next_sequence(connection)
            assigned_id = note_id or f"sim-{sequence:08d}"
            connection.execute(
                """
                INSERT INTO notes (
                    note_id, sequence_number, kind, title, updated_at,
                    mime_type, content, deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    assigned_id,
                    sequence,
                    NoteKind.TEXT.value,
                    title,
                    self._timestamp(),
                    "text/plain; charset=utf-8",
                    content.encode("utf-8"),
                ),
            )
            connection.commit()
        note = self._get_note_without_fault(assigned_id)
        assert note is not None
        return note

    def replace_audio_content(self, note_id: str, content: bytes) -> None:
        """Model a remote identity/content anomaly for integrity tests."""

        if not content:
            raise ValueError("audio content must not be empty")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE notes SET content = ?, updated_at = ?
                WHERE note_id = ? AND deleted = 0 AND kind = ?
                """,
                (bytes(content), self._timestamp(), note_id, NoteKind.AUDIO.value),
            )
            if cursor.rowcount != 1:
                raise ValueError("audio note does not exist")
            connection.commit()

    def set_list_duplicates(self, note_id: str, *, extra_copies: int) -> None:
        if extra_copies < 0:
            raise ValueError("extra_copies must be non-negative")
        with self._lock:
            if extra_copies:
                self._duplicates[note_id] = extra_copies
            else:
                self._duplicates.pop(note_id, None)

    def _verify_created(
        self, note_id: str, *, title: str, content: str
    ) -> MutationReceipt:
        try:
            note = self.get_note(note_id)
            observed = hashlib.sha256(self.read_note_content(note_id)).hexdigest()
        except SimulatedProviderError:
            return MutationReceipt(
                MutationOutcome.UNKNOWN,
                note=self._get_note_without_fault(note_id),
                detail="create_verification_failed",
            )
        desired = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if note is not None and note.title == title and observed == desired:
            return MutationReceipt(
                MutationOutcome.VERIFIED,
                note=note,
                observed_content_sha256=observed,
            )
        return MutationReceipt(
            MutationOutcome.CONFLICT,
            note=note,
            observed_content_sha256=observed,
            detail="created_note_differs",
        )

    def _verify_updated(
        self, note_id: str, *, title: str, content: str
    ) -> MutationReceipt:
        try:
            note = self.get_note(note_id)
            if note is None:
                return MutationReceipt(
                    MutationOutcome.CONFLICT,
                    detail="note_deleted_after_update",
                )
            observed = hashlib.sha256(self.read_note_content(note_id)).hexdigest()
        except SimulatedProviderError:
            return MutationReceipt(
                MutationOutcome.UNKNOWN,
                note=self._get_note_without_fault(note_id),
                detail="update_verification_failed",
            )
        desired = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if note.title == title and observed == desired:
            return MutationReceipt(
                MutationOutcome.VERIFIED,
                note=note,
                observed_content_sha256=observed,
            )
        return MutationReceipt(
            MutationOutcome.CONFLICT,
            note=note,
            observed_content_sha256=observed,
            detail="updated_note_differs",
        )

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS counters (
                    name TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO counters (name, value) VALUES ('note', 0);
                CREATE TABLE IF NOT EXISTS notes (
                    note_id TEXT PRIMARY KEY,
                    sequence_number INTEGER NOT NULL UNIQUE,
                    kind TEXT NOT NULL CHECK (kind IN ('audio', 'text')),
                    title TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    mime_type TEXT,
                    content BLOB NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0, 1))
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @staticmethod
    def _next_sequence(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "UPDATE counters SET value = value + 1 WHERE name = 'note' RETURNING value"
        ).fetchone()
        if row is None:
            raise RuntimeError("simulator note counter is unavailable")
        return int(row[0])

    @staticmethod
    def _fetch_row(
        connection: sqlite3.Connection, note_id: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT note_id, sequence_number, kind, title, updated_at,
                   mime_type, content, deleted
            FROM notes WHERE note_id = ? AND deleted = 0
            """,
            (note_id,),
        ).fetchone()

    def _get_note_without_fault(self, note_id: str | None) -> RemoteNote | None:
        if note_id is None:
            return None
        with self._lock, self._connect() as connection:
            row = self._fetch_row(connection, note_id)
        return self._remote_note(row) if row is not None else None

    @staticmethod
    def _remote_note(row: sqlite3.Row) -> RemoteNote:
        return RemoteNote(
            note_id=str(row["note_id"]),
            kind=NoteKind(str(row["kind"])),
            title=str(row["title"]),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            mime_type=str(row["mime_type"]) if row["mime_type"] else None,
        )

    def _timestamp(self) -> str:
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()

    @staticmethod
    def _validate_digest(value: str) -> None:
        if not _SHA256.fullmatch(value):
            raise PermanentDependencyError(
                "expected_content_sha256 must be a lowercase SHA-256 digest"
            )
