from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import sqlite3

import pytest

from lightphone_gpt.models import (
    AppliedCreateVerification,
    EnqueueOutcome,
    JobState,
    NoteKind,
    RemoteNote,
    SourceCleanupState,
)
from lightphone_gpt.store import (
    Store,
    _LATEST_SCHEMA_VERSION,
    _SCHEMA_V1,
    _SCHEMA_V2,
    _SCHEMA_V3,
    _SCHEMA_V4,
    _SCHEMA_V5,
    _SCHEMA_V6,
    _SCHEMA_V7,
    _SCHEMA_V8,
    _SCHEMA_V9,
    _SCHEMA_V10,
    _SCHEMA_V11,
)


NOW = datetime(2026, 10, 15, 8, 0, tzinfo=timezone.utc)


def audio_note(note_id: str, *, minute: int = 0) -> RemoteNote:
    return RemoteNote(
        note_id,
        NoteKind.AUDIO,
        "Recording",
        NOW + timedelta(minutes=minute),
        "audio/mp4",
    )


def text_note(note_id: str, *, minute: int = 0, title: str = "Text") -> RemoteNote:
    return RemoteNote(
        note_id,
        NoteKind.TEXT,
        title,
        NOW + timedelta(minutes=minute),
        "text/plain; charset=utf-8",
    )


def applied_verification(
    note_id: str,
    content_sha256: str,
    *,
    when: datetime = NOW,
) -> AppliedCreateVerification:
    return AppliedCreateVerification(
        method="light_read_only_v1",
        output_note_id=note_id,
        observed_content_sha256=content_sha256,
        observed_title_sha256=hashlib.sha256(b"AI READY").hexdigest(),
        observed_kind=NoteKind.TEXT,
        remote_updated_at=when,
        started_at=when,
        completed_at=when + timedelta(seconds=1),
    )


def create_v8_schema(connection: sqlite3.Connection) -> None:
    for schema in (
        _SCHEMA_V1,
        _SCHEMA_V2,
        _SCHEMA_V3,
        _SCHEMA_V4,
        _SCHEMA_V5,
        _SCHEMA_V6,
        _SCHEMA_V7,
        _SCHEMA_V8,
    ):
        connection.executescript(schema)


def test_sqlite_safety_pragmas_and_baseline(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert store.connection.execute("PRAGMA secure_delete").fetchone()[0] == 1
        assert store.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

        assert (
            store.establish_baseline(
                [audio_note("old"), text_note("old-text"), text_note("reply", title="AI READY")],
                now=NOW,
            )
            == 3
        )
        old = store.get_job("old")
        assert old is not None
        assert old.state == JobState.IGNORED
        assert old.audio is None
        assert old.audio_sha256 is None
        assert old.source_kind is NoteKind.AUDIO
        assert store.get_job("old-text").source_kind is NoteKind.TEXT
        assert store.get_job("reply").state is JobState.IGNORED
        assert store.establish_baseline([audio_note("another")], now=NOW) == 0


def test_note_id_is_identity_and_hash_change_is_integrity_failure(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        note = audio_note("n1")
        assert store.enqueue_audio(note, b"first", now=NOW) == EnqueueOutcome.ENQUEUED
        assert store.enqueue_audio(note, b"first", now=NOW) == EnqueueOutcome.DUPLICATE
        assert (
            store.enqueue_audio(note, b"different", now=NOW)
            == EnqueueOutcome.INTEGRITY_ERROR
        )
        job = store.get_job("n1")
        assert job is not None
        assert job.state == JobState.FAILED
        assert job.audio is None
        assert "changed" in (job.last_error or "")


def test_cleanup_policy_schedules_only_publications_completed_while_enabled(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        store.enqueue_audio(audio_note("old"), b"old", now=NOW)
        store.mark_question_transcribed("old", question="Old", now=NOW)
        store.mark_answered(
            "old", answer="answer", sections=("answer",), response_id="r0", now=NOW
        )
        store.mark_published("old", output_note_id="reply", now=NOW)
        assert store.source_cleanup("old") is None

        store.configure_source_cleanup(enabled=True, now=NOW + timedelta(seconds=1))
        store.enqueue_audio(
            audio_note("new", minute=1), b"new", now=NOW + timedelta(minutes=1)
        )
        store.mark_question_transcribed(
            "new", question="New", now=NOW + timedelta(minutes=1)
        )
        store.mark_answered(
            "new",
            answer="answer",
            sections=("answer",),
            response_id="r1",
            now=NOW + timedelta(minutes=1),
        )
        store.mark_published(
            "new", output_note_id="reply", now=NOW + timedelta(minutes=1)
        )

        cleanup = store.source_cleanup("new")
        assert cleanup is not None
        assert cleanup.state is SourceCleanupState.PENDING
        assert cleanup.source_kind is NoteKind.AUDIO
        assert cleanup.source_sha256 == hashlib.sha256(b"new").hexdigest()
        assert store.source_cleanup_status() == {
            "enabled": True,
            "pending": 1,
            "uncertain": 0,
            "conflicts": 0,
            "confirmed": 0,
            "unresolved": 1,
            "blocked": False,
        }


def test_schema_v10_defaults_source_cleanup_off_without_backfill(tmp_path):
    database = tmp_path / "v9.sqlite3"
    connection = sqlite3.connect(database)
    create_v8_schema(connection)
    connection.executescript(_SCHEMA_V9)
    connection.execute("PRAGMA user_version = 9")
    connection.commit()
    connection.close()

    with Store(database) as store:
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == _LATEST_SCHEMA_VERSION
        assert store.source_cleanup_status() == {
            "enabled": False,
            "pending": 0,
            "uncertain": 0,
            "conflicts": 0,
            "confirmed": 0,
            "unresolved": 0,
            "blocked": False,
        }


def test_schema_v12_rebuild_preserves_jobs_and_foreign_key_children(tmp_path):
    database = tmp_path / "v11.sqlite3"
    connection = sqlite3.connect(database)
    create_v8_schema(connection)
    connection.executescript(_SCHEMA_V9)
    connection.executescript(_SCHEMA_V10)
    connection.executescript(_SCHEMA_V11)
    timestamp = NOW.isoformat(timespec="microseconds")
    connection.execute(
        "INSERT INTO jobs (note_id, queue_seq, state, route_kind, "
        "accepted_output_epoch, source_kind, notifier_event_sequence, "
        "discovered_at, updated_at, completed_at) "
        "VALUES ('published', 1, 'published', 'question', 0, 'audio', 7, "
        "?, ?, ?)",
        (timestamp, timestamp, timestamp),
    )
    publication = hashlib.sha256(b"uncertain answer").hexdigest()
    title = hashlib.sha256(b"AI READY").hexdigest()
    previous = hashlib.sha256(b"previous answer").hexdigest()
    connection.execute(
        "INSERT INTO jobs (note_id, queue_seq, state, route_kind, answer, "
        "publication_content, publication_sha256, publication_title_sha256, "
        "publication_output_mode, publication_attempts, accepted_output_epoch, "
        "accepted_output_note_id, failure_stage, resume_state, source_kind, "
        "discovered_at, updated_at) VALUES "
        "('stalled', 2, 'failed', 'question', 'uncertain answer', "
        "'uncertain answer', ?, ?, 'rolling', 1, 0, 'reply', "
        "'publication', 'answered', 'audio', ?, ?)",
        (publication, title, timestamp, timestamp),
    )
    connection.execute(
        "INSERT INTO output_surfaces (note_id, epoch, content_sha256, "
        "created_at) VALUES ('reply', 0, ?, ?)",
        (previous, timestamp),
    )
    connection.execute(
        "UPDATE bridge_state SET enrolled_at = ?, active_output_note_id = "
        "'reply', output_epoch = 0, output_gate = 'update_ambiguous', "
        "output_gate_job_note_id = 'stalled', output_gate_at = ?, "
        "notifier_sequence = 7 WHERE singleton = 1",
        (timestamp, timestamp),
    )
    connection.execute(
        "INSERT INTO job_recovery_audit (retried_at, source_note_id, "
        "failure_stage, prior_attempts, resumed_state) "
        "VALUES (?, 'published', 'model', 1, 'transcribed')",
        (timestamp,),
    )
    connection.execute(
        "INSERT INTO source_cleanups (source_note_id, source_kind, "
        "source_sha256, state, scheduled_at, updated_at) "
        "VALUES ('published', 'audio', ?, 'pending', ?, ?)",
        ("a" * 64, timestamp, timestamp),
    )
    connection.execute(
        "INSERT INTO publication_recovery_audit (gated_at, decided_at, gate, "
        "source_note_id, publication_sha256, publication_title_sha256, "
        "publication_attempts, output_epoch, decision, output_mode) "
        "VALUES (?, ?, 'create_ambiguous', 'stalled', ?, ?, 1, 0, "
        "'not_applied', 'rolling')",
        (timestamp, timestamp, publication, title),
    )
    connection.execute("PRAGMA user_version = 11")
    connection.commit()
    connection.close()

    with Store(database) as store:
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 12
        published = store.get_job("published")
        stalled = store.get_job("stalled")
        assert published is not None
        assert store.connection.execute(
            "SELECT notifier_event_sequence FROM jobs "
            "WHERE note_id = 'published'"
        ).fetchone()[0] == 7
        assert stalled is not None and stalled.publication_attempts == 1
        assert store.connection.execute(
            "SELECT COUNT(*) FROM job_recovery_audit "
            "WHERE source_note_id = 'published'"
        ).fetchone()[0] == 1
        assert store.connection.execute(
            "SELECT COUNT(*) FROM source_cleanups "
            "WHERE source_note_id = 'published'"
        ).fetchone()[0] == 1
        assert store.connection.execute(
            "SELECT COUNT(*) FROM publication_recovery_audit "
            "WHERE source_note_id = 'stalled'"
        ).fetchone()[0] == 1
        state = store.connection.execute(
            "SELECT active_output_note_id, output_epoch, output_gate, "
            "output_gate_job_note_id, notifier_sequence FROM bridge_state"
        ).fetchone()
        assert tuple(state) == ("reply", 0, "update_ambiguous", "stalled", 7)
        assert store.connection.execute(
            "SELECT content_sha256 FROM output_surfaces WHERE note_id = 'reply'"
        ).fetchone()[0] == previous
        assert store.connection.execute("PRAGMA foreign_key_check").fetchall() == []
        columns = {
            row["name"]
            for row in store.connection.execute("PRAGMA table_info(jobs)")
        }
        assert {
            "admitted_output_gate",
            "admitted_output_gate_job_note_id",
            "admitted_assistant_gate",
            "admitted_assistant_gate_job_note_id",
        } <= columns


def test_continue_consumes_section_only_after_verified_publication(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        store.enqueue_audio(audio_note("q"), b"q", now=NOW)
        store.mark_question_transcribed("q", question="Question", now=NOW)
        store.mark_answered(
            "q",
            answer="full",
            sections=("first", "second", "third"),
            response_id="r1",
            now=NOW,
        )
        assert store.conversation_turns(1, before_queue_seq=999) == []
        store.mark_published("q", output_note_id="reply", now=NOW)
        assert store.conversation_turns(1, before_queue_seq=999) == [
            ("Question", "first")
        ]

        store.enqueue_audio(audio_note("c1", minute=1), b"c1", now=NOW)
        assert (
            store.mark_continue(
                "c1", now=NOW, exhausted_answer="Nothing left."
            )
            == "second"
        )
        source = store.get_job("q")
        assert source is not None
        assert source.next_section_index == 1
        assert source.delivered_sections == 1
        assert store.conversation_turns(1, before_queue_seq=999) == [
            ("Question", "first")
        ]

        store.mark_published("c1", output_note_id="reply", now=NOW)
        source = store.get_job("q")
        assert source is not None
        assert source.next_section_index == 2
        assert source.delivered_sections == 2
        assert store.conversation_turns(1, before_queue_seq=999) == [
            ("Question", "first\n\nsecond")
        ]

        store.enqueue_audio(audio_note("c2", minute=2), b"c2", now=NOW)
        assert (
            store.mark_continue(
                "c2", now=NOW, exhausted_answer="Nothing left."
            )
            == "third"
        )
        store.mark_published("c2", output_note_id="reply", now=NOW)
        store.enqueue_audio(audio_note("c3", minute=3), b"c3", now=NOW)
        assert (
            store.mark_continue(
                "c3", now=NOW, exhausted_answer="Nothing left."
            )
            == "Nothing left."
        )


def test_continue_never_falls_back_to_an_older_answer(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        store.enqueue_audio(audio_note("older"), b"older", now=NOW)
        store.mark_question_transcribed("older", question="Older", now=NOW)
        store.mark_answered(
            "older",
            answer="old full",
            sections=("old first", "old second"),
            response_id="r-old",
            now=NOW,
        )
        store.mark_published("older", output_note_id="reply", now=NOW)

        store.enqueue_audio(audio_note("latest", minute=1), b"latest", now=NOW)
        store.mark_question_transcribed("latest", question="Latest", now=NOW)
        store.mark_answered(
            "latest",
            answer="latest only",
            sections=("latest only",),
            response_id="r-latest",
            now=NOW,
        )
        store.mark_published("latest", output_note_id="reply", now=NOW)

        store.enqueue_audio(audio_note("continue", minute=2), b"continue", now=NOW)
        assert (
            store.mark_continue(
                "continue", now=NOW, exhausted_answer="Nothing left."
            )
            == "Nothing left."
        )
        continuation = store.get_job("continue")
        assert continuation is not None
        assert continuation.source_job_note_id is None
        assert store.get_job("older").next_section_index == 1  # type: ignore[union-attr]


def test_schema_v1_is_migrated_and_request_identity_is_backfilled(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(_SCHEMA_V1)
    encoded = NOW.isoformat(timespec="microseconds")
    connection.execute(
        """
        INSERT INTO jobs (
            note_id, queue_seq, state, audio_sha256,
            accepted_output_epoch, discovered_at, updated_at, completed_at
        ) VALUES ('legacy', 1, 'ignored', ?, 0, ?, ?, ?)
        """,
        ("a" * 64, encoded, encoded, encoded),
    )
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    with Store(database) as store:
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == _LATEST_SCHEMA_VERSION
        assert store.has_seen_request("legacy")
        columns = {
            row["name"]
            for row in store.connection.execute("PRAGMA table_info(jobs)").fetchall()
        }
        assert {"delivered_sections", "source_section_index"} <= columns
        assert "source_kind" in columns
        output_columns = {
            row["name"]
            for row in store.connection.execute(
                "PRAGMA table_info(output_surfaces)"
            ).fetchall()
        }
        assert {"missing_observations", "missing_since"} <= output_columns
        bridge_columns = {
            row["name"]
            for row in store.connection.execute(
                "PRAGMA table_info(bridge_state)"
            ).fetchall()
        }
        assert {
            "assistant_gate",
            "assistant_gate_job_note_id",
            "assistant_gate_at",
        } <= bridge_columns
        assert store.connection.execute(
            "SELECT COUNT(*) FROM model_usage_days"
        ).fetchone()[0] == 0
        assert {
            "output_gate",
            "output_gate_job_note_id",
            "output_gate_at",
            "output_gate_readback_attempts",
            "text_ingress_enrolled_at",
        } <= bridge_columns
        ledger_columns = {
            row["name"]
            for row in store.connection.execute(
                "PRAGMA table_info(request_ledger)"
            ).fetchall()
        }
        assert "source_kind" in ledger_columns
        assert store.get_job("legacy").source_kind is NoteKind.AUDIO


def test_schema_v9_preserves_existing_gate_and_starts_readback_counter_at_zero(
    tmp_path,
):
    database = tmp_path / "v8.sqlite3"
    connection = sqlite3.connect(database)
    create_v8_schema(connection)
    connection.execute(
        "UPDATE bridge_state SET output_gate = 'update_ambiguous', "
        "output_gate_job_note_id = 'source', output_gate_at = ?",
        (NOW.isoformat(timespec="microseconds"),),
    )
    connection.execute("PRAGMA user_version = 8")
    connection.commit()
    connection.close()

    with Store(database) as store:
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == _LATEST_SCHEMA_VERSION
        state = store.connection.execute(
            "SELECT output_gate, output_gate_job_note_id, "
            "output_gate_readback_attempts FROM bridge_state"
        ).fetchone()
        assert tuple(state) == ("update_ambiguous", "source", 0)
        assert store.connection.execute(
            "SELECT COUNT(*) FROM update_recovery_audit"
        ).fetchone()[0] == 0


def test_schema_v9_restores_exact_legacy_stale_read_failure_to_safe_gate(tmp_path):
    database = tmp_path / "legacy-stale-update.sqlite3"
    connection = sqlite3.connect(database)
    create_v8_schema(connection)
    timestamp = NOW.isoformat(timespec="microseconds")
    previous_digest = hashlib.sha256(b"previous reply").hexdigest()
    publication = "desired reply"
    publication_digest = hashlib.sha256(publication.encode()).hexdigest()
    title_digest = hashlib.sha256(b"AI READY").hexdigest()
    connection.execute(
        "UPDATE bridge_state SET enrolled_at = ?, active_output_note_id = 'reply', "
        "output_epoch = 0 WHERE singleton = 1",
        (timestamp,),
    )
    connection.execute(
        "INSERT INTO output_surfaces(note_id, epoch, content_sha256, created_at) "
        "VALUES ('reply', 0, ?, ?)",
        (previous_digest, timestamp),
    )
    connection.execute(
        """
        INSERT INTO request_ledger(
            note_id, first_seen_at, last_seen_at, remote_updated_at,
            audio_sha256, disposition, source_kind
        ) VALUES ('source', ?, ?, ?, ?, 'accepted', 'audio')
        """,
        (timestamp, timestamp, timestamp, "a" * 64),
    )
    connection.execute(
        """
        INSERT INTO jobs(
            note_id, queue_seq, state, remote_updated_at, mime_type,
            source_kind, audio_sha256, route_kind, transcript, answer,
            answer_sections, publication_content, publication_sha256,
            publication_title_sha256, publication_output_mode,
            accepted_output_epoch, accepted_output_note_id,
            failure_stage, resume_state, retryable, last_error,
            publication_attempts, discovered_at, updated_at, completed_at
        ) VALUES (
            'source', 1, 'failed', ?, 'audio/mp4', 'audio', ?, 'question',
            'Question', 'Answer', '["Answer"]', ?, ?, ?, 'rolling',
            0, 'reply', 'publication', 'answered', 0,
            'remote read-back confirmed update was not applied',
            1, ?, ?, ?
        )
        """,
        (
            timestamp,
            "a" * 64,
            publication,
            publication_digest,
            title_digest,
            timestamp,
            timestamp,
            timestamp,
        ),
    )
    connection.execute("PRAGMA user_version = 8")
    connection.commit()
    connection.close()

    with Store(database) as store:
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == _LATEST_SCHEMA_VERSION
        candidate = store.ambiguous_update_candidate()
        assert candidate is not None
        assert candidate.source_note_id == "source"
        assert candidate.output_note_id == "reply"
        assert candidate.publication_sha256 == publication_digest
        assert candidate.previous_content_sha256 == previous_digest
        assert candidate.publication_attempts == 1
        assert candidate.readback_attempts == 0
        assert store.get_job("source").state is JobState.FAILED


def test_text_identity_is_permanent_and_content_changes_fail_closed(tmp_path):
    database = tmp_path / "bridge.sqlite3"
    with Store(database) as store:
        store.establish_baseline([], now=NOW)
        note = text_note("text-1")
        assert (
            store.enqueue_text(note, b"Assistant, original", now=NOW)
            is EnqueueOutcome.ENQUEUED
        )
        assert (
            store.enqueue_text(note, b"Assistant, original", now=NOW)
            is EnqueueOutcome.DUPLICATE
        )
        store.mark_ignored("text-1", reason="test_cleanup", now=NOW)
        assert (
            store.delete_expired_metadata(completed_before=NOW + timedelta(days=91))
            == 1
        )
        assert store.get_job("text-1") is None
        assert (
            store.enqueue_text(
                text_note("text-1", minute=1),
                b"Assistant, changed",
                now=NOW + timedelta(minutes=1),
            )
            is EnqueueOutcome.INTEGRITY_ERROR
        )
        failed = store.get_job("text-1")
        assert failed.state is JobState.FAILED
        assert failed.source_kind is NoteKind.TEXT
        assert failed.audio is None


def test_schema_v4_backfills_existing_bridge_output_identity(tmp_path):
    database = tmp_path / "v3.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(_SCHEMA_V1)
    connection.executescript(_SCHEMA_V2)
    connection.executescript(_SCHEMA_V3)
    encoded = NOW.isoformat(timespec="microseconds")
    digest = "b" * 64
    connection.execute(
        """
        INSERT INTO jobs (
            note_id, queue_seq, state, publication_sha256, output_note_id,
            accepted_output_epoch, discovered_at, updated_at, completed_at
        ) VALUES ('source', 1, 'published', ?, 'old-output', 0, ?, ?, ?)
        """,
        (digest, encoded, encoded, encoded),
    )
    connection.execute("PRAGMA user_version = 3")
    connection.commit()
    connection.close()

    with Store(database) as store:
        assert not store.is_text_ingress_enrolled()
        assert store.is_known_output_note("old-output")
        row = store.connection.execute(
            "SELECT disposition, source_kind, audio_sha256 "
            "FROM request_ledger WHERE note_id = 'old-output'"
        ).fetchone()
        assert tuple(row) == ("bridge_output", "text", digest)


def test_ambiguous_create_recovery_applied_is_exact_atomic_and_audited(tmp_path):
    database = tmp_path / "bridge.sqlite3"
    payload = "stored answer payload"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    with Store(database) as store:
        store.establish_baseline([], now=NOW)
        store.enqueue_audio(audio_note("source"), b"audio", now=NOW)
        store.mark_question_transcribed("source", question="Question", now=NOW)
        store.mark_answered(
            "source", answer="Answer", sections=("Answer",), response_id="r", now=NOW
        )
        store.prepare_publication(
            "source",
            content=payload,
            title="AI READY",
            output_mode="rolling",
            now=NOW,
        )
        store.increment_attempt("source", "publication", now=NOW)
        store.remember_output_note(
            "remote-output",
            source_note_id="source",
            output_epoch=0,
            publication_attempt=1,
            content_sha256=digest,
            now=NOW,
        )
        store.mark_publication_ambiguous(
            "source", gate="create_ambiguous", error="unknown", now=NOW
        )

        candidate = store.ambiguous_create_candidate()
        assert candidate is not None
        assert candidate.source_note_id == "source"
        assert candidate.publication_sha256 == digest
        assert candidate.publication_title_sha256 == hashlib.sha256(
            b"AI READY"
        ).hexdigest()
        assert candidate.output_epoch == 0
        assert candidate.output_mode == "rolling"
        assert candidate.provisional_output_note_id == "remote-output"

        with pytest.raises(ValueError, match="publication receipt"):
            store.resolve_ambiguous_create(
                source_note_id="source",
                expected_publication_sha256=digest,
                expected_output_epoch=0,
                decision="applied",
                verification=applied_verification("different-output", digest),
                now=NOW + timedelta(seconds=30),
            )
        assert store.output_gate() == "create_ambiguous"
        assert store.connection.execute(
            "SELECT COUNT(*) FROM publication_recovery_audit"
        ).fetchone()[0] == 0

        result = store.resolve_ambiguous_create(
            source_note_id="source",
            expected_publication_sha256=digest,
            expected_output_epoch=0,
            decision="applied",
            verification=applied_verification("remote-output", digest),
            now=NOW + timedelta(minutes=1),
        )

        assert result.audit_id == 1
        assert result.decision == "applied"
        assert result.output_mode == "rolling"
        assert store.output_gate() is None
        assert store.get_job("source").state is JobState.PUBLISHED
        assert store.get_job("source").output_note_id == "remote-output"
        assert store.output_snapshot().note_id == "remote-output"
        assert store.output_snapshot().content_sha256 == digest
        assert store.is_known_output_note("remote-output")
        audit = store.connection.execute(
            "SELECT gated_at, gate, source_note_id, publication_sha256, "
            "publication_attempts, output_epoch, decision, output_mode, "
            "output_note_id, observed_content_sha256 "
            "FROM publication_recovery_audit"
        ).fetchone()
        assert tuple(audit) == (
            NOW.isoformat(timespec="microseconds"),
            "create_ambiguous",
            "source",
            digest,
            1,
            0,
            "applied",
            "rolling",
            "remote-output",
            digest,
        )


def test_ambiguous_create_recovery_mismatch_leaves_gate_untouched(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        store.enqueue_audio(audio_note("source"), b"audio", now=NOW)
        store.mark_question_transcribed("source", question="Question", now=NOW)
        store.mark_answered(
            "source", answer="Answer", sections=("Answer",), response_id=None, now=NOW
        )
        _, digest, _ = store.prepare_publication(
            "source",
            content="payload",
            title="AI READY",
            output_mode="rolling",
            now=NOW,
        )
        store.increment_attempt("source", "publication", now=NOW)
        store.mark_publication_ambiguous(
            "source", gate="create_ambiguous", error="unknown", now=NOW
        )

        with pytest.raises(ValueError, match="do not match"):
            store.resolve_ambiguous_create(
                source_note_id="source",
                expected_publication_sha256="f" * 64,
                expected_output_epoch=0,
                decision="not_applied",
                verification=None,
                now=NOW + timedelta(minutes=1),
            )

        assert digest != "f" * 64
        assert store.output_gate() == "create_ambiguous"
        assert store.get_job("source").state is JobState.FAILED
        assert store.connection.execute(
            "SELECT COUNT(*) FROM publication_recovery_audit"
        ).fetchone()[0] == 0


def test_v4_active_create_gate_without_stored_mode_fails_closed(tmp_path):
    database = tmp_path / "v4-active-gate.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(_SCHEMA_V1)
    connection.executescript(_SCHEMA_V2)
    connection.executescript(_SCHEMA_V3)
    connection.executescript(_SCHEMA_V4)
    encoded = NOW.isoformat(timespec="microseconds")
    payload = "legacy stored payload"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    connection.execute(
        """
        INSERT INTO jobs (
            note_id, queue_seq, state, source_kind, route_kind,
            transcript, answer, answer_sections, publication_content,
            publication_sha256, accepted_output_epoch, failure_stage,
            resume_state, retryable, publication_attempts,
            discovered_at, updated_at, completed_at
        ) VALUES (
            'legacy-source', 1, 'failed', 'audio', 'question',
            'Question', 'Answer', '["Answer"]', ?, ?, 0, 'publication',
            'answered', 0, 1, ?, ?, ?
        )
        """,
        (payload, digest, encoded, encoded, encoded),
    )
    connection.execute(
        """
        UPDATE bridge_state
           SET output_gate = 'create_ambiguous',
               output_gate_job_note_id = 'legacy-source',
               output_gate_at = ?, output_epoch = 0
         WHERE singleton = 1
        """,
        (encoded,),
    )
    connection.execute("PRAGMA user_version = 4")
    connection.commit()
    connection.close()

    with Store(database) as store:
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == _LATEST_SCHEMA_VERSION
        assert store.get_job("legacy-source").publication_output_mode is None
        with pytest.raises(RuntimeError, match="inconsistent"):
            store.ambiguous_create_candidate()
        with pytest.raises(ValueError, match="do not match"):
            store.resolve_ambiguous_create(
                source_note_id="legacy-source",
                expected_publication_sha256=digest,
                expected_output_epoch=0,
                decision="not_applied",
                verification=None,
                now=NOW + timedelta(minutes=1),
            )
        assert store.output_gate() == "create_ambiguous"
        assert store.connection.execute(
            "SELECT COUNT(*) FROM publication_recovery_audit"
        ).fetchone()[0] == 0


def test_remember_output_note_never_reclassifies_an_input_identity(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([audio_note("unrelated")], now=NOW)
        store.enqueue_audio(audio_note("source", minute=1), b"audio", now=NOW)
        store.mark_question_transcribed("source", question="Question", now=NOW)
        store.mark_answered(
            "source", answer="Answer", sections=("Answer",), response_id=None, now=NOW
        )
        store.prepare_publication(
            "source",
            content="payload",
            title="AI READY",
            output_mode="per_turn",
            now=NOW,
        )
        store.increment_attempt("source", "publication", now=NOW)

        with pytest.raises(ValueError, match="conflicts"):
            store.remember_output_note(
                "unrelated",
                source_note_id="source",
                output_epoch=0,
                publication_attempt=1,
                content_sha256=None,
                now=NOW,
            )

        ledger = store.connection.execute(
            "SELECT disposition, source_kind FROM request_ledger "
            "WHERE note_id = 'unrelated'"
        ).fetchone()
        assert tuple(ledger) == ("baseline", "audio")
        assert store.connection.execute(
            "SELECT COUNT(*) FROM provisional_output_identities"
        ).fetchone()[0] == 0


def test_retained_prior_output_id_cannot_bind_to_new_same_payload_attempt(tmp_path):
    database = tmp_path / "bridge.sqlite3"
    payload = "same publication payload"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    with Store(database) as store:
        store.establish_baseline([], now=NOW)
        store.enqueue_audio(audio_note("old-source"), b"old", now=NOW)
        old_conversation = store.mark_question_transcribed(
            "old-source", question="Old", now=NOW
        )
        store.mark_answered(
            "old-source", answer="Answer", sections=("Answer",), response_id=None, now=NOW
        )
        store.prepare_publication(
            "old-source",
            content=payload,
            title="AI READY",
            output_mode="per_turn",
            now=NOW,
        )
        store.increment_attempt("old-source", "publication", now=NOW)
        store.remember_output_note(
            "retained-output",
            source_note_id="old-source",
            output_epoch=0,
            publication_attempt=1,
            content_sha256=digest,
            now=NOW,
        )
        store.mark_published(
            "old-source", output_note_id="retained-output", now=NOW
        )
        store.connection.execute(
            "UPDATE conversations SET closed_at = ? WHERE id = ?",
            ((NOW + timedelta(minutes=1)).isoformat(), old_conversation),
        )
        store.connection.execute(
            "UPDATE bridge_state SET active_conversation_id = NULL "
            "WHERE singleton = 1"
        )
        assert store.delete_expired_metadata(
            completed_before=NOW + timedelta(minutes=2)
        ) == 1
        assert store.get_job("old-source") is None
        assert store.is_known_output_note("retained-output")

        store.enqueue_audio(audio_note("new-source", minute=3), b"new", now=NOW)
        store.mark_question_transcribed("new-source", question="New", now=NOW)
        store.mark_answered(
            "new-source", answer="Answer", sections=("Answer",), response_id=None, now=NOW
        )
        store.prepare_publication(
            "new-source",
            content=payload,
            title="AI READY",
            output_mode="per_turn",
            now=NOW,
        )
        store.increment_attempt("new-source", "publication", now=NOW)
        store.mark_publication_ambiguous(
            "new-source", gate="create_ambiguous", error="unknown", now=NOW
        )

        with pytest.raises(ValueError, match="gated publication"):
            store.resolve_ambiguous_create(
                source_note_id="new-source",
                expected_publication_sha256=digest,
                expected_output_epoch=0,
                decision="applied",
                verification=applied_verification("retained-output", digest),
                now=NOW + timedelta(minutes=4),
            )
        assert store.output_gate() == "create_ambiguous"

        result = store.resolve_ambiguous_create(
            source_note_id="new-source",
            expected_publication_sha256=digest,
            expected_output_epoch=0,
            decision="applied",
            verification=applied_verification("new-untracked-output", digest),
            now=NOW + timedelta(minutes=4),
        )
        assert result.output_mode == "per_turn"
        assert store.output_snapshot().note_id is None
        assert store.get_job("new-source").output_note_id == "new-untracked-output"


def test_oversized_rejection_never_persists_audio_blob(tmp_path):
    content = b"too-large-private-recording"
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        outcome = store.reject_oversized_audio(
            audio_note("large"),
            audio_sha256=hashlib.sha256(content).hexdigest(),
            now=NOW,
        )
        assert outcome == EnqueueOutcome.REJECTED
        job = store.get_job("large")
        assert job is not None
        assert job.state == JobState.FAILED
        assert job.failure_stage == "intake"
        assert job.audio is None
        assert store.has_seen_request("large")
        assert store.connection.execute(
            "SELECT length(audio_blob) FROM jobs WHERE note_id = 'large'"
        ).fetchone()[0] is None


def test_cleared_audio_is_removed_from_live_database_and_wal(tmp_path):
    database = tmp_path / "bridge.sqlite3"
    marker = b"private-audio-marker-1c69f87e" * 500
    with Store(database) as store:
        store.establish_baseline([], now=NOW)
        assert (
            store.enqueue_audio(audio_note("private"), marker, now=NOW)
            == EnqueueOutcome.ENQUEUED
        )
        store.mark_ignored("private", reason="prefix_missing", now=NOW)
        store.checkpoint()

        assert store.get_job("private").audio is None  # type: ignore[union-attr]
        on_disk = b"".join(
            path.read_bytes()
            for path in tmp_path.glob("bridge.sqlite3*")
            if path.is_file()
        )
        assert marker not in on_disk


def test_cleared_unaccepted_text_is_removed_from_live_database_and_wal(tmp_path):
    database = tmp_path / "bridge.sqlite3"
    marker = b"private-text-marker-91f0f991" * 500
    with Store(database) as store:
        store.establish_baseline([], now=NOW)
        assert store.ignore_text_at_intake(
            text_note("private-text"),
            content_sha256=hashlib.sha256(marker).hexdigest(),
            now=NOW,
        ) is EnqueueOutcome.IGNORED
        store.checkpoint()

        job = store.get_job("private-text")
        assert job.audio is None and job.transcript is None
        assert job.audio_sha256 is None
        ledger = store.connection.execute(
            "SELECT audio_sha256, disposition FROM request_ledger "
            "WHERE note_id = 'private-text'"
        ).fetchone()
        assert tuple(ledger) == (None, "ignored")
        on_disk = b"".join(
            path.read_bytes()
            for path in tmp_path.glob("bridge.sqlite3*")
            if path.is_file()
        )
        assert marker not in on_disk


def test_v3_enrolled_store_requires_one_time_text_ingress_baseline(tmp_path):
    database = tmp_path / "v3-enrolled.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(_SCHEMA_V1)
    connection.executescript(_SCHEMA_V2)
    connection.executescript(_SCHEMA_V3)
    encoded = NOW.isoformat(timespec="microseconds")
    connection.execute(
        "UPDATE bridge_state SET enrolled_at = ? WHERE singleton = 1",
        (encoded,),
    )
    connection.execute("PRAGMA user_version = 3")
    connection.commit()
    connection.close()

    with Store(database) as store:
        assert store.is_enrolled()
        assert not store.is_text_ingress_enrolled()
        assert store.establish_text_ingress_baseline(
            [text_note("old-text")], now=NOW
        ) == 1
        assert store.is_text_ingress_enrolled()
        old = store.get_job("old-text")
        assert old is not None and old.state is JobState.IGNORED
        assert old.audio is None and old.audio_sha256 is None
        assert store.establish_text_ingress_baseline(
            [text_note("later-text")], now=NOW
        ) == 0


def test_retiring_output_advances_generation(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        initial = store.output_snapshot()
        assert initial.epoch == 0 and initial.note_id is None
        assert store.register_created_output(
            expected_epoch=0,
            note_id="reply",
            content_sha256="a" * 64,
            now=NOW,
        )
        assert store.retire_output("reply", reason="deleted", now=NOW)
        retired = store.output_snapshot()
        assert retired.epoch == 1 and retired.note_id is None
        assert not store.register_created_output(
            expected_epoch=0,
            note_id="late",
            content_sha256="b" * 64,
            now=NOW,
        )


def test_ambiguous_update_old_readbacks_are_bounded_and_do_not_clear_gate(tmp_path):
    database = tmp_path / "bridge.sqlite3"
    old_digest = hashlib.sha256(b"old reply").hexdigest()
    with Store(database) as store:
        store.establish_baseline([], now=NOW)
        store.enqueue_audio(audio_note("question"), b"audio", now=NOW)
        assert store.get_job("question").accepted_output_note_id is None
        assert store.register_created_output(
            expected_epoch=0,
            note_id="reply",
            content_sha256=old_digest,
            now=NOW,
        )
        store.mark_question_transcribed("question", question="Question", now=NOW)
        store.mark_answered(
            "question",
            answer="Answer",
            sections=("Answer",),
            response_id="response",
            now=NOW,
        )
        _, desired_digest, _ = store.prepare_publication(
            "question",
            content="new reply",
            title="AI READY",
            output_mode="rolling",
            now=NOW,
        )
        assert store.increment_attempt("question", "publication", now=NOW) == 1
        store.mark_publication_ambiguous(
            "question",
            gate="update_ambiguous",
            error="update outcome unknown",
            now=NOW,
        )

        assert store.resolve_unapplied_publication(
            output_note_id="reply",
            observed_content_sha256=old_digest,
            maximum_readbacks=2,
            now=NOW + timedelta(seconds=1),
        ) == "pending"
        assert store.get_job("question").accepted_output_note_id == "reply"
        assert not store.update_readback_exhausted(maximum=2)
        assert store.resolve_unapplied_publication(
            output_note_id="reply",
            observed_content_sha256=old_digest,
            maximum_readbacks=2,
            now=NOW + timedelta(seconds=2),
        ) == "unknown"
        assert store.update_readback_exhausted(maximum=2)
        assert store.output_gate() == "update_ambiguous"
        failed = store.get_job("question")
        assert failed is not None and failed.state is JobState.FAILED
        assert failed.publication_attempts == 1
        assert failed.last_error == (
            "update outcome remains unknown after bounded read-back"
        )

        # Repeated callers cannot consume more attempts or reinterpret the old
        # value as proof that the one mutation did not apply.
        assert store.resolve_unapplied_publication(
            output_note_id="reply",
            observed_content_sha256=old_digest,
            maximum_readbacks=2,
            now=NOW + timedelta(seconds=3),
        ) == "unknown"
        assert store.connection.execute(
            "SELECT output_gate_readback_attempts FROM bridge_state"
        ).fetchone()[0] == 2

        invalid_verification = replace(
            applied_verification("reply", desired_digest),
            method=None,  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="does not match the active gate"):
            store.resolve_ambiguous_update_applied(
                source_note_id="question",
                expected_publication_sha256=desired_digest,
                expected_output_epoch=0,
                expected_output_note_id="reply",
                expected_previous_content_sha256=old_digest,
                expected_readback_attempts=2,
                verification=invalid_verification,
                now=NOW + timedelta(seconds=4),
            )
        assert store.output_gate() == "update_ambiguous"
        assert store.connection.execute(
            "SELECT COUNT(*) FROM update_recovery_audit"
        ).fetchone()[0] == 0
        assert store.latest_published_sequence() == 0

        assert store.reconcile_applied_publication(
            output_note_id="reply",
            observed_content_sha256=desired_digest,
            now=NOW + timedelta(seconds=5),
        ) == "question"
        assert store.get_job("question").state is JobState.PUBLISHED
        assert store.output_gate() is None
        assert not store.update_readback_exhausted(maximum=2)
        assert store.latest_published_sequence() == 1
        assert store.reconcile_applied_publication(
            output_note_id="reply",
            observed_content_sha256=desired_digest,
            now=NOW + timedelta(seconds=6),
        ) is None
        assert store.latest_published_sequence() == 1


def test_exhausted_old_value_quarantines_only_the_ambiguous_delivery(tmp_path):
    old_digest = hashlib.sha256(b"old reply").hexdigest()
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        store.configure_source_cleanup(enabled=True, now=NOW)
        store.enqueue_audio(audio_note("ambiguous"), b"audio", now=NOW)
        assert store.register_created_output(
            expected_epoch=0,
            note_id="reply",
            content_sha256=old_digest,
            now=NOW,
        )
        store.mark_question_transcribed(
            "ambiguous", question="Question", now=NOW
        )
        store.mark_answered(
            "ambiguous",
            answer="Stored answer",
            sections=("Stored answer",),
            response_id="response",
            now=NOW,
        )
        store.prepare_publication(
            "ambiguous",
            content="new reply",
            title="AI READY",
            output_mode="rolling",
            now=NOW,
        )
        assert store.bind_rolling_publication_target(
            "ambiguous",
            output_note_id="reply",
            expected_output_epoch=0,
            expected_content_sha256=old_digest,
            now=NOW,
        )
        assert store.increment_attempt("ambiguous", "publication", now=NOW) == 1
        store.mark_publication_ambiguous(
            "ambiguous",
            gate="update_ambiguous",
            error="update outcome unknown",
            now=NOW,
        )
        store.enqueue_audio(
            audio_note("later-one", minute=1),
            b"later audio one",
            now=NOW + timedelta(minutes=1),
        )
        store.enqueue_audio(
            audio_note("later-two", minute=2),
            b"later audio two",
            now=NOW + timedelta(minutes=2),
        )

        assert not store.abandon_stale_ambiguous_update(
            output_note_id="reply",
            observed_content_sha256=old_digest,
            maximum_readbacks=2,
            now=NOW + timedelta(seconds=1),
        )
        assert store.resolve_unapplied_publication(
            output_note_id="reply",
            observed_content_sha256=old_digest,
            maximum_readbacks=2,
            now=NOW + timedelta(seconds=1),
        ) == "pending"
        assert store.resolve_unapplied_publication(
            output_note_id="reply",
            observed_content_sha256=old_digest,
            maximum_readbacks=2,
            now=NOW + timedelta(seconds=2),
        ) == "unknown"
        store.connection.execute(
            "UPDATE jobs SET publication_attempts = 2 "
            "WHERE note_id = 'ambiguous'"
        )
        assert not store.abandon_stale_ambiguous_update(
            output_note_id="reply",
            observed_content_sha256=old_digest,
            maximum_readbacks=2,
            now=NOW + timedelta(seconds=3),
        )
        assert store.output_gate() == "update_ambiguous"
        store.connection.execute(
            "UPDATE jobs SET publication_attempts = 1 "
            "WHERE note_id = 'ambiguous'"
        )
        assert not store.abandon_stale_ambiguous_update(
            output_note_id="reply",
            observed_content_sha256="f" * 64,
            maximum_readbacks=2,
            now=NOW + timedelta(seconds=4),
        )
        assert store.abandon_stale_ambiguous_update(
            output_note_id="reply",
            observed_content_sha256=old_digest,
            maximum_readbacks=2,
            now=NOW + timedelta(seconds=5),
        )

        ambiguous = store.get_job("ambiguous")
        assert ambiguous is not None and ambiguous.state is JobState.FAILED
        assert ambiguous.answer == "Stored answer"
        assert ambiguous.response_id == "response"
        assert ambiguous.publication_attempts == 1
        assert ambiguous.output_note_id is None
        assert ambiguous.delivered_sections == 0
        assert ambiguous.last_error == (
            "uncertain update delivery abandoned after bounded non-desired read-back"
        )
        assert store.latest_published_sequence() == 0
        assert store.connection.execute(
            "SELECT notifier_event_sequence FROM jobs "
            "WHERE note_id = 'ambiguous'"
        ).fetchone()["notifier_event_sequence"] is None
        assert store.source_cleanup("ambiguous") is None
        assert store.source_cleanup_status()["enabled"] is True
        assert store.source_cleanup_status()["pending"] == 0
        assert store.output_gate() is None
        assert store.output_snapshot().epoch == 1
        assert store.output_snapshot().note_id is None
        assert store.is_known_output_note("reply")
        surface = store.connection.execute(
            "SELECT retired_reason FROM output_surfaces WHERE note_id = 'reply'"
        ).fetchone()
        assert surface["retired_reason"] == "ambiguous_old_value"
        later_one = store.get_job("later-one")
        later_two = store.get_job("later-two")
        assert later_one is not None and later_one.state is JobState.QUEUED
        assert later_two is not None and later_two.state is JobState.QUEUED
        assert later_one.accepted_output_epoch == 1
        assert later_two.accepted_output_epoch == 1
        assert later_one.accepted_output_note_id is None
        assert later_two.accepted_output_note_id is None
        assert ambiguous.conversation_id is not None
        assert store.conversation_turns(
            ambiguous.conversation_id,
            before_queue_seq=later_two.queue_seq + 1,
        ) == []
        assert store.mark_continue(
            "later-one",
            now=NOW + timedelta(minutes=3),
            exhausted_answer="No prior delivered answer.",
        ) == "No prior delivered answer."
        continued = store.get_job("later-one")
        assert continued is not None
        assert continued.source_job_note_id is None
        assert continued.source_section_index is None


def test_assistant_gate_preserves_and_explicitly_resumes_oldest_turn(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        store.enqueue_text(
            text_note("question"), b"Assistant, private question", now=NOW
        )
        store.mark_question_transcribed(
            "question", question="private question", now=NOW
        )

        store.pause_assistant(
            "question",
            gate="authorization_required",
            error="authorization required",
            now=NOW,
        )

        assert store.assistant_gate() == "authorization_required"
        assert store.assistant_gate_status() == {
            "gate": "authorization_required",
            "source_note_id": "question",
            "gated_at": NOW,
        }
        paused = store.get_job("question")
        assert paused is not None
        assert paused.state is JobState.FAILED
        assert paused.completed_at is None
        assert store.pending_job_count() == 1
        assert store.oldest_actionable(now=NOW) is None

        assert store.clear_assistant_gate(
            expected_gate="authorization_required",
            expected_note_id="question",
            now=NOW + timedelta(minutes=1),
        )
        resumed = store.get_job("question")
        assert resumed is not None and resumed.state is JobState.TRANSCRIBED
        assert resumed.model_attempts == 0
        assert store.assistant_gate() is None
        assert store.oldest_actionable(now=NOW) is not None


def test_assistant_gate_recovery_requires_exact_identifiers(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        store.enqueue_audio(audio_note("question"), b"audio", now=NOW)
        store.mark_question_transcribed("question", question="Question", now=NOW)
        store.pause_assistant(
            "question",
            gate="credential_ambiguous",
            error="credential state uncertain",
            now=NOW,
        )

        with pytest.raises(ValueError, match="identifiers"):
            store.clear_assistant_gate(
                expected_gate="authorization_required",
                expected_note_id="question",
                now=NOW,
            )
        assert store.assistant_gate() == "credential_ambiguous"


def test_terminal_job_retry_requires_exact_safe_state_and_is_audited(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        store.enqueue_audio(audio_note("question"), b"private audio", now=NOW)
        assert store.increment_attempt("question", "transcription", now=NOW) == 1
        store.mark_failure(
            "question",
            stage="transcription",
            resume_state=JobState.QUEUED,
            error="temporary dependency exhausted",
            retryable=False,
            manual_retry_allowed=True,
            preserve_audio=True,
            now=NOW,
        )

        listed, total = store.list_job_recovery_candidates()
        assert total == 1 and listed[0].source_note_id == "question"
        assert store.manual_recovery_pending()
        assert store.pending_job_count() == 1
        assert store.delete_expired_metadata(
            completed_before=NOW + timedelta(days=365)
        ) == 0
        with pytest.raises(ValueError, match="identifiers"):
            store.retry_terminal_job(
                source_note_id="question",
                expected_stage="transcription",
                expected_attempts=2,
                now=NOW + timedelta(minutes=1),
            )

        result = store.retry_terminal_job(
            source_note_id="question",
            expected_stage="transcription",
            expected_attempts=1,
            now=NOW + timedelta(minutes=1),
        )

        assert result.audit_id == 1
        assert result.resumed_state is JobState.QUEUED
        restored = store.get_job("question")
        assert restored is not None
        assert restored.audio == b"private audio"
        assert restored.transcription_attempts == 0
        assert restored.manual_retry_allowed is False
        assert not store.manual_recovery_pending()
        audit = store.connection.execute(
            "SELECT source_note_id, failure_stage, prior_attempts, resumed_state "
            "FROM job_recovery_audit"
        ).fetchone()
        assert tuple(audit) == ("question", "transcription", 1, "queued")


def test_ignored_audio_recovery_is_exact_atomic_audited_and_idempotent(tmp_path):
    payload = b"private retained recording"
    digest = hashlib.sha256(payload).hexdigest()
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        original = audio_note("ignored-audio")
        store.enqueue_audio(original, payload, now=NOW)
        assert store.increment_attempt(
            "ignored-audio", "transcription", now=NOW
        ) == 1
        store.mark_ignored(
            "ignored-audio", reason="prefix_missing", now=NOW
        )
        before = store.get_job("ignored-audio")
        assert before is not None and before.state is JobState.IGNORED
        assert before.audio is None and before.transcription_attempts == 1

        candidate = store.ignored_audio_recovery_candidate("ignored-audio")
        assert candidate is not None
        assert candidate.retained_sha256 == digest
        assert candidate.transcription_attempts == 1

        output_digest = hashlib.sha256(b"prior output").hexdigest()
        assert store.register_created_output(
            expected_epoch=0,
            note_id="current-output",
            content_sha256=output_digest,
            now=NOW + timedelta(seconds=1),
        )
        refreshed = audio_note("ignored-audio", minute=2)
        first = store.restage_ignored_audio(
            refreshed,
            payload,
            expected_sha256=digest,
            expected_state="ignored",
            expected_source_kind="audio",
            expected_ignored_reason="prefix_missing",
            expected_transcription_attempts=1,
            expected_model_attempts=0,
            expected_publication_attempts=0,
            now=NOW + timedelta(minutes=2),
        )

        assert first.already_restaged is False
        assert first.audit_id == 1
        assert first.prior_transcription_attempts == 1
        assert first.queue_seq == before.queue_seq
        assert first.accepted_output_note_id == "current-output"
        restored = store.get_job("ignored-audio")
        assert restored is not None and restored.state is JobState.QUEUED
        assert restored.queue_seq == before.queue_seq
        assert restored.audio == payload
        assert restored.transcription_attempts == 0
        assert restored.model_attempts == 0
        assert restored.publication_attempts == 0
        assert restored.accepted_output_epoch == 0
        assert restored.accepted_output_note_id == "current-output"
        assert restored.output_note_id is None
        assert store.connection.execute(
            "SELECT ignored_reason FROM jobs WHERE note_id = ?",
            ("ignored-audio",),
        ).fetchone()[0] == "prefix_missing_recovery_queued"
        audit = store.connection.execute(
            "SELECT source_note_id, failure_stage, prior_attempts, resumed_state "
            "FROM job_recovery_audit"
        ).fetchone()
        assert tuple(audit) == (
            "ignored-audio",
            "transcription",
            1,
            "queued",
        )

        assert store.ignored_audio_recovery_already_restaged(
            "ignored-audio",
            expected_sha256=digest,
            expected_transcription_attempts=1,
            expected_model_attempts=0,
            expected_publication_attempts=0,
        )
        repeated = store.restage_ignored_audio(
            refreshed,
            payload,
            expected_sha256=digest,
            expected_state="ignored",
            expected_source_kind="audio",
            expected_ignored_reason="prefix_missing",
            expected_transcription_attempts=1,
            expected_model_attempts=0,
            expected_publication_attempts=0,
            now=NOW + timedelta(minutes=3),
        )
        assert repeated.already_restaged is True
        assert repeated.audit_id == first.audit_id
        assert repeated.restaged_at == first.restaged_at
        assert store.connection.execute(
            "SELECT COUNT(*) FROM job_recovery_audit"
        ).fetchone()[0] == 1

        store.mark_question_transcribed(
            "ignored-audio",
            question="Accepted question",
            now=NOW + timedelta(minutes=4),
        )
        assert store.connection.execute(
            "SELECT ignored_reason FROM jobs WHERE note_id = ?",
            ("ignored-audio",),
        ).fetchone()[0] is None


def test_ignored_audio_recovery_fails_closed_on_any_identity_mismatch(tmp_path):
    payload = b"private retained recording"
    digest = hashlib.sha256(payload).hexdigest()
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        note = audio_note("ignored-audio")
        store.enqueue_audio(note, payload, now=NOW)
        store.increment_attempt("ignored-audio", "transcription", now=NOW)
        store.mark_ignored(
            "ignored-audio", reason="prefix_missing", now=NOW
        )

        with pytest.raises(ValueError, match="identity"):
            store.restage_ignored_audio(
                note,
                payload,
                expected_sha256="f" * 64,
                expected_state="ignored",
                expected_source_kind="audio",
                expected_ignored_reason="prefix_missing",
                expected_transcription_attempts=1,
                expected_model_attempts=0,
                expected_publication_attempts=0,
                now=NOW + timedelta(minutes=1),
            )
        with pytest.raises(ValueError, match="expectations"):
            store.restage_ignored_audio(
                note,
                payload,
                expected_sha256=digest,
                expected_state="ignored",
                expected_source_kind="audio",
                expected_ignored_reason="prefix_missing",
                expected_transcription_attempts=0,
                expected_model_attempts=0,
                expected_publication_attempts=0,
                now=NOW + timedelta(minutes=1),
            )

        unchanged = store.get_job("ignored-audio")
        assert unchanged is not None and unchanged.state is JobState.IGNORED
        assert unchanged.audio is None
        assert unchanged.transcription_attempts == 1
        assert store.connection.execute(
            "SELECT COUNT(*) FROM job_recovery_audit"
        ).fetchone()[0] == 0


def test_ignored_audio_idempotence_rejects_generic_transcription_retry(tmp_path):
    payload = b"generic retry audio"
    digest = hashlib.sha256(payload).hexdigest()
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        note = audio_note("ordinary-retry")
        store.enqueue_audio(note, payload, now=NOW)
        store.increment_attempt("ordinary-retry", "transcription", now=NOW)
        store.mark_failure(
            "ordinary-retry",
            stage="transcription",
            resume_state=JobState.QUEUED,
            error="ordinary exhausted transcription",
            retryable=False,
            manual_retry_allowed=True,
            preserve_audio=True,
            now=NOW,
        )
        generic = store.retry_terminal_job(
            source_note_id="ordinary-retry",
            expected_stage="transcription",
            expected_attempts=1,
            now=NOW + timedelta(minutes=1),
        )
        assert generic.audit_id == 1
        retried = store.get_job("ordinary-retry")
        assert retried is not None and retried.state is JobState.QUEUED
        assert retried.audio == payload and retried.transcription_attempts == 0

        assert not store.ignored_audio_recovery_already_restaged(
            "ordinary-retry",
            expected_sha256=digest,
            expected_transcription_attempts=1,
            expected_model_attempts=0,
            expected_publication_attempts=0,
        )
        with pytest.raises(RuntimeError, match="restaged audio recovery state"):
            store.restage_ignored_audio(
                note,
                payload,
                expected_sha256=digest,
                expected_state="ignored",
                expected_source_kind="audio",
                expected_ignored_reason="prefix_missing",
                expected_transcription_attempts=1,
                expected_model_attempts=0,
                expected_publication_attempts=0,
                now=NOW + timedelta(minutes=2),
            )
        assert store.connection.execute(
            "SELECT COUNT(*) FROM job_recovery_audit"
        ).fetchone()[0] == 1


def test_accepted_control_routes_clear_ignored_audio_recovery_marker(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        for note_id in ("new", "continue"):
            store.enqueue_audio(audio_note(note_id), note_id.encode(), now=NOW)
            store.connection.execute(
                "UPDATE jobs SET ignored_reason = ? WHERE note_id = ?",
                ("prefix_missing_recovery_queued", note_id),
            )

        store.mark_new_conversation(
            "new", now=NOW + timedelta(minutes=1), answer="Started"
        )
        store.mark_continue(
            "continue",
            now=NOW + timedelta(minutes=2),
            exhausted_answer="No more",
        )
        rows = store.connection.execute(
            "SELECT note_id, ignored_reason FROM jobs ORDER BY note_id"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("continue", None),
            ("new", None),
        ]


def test_job_recovery_audit_is_removed_with_expired_job_metadata(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        store.enqueue_audio(audio_note("question"), b"audio", now=NOW)
        store.increment_attempt("question", "transcription", now=NOW)
        store.mark_failure(
            "question",
            stage="transcription",
            resume_state=JobState.QUEUED,
            error="temporary dependency exhausted",
            retryable=False,
            manual_retry_allowed=True,
            preserve_audio=True,
            now=NOW,
        )
        store.retry_terminal_job(
            source_note_id="question",
            expected_stage="transcription",
            expected_attempts=1,
            now=NOW + timedelta(minutes=1),
        )
        store.mark_ignored(
            "question", reason="test_completed", now=NOW + timedelta(minutes=2)
        )

        assert store.delete_expired_metadata(
            completed_before=NOW + timedelta(days=1)
        ) == 1
        assert store.connection.execute(
            "SELECT COUNT(*) FROM job_recovery_audit"
        ).fetchone()[0] == 0


def test_interrupted_model_attempt_installs_exact_ambiguity_gate(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW)
        store.enqueue_audio(audio_note("question"), b"audio", now=NOW)
        store.mark_question_transcribed("question", question="Question", now=NOW)
        assert store.begin_model_attempt("question", now=NOW) == 1

        assert store.gate_interrupted_model(now=NOW + timedelta(seconds=1))
        assert store.assistant_gate() == "inference_ambiguous"
        failed = store.get_job("question")
        assert failed is not None
        assert failed.state is JobState.FAILED
        assert failed.manual_retry_allowed is False
        assert failed.completed_at is None
        assert not store.gate_interrupted_model(now=NOW + timedelta(seconds=2))


def test_model_call_daily_limit_is_atomic_and_day_scoped(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        assert store.reserve_model_call(now=NOW, daily_limit=2)
        assert store.reserve_model_call(now=NOW, daily_limit=2)
        assert not store.reserve_model_call(now=NOW, daily_limit=2)
        assert store.model_calls_for_day(now=NOW) == 2

        tomorrow = NOW + timedelta(days=1)
        assert store.reserve_model_call(now=tomorrow, daily_limit=2)
        assert store.model_calls_for_day(now=tomorrow) == 1


def test_known_output_note_ids_include_active_retired_and_per_turn(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.register_created_output(
            expected_epoch=0,
            note_id="rolling",
            content_sha256="a" * 64,
            now=NOW,
        )
        assert store.retire_output("rolling", reason="deleted", now=NOW)
        assert store.known_output_note_ids() == ("rolling",)
