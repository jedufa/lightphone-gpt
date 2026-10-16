from datetime import datetime, timedelta, timezone

from lightphone_gpt.models import EnqueueOutcome, NoteKind, RemoteNote
from lightphone_gpt.retention import RetentionPolicy, apply_retention
from lightphone_gpt.store import Store


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def note(note_id: str) -> RemoteNote:
    return RemoteNote(note_id, NoteKind.AUDIO, "Recording", START, "audio/mp4")


def test_closed_content_expires_but_active_conversation_is_kept(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=START)
        store.enqueue_audio(note("old"), b"old", now=START)
        store.mark_question_transcribed("old", question="old question", now=START)
        store.mark_answered(
            "old",
            answer="old answer",
            sections=("old answer",),
            response_id="old-response",
            now=START,
        )
        store.mark_published("old", output_note_id="reply", now=START)

        later = START + timedelta(days=1)
        store.enqueue_audio(note("reset"), b"reset", now=later)
        store.mark_new_conversation(
            "reset", now=later, answer="New conversation started."
        )
        store.mark_published("reset", output_note_id="reply", now=later)

        report = apply_retention(
            store,
            now=START + timedelta(days=32),
            policy=RetentionPolicy(closed_conversation_days=30, metadata_days=90),
        )
        assert report.conversations_redacted == 1
        assert store.get_job("old").transcript is None
        assert store.get_job("old").answer is None
        assert store.get_job("reset").answer == "New conversation started."


def test_metadata_expiry_does_not_delete_active_conversation(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([note("baseline")], now=START)
        store.enqueue_audio(note("active"), b"active", now=START)
        store.mark_question_transcribed("active", question="question", now=START)
        store.mark_answered(
            "active", answer="answer", sections=("answer",), response_id=None, now=START
        )
        store.mark_published("active", output_note_id="reply", now=START)

        report = apply_retention(
            store,
            now=START + timedelta(days=91),
            policy=RetentionPolicy(closed_conversation_days=30, metadata_days=90),
        )
        assert report.jobs_deleted == 1  # baseline metadata only
        assert store.get_job("baseline") is None
        assert store.get_job("active") is not None
        assert store.has_seen_request("baseline")


def test_pruned_job_cannot_be_replayed_as_a_new_request(tmp_path):
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=START)
        old = note("old")
        assert store.enqueue_audio(old, b"private", now=START) == EnqueueOutcome.ENQUEUED
        store.mark_ignored("old", reason="prefix_missing", now=START)

        report = apply_retention(
            store,
            now=START + timedelta(days=91),
            policy=RetentionPolicy(closed_conversation_days=30, metadata_days=90),
        )
        assert report.jobs_deleted == 1
        assert store.get_job("old") is None
        assert store.has_seen_request("old")
        assert store.enqueue_audio(old, b"private", now=START) == EnqueueOutcome.DUPLICATE
        assert store.get_job("old") is None
