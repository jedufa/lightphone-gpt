from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import sqlite3

import pytest

from lightphone_gpt.chatgpt_errors import (
    ChatGPTAccountNotEligibleError,
    ChatGPTAmbiguousOutcomeError,
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTCredentialAmbiguousError,
    ChatGPTPlanUsageLimitError,
)
from lightphone_gpt.contracts import TransientDependencyError
from lightphone_gpt.models import (
    AppliedCreateVerification,
    AssistantResponse,
    DeletionReceipt,
    EnqueueOutcome,
    JobState,
    MutationOutcome,
    MutationReceipt,
    NoteKind,
    RemoteNote,
    RouteKind,
    SourceCleanupState,
)
from lightphone_gpt.service import database_status
from lightphone_gpt.store import Store
from lightphone_gpt.store import _SCHEMA_V1, _SCHEMA_V2, _SCHEMA_V3
from lightphone_gpt.worker import Worker, WorkerConfig


START = datetime(2026, 10, 15, 10, 0, tzinfo=timezone.utc)


def applied_verification(
    note_id: str,
    content_sha256: str,
    *,
    when: datetime,
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


class Clock:
    def __init__(self) -> None:
        self.value = START

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: int) -> None:
        self.value += timedelta(**kwargs)


class FakeProvider:
    def __init__(self) -> None:
        self.notes: dict[str, RemoteNote] = {}
        self.content: dict[str, bytes] = {}
        self.next_id = 1
        self.create_calls = 0
        self.list_calls = 0
        self.update_calls = 0
        self.download_calls = 0
        self.delete_source_calls = 0
        self.read_calls: list[str] = []
        self.create_outcome = MutationOutcome.VERIFIED
        self.update_outcome = MutationOutcome.VERIFIED
        self.crash_after_create = False
        self.crash_after_update = False
        self.unknown_after_create = False
        self.unknown_after_update = False
        self.delete_outcome = MutationOutcome.VERIFIED
        self.delete_attempted = True
        self.crash_after_delete = False
        self.get_failures = 0

    def add_audio(
        self,
        note_id: str,
        transcript: str,
        *,
        when: datetime,
        title: str = "Recording",
    ) -> None:
        self.notes[note_id] = RemoteNote(
            note_id, NoteKind.AUDIO, title, when, "audio/mp4"
        )
        self.content[note_id] = transcript.encode()

    def add_text(
        self,
        note_id: str,
        content: str,
        *,
        when: datetime,
        title: str = "Text note",
    ) -> None:
        self.notes[note_id] = RemoteNote(
            note_id,
            NoteKind.TEXT,
            title,
            when,
            "text/plain; charset=utf-8",
        )
        self.content[note_id] = content.encode()

    def rename(self, note_id: str, title: str, *, when: datetime) -> None:
        current = self.notes[note_id]
        self.notes[note_id] = RemoteNote(
            current.note_id,
            current.kind,
            title,
            when,
            current.mime_type,
        )

    def list_notes(self):
        self.list_calls += 1
        return list(self.notes.values())

    def get_note(self, note_id: str):
        if self.get_failures:
            self.get_failures -= 1
            raise TransientDependencyError("temporary metadata read failure")
        return self.notes.get(note_id)

    def read_note_content(self, note_id: str) -> bytes:
        self.read_calls.append(note_id)
        return self.content[note_id]

    def download_audio(self, note_id: str) -> bytes:
        self.download_calls += 1
        return self.content[note_id]

    def create_text_note(self, *, title: str, content: str) -> MutationReceipt:
        self.create_calls += 1
        if self.create_outcome != MutationOutcome.VERIFIED:
            return MutationReceipt(self.create_outcome, detail="injected outcome")
        note_id = f"reply-{self.next_id}"
        self.next_id += 1
        note = RemoteNote(note_id, NoteKind.TEXT, title, START)
        raw = content.encode()
        self.notes[note_id] = note
        self.content[note_id] = raw
        if self.unknown_after_create:
            self.unknown_after_create = False
            return MutationReceipt(
                MutationOutcome.UNKNOWN,
                note,
                hashlib.sha256(raw).hexdigest(),
                "injected unknown after create",
            )
        if self.crash_after_create:
            raise SystemExit("injected crash after create")
        return MutationReceipt(
            MutationOutcome.VERIFIED,
            note,
            hashlib.sha256(raw).hexdigest(),
        )

    def update_text_note(
        self,
        note_id: str,
        *,
        title: str,
        content: str,
        expected_title: str,
        expected_content_sha256: str,
    ) -> MutationReceipt:
        self.update_calls += 1
        if self.update_outcome != MutationOutcome.VERIFIED:
            return MutationReceipt(self.update_outcome, detail="injected outcome")
        if note_id not in self.notes:
            return MutationReceipt(MutationOutcome.CONFLICT, detail="missing")
        if self.notes[note_id].title != expected_title:
            return MutationReceipt(
                MutationOutcome.CONFLICT,
                self.notes[note_id],
                detail="title_changed",
            )
        observed = hashlib.sha256(self.content[note_id]).hexdigest()
        if observed != expected_content_sha256:
            return MutationReceipt(
                MutationOutcome.CONFLICT,
                self.notes[note_id],
                observed,
                "changed",
            )
        raw = content.encode()
        note = RemoteNote(note_id, NoteKind.TEXT, title, START)
        self.notes[note_id] = note
        self.content[note_id] = raw
        if self.unknown_after_update:
            self.unknown_after_update = False
            return MutationReceipt(
                MutationOutcome.UNKNOWN,
                note,
                hashlib.sha256(raw).hexdigest(),
                "injected unknown after update",
            )
        if self.crash_after_update:
            raise SystemExit("injected crash after update")
        return MutationReceipt(
            MutationOutcome.VERIFIED,
            note,
            hashlib.sha256(raw).hexdigest(),
        )

    def delete(self, note_id: str) -> None:
        self.notes.pop(note_id, None)
        self.content.pop(note_id, None)

    def delete_source_note(
        self,
        note_id: str,
        *,
        expected_kind: NoteKind,
        expected_content_sha256: str,
    ) -> DeletionReceipt:
        self.delete_source_calls += 1
        note = self.notes.get(note_id)
        if note is None:
            return DeletionReceipt(
                MutationOutcome.VERIFIED,
                deletion_attempted=False,
                detail="source_already_missing",
            )
        observed = hashlib.sha256(self.content[note_id]).hexdigest()
        if note.kind is not expected_kind or observed != expected_content_sha256:
            return DeletionReceipt(
                MutationOutcome.CONFLICT,
                deletion_attempted=False,
                note=note,
                observed_content_sha256=observed,
                detail="source_changed",
            )
        if self.delete_outcome is MutationOutcome.VERIFIED:
            self.delete(note_id)
        if self.crash_after_delete:
            self.delete(note_id)
            raise SystemExit("injected crash after source delete")
        return DeletionReceipt(
            self.delete_outcome,
            deletion_attempted=self.delete_attempted,
            note=None if self.delete_outcome is MutationOutcome.VERIFIED else note,
            observed_content_sha256=observed,
            detail="injected source delete outcome",
        )


class DecodeTranscriber:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio: bytes, *, mime_type: str | None = None) -> str:
        self.calls += 1
        return audio.decode()


class RaisingTranscriber:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def transcribe(self, audio: bytes, *, mime_type: str | None = None) -> str:
        del audio, mime_type
        self.calls += 1
        raise self.error


class FakeAssistant:
    def __init__(self, answer: str = "A useful answer.") -> None:
        self.answer = answer
        self.calls: list[list] = []
        self.failures = 0

    def respond(self, messages):
        self.calls.append(list(messages))
        if self.failures:
            self.failures -= 1
            raise TransientDependencyError("temporary model failure")
        return AssistantResponse(self.answer, f"response-{len(self.calls)}")


class RaisingAssistant(FakeAssistant):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def respond(self, messages):
        self.calls.append(list(messages))
        raise self.error


def make_worker(tmp_path, *, answer="A useful answer.", config=None):
    store = Store(tmp_path / "bridge.sqlite3")
    provider = FakeProvider()
    transcriber = DecodeTranscriber()
    assistant = FakeAssistant(answer)
    clock = Clock()
    worker = Worker(
        store=store,
        provider=provider,
        transcriber=transcriber,
        assistant=assistant,
        config=config or WorkerConfig(retry_delay=timedelta(seconds=1)),
        clock=clock,
    )
    assert worker.poll().baselined == 0
    return store, provider, transcriber, assistant, clock, worker


def ask(provider, clock, worker, note_id, words):
    clock.advance(minutes=1)
    provider.add_audio(note_id, f"Assistant, {words}", when=clock.value)
    assert worker.poll().enqueued == 1
    worker.drain()


def ask_text(provider, clock, worker, note_id, words):
    clock.advance(minutes=1)
    provider.add_text(note_id, f"Assistant, {words}", when=clock.value)
    assert worker.poll().enqueued == 1
    worker.drain()


def test_end_to_end_question_is_durable_and_published(tmp_path):
    store, provider, transcriber, assistant, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "What is next?")
    job = store.get_job("q1")
    assert job.state == JobState.PUBLISHED
    assert job.audio is None
    assert job.transcript == "What is next?"
    assert job.response_id == "response-1"
    assert provider.create_calls == 1
    assert b"YOU SAID\nWhat is next?" in provider.content[job.output_note_id]
    assert provider.notes[job.output_note_id].title == "AI READY 10:01"


def test_rolling_output_title_updates_to_each_local_completion_minute(tmp_path):
    store, provider, _transcriber, _assistant, clock, worker = make_worker(tmp_path)

    ask(provider, clock, worker, "q1", "First")
    first = store.get_job("q1")
    assert first is not None and first.output_note_id is not None
    output_id = first.output_note_id
    assert provider.notes[output_id].title == "AI READY 10:01"

    ask(provider, clock, worker, "q2", "Second")
    second = store.get_job("q2")
    assert second is not None and second.output_note_id == output_id
    assert provider.notes[output_id].title == "AI READY 10:02"


def test_output_title_uses_explicit_nondefault_timezone(tmp_path):
    settings = WorkerConfig(
        local_timezone="America/New_York",
        retry_delay=timedelta(seconds=1),
    )
    store, provider, _transcriber, _assistant, clock, worker = make_worker(
        tmp_path, config=settings
    )

    ask(provider, clock, worker, "q1", "Local time")
    job = store.get_job("q1")
    assert job is not None and job.output_note_id is not None
    assert provider.notes[job.output_note_id].title == "AI READY 06:01"


def test_safe_publication_precheck_retry_keeps_original_completion_title(tmp_path):
    store, provider, _transcriber, _assistant, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "question"
    assert worker.run_once().detail == "answered"
    provider.get_failures = 1
    assert worker.run_once().detail == "retry_scheduled"
    prepared = store.get_job("q2")
    assert prepared is not None
    assert prepared.publication_title_sha256 == hashlib.sha256(
        b"AI READY 10:02"
    ).hexdigest()

    clock.advance(minutes=10)
    assert worker.run_once().detail == "published"
    published = store.get_job("q2")
    assert published is not None and published.output_note_id is not None
    assert provider.notes[published.output_note_id].title == "AI READY 10:02"


def test_title_patch_loss_recovers_answer_from_content_with_old_valid_title(tmp_path):
    store, provider, _transcriber, _assistant, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")
    first = store.get_job("q1")
    assert first is not None and first.output_note_id is not None
    output_id = first.output_note_id
    previous_title = provider.notes[output_id].title

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "question"
    assert worker.run_once().detail == "answered"
    provider.unknown_after_update = True
    assert worker.run_once().detail == "unknown"
    provider.rename(output_id, previous_title, when=clock.value)

    assert worker.poll().output_retired == "recovered"
    recovered = store.get_job("q2")
    assert recovered is not None and recovered.state is JobState.PUBLISHED
    assert provider.notes[output_id].title == previous_title


def test_enabled_source_cleanup_deletes_exact_source_only_after_publication(tmp_path):
    settings = WorkerConfig(
        delete_source_after_publish=True,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, _transcriber, _assistant, clock, worker = make_worker(
        tmp_path, config=settings
    )
    clock.advance(minutes=1)
    provider.add_audio("source", "Assistant, clean this up", when=clock.value)
    assert worker.poll().enqueued == 1

    reports = worker.drain()
    assert reports[-1].detail == "source_cleanup_verification_pending"
    assert worker.poll().source_cleanup == "confirmed"

    job = store.get_job("source")
    cleanup = store.source_cleanup("source")
    assert job is not None and job.state is JobState.PUBLISHED
    assert cleanup is not None and cleanup.state is SourceCleanupState.CONFIRMED
    assert provider.delete_source_calls == 1
    assert "source" not in provider.notes
    assert job.output_note_id in provider.notes


def test_source_cleanup_policy_off_preserves_published_source(tmp_path):
    store, provider, _transcriber, _assistant, clock, worker = make_worker(tmp_path)

    ask(provider, clock, worker, "source", "keep the source")

    assert store.source_cleanup("source") is None
    assert provider.delete_source_calls == 0
    assert "source" in provider.notes


def test_enabled_source_cleanup_also_removes_exact_text_request(tmp_path):
    settings = WorkerConfig(
        delete_source_after_publish=True,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, transcriber, _assistant, clock, worker = make_worker(
        tmp_path, config=settings
    )
    clock.advance(minutes=1)
    provider.add_text("text-source", "Assistant, clean up text", when=clock.value)
    assert worker.poll().enqueued == 1

    worker.drain()
    assert worker.poll().source_cleanup == "confirmed"

    cleanup = store.source_cleanup("text-source")
    assert cleanup is not None and cleanup.state is SourceCleanupState.CONFIRMED
    assert cleanup.source_kind is NoteKind.TEXT
    assert "text-source" not in provider.notes
    assert transcriber.calls == 0


def test_source_cleanup_content_change_fails_closed_without_delete(tmp_path):
    settings = WorkerConfig(
        delete_source_after_publish=True,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, _transcriber, _assistant, clock, worker = make_worker(
        tmp_path, config=settings
    )
    clock.advance(minutes=1)
    provider.add_audio("source", "Assistant, original", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "question"
    assert worker.run_once().detail == "answered"
    assert worker.run_once().detail == "published"
    provider.content["source"] = b"changed after acceptance"

    report = worker.run_once()

    cleanup = store.source_cleanup("source")
    assert report.detail == "source_cleanup_conflict"
    assert cleanup is not None and cleanup.state is SourceCleanupState.CONFLICT
    assert provider.delete_source_calls == 1
    assert "source" in provider.notes


def test_crash_after_source_delete_is_reconciled_without_second_delete(tmp_path):
    settings = WorkerConfig(
        delete_source_after_publish=True,
        retry_delay=timedelta(seconds=1),
        confirm_source_deletion_after=2,
    )
    store, provider, transcriber, assistant, clock, worker = make_worker(
        tmp_path, config=settings
    )
    clock.advance(minutes=1)
    provider.add_audio("source", "Assistant, crash safely", when=clock.value)
    assert worker.poll().enqueued == 1
    assert [worker.run_once().detail for _ in range(3)] == [
        "question",
        "answered",
        "published",
    ]
    provider.crash_after_delete = True

    with pytest.raises(SystemExit, match="injected crash"):
        worker.run_once()

    attempted = store.source_cleanup("source")
    assert attempted is not None
    assert attempted.state is SourceCleanupState.ATTEMPTED
    assert attempted.delete_attempts == 1
    provider.crash_after_delete = False
    restarted = Worker(
        store=store,
        provider=provider,
        transcriber=transcriber,
        assistant=assistant,
        config=settings,
        clock=clock,
    )
    assert restarted.poll().source_cleanup == "missing"
    clock.advance(seconds=1)
    assert restarted.poll().source_cleanup == "confirmed"
    assert provider.delete_source_calls == 1
    confirmed = store.source_cleanup("source")
    assert confirmed is not None
    assert confirmed.state is SourceCleanupState.CONFIRMED


def test_unknown_source_delete_is_never_reissued_while_source_remains(tmp_path):
    settings = WorkerConfig(
        delete_source_after_publish=True,
        max_source_delete_readbacks=2,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, _transcriber, _assistant, clock, worker = make_worker(
        tmp_path, config=settings
    )
    clock.advance(minutes=1)
    provider.add_audio("source", "Assistant, ambiguous delete", when=clock.value)
    assert worker.poll().enqueued == 1
    provider.delete_outcome = MutationOutcome.UNKNOWN
    provider.delete_attempted = True
    worker.drain()

    assert provider.delete_source_calls == 1
    assert worker.poll().source_cleanup == "present"
    clock.advance(seconds=1)
    assert worker.poll().source_cleanup == "exhausted"
    clock.advance(seconds=1)
    assert worker.poll().source_cleanup == "exhausted"
    assert worker.drain() == []
    assert provider.delete_source_calls == 1
    cleanup = store.source_cleanup("source")
    assert cleanup is not None
    assert cleanup.state is SourceCleanupState.ATTEMPTED
    assert cleanup.delete_attempts == 1
    assert cleanup.readback_attempts == 2


def test_proven_precheck_failure_retries_before_any_delete(tmp_path):
    settings = WorkerConfig(
        delete_source_after_publish=True,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, _transcriber, _assistant, clock, worker = make_worker(
        tmp_path, config=settings
    )
    clock.advance(minutes=1)
    provider.add_audio("source", "Assistant, retry safe precheck", when=clock.value)
    assert worker.poll().enqueued == 1
    provider.delete_outcome = MutationOutcome.UNKNOWN
    provider.delete_attempted = False
    reports = worker.drain()
    assert reports[-1].detail == "source_cleanup_retry_scheduled"
    pending = store.source_cleanup("source")
    assert pending is not None and pending.state is SourceCleanupState.PENDING
    assert pending.delete_attempts == 0

    provider.delete_outcome = MutationOutcome.VERIFIED
    provider.delete_attempted = True
    clock.advance(seconds=1)
    assert worker.run_once().detail == "source_cleanup_verification_pending"
    assert worker.poll().source_cleanup == "confirmed"
    assert provider.delete_source_calls == 2
    assert store.source_cleanup("source").state is SourceCleanupState.CONFIRMED


def test_audio_only_spoken_vocative_is_reconstructed_and_published(tmp_path):
    store, provider, transcriber, assistant, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_audio(
        "spoken-vocative",
        "Do you, Assistant, remember the earlier answer?",
        when=clock.value,
    )

    assert worker.poll().enqueued == 1
    worker.drain()

    job = store.get_job("spoken-vocative")
    assert job.state is JobState.PUBLISHED
    assert job.transcript == "do you remember the earlier answer?"
    assert transcriber.calls == 1
    assert len(assistant.calls) == 1
    assert provider.create_calls == 1


def test_same_vocative_form_in_text_stays_strictly_ignored(tmp_path):
    store, provider, transcriber, assistant, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_text(
        "text-vocative",
        "Do you, Assistant, process this text?",
        when=clock.value,
    )

    report = worker.poll()

    assert report.enqueued == 0
    assert report.ignored == 1
    assert worker.drain() == []
    job = store.get_job("text-vocative")
    assert job.state is JobState.IGNORED
    assert transcriber.calls == 0
    assert not assistant.calls
    assert provider.create_calls == 0


def test_unprefixed_recording_is_forgotten_and_never_sent_to_model(tmp_path):
    store, provider, transcriber, assistant, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_audio("ordinary", "private ordinary note", when=clock.value)
    worker.poll()
    worker.drain()
    job = store.get_job("ordinary")
    assert job.state == JobState.IGNORED
    assert job.audio is None and job.transcript is None
    assert not assistant.calls
    assert provider.create_calls == 0


def test_text_question_and_follow_up_skip_speech_transcription(tmp_path):
    store, provider, transcriber, assistant, clock, worker = make_worker(tmp_path)

    ask_text(provider, clock, worker, "text-1", "First text question")
    ask_text(provider, clock, worker, "text-2", "Follow up in text")

    assert transcriber.calls == 0
    assert store.get_job("text-1").source_kind is NoteKind.TEXT
    assert store.get_job("text-1").state is JobState.PUBLISHED
    assert store.get_job("text-2").state is JobState.PUBLISHED
    prompt = [(message.role, message.content) for message in assistant.calls[-1]]
    assert ("user", "First text question") in prompt
    assert ("assistant", "A useful answer.") in prompt
    assert ("user", "Follow up in text") in prompt


def test_text_continue_and_new_conversation_are_deterministic(tmp_path):
    long_answer = " ".join(f"item{i}" for i in range(250))
    store, provider, transcriber, assistant, clock, worker = make_worker(
        tmp_path, answer=long_answer
    )
    ask_text(provider, clock, worker, "text-1", "Give me the list")
    original = store.get_job("text-1")
    assert len(original.answer_sections) >= 2
    model_calls = len(assistant.calls)

    ask_text(provider, clock, worker, "text-continue", "continue")
    assert len(assistant.calls) == model_calls
    assert store.get_job("text-continue").answer == original.answer_sections[1]

    ask_text(provider, clock, worker, "text-reset", "new conversation")
    assert len(assistant.calls) == model_calls
    ask_text(provider, clock, worker, "text-fresh", "Fresh question")
    latest_prompt = [message.content for message in assistant.calls[-1]]
    assert "Give me the list" not in latest_prompt
    assert "Fresh question" in latest_prompt
    assert transcriber.calls == 0


def test_unprefixed_text_is_cleared_and_never_sent_to_assistant(tmp_path):
    store, provider, transcriber, assistant, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_text("ordinary-text", "private ordinary text", when=clock.value)

    report = worker.poll()
    assert report.enqueued == 0
    assert report.ignored == 1
    assert worker.drain() == []

    job = store.get_job("ordinary-text")
    assert job.state is JobState.IGNORED
    assert job.audio is None and job.transcript is None
    assert transcriber.calls == 0
    assert not assistant.calls
    assert provider.create_calls == 0


def test_editing_ignored_text_to_add_prefix_does_not_submit_same_id(tmp_path):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_text("ordinary-text", "private ordinary text", when=clock.value)
    assert worker.poll().ignored == 1
    reads = list(provider.read_calls)

    clock.advance(minutes=1)
    provider.add_text(
        "ordinary-text", "Assistant, now submit", when=clock.value
    )
    report = worker.poll()

    assert report.duplicates == 1
    assert provider.read_calls == reads
    assert store.get_job("ordinary-text").state is JobState.IGNORED
    assert assistant.calls == []


def test_text_duplicate_and_content_change_fail_closed(tmp_path):
    store, provider, transcriber, assistant, clock, worker = make_worker(tmp_path)
    ask_text(provider, clock, worker, "text-1", "Original question")
    original_reads = provider.read_calls.count("text-1")

    report = worker.poll()
    assert report.duplicates >= 1
    assert provider.read_calls.count("text-1") == original_reads

    clock.advance(minutes=1)
    provider.add_text("text-1", "Assistant, changed question", when=clock.value)
    report = worker.poll()
    assert report.integrity_errors == 1
    assert store.get_job("text-1").state is JobState.FAILED
    assert store.get_job("text-1").audio is None
    assert len(assistant.calls) == 1
    assert transcriber.calls == 0


def test_audio_and_text_share_updated_at_then_id_queue_order(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    later = clock.value + timedelta(minutes=2)
    earlier = clock.value + timedelta(minutes=1)
    provider.add_audio("z-later", "Assistant, later", when=later)
    provider.add_audio("b-same", "Assistant, second at same time", when=earlier)
    provider.add_text("a-same", "Assistant, first at same time", when=earlier)

    assert worker.poll().enqueued == 3
    assert [job.note_id for job in store.list_jobs()] == [
        "a-same",
        "b-same",
        "z-later",
    ]


def test_duplicate_titles_do_not_merge_distinct_audio_and_text_ids(tmp_path):
    store, provider, transcriber, assistant, clock, worker = make_worker(tmp_path)
    when = clock.value + timedelta(minutes=1)
    provider.add_text(
        "text-id", "Assistant, typed", when=when, title="Synthetic duplicate title"
    )
    provider.add_audio(
        "audio-id", "Assistant, spoken", when=when, title="Synthetic duplicate title"
    )

    assert worker.poll().enqueued == 2
    worker.drain()

    assert {job.note_id for job in store.list_jobs()} == {"audio-id", "text-id"}
    assert all(job.state is JobState.PUBLISHED for job in store.list_jobs())
    assert transcriber.calls == 1
    assert len(assistant.calls) == 2


def test_title_rename_keeps_same_note_identity_and_does_not_replay(tmp_path):
    store, provider, transcriber, assistant, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "stable-id", "Original")
    original_sequence = store.get_job("stable-id").queue_seq
    original_calls = len(assistant.calls)

    clock.advance(minutes=1)
    provider.rename("stable-id", "Completely different title", when=clock.value)
    report = worker.poll()

    assert report.duplicates >= 1
    assert store.get_job("stable-id").queue_seq == original_sequence
    assert len(assistant.calls) == original_calls
    assert transcriber.calls == 1


def test_ai_ready_title_alone_does_not_make_user_input_bridge_output(tmp_path):
    store, provider, transcriber, assistant, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_text(
        "user-owned-id",
        "Assistant, title collision",
        when=clock.value,
        title="AI READY",
    )

    assert worker.poll().enqueued == 1
    worker.drain()

    assert store.get_job("user-owned-id").state is JobState.PUBLISHED
    assert not store.is_known_output_note("user-owned-id")
    assert transcriber.calls == 0
    assert len(assistant.calls) == 1


def test_existing_text_and_reply_notes_are_baselined_without_reads(tmp_path):
    store = Store(tmp_path / "bridge.sqlite3")
    provider = FakeProvider()
    provider.add_text("old-text", "Assistant, do not replay", when=START)
    provider.add_text(
        "old-reply", "Assistant, still do not replay", when=START, title="AI READY"
    )
    provider.add_audio("old-audio", "Assistant, do not replay", when=START)
    transcriber = DecodeTranscriber()
    assistant = FakeAssistant()
    worker = Worker(
        store=store,
        provider=provider,
        transcriber=transcriber,
        assistant=assistant,
        clock=Clock(),
    )

    assert worker.poll().baselined == 3
    assert provider.read_calls == []
    assert provider.download_calls == 0
    assert all(job.state is JobState.IGNORED for job in store.list_jobs())
    assert worker.drain() == []
    assert not assistant.calls


def test_v3_enrolled_store_baselines_existing_text_before_content_reads(tmp_path):
    database = tmp_path / "v3.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(_SCHEMA_V1)
    connection.executescript(_SCHEMA_V2)
    connection.executescript(_SCHEMA_V3)
    connection.execute(
        "UPDATE bridge_state SET enrolled_at = ? WHERE singleton = 1",
        (START.isoformat(timespec="microseconds"),),
    )
    connection.execute("PRAGMA user_version = 3")
    connection.commit()
    connection.close()

    store = Store(database)
    provider = FakeProvider()
    provider.add_text(
        "old-text", "Assistant, do not replay", when=START, title="Old note"
    )
    transcriber = DecodeTranscriber()
    assistant = FakeAssistant()
    clock = Clock()
    worker = Worker(
        store=store,
        provider=provider,
        transcriber=transcriber,
        assistant=assistant,
        clock=clock,
    )

    report = worker.poll()
    assert report.baselined == 1
    assert provider.read_calls == []
    assert store.get_job("old-text").state is JobState.IGNORED
    assert worker.drain() == []
    assert assistant.calls == []

    clock.advance(minutes=1)
    provider.add_text("new-text", "Assistant, process me", when=clock.value)
    assert worker.poll().enqueued == 1
    worker.drain()
    assert store.get_job("new-text").state is JobState.PUBLISHED


def test_baselined_audio_identity_changing_to_text_is_quarantined(tmp_path):
    store = Store(tmp_path / "bridge.sqlite3")
    provider = FakeProvider()
    provider.add_audio("stable-id", "old audio", when=START)
    worker = Worker(
        store=store,
        provider=provider,
        transcriber=DecodeTranscriber(),
        assistant=FakeAssistant(),
        clock=Clock(),
    )
    assert worker.poll().baselined == 1

    provider.add_text(
        "stable-id",
        "Assistant, changed kind",
        when=START + timedelta(minutes=1),
    )
    report = worker.poll()

    assert report.integrity_errors == 1
    assert provider.read_calls == []
    changed = store.get_job("stable-id")
    assert changed is not None and changed.state is JobState.FAILED
    assert changed.failure_stage == "integrity"
    assert changed.audio is None
    assert worker.poll().duplicates == 1
    assert provider.read_calls == []


def test_baselined_text_identity_changing_to_audio_is_quarantined_once(tmp_path):
    store = Store(tmp_path / "bridge.sqlite3")
    provider = FakeProvider()
    provider.add_text("stable-id", "old text", when=START)
    worker = Worker(
        store=store,
        provider=provider,
        transcriber=DecodeTranscriber(),
        assistant=FakeAssistant(),
        clock=Clock(),
    )
    assert worker.poll().baselined == 1

    provider.add_audio(
        "stable-id",
        "Assistant, changed kind",
        when=START + timedelta(minutes=1),
    )
    assert worker.poll().integrity_errors == 1
    assert provider.download_calls == 0
    assert store.get_job("stable-id").state is JobState.FAILED

    assert worker.poll().duplicates == 1
    assert provider.download_calls == 0


def test_known_per_turn_output_is_never_admitted_as_text_input(tmp_path):
    config = WorkerConfig(
        output_mode="per_turn", retry_delay=timedelta(seconds=1)
    )
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    ask_text(provider, clock, worker, "text-1", "First question")
    output_note_id = store.get_job("text-1").output_note_id
    assert output_note_id is not None
    provider.read_calls.clear()

    clock.advance(minutes=1)
    provider.add_text("text-2", "Assistant, Second question", when=clock.value)
    assert worker.poll().enqueued == 1

    assert output_note_id not in provider.read_calls
    assert store.get_job(output_note_id) is None
    worker.drain()
    assert len(assistant.calls) == 2

    clock.advance(days=91)
    store.delete_expired_metadata(completed_before=clock.value)
    provider.read_calls.clear()
    worker.poll()
    assert output_note_id not in provider.read_calls
    assert store.is_known_output_note(output_note_id)


def test_continue_reveals_stored_section_without_model_call(tmp_path):
    long_answer = " ".join(f"item{i}" for i in range(250))
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, answer=long_answer
    )
    ask(provider, clock, worker, "q1", "Give me the list")
    original = store.get_job("q1")
    assert len(original.answer_sections) >= 2
    calls = len(assistant.calls)

    ask(provider, clock, worker, "continue", "continue")
    continued = store.get_job("continue")
    assert len(assistant.calls) == calls
    assert continued.answer == original.answer_sections[1]
    assert continued.state == JobState.PUBLISHED


def test_new_conversation_resets_model_context(tmp_path):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First question")
    ask(provider, clock, worker, "q2", "Follow up")
    second_roles = [(message.role, message.content) for message in assistant.calls[-1]]
    assert ("user", "First question") in second_roles
    assert ("assistant", "A useful answer.") in second_roles

    ask(provider, clock, worker, "reset", "new conversation")
    assert len(assistant.calls) == 2
    ask(provider, clock, worker, "q3", "Fresh question")
    third_text = [message.content for message in assistant.calls[-1]]
    assert "First question" not in third_text
    assert "Fresh question" in third_text


def test_undelivered_answer_is_excluded_after_reviewed_create_gate_clear(tmp_path):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    provider.create_outcome = MutationOutcome.UNKNOWN
    ask(provider, clock, worker, "q1", "Never delivered")
    assert store.get_job("q1").state == JobState.FAILED
    assert store.output_gate() == "create_ambiguous"
    candidate = store.ambiguous_create_candidate()
    assert candidate is not None
    store.resolve_ambiguous_create(
        source_note_id=candidate.source_note_id,
        expected_publication_sha256=candidate.publication_sha256,
        expected_output_epoch=candidate.output_epoch,
        decision="not_applied",
        verification=None,
        now=clock.value,
    )

    provider.create_outcome = MutationOutcome.VERIFIED
    ask(provider, clock, worker, "q2", "Visible question")
    prompt_text = [message.content for message in assistant.calls[-1]]
    assert "Never delivered" not in prompt_text
    assert "Visible question" in prompt_text


def test_verified_create_without_exact_readback_hash_is_not_published(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    create = provider.create_text_note

    def incomplete_create(*, title: str, content: str) -> MutationReceipt:
        receipt = create(title=title, content=content)
        return MutationReceipt(MutationOutcome.VERIFIED, note=receipt.note)

    provider.create_text_note = incomplete_create  # type: ignore[method-assign]
    ask(provider, clock, worker, "q1", "Question")

    job = store.get_job("q1")
    assert job is not None
    assert job.state == JobState.FAILED
    assert job.failure_stage == "publication"
    assert job.output_note_id is None


def test_malformed_create_receipt_identity_is_still_excluded_from_input(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    create = provider.create_text_note

    def malformed_create(*, title: str, content: str) -> MutationReceipt:
        receipt = create(title=title, content=content)
        assert receipt.note is not None
        malformed = RemoteNote(
            receipt.note.note_id,
            NoteKind.AUDIO,
            "unexpected title",
            receipt.note.updated_at,
            "audio/mp4",
        )
        provider.notes[malformed.note_id] = malformed
        return MutationReceipt(
            MutationOutcome.VERIFIED,
            note=malformed,
            observed_content_sha256=receipt.observed_content_sha256,
        )

    provider.create_text_note = malformed_create  # type: ignore[method-assign]
    ask(provider, clock, worker, "q1", "Question")

    returned_id = "reply-1"
    assert store.get_job("q1").state is JobState.FAILED
    assert store.output_gate() == "create_ambiguous"
    assert store.is_known_output_note(returned_id)
    list_calls = provider.list_calls
    report = worker.poll()
    assert report.output_gate == "create_ambiguous"
    assert provider.list_calls == list_calls + 1
    assert store.get_job(returned_id) is None


@pytest.mark.parametrize("invalid_return", [None, {"outcome": "verified"}])
def test_invalid_create_return_sets_gate_before_created_note_can_be_ingested(
    tmp_path, invalid_return
):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    create = provider.create_text_note

    def create_then_return_invalid(*, title: str, content: str):
        create(title=title, content=content)
        return invalid_return

    provider.create_text_note = create_then_return_invalid  # type: ignore[method-assign]
    ask(provider, clock, worker, "q1", "Question")

    assert store.get_job("q1").state is JobState.FAILED
    assert store.output_gate() == "create_ambiguous"
    assert "reply-1" in provider.notes
    list_calls = provider.list_calls
    report = worker.poll()
    assert report.error is None
    assert report.output_gate == "create_ambiguous"
    assert provider.list_calls == list_calls + 1
    assert store.get_job("reply-1") is None


@pytest.mark.parametrize("bad_note_id", [None, "", 17])
def test_verified_create_receipt_requires_nonempty_string_note_id(
    tmp_path, bad_note_id
):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    create = provider.create_text_note

    def create_with_bad_identity(*, title: str, content: str) -> MutationReceipt:
        receipt = create(title=title, content=content)
        assert receipt.note is not None
        malformed = RemoteNote(
            bad_note_id,  # type: ignore[arg-type]
            NoteKind.TEXT,
            title,
            receipt.note.updated_at,
            "text/plain",
        )
        return MutationReceipt(
            MutationOutcome.VERIFIED,
            malformed,
            receipt.observed_content_sha256,
        )

    provider.create_text_note = create_with_bad_identity  # type: ignore[method-assign]
    ask(provider, clock, worker, "q1", "Question")

    assert store.get_job("q1").state is JobState.FAILED
    assert store.output_gate() == "create_ambiguous"


def test_invalid_create_outcome_still_permanently_excludes_returned_identity(
    tmp_path,
):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    create = provider.create_text_note

    def create_with_invalid_outcome(*, title: str, content: str) -> MutationReceipt:
        receipt = create(title=title, content=content)
        assert receipt.note is not None
        return MutationReceipt(  # type: ignore[arg-type]
            "bogus",
            receipt.note,
            receipt.observed_content_sha256,
        )

    provider.create_text_note = create_with_invalid_outcome  # type: ignore[method-assign]
    ask(provider, clock, worker, "q1", "Question")

    assert store.get_job("q1").state is JobState.FAILED
    assert store.output_gate() == "create_ambiguous"
    assert store.is_known_output_note("reply-1")


def test_non_note_create_metadata_sets_gate_without_dereference(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    create = provider.create_text_note

    def create_with_non_note(*, title: str, content: str) -> MutationReceipt:
        receipt = create(title=title, content=content)
        return MutationReceipt(  # type: ignore[arg-type]
            MutationOutcome.VERIFIED,
            {"note_id": "reply-1"},
            receipt.observed_content_sha256,
        )

    provider.create_text_note = create_with_non_note  # type: ignore[method-assign]
    ask(provider, clock, worker, "q1", "Question")

    assert store.get_job("q1").state is JobState.FAILED
    assert store.output_gate() == "create_ambiguous"


def test_verified_update_without_exact_readback_hash_is_not_published(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")
    update = provider.update_text_note

    def incomplete_update(
        note_id: str,
        *,
        title: str,
        content: str,
        expected_content_sha256: str,
    ) -> MutationReceipt:
        receipt = update(
            note_id,
            title=title,
            content=content,
            expected_content_sha256=expected_content_sha256,
        )
        return MutationReceipt(MutationOutcome.VERIFIED, note=receipt.note)

    provider.update_text_note = incomplete_update  # type: ignore[method-assign]
    ask(provider, clock, worker, "q2", "Second")

    job = store.get_job("q2")
    assert job is not None
    assert job.state == JobState.FAILED
    assert job.failure_stage == "publication"
    assert job.output_note_id is None


def test_conflict_recovery_requires_exact_output_metadata(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")

    def misleading_conflict(
        note_id: str,
        *,
        title: str,
        content: str,
        expected_content_sha256: str,
    ) -> MutationReceipt:
        del title, expected_content_sha256
        return MutationReceipt(
            MutationOutcome.CONFLICT,
            note=RemoteNote(note_id, NoteKind.TEXT, "RENAMED", START),
            observed_content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            detail="injected misleading conflict",
        )

    provider.update_text_note = misleading_conflict  # type: ignore[method-assign]
    ask(provider, clock, worker, "q2", "Second")

    assert store.get_job("q2").state == JobState.FAILED
    assert store.output_gate() == "update_ambiguous"


def test_ambiguous_continue_consumes_section_only_after_verified_readback(tmp_path):
    long_answer = " ".join(f"item{i}" for i in range(250))
    store, provider, _, _, clock, worker = make_worker(tmp_path, answer=long_answer)
    ask(provider, clock, worker, "q1", "Give me the list")
    original = store.get_job("q1")
    expected = original.answer_sections[1]

    provider.update_outcome = MutationOutcome.UNKNOWN
    ask(provider, clock, worker, "c1", "continue")
    assert store.get_job("c1").state == JobState.FAILED
    original = store.get_job("q1")
    assert original.next_section_index == 1
    assert original.delivered_sections == 1

    gated = store.get_job("c1")
    assert gated.publication_content is not None
    provider.content[gated.accepted_output_note_id] = gated.publication_content.encode()
    assert worker.poll().output_retired == "recovered"
    assert store.get_job("c1").state is JobState.PUBLISHED
    original = store.get_job("q1")
    assert original.next_section_index == 2
    assert original.delivered_sections == 2
    assert gated.answer == expected


def test_oversized_audio_is_rejected_before_store_persistence(tmp_path):
    config = WorkerConfig(max_audio_bytes=8, retry_delay=timedelta(seconds=1))
    store, provider, transcriber, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    clock.advance(minutes=1)
    provider.add_audio("large", "Assistant, this is too large", when=clock.value)

    report = worker.poll()
    assert report.rejected == 1
    job = store.get_job("large")
    assert job.state == JobState.FAILED
    assert job.failure_stage == "intake"
    assert job.audio is None
    assert transcriber.calls == 0
    assert not assistant.calls


def test_oversized_text_is_rejected_before_store_persistence(tmp_path):
    config = WorkerConfig(max_text_bytes=8, retry_delay=timedelta(seconds=1))
    store, provider, transcriber, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    clock.advance(minutes=1)
    provider.add_text("large-text", "Assistant, this is too large", when=clock.value)

    report = worker.poll()

    assert report.rejected == 1
    job = store.get_job("large-text")
    assert job.state is JobState.FAILED
    assert job.failure_stage == "intake"
    assert job.audio is None and job.transcript is None
    assert job.audio_sha256 is None
    assert transcriber.calls == 0
    assert not assistant.calls


def test_invalid_text_is_rejected_without_content_or_digest(tmp_path):
    store, provider, transcriber, assistant, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.notes["invalid-text"] = RemoteNote(
        "invalid-text",
        NoteKind.TEXT,
        "Invalid text",
        clock.value,
        "text/plain",
    )
    provider.content["invalid-text"] = b"private-invalid-\xff-text"

    report = worker.poll()

    assert report.rejected == 1
    job = store.get_job("invalid-text")
    assert job is not None and job.state is JobState.FAILED
    assert job.audio is None and job.audio_sha256 is None
    assert transcriber.calls == 0
    assert assistant.calls == []


def test_request_ledger_avoids_redownload_after_job_pruning(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_audio("ordinary", "ordinary recording", when=clock.value)
    worker.poll()
    worker.drain()
    downloads = provider.download_calls
    clock.advance(days=91)
    assert store.delete_expired_metadata(completed_before=clock.value) == 1
    assert store.get_job("ordinary") is None

    report = worker.poll()
    assert report.duplicates == 1
    assert provider.download_calls == downloads
    assert store.get_job("ordinary") is None


def test_transient_model_failure_retries_exact_turn_and_publishes_once(tmp_path):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    assistant.failures = 1
    clock.advance(minutes=1)
    provider.add_audio("q1", "Assistant, retry me", when=clock.value)
    worker.poll()
    worker.run_once()  # transcription
    worker.run_once()  # first model attempt
    failed = store.get_job("q1")
    assert failed.state == JobState.FAILED and failed.retryable
    assert worker.run_once().worked is False
    clock.advance(seconds=1)
    worker.drain()
    assert store.get_job("q1").state == JobState.PUBLISHED
    assert len(assistant.calls) == 2
    assert provider.create_calls == 1


def test_transient_model_backoff_is_exponential_capped_and_blocks_overtaking(
    tmp_path,
):
    config = WorkerConfig(
        max_model_attempts=4,
        retry_delay=timedelta(seconds=10),
        retry_max_delay=timedelta(seconds=25),
    )
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    assistant.failures = 10
    clock.advance(minutes=1)
    provider.add_audio("q1", "Assistant, retry safely", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "question"

    expected_delays = (10, 20, 25)
    for attempt, seconds in enumerate(expected_delays, start=1):
        before = clock.value
        report = worker.run_once()
        assert report.detail == "retry_scheduled"
        failed = store.get_job("q1")
        assert failed is not None
        assert failed.model_attempts == attempt
        assert failed.next_retry_at == before + timedelta(seconds=seconds)
        assert failed.manual_retry_allowed is False
        assert worker.run_once().worked is False
        clock.advance(seconds=seconds)

    terminal = worker.run_once()
    assert terminal.detail == "failed"
    failed = store.get_job("q1")
    assert failed is not None
    assert failed.model_attempts == 4
    assert failed.retryable is False
    assert failed.manual_retry_allowed is True
    assert failed.transcript == "retry safely"
    assert failed.answer is None and failed.response_id is None

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, must wait", when=clock.value)
    assert worker.poll().enqueued == 1
    staged = worker.run_once()
    assert staged.worked is True
    assert staged.detail == "reset_probe_deferred"
    assert len(assistant.calls) == 4
    blocked = worker.run_once()
    assert blocked.worked is False
    assert blocked.detail == "job_recovery_required"
    assert store.get_job("q2").state is JobState.QUEUED
    assert store.get_job("q2").route_kind is RouteKind.QUESTION


def test_text_reset_recovers_manual_transcription_stall_after_restart(tmp_path):
    config = WorkerConfig(
        max_transcription_attempts=1,
    )
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    worker.transcriber = RaisingTranscriber(
        TransientDependencyError("persistent STT failure")
    )
    clock.advance(minutes=1)
    provider.add_audio("stalled", "Assistant, never decoded", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "failed"
    assert store.get_job("stalled").manual_retry_allowed is True

    clock.advance(minutes=1)
    provider.add_text("deferred", "Assistant, do not answer", when=clock.value)
    clock.advance(seconds=1)
    provider.add_text("reset", "Assistant, reset", when=clock.value)
    assert worker.poll().enqueued == 2
    assert worker.run_once().detail == "reset_probe_deferred"
    assert store.get_job("deferred").route_kind is RouteKind.QUESTION

    store.close()
    reopened = Store(tmp_path / "bridge.sqlite3")
    worker.store = reopened
    try:
        reports = worker.drain()
        assert any(report.detail == "reset" for report in reports)
        assert reopened.get_job("stalled").failure_stage == "superseded"
        assert reopened.get_job("stalled").audio is None
        assert reopened.get_job("deferred").failure_stage == "superseded"
        assert reopened.get_job("reset").state is JobState.PUBLISHED
        assert len(assistant.calls) == 0
        assert provider.create_calls == 1
    finally:
        reopened.close()


def test_audio_reset_recovers_manual_model_stall_after_restart(tmp_path):
    config = WorkerConfig(
        max_model_attempts=1,
    )
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    assistant.failures = 1
    clock.advance(minutes=1)
    provider.add_text("stalled", "Assistant, fail once", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "question"
    assert worker.run_once().detail == "failed"
    assert store.get_job("stalled").manual_retry_allowed is True

    clock.advance(minutes=1)
    provider.add_audio("reset", "Assistant, reset", when=clock.value)
    assert worker.poll().enqueued == 1
    store.close()
    reopened = Store(tmp_path / "bridge.sqlite3")
    worker.store = reopened
    try:
        worker.drain()
        assert reopened.get_job("stalled").failure_stage == "superseded"
        assert reopened.get_job("reset").state is JobState.PUBLISHED
        assert len(assistant.calls) == 1
        assert provider.create_calls == 1
    finally:
        reopened.close()


def test_text_reset_interrupts_future_retry_after_deferred_audio_and_restart(
    tmp_path,
):
    config = WorkerConfig(
        max_model_attempts=3,
        retry_delay=timedelta(hours=1),
        retry_max_delay=timedelta(hours=1),
    )
    store, provider, transcriber, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    assistant.failures = 1
    clock.advance(minutes=1)
    provider.add_audio("stalled", "Assistant, retry later", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "question"
    assert worker.run_once().detail == "retry_scheduled"
    retry = store.get_job("stalled")
    assert retry.retryable is True and retry.next_retry_at > clock.value

    clock.advance(minutes=1)
    provider.add_audio("deferred", "Assistant, do not answer", when=clock.value)
    clock.advance(seconds=1)
    provider.add_text("reset", "Assistant reset", when=clock.value)
    assert worker.poll().enqueued == 2
    assert worker.run_once().detail == "reset_probe_deferred"
    assert store.get_job("deferred").audio is None
    assert transcriber.calls == 2

    store.close()
    reopened = Store(tmp_path / "bridge.sqlite3")
    worker.store = reopened
    try:
        worker.drain()
        assert reopened.get_job("stalled").failure_stage == "superseded"
        assert reopened.get_job("deferred").failure_stage == "superseded"
        assert reopened.get_job("reset").state is JobState.PUBLISHED
        assert len(assistant.calls) == 1
        assert provider.create_calls == 1
    finally:
        reopened.close()


def test_future_retry_reserves_only_one_full_queue_reset_intake_slot(tmp_path):
    config = WorkerConfig(
        max_pending_jobs=1,
        retry_delay=timedelta(hours=1),
        retry_max_delay=timedelta(hours=1),
    )
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    assistant.failures = 1
    clock.advance(minutes=1)
    provider.add_audio("stalled", "Assistant, retry later", when=clock.value)
    assert worker.poll().enqueued == 1
    worker.run_once()
    assert worker.run_once().detail == "retry_scheduled"
    assert store.pending_job_count() == 1

    clock.advance(minutes=1)
    provider.add_audio("reset", "Assistant, reset", when=clock.value)
    clock.advance(seconds=1)
    provider.add_text("later", "Assistant, remain outside", when=clock.value)
    intake = worker.poll()
    assert intake.enqueued == 1
    assert intake.queue_limited is True
    assert store.get_job("reset") is not None
    assert store.get_job("later") is None
    assert store.pending_job_count() == 2
    repeated = worker.poll()
    assert repeated.enqueued == 0
    assert repeated.queue_limited is True
    assert store.pending_job_count() == 2

    worker.drain()
    assert store.get_job("reset").state is JobState.PUBLISHED
    assert len(assistant.calls) == 1


def test_full_queue_reset_remains_discoverable_after_ordinary_probe_and_restart(
    tmp_path,
):
    config = WorkerConfig(
        max_pending_jobs=1,
        retry_delay=timedelta(hours=1),
        retry_max_delay=timedelta(hours=1),
    )
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    assistant.failures = 1
    clock.advance(minutes=1)
    provider.add_audio("stalled", "Assistant, retry later", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "question"
    assert worker.run_once().detail == "retry_scheduled"

    clock.advance(minutes=1)
    provider.add_text("ordinary", "Assistant, keep this pending", when=clock.value)
    clock.advance(seconds=1)
    provider.add_audio("reset", "Assistant, reset", when=clock.value)
    first = worker.poll()
    assert first.enqueued == 1
    assert first.queue_limited is True
    assert store.get_job("ordinary") is not None
    assert store.get_job("reset") is None

    # Repeated polling cannot consume another slot until the existing probe has
    # been routed locally.
    repeated = worker.poll()
    assert repeated.enqueued == 0
    assert repeated.queue_limited is True
    assert store.pending_job_count() == 2
    assert worker.run_once().detail == "reset_probe_deferred"

    store.close()
    reopened = Store(tmp_path / "bridge.sqlite3")
    worker.store = reopened
    try:
        second = worker.poll()
        assert second.enqueued == 1
        assert reopened.get_job("reset") is not None
        assert reopened.pending_job_count() == 3
        worker.drain()
        assert reopened.get_job("stalled").failure_stage == "superseded"
        assert reopened.get_job("ordinary").failure_stage == "superseded"
        assert reopened.get_job("reset").state is JobState.PUBLISHED
        assert len(assistant.calls) == 1
    finally:
        reopened.close()


def test_full_queue_inference_reset_reaches_past_ordinary_probe_after_restart(
    tmp_path,
):
    config = WorkerConfig(max_pending_jobs=1)
    store, provider, _, _, clock, worker = make_worker(tmp_path, config=config)
    ambiguous = RaisingAssistant(ChatGPTAmbiguousOutcomeError("stream"))
    worker.assistant = ambiguous
    clock.advance(minutes=1)
    provider.add_audio("stalled", "Assistant, uncertain model", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "question"
    assert worker.run_once().detail == "assistant_paused"
    assert store.assistant_gate() == "inference_ambiguous"

    clock.advance(minutes=1)
    provider.add_text("ordinary", "Assistant, keep this pending", when=clock.value)
    clock.advance(seconds=1)
    provider.add_audio("reset", "Assistant, reset", when=clock.value)
    first = worker.poll()
    assert first.enqueued == 1
    assert first.queue_limited is True
    repeated = worker.poll()
    assert repeated.enqueued == 0
    assert repeated.queue_limited is True
    assert store.pending_job_count() == 2
    assert worker.run_once().detail == "question"
    assert store.get_job("ordinary").state is JobState.TRANSCRIBED

    store.close()
    reopened = Store(tmp_path / "bridge.sqlite3")
    worker.store = reopened
    try:
        second = worker.poll()
        assert second.enqueued == 1
        assert reopened.get_job("reset") is not None
        worker.drain()
        assert reopened.assistant_gate() is None
        assert reopened.get_job("stalled").state is JobState.FAILED
        assert reopened.get_job("ordinary").failure_stage == "superseded"
        assert reopened.get_job("reset").state is JobState.PUBLISHED
        assert len(ambiguous.calls) == 1
    finally:
        reopened.close()


@pytest.mark.parametrize("gate_kind", ["create_ambiguous", "update_ambiguous"])
def test_full_queue_output_reset_reaches_past_failed_audio_probes_after_restart(
    tmp_path, gate_kind,
):
    config = WorkerConfig(
        max_pending_jobs=1,
        retry_delay=timedelta(hours=1),
        retry_max_delay=timedelta(hours=1),
    )
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    if gate_kind == "update_ambiguous":
        ask(provider, clock, worker, "seed", "Create the rolling surface")
        provider.update_outcome = MutationOutcome.UNKNOWN
    else:
        provider.create_outcome = MutationOutcome.UNKNOWN
    ask(provider, clock, worker, "stalled", "Uncertain publication")
    assert store.output_gate() == gate_kind

    worker.transcriber = RaisingTranscriber(
        TransientDependencyError("temporary transcription failure")
    )
    clock.advance(minutes=1)
    provider.add_audio("probe-1", "Assistant, first probe", when=clock.value)
    clock.advance(seconds=1)
    provider.add_audio("probe-2", "Assistant, second probe", when=clock.value)
    clock.advance(seconds=1)
    provider.add_text("reset", "Assistant, reset", when=clock.value)

    first = worker.poll()
    assert first.enqueued == 1
    assert first.queue_limited is True
    assert worker.poll().enqueued == 0
    assert worker.run_once().detail == "retry_scheduled"

    second = worker.poll()
    assert second.enqueued == 1
    assert second.queue_limited is True
    assert worker.run_once().detail == "retry_scheduled"
    assert store.get_job("reset") is None

    store.close()
    reopened = Store(tmp_path / "bridge.sqlite3")
    worker.store = reopened
    try:
        third = worker.poll()
        assert third.enqueued == 1
        assert reopened.get_job("reset") is not None
        provider.create_outcome = MutationOutcome.VERIFIED
        worker.drain()
        assert reopened.output_gate() is None
        assert reopened.get_job("stalled").state is JobState.FAILED
        assert reopened.get_job("probe-1").failure_stage == "superseded"
        assert reopened.get_job("probe-2").failure_stage == "superseded"
        assert reopened.get_job("reset").state is JobState.PUBLISHED
        assert provider.create_calls == 2
        assert provider.update_calls == (
            1 if gate_kind == "update_ambiguous" else 0
        )
        assert len(assistant.calls) == (
            2 if gate_kind == "update_ambiguous" else 1
        )
    finally:
        reopened.close()


def test_local_reset_probe_defers_new_conversation_effect_until_order_resumes(
    tmp_path,
):
    config = WorkerConfig(
        retry_delay=timedelta(hours=1),
        retry_max_delay=timedelta(hours=1),
    )
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    ask(provider, clock, worker, "first", "Establish context")
    original_conversation = store.get_job("first").conversation_id
    assistant.failures = 1
    clock.advance(minutes=1)
    provider.add_audio("stalled", "Assistant, retry later", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "question"
    assert worker.run_once().detail == "retry_scheduled"

    clock.advance(minutes=1)
    provider.add_text(
        "deferred-new", "Assistant, new conversation", when=clock.value
    )
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "reset_probe_deferred"
    deferred = store.get_job("deferred-new")
    assert deferred.state is JobState.QUEUED
    assert deferred.route_kind is RouteKind.NEW_CONVERSATION
    assert store.connection.execute(
        "SELECT active_conversation_id FROM bridge_state"
    ).fetchone()[0] == original_conversation

    clock.advance(hours=1)
    worker.drain()
    assert store.get_job("stalled").state is JobState.PUBLISHED
    assert store.get_job("deferred-new").state is JobState.PUBLISHED
    assert store.connection.execute(
        "SELECT active_conversation_id FROM bridge_state"
    ).fetchone()[0] != original_conversation


def test_exhausted_transcription_preserves_exact_source_for_audited_retry(
    tmp_path,
):
    config = WorkerConfig(
        max_transcription_attempts=3,
        retry_delay=timedelta(seconds=1),
        retry_max_delay=timedelta(seconds=2),
    )
    store, provider, _, _, clock, worker = make_worker(tmp_path, config=config)
    failing = RaisingTranscriber(TransientDependencyError("temporary STT outage"))
    worker.transcriber = failing
    clock.advance(minutes=1)
    provider.add_audio("q1", "Assistant, preserved source", when=clock.value)
    assert worker.poll().enqueued == 1

    assert worker.run_once().detail == "retry_scheduled"
    clock.advance(seconds=1)
    assert worker.run_once().detail == "retry_scheduled"
    clock.advance(seconds=2)
    assert worker.run_once().detail == "failed"

    failed = store.get_job("q1")
    assert failed is not None
    assert failed.audio == b"Assistant, preserved source"
    assert failed.transcript is None and failed.answer is None
    assert failed.manual_retry_allowed is True
    candidate = store.job_recovery_candidate("q1")
    assert candidate is not None
    assert candidate.failure_stage == "transcription"
    assert candidate.attempts == 3

    retried = store.retry_terminal_job(
        source_note_id="q1",
        expected_stage="transcription",
        expected_attempts=3,
        now=clock.value,
    )
    assert retried.prior_attempts == 3
    restored = store.get_job("q1")
    assert restored is not None
    assert restored.state is JobState.QUEUED
    assert restored.audio == b"Assistant, preserved source"
    assert restored.transcription_attempts == 0
    worker.transcriber = DecodeTranscriber()
    worker.drain()
    assert store.get_job("q1").state is JobState.PUBLISHED


def test_interrupted_completed_model_response_is_gated_before_any_repeat(
    tmp_path, monkeypatch
):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_audio("q1", "Assistant, call once", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "question"

    def process_death(*_args, **_kwargs):
        raise SystemExit("injected death after model response")

    with monkeypatch.context() as patch:
        patch.setattr(store, "mark_answered", process_death)
        with pytest.raises(SystemExit, match="injected death"):
            worker.run_once()

    assert len(assistant.calls) == 1
    in_flight = store.get_job("q1")
    assert in_flight is not None and in_flight.state is JobState.TRANSCRIBED
    report = worker.run_once()
    assert report.worked is False and report.detail == "assistant_gate"
    assert store.assistant_gate() == "inference_ambiguous"
    assert len(assistant.calls) == 1


def test_unknown_create_is_terminal_and_never_blind_retried(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    provider.create_outcome = MutationOutcome.UNKNOWN
    ask(provider, clock, worker, "q1", "Do it")
    job = store.get_job("q1")
    assert job.state == JobState.FAILED
    assert not job.retryable
    assert store.output_gate() == "create_ambiguous"
    assert provider.create_calls == 1
    assert worker.run_once().worked is False
    assert provider.create_calls == 1

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Later", when=clock.value)
    list_calls = provider.list_calls
    report = worker.poll()
    assert report.output_gate == "create_ambiguous"
    assert report.enqueued == 1
    assert provider.list_calls == list_calls + 1
    provider.create_outcome = MutationOutcome.VERIFIED
    worker.drain()
    assert store.get_job("q2").state is JobState.PUBLISHED
    assert store.get_job("q1").state is JobState.FAILED
    assert store.output_gate() is None
    assert provider.create_calls == 2

    store.close()
    reopened = Store(tmp_path / "bridge.sqlite3")
    try:
        worker.store = reopened
        assert reopened.output_gate() is None
        assert worker.run_once().worked is False
        assert provider.create_calls == 2
    finally:
        reopened.close()


def test_unknown_after_create_allows_later_turn_without_repeating_unknown(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    provider.unknown_after_create = True
    ask(provider, clock, worker, "q1", "First")

    assert store.get_job("q1").state == JobState.FAILED
    assert store.output_gate() == "create_ambiguous"
    assert provider.create_calls == 1
    assert len([note for note in provider.notes.values() if note.kind is NoteKind.TEXT]) == 1

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    list_calls = provider.list_calls
    report = worker.poll()
    assert report.output_gate == "create_ambiguous"
    assert report.enqueued == 1
    assert provider.list_calls == list_calls + 1
    worker.drain()
    assert store.get_job("q2").state is JobState.PUBLISHED
    assert store.get_job("q1").state is JobState.FAILED
    assert store.output_gate() is None
    assert provider.create_calls == 2


def test_new_audio_request_supersedes_ambiguous_create_after_restart(tmp_path):
    settings = WorkerConfig(
        delete_source_after_publish=True,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, transcriber, assistant, clock, worker = make_worker(
        tmp_path, config=settings
    )
    provider.create_outcome = MutationOutcome.UNKNOWN
    ask(provider, clock, worker, "stalled", "First uncertain create")
    assert store.output_gate() == "create_ambiguous"

    clock.advance(minutes=1)
    provider.add_audio("ordinary", "background recording", when=clock.value)
    clock.advance(seconds=1)
    provider.add_audio("interrupter", "Assistant, Start again", when=clock.value)
    intake = worker.poll()
    assert intake.enqueued == 2
    association = store.connection.execute(
        "SELECT admitted_output_gate, admitted_output_gate_job_note_id "
        "FROM jobs WHERE note_id = 'interrupter'"
    ).fetchone()
    assert tuple(association) == ("create_ambiguous", "stalled")

    store.close()
    reopened = Store(tmp_path / "bridge.sqlite3")
    worker.store = reopened
    try:
        provider.create_outcome = MutationOutcome.VERIFIED
        reports = worker.drain()
        assert any(report.detail == "publication_superseded" for report in reports)
        assert reopened.output_gate() is None
        assert reopened.get_job("ordinary").state is JobState.IGNORED
        assert reopened.get_job("stalled").state is JobState.FAILED
        assert reopened.get_job("interrupter").state is JobState.PUBLISHED
        assert provider.create_calls == 2
        assert len(assistant.calls) == 2
        assert reopened.source_cleanup("stalled") is None
        assert reopened.connection.execute(
            "SELECT notifier_event_sequence FROM jobs WHERE note_id = 'stalled'"
        ).fetchone()[0] is None
    finally:
        reopened.close()


def test_only_request_admitted_under_exact_gate_can_interrupt(tmp_path):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    provider.create_outcome = MutationOutcome.UNKNOWN
    clock.advance(minutes=1)
    provider.add_audio("stalled", "Assistant, First", when=clock.value)
    clock.advance(seconds=1)
    provider.add_audio("same-batch", "Assistant, Too early", when=clock.value)
    assert worker.poll().enqueued == 2
    worker.drain()
    assert store.output_gate() == "create_ambiguous"
    assert store.get_job("same-batch").state is JobState.QUEUED
    assert store.connection.execute(
        "SELECT admitted_output_gate FROM jobs WHERE note_id = 'same-batch'"
    ).fetchone()[0] is None

    clock.advance(minutes=1)
    provider.add_text("newer", "Assistant, Use this request", when=clock.value)
    assert worker.poll().enqueued == 1
    provider.create_outcome = MutationOutcome.VERIFIED
    worker.drain()

    assert store.output_gate() is None
    same_batch = store.get_job("same-batch")
    assert same_batch.state is JobState.FAILED
    assert same_batch.failure_stage == "superseded"
    assert store.get_job("newer").state is JobState.PUBLISHED
    assert len(assistant.calls) == 2


def test_safe_gate_reserves_one_intake_slot_when_queue_is_full(tmp_path):
    settings = WorkerConfig(
        max_pending_jobs=1,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, _, _, clock, worker = make_worker(tmp_path, config=settings)
    provider.create_outcome = MutationOutcome.UNKNOWN
    clock.advance(minutes=1)
    provider.add_audio("stalled", "Assistant, First", when=clock.value)
    assert worker.poll().enqueued == 1
    backlog = RemoteNote(
        "pre-gate-backlog",
        NoteKind.AUDIO,
        "Recording",
        clock.value,
        "audio/mp4",
    )
    assert (
        store.enqueue_audio(
            backlog,
            b"Assistant, stale backlog",
            now=clock.value,
        )
        is EnqueueOutcome.ENQUEUED
    )
    worker.drain()
    assert store.output_gate() == "create_ambiguous"
    assert store.pending_job_count() == 1

    clock.advance(minutes=1)
    provider.add_audio("recovery", "Assistant, Recover", when=clock.value)
    intake = worker.poll()
    assert intake.enqueued == 1
    assert store.pending_job_count() == 2
    provider.create_outcome = MutationOutcome.VERIFIED
    worker.drain()
    assert store.get_job("pre-gate-backlog").failure_stage == "superseded"
    assert store.get_job("recovery").state is JobState.PUBLISHED


def test_new_text_request_supersedes_ambiguous_update_on_fresh_surface(tmp_path):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "first", "First answer")
    old_output = store.get_job("first").output_note_id

    provider.update_outcome = MutationOutcome.UNKNOWN
    ask(provider, clock, worker, "stalled", "Uncertain update")
    assert store.output_gate() == "update_ambiguous"
    assert provider.update_calls == 1

    clock.advance(minutes=1)
    provider.add_text("interrupter", "Assistant, Recover now", when=clock.value)
    assert worker.poll().enqueued == 1
    provider.update_outcome = MutationOutcome.VERIFIED
    worker.drain()

    assert store.output_gate() is None
    assert store.get_job("stalled").state is JobState.FAILED
    replacement = store.get_job("interrupter")
    assert replacement.state is JobState.PUBLISHED
    assert replacement.output_note_id != old_output
    assert store.is_known_output_note(old_output)
    assert old_output in provider.notes
    assert provider.update_calls == 1
    assert provider.create_calls == 2
    assert len(assistant.calls) == 3


def test_assistant_reset_abandons_uncertain_model_without_replaying_it(tmp_path):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    worker.assistant = RaisingAssistant(ChatGPTAmbiguousOutcomeError("stream"))
    clock.advance(minutes=1)
    provider.add_audio("stalled", "Assistant, First", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "question"
    assert worker.run_once().detail == "assistant_paused"
    assert store.assistant_gate() == "inference_ambiguous"

    clock.advance(minutes=1)
    provider.add_audio("pending", "Assistant, Do not run this", when=clock.value)
    clock.advance(seconds=1)
    provider.add_audio("reset", "Assistant, reset", when=clock.value)
    assert worker.poll().enqueued == 2
    worker.assistant = assistant
    reports = worker.drain()

    assert any(report.detail == "reset" for report in reports)
    assert store.assistant_gate() is None
    assert store.get_job("stalled").state is JobState.FAILED
    assert store.get_job("pending").state is JobState.FAILED
    assert store.get_job("pending").failure_stage == "superseded"
    reset = store.get_job("reset")
    assert reset.route_kind is RouteKind.RESET
    assert reset.state is JobState.PUBLISHED
    assert len(assistant.calls) == 0


def test_reset_cannot_clear_authorization_gate(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    worker.assistant = RaisingAssistant(ChatGPTAuthorizationRequiredError("renew"))
    clock.advance(minutes=1)
    provider.add_audio("stalled", "Assistant, First", when=clock.value)
    assert worker.poll().enqueued == 1
    worker.run_once()
    worker.run_once()
    assert store.assistant_gate() == "authorization_required"

    clock.advance(minutes=1)
    provider.add_audio("reset", "Assistant, reset", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "assistant_gate"
    assert store.assistant_gate() == "authorization_required"
    assert store.get_job("reset").state is JobState.QUEUED


@pytest.mark.parametrize(
    ("stored_mode", "runtime_mode", "expects_rolling_surface"),
    [
        pytest.param("rolling", "per_turn", True, id="rolling-to-per-turn"),
        pytest.param("per_turn", "rolling", False, id="per-turn-to-rolling"),
    ],
)
def test_stored_publication_mode_controls_recovery_across_config_drift(
    tmp_path,
    stored_mode,
    runtime_mode,
    expects_rolling_surface,
):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_audio("q1", "Assistant, Question", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().state is JobState.TRANSCRIBED
    assert worker.run_once().state is JobState.ANSWERED
    payload = "durably prepared publication"
    _, digest, persisted_mode = store.prepare_publication(
        "q1",
        content=payload,
        title="AI READY",
        output_mode=stored_mode,
        now=clock.value,
    )
    assert persisted_mode == stored_mode

    worker.config = WorkerConfig(
        output_mode=runtime_mode,
        retry_delay=timedelta(seconds=1),
    )
    provider.unknown_after_create = True
    report = worker.run_once()
    assert report.detail == "unknown"
    assert report.state is JobState.FAILED
    assert store.output_gate() == "create_ambiguous"

    candidate = store.ambiguous_create_candidate()
    assert candidate is not None
    assert candidate.output_mode == stored_mode
    assert candidate.publication_sha256 == digest
    assert candidate.provisional_output_note_id == "reply-1"
    assert candidate.provisional_content_sha256 == digest
    result = store.resolve_ambiguous_create(
        source_note_id="q1",
        expected_publication_sha256=digest,
        expected_output_epoch=0,
        decision="applied",
        verification=applied_verification("reply-1", digest, when=clock.value),
        now=clock.value,
    )
    assert result.output_mode == stored_mode
    assert (store.output_snapshot().note_id == "reply-1") is expects_rolling_surface
    audit_mode = store.connection.execute(
        "SELECT output_mode FROM publication_recovery_audit WHERE id = ?",
        (result.audit_id,),
    ).fetchone()[0]
    assert audit_mode == stored_mode


def test_create_gate_discards_non_assistant_text_without_persisting_content(tmp_path):
    database = tmp_path / "bridge.sqlite3"
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    provider.create_outcome = MutationOutcome.UNKNOWN
    ask(provider, clock, worker, "q1", "Create an ambiguous result")
    marker = "private-text-marker-e47f8d2a" * 300
    clock.advance(minutes=1)
    provider.add_text("private", marker, when=clock.value)

    list_calls = provider.list_calls
    report = worker.poll()
    assert report.output_gate == "create_ambiguous"
    assert provider.list_calls == list_calls + 1
    assert "private" in provider.read_calls
    private = store.get_job("private")
    assert private is not None and private.state is JobState.IGNORED
    assert marker.encode() not in database.read_bytes()

    candidate = store.ambiguous_create_candidate()
    assert candidate is not None
    store.resolve_ambiguous_create(
        source_note_id=candidate.source_note_id,
        expected_publication_sha256=candidate.publication_sha256,
        expected_output_epoch=candidate.output_epoch,
        decision="not_applied",
        verification=None,
        now=clock.value,
    )
    report = worker.poll()
    assert report.output_gate is None
    assert report.ignored == 0
    assert "private" in provider.read_calls
    private = store.get_job("private")
    assert private is not None and private.state is JobState.IGNORED
    assert private.audio is None and private.transcript is None
    assert private.audio_sha256 is None
    assert worker.drain() == []
    assert len(assistant.calls) == 1
    store.checkpoint()
    stored = b"".join(
        path.read_bytes()
        for path in tmp_path.glob("bridge.sqlite3*")
        if path.is_file()
    )
    assert marker.encode() not in stored


def test_same_poll_private_text_is_never_staged_before_later_create_gate(tmp_path):
    database = tmp_path / "bridge.sqlite3"
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    provider.create_outcome = MutationOutcome.UNKNOWN
    marker = "same-poll-private-marker-3cb8c781" * 300
    clock.advance(minutes=1)
    provider.add_audio("q1", "Assistant, trigger gate", when=clock.value)
    clock.advance(seconds=1)
    provider.add_text("private", marker, when=clock.value)

    report = worker.poll()
    assert report.enqueued == 1
    assert report.ignored == 1
    private = store.get_job("private")
    assert private is not None and private.state is JobState.IGNORED
    assert private.audio is None and private.transcript is None
    assert private.audio_sha256 is None
    worker.drain()
    assert store.output_gate() == "create_ambiguous"
    store.checkpoint()
    stored = b"".join(
        path.read_bytes()
        for path in tmp_path.glob("bridge.sqlite3*")
        if path.is_file()
    )
    assert marker.encode() not in stored


def test_unknown_update_blocks_later_turn_until_readback_recovers(tmp_path):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    clock.advance(minutes=1)
    provider.add_audio("q3", "Assistant, Third", when=clock.value)
    assert worker.poll().enqueued == 2
    provider.unknown_after_update = True

    worker.drain()
    assert store.get_job("q2").state == JobState.FAILED
    assert store.get_job("q3").state == JobState.QUEUED
    assert store.output_gate() == "update_ambiguous"
    assert len(assistant.calls) == 2
    assert provider.update_calls == 1

    report = worker.poll()
    assert report.output_retired == "recovered"
    assert store.output_gate() is None
    assert store.get_job("q2").state == JobState.PUBLISHED

    worker.drain()
    assert store.get_job("q3").state == JobState.PUBLISHED
    assert provider.update_calls == 2


def test_eventually_visible_update_stays_gated_then_recovers_without_rewrite(tmp_path):
    config = WorkerConfig(
        max_update_readback_attempts=3,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    ask(provider, clock, worker, "q1", "First")

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    clock.advance(minutes=1)
    provider.add_audio("q3", "Assistant, Third", when=clock.value)
    assert worker.poll().enqueued == 2
    provider.update_outcome = MutationOutcome.UNKNOWN
    worker.drain()

    assert store.output_gate() == "update_ambiguous"
    assert store.get_job("q2").state == JobState.FAILED
    assert store.get_job("q3").state == JobState.QUEUED
    assert provider.update_calls == 1
    assert worker.poll().output_retired == "pending"
    assert worker.poll().output_retired == "pending"
    assert store.output_gate() == "update_ambiguous"
    assert store.get_job("q3").state == JobState.QUEUED
    assert provider.update_calls == 1

    gated = store.get_job("q2")
    assert gated.publication_content is not None
    provider.content[gated.accepted_output_note_id] = gated.publication_content.encode()
    report = worker.poll()
    assert report.output_retired == "recovered"
    assert store.output_gate() is None
    assert store.get_job("q2").state == JobState.PUBLISHED
    assert provider.update_calls == 1

    provider.update_outcome = MutationOutcome.VERIFIED
    worker.drain()
    assert store.get_job("q3").state == JobState.PUBLISHED
    assert provider.update_calls == 2
    assert len(assistant.calls) == 3


def test_alternating_missing_and_old_reads_share_bounded_recovery_window(tmp_path):
    config = WorkerConfig(
        max_update_readback_attempts=3,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    ask(provider, clock, worker, "q1", "First")
    first_output = store.get_job("q1").output_note_id
    old_note = provider.notes[first_output]
    old_content = provider.content[first_output]

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    assert worker.poll().enqueued == 1
    provider.update_outcome = MutationOutcome.UNKNOWN
    worker.drain()

    # The durable counter covers non-desired confirmation reads. Missing and
    # exact-old observations may alternate; only the final extra exact-old
    # sample authorizes local quarantine of the rolling surface.
    del provider.notes[first_output]
    del provider.content[first_output]
    assert worker.poll().output_retired == "pending"

    provider.notes[first_output] = old_note
    provider.content[first_output] = old_content
    assert worker.poll().output_retired == "pending"

    del provider.notes[first_output]
    del provider.content[first_output]
    assert worker.poll().output_retired == "pending"
    assert store.update_readback_exhausted(maximum=3)
    assert store.output_gate() == "update_ambiguous"

    provider.notes[first_output] = old_note
    provider.content[first_output] = old_content
    final = worker.poll()
    assert final.output_retired == "abandoned"
    assert final.output_gate is None
    q2 = store.get_job("q2")
    assert q2.state is JobState.FAILED
    assert q2.publication_attempts == 1
    assert q2.output_note_id is None
    assert store.output_snapshot().epoch == 1
    assert store.output_snapshot().note_id is None
    assert store.latest_published_sequence() == 1
    assert provider.create_calls == 1
    assert provider.update_calls == 1
    assert len(assistant.calls) == 2
    status = database_status(store)
    assert status["output_blocked"] is False
    assert status["jobs_terminal_failed"] == 1
    assert status["jobs_unresolved_failed"] == 0


def test_old_update_readback_exhaustion_quarantines_surface_and_resumes_queue(
    tmp_path,
):
    config = WorkerConfig(
        max_update_readback_attempts=2,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, transcriber, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    ask(provider, clock, worker, "q1", "First")
    first_output = store.get_job("q1").output_note_id

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    assert worker.poll().enqueued == 1
    provider.update_outcome = MutationOutcome.UNKNOWN
    worker.drain()
    q2_response_id = store.get_job("q2").response_id

    # A known-identity update gate pauses processing, but must not blind the
    # intake loop to requests that arrive afterward.
    clock.advance(minutes=1)
    provider.add_audio("q3", "Assistant, Third", when=clock.value)

    first_readback = worker.poll()
    assert first_readback.output_retired == "pending"
    assert first_readback.output_gate == "update_ambiguous"
    assert first_readback.enqueued == 1
    assert store.get_job("q3").state is JobState.QUEUED
    assert worker.poll().output_retired == "unknown"
    assert store.output_gate() == "update_ambiguous"
    assert store.update_readback_exhausted(maximum=2)
    assert store.get_job("q2").last_error == (
        "update outcome remains unknown after bounded read-back"
    )
    reads = len(provider.read_calls)
    abandoned = worker.poll()
    assert abandoned.output_retired == "abandoned"
    assert abandoned.output_gate is None
    assert len(provider.read_calls) == reads + 1
    assert provider.update_calls == 1
    assert store.get_job("q3").state is JobState.QUEUED
    assert len(assistant.calls) == 2
    assert store.output_gate() is None
    assert store.output_snapshot().note_id is None
    assert store.output_snapshot().epoch == 1
    q2 = store.get_job("q2")
    assert q2.state is JobState.FAILED
    assert q2.publication_attempts == 1
    assert q2.response_id == q2_response_id
    assert q2.last_error == (
        "uncertain update delivery abandoned after bounded non-desired read-back"
    )
    assert store.get_job("q3").accepted_output_epoch == 1
    assert store.is_known_output_note(first_output)
    assert first_output in provider.notes

    store.close()
    reopened = Store(tmp_path / "bridge.sqlite3")
    try:
        restarted = Worker(
            store=reopened,
            provider=provider,
            transcriber=transcriber,
            assistant=assistant,
            config=config,
            clock=clock,
        )
        provider.update_outcome = MutationOutcome.VERIFIED
        restarted.drain()
        q2 = reopened.get_job("q2")
        q3 = reopened.get_job("q3")
        assert q2.state is JobState.FAILED
        assert q3.state is JobState.PUBLISHED
        assert q2.response_id == q2_response_id
        assert q2.output_note_id is None
        assert q3.output_note_id != first_output
        assert provider.create_calls == 2
        assert provider.update_calls == 1
        assert len(assistant.calls) == 3

        # If the quarantined mutation becomes visible late, its identity stays
        # bridge-owned and cannot be mistaken for a fresh handset request. The
        # worker neither recreates nor republishes the abandoned answer.
        assert q2.publication_content is not None
        provider.content[first_output] = q2.publication_content.encode()
        late = restarted.poll()
        assert late.enqueued == 0
        assert reopened.output_snapshot().note_id == q3.output_note_id
        assert provider.create_calls == 2
        assert provider.update_calls == 1
        assert len(assistant.calls) == 3
    finally:
        reopened.close()


def test_deleted_reply_stays_deleted_until_a_new_request(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")
    first_output = store.get_job("q1").output_note_id
    provider.delete(first_output)

    report = worker.poll()
    assert report.output_retired == "missing"
    assert store.output_snapshot().note_id == first_output
    report = worker.poll()
    assert report.output_retired == "deleted"
    assert provider.create_calls == 1
    assert worker.run_once().worked is False
    assert first_output not in provider.notes

    ask(provider, clock, worker, "q2", "A new request")
    second_output = store.get_job("q2").output_note_id
    assert second_output != first_output
    assert provider.create_calls == 2


def test_transient_missing_reply_does_not_retire_or_recreate_it(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")
    first_output = store.get_job("q1").output_note_id
    saved_note = provider.notes.pop(first_output)
    saved_content = provider.content.pop(first_output)

    report = worker.poll()
    assert report.output_retired == "missing"
    assert store.output_snapshot().note_id == first_output
    assert provider.create_calls == 1

    provider.notes[first_output] = saved_note
    provider.content[first_output] = saved_content
    assert worker.poll().output_retired is None
    assert store.output_snapshot().note_id == first_output

    provider.notes.pop(first_output)
    provider.content.pop(first_output)
    assert worker.poll().output_retired == "missing"
    assert store.output_snapshot().note_id == first_output
    assert provider.create_calls == 1


def test_new_request_during_first_missing_poll_preserves_answer_until_delete_confirmed(
    tmp_path,
):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")
    first_output = store.get_job("q1").output_note_id
    provider.delete(first_output)

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    first_poll = worker.poll()
    assert first_poll.output_retired == "missing"
    assert first_poll.enqueued == 1

    worker.drain()
    answered = store.get_job("q2")
    assert answered.state is JobState.ANSWERED
    assert answered.answer == "A useful answer."
    assert answered.publication_attempts == 0
    assert len(assistant.calls) == 2
    assert provider.update_calls == 0
    assert provider.create_calls == 1
    blocked = worker.run_once()
    assert blocked.worked is False
    assert blocked.detail == "output_missing_pending"

    # The missing observation and answer both survive a worker restart.
    store.close()
    reopened = Store(tmp_path / "bridge.sqlite3")
    worker.store = reopened
    try:
        second_poll = worker.poll()
        assert second_poll.output_retired == "deleted"
        rebased = reopened.get_job("q2")
        assert rebased.state is JobState.ANSWERED
        assert rebased.answer == "A useful answer."
        assert rebased.accepted_output_epoch == 1
        assert rebased.accepted_output_note_id is None
        assert provider.create_calls == 1

        worker.drain()
        published = reopened.get_job("q2")
        assert published.state is JobState.PUBLISHED
        assert published.output_note_id != first_output
        assert provider.create_calls == 2
        assert provider.update_calls == 0
        assert len(assistant.calls) == 2
    finally:
        reopened.close()


def test_missing_update_target_waits_for_two_polls_then_reuses_stored_answer(
    tmp_path,
):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")
    first_output = store.get_job("q1").output_note_id

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().state is JobState.TRANSCRIBED
    assert worker.run_once().state is JobState.ANSWERED
    provider.delete(first_output)

    raced = worker.run_once()
    assert raced.detail == "missing"
    assert raced.state is JobState.FAILED
    assert store.output_gate() == "update_ambiguous"
    assert store.output_snapshot().note_id == first_output
    assert provider.create_calls == 1

    first_poll = worker.poll()
    assert first_poll.output_retired == "missing"
    assert first_poll.output_gate == "update_ambiguous"
    assert store.output_snapshot().note_id == first_output
    assert provider.create_calls == 1

    second_poll = worker.poll()
    assert second_poll.output_retired == "deleted"
    assert second_poll.output_gate is None
    restored = store.get_job("q2")
    assert restored.state is JobState.ANSWERED
    assert restored.publication_attempts == 0
    assert store.output_snapshot().note_id is None

    worker.drain()
    published = store.get_job("q2")
    assert published.state is JobState.PUBLISHED
    assert published.output_note_id != first_output
    assert provider.create_calls == 2
    assert len(assistant.calls) == 2


def test_missing_read_after_update_interrupted_before_gate_never_retries(
    tmp_path, monkeypatch
):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")
    first_output = store.get_job("q1").output_note_id

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().state is JobState.TRANSCRIBED
    assert worker.run_once().state is JobState.ANSWERED
    provider.delete(first_output)

    original_gate = store.mark_publication_ambiguous

    def fail_before_gate(*_args, **_kwargs):
        raise sqlite3.OperationalError("injected gate persistence failure")

    monkeypatch.setattr(store, "mark_publication_ambiguous", fail_before_gate)
    with pytest.raises(sqlite3.OperationalError, match="gate persistence failure"):
        worker.run_once()
    interrupted = store.get_job("q2")
    assert interrupted.state is JobState.ANSWERED
    assert interrupted.publication_attempts == 1
    assert store.output_gate() is None
    monkeypatch.setattr(store, "mark_publication_ambiguous", original_gate)

    assert worker.poll().output_retired == "pending"
    assert worker.poll().output_retired == "pending"
    gated = store.get_job("q2")
    assert gated.state is JobState.FAILED
    assert gated.accepted_output_epoch == 0
    assert gated.accepted_output_note_id == first_output
    assert gated.publication_attempts == 1
    assert store.output_gate() == "update_ambiguous"
    assert worker.run_once().detail == "output_gate"
    assert provider.create_calls == 1
    assert provider.update_calls == 1
    assert len(assistant.calls) == 2


def test_missing_unknown_update_rebases_stored_answer_and_later_intake(
    tmp_path,
):
    config = WorkerConfig(
        max_update_readback_attempts=1,
        confirm_output_deletion_after=2,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, transcriber, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    ask(provider, clock, worker, "q1", "First")
    first_output = store.get_job("q1").output_note_id

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().state is JobState.TRANSCRIBED
    assert worker.run_once().state is JobState.ANSWERED
    provider.unknown_after_update = True
    assert worker.run_once().detail == "unknown"
    assert provider.update_calls == 1
    assert provider.create_calls == 1
    assert store.output_gate() == "update_ambiguous"
    q2_response_id = store.get_job("q2").response_id

    # The exact update target remains absent. Keep admitting later requests
    # while the bounded read-only confirmation window runs.
    provider.notes.pop(first_output)
    provider.content.pop(first_output)
    clock.advance(minutes=1)
    provider.add_audio("q3", "Assistant, Third", when=clock.value)
    pending = worker.poll()
    assert pending.output_retired == "pending"
    assert pending.output_gate == "update_ambiguous"
    assert pending.enqueued == 1
    assert store.get_job("q3").state is JobState.QUEUED
    assert store.get_job("q2").publication_attempts == 1

    store.close()
    reopened = Store(tmp_path / "bridge.sqlite3")
    worker.store = reopened
    try:
        recovered = worker.poll()
        assert recovered.output_retired == "deleted"
        assert recovered.output_gate is None
        assert reopened.output_gate() is None
        gated = reopened.get_job("q2")
        assert gated.state is JobState.ANSWERED
        assert gated.publication_attempts == 0
        assert gated.response_id == q2_response_id
        assert reopened.get_job("q3").accepted_output_epoch == 1
        assert reopened.get_job("q3").accepted_output_note_id is None
        assert reopened.is_known_output_note(first_output)
        assert provider.update_calls == 1
        assert provider.create_calls == 1
        assert len(assistant.calls) == 2

        worker.drain()

        q2 = reopened.get_job("q2")
        q3 = reopened.get_job("q3")
        assert q2.state is JobState.PUBLISHED
        assert q3.state is JobState.PUBLISHED
        assert q2.response_id == q2_response_id
        assert q2.output_note_id != first_output
        assert q3.output_note_id == q2.output_note_id
        assert provider.create_calls == 2
        assert provider.update_calls == 2
        assert len(assistant.calls) == 3
    finally:
        reopened.close()


def test_temporarily_missing_update_target_retries_same_stored_answer(tmp_path):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")
    first_output = store.get_job("q1").output_note_id

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().state is JobState.TRANSCRIBED
    assert worker.run_once().state is JobState.ANSWERED
    saved_note = provider.notes.pop(first_output)
    saved_content = provider.content.pop(first_output)

    assert worker.run_once().detail == "missing"
    assert store.output_gate() == "update_ambiguous"
    provider.notes[first_output] = saved_note
    provider.content[first_output] = saved_content

    report = worker.poll()
    assert report.output_retired == "retry"
    assert report.output_gate is None
    restored = store.get_job("q2")
    assert restored.state is JobState.ANSWERED
    assert restored.publication_attempts == 0

    worker.drain()
    published = store.get_job("q2")
    assert published.state is JobState.PUBLISHED
    assert published.output_note_id == first_output
    assert provider.create_calls == 1
    assert provider.update_calls == 2
    assert len(assistant.calls) == 2


def test_post_write_missing_receipt_never_retries_update(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().state is JobState.TRANSCRIBED
    assert worker.run_once().state is JobState.ANSWERED

    def mutate_then_report_missing(
        note_id: str,
        *,
        title: str,
        content: str,
        expected_title: str,
        expected_content_sha256: str,
    ) -> MutationReceipt:
        provider.update_calls += 1
        provider.notes[note_id] = RemoteNote(note_id, NoteKind.TEXT, title, START)
        provider.content[note_id] = content.encode()
        return MutationReceipt(
            MutationOutcome.CONFLICT,
            detail="note_missing_after_write",
        )

    provider.update_text_note = mutate_then_report_missing  # type: ignore[method-assign]
    result = worker.run_once()
    assert result.detail == "conflict"
    failed = store.get_job("q2")
    assert failed.state is JobState.FAILED
    assert failed.retryable is False
    assert store.output_gate() is None
    assert worker.run_once().worked is False
    assert provider.update_calls == 1


def test_external_reply_edit_is_not_overwritten(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")
    output = store.get_job("q1").output_note_id
    provider.content[output] = b"user content"
    report = worker.poll()
    assert report.output_retired == "modified"
    assert provider.content[output] == b"user content"
    assert provider.update_calls == 0


def test_crash_after_create_never_causes_duplicate_create(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_audio("q1", "Assistant, First", when=clock.value)
    worker.poll()
    worker.run_once()
    worker.run_once()
    provider.crash_after_create = True
    try:
        worker.run_once()
    except SystemExit:
        pass
    else:
        raise AssertionError("expected simulated process death")
    assert provider.create_calls == 1
    assert store.get_job("q1").state == JobState.ANSWERED

    provider.crash_after_create = False
    worker.run_once()
    assert provider.create_calls == 1
    assert store.get_job("q1").state == JobState.FAILED


def test_disk_full_before_create_does_not_perform_a_cloud_write(
    tmp_path, monkeypatch
):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_audio("q1", "Assistant, First", when=clock.value)
    worker.poll()
    worker.run_once()
    worker.run_once()
    original_increment = store.increment_attempt

    def disk_full(*_args, **_kwargs):
        raise sqlite3.OperationalError("database or disk is full")

    with monkeypatch.context() as patch:
        patch.setattr(store, "increment_attempt", disk_full)
        with pytest.raises(sqlite3.OperationalError, match="disk.*full"):
            worker.run_once()

    assert provider.create_calls == 0
    assert store.get_job("q1").state is JobState.ANSWERED
    assert worker.run_once().state is JobState.PUBLISHED
    assert provider.create_calls == 1
    assert original_increment is not None


def test_disk_full_after_create_never_blindly_retries_the_create(
    tmp_path, monkeypatch
):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_audio("q1", "Assistant, First", when=clock.value)
    worker.poll()
    worker.run_once()
    worker.run_once()

    def disk_full(*_args, **_kwargs):
        raise sqlite3.OperationalError("database or disk is full")

    with monkeypatch.context() as patch:
        patch.setattr(store, "remember_output_note", disk_full)
        with pytest.raises(sqlite3.OperationalError, match="disk.*full"):
            worker.run_once()

    assert provider.create_calls == 1
    assert store.get_job("q1").state is JobState.ANSWERED
    report = worker.run_once()
    assert report.detail == "output_gate"
    assert report.worked is False
    assert store.get_job("q1").state is JobState.FAILED
    assert store.output_gate() == "create_ambiguous"
    assert provider.create_calls == 1


def test_poll_recovers_crash_after_verified_update(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    ask(provider, clock, worker, "q1", "First")

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    worker.poll()
    worker.run_once()
    worker.run_once()
    provider.crash_after_update = True
    try:
        worker.run_once()
    except SystemExit:
        pass
    else:
        raise AssertionError("expected simulated process death")
    assert store.get_job("q2").state == JobState.ANSWERED

    provider.crash_after_update = False
    report = worker.poll()
    assert report.output_retired == "recovered"
    assert store.get_job("q2").state == JobState.PUBLISHED
    assert provider.update_calls == 1


def test_interrupted_update_with_old_readback_is_never_called_twice(tmp_path):
    store, provider, _, _, clock, worker = make_worker(tmp_path)
    clock.advance(minutes=1)
    provider.add_audio("q1", "Assistant, First", when=clock.value)
    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    assert worker.poll().enqueued == 2
    assert store.get_job("q2").accepted_output_note_id is None

    assert worker.run_once().state is JobState.TRANSCRIBED
    assert worker.run_once().state is JobState.ANSWERED
    assert worker.run_once().state is JobState.PUBLISHED
    output_note_id = store.get_job("q1").output_note_id
    assert worker.run_once().state is JobState.TRANSCRIBED
    assert worker.run_once().state is JobState.ANSWERED

    def die_before_remote_result(*_args, **_kwargs):
        provider.update_calls += 1
        raise SystemExit("injected update interruption")

    provider.update_text_note = die_before_remote_result  # type: ignore[method-assign]
    with pytest.raises(SystemExit, match="update interruption"):
        worker.run_once()
    assert provider.update_calls == 1
    interrupted = store.get_job("q2")
    assert interrupted.publication_attempts == 1
    assert interrupted.accepted_output_note_id == output_note_id

    report = worker.run_once()
    assert report.worked is False
    assert report.detail == "output_gate"
    assert store.output_gate() == "update_ambiguous"
    assert provider.update_calls == 1
    assert worker.poll().output_retired == "pending"
    assert provider.update_calls == 1


@pytest.mark.parametrize(
    ("error", "gate"),
    [
        (ChatGPTAuthorizationRequiredError("renew"), "authorization_required"),
        (ChatGPTPlanUsageLimitError("limit"), "plan_limit"),
        (ChatGPTCredentialAmbiguousError("rotation"), "credential_ambiguous"),
        (ChatGPTAmbiguousOutcomeError("stream"), "inference_ambiguous"),
        (ChatGPTAccountNotEligibleError("account"), "account_not_eligible"),
        (ChatGPTConfigurationError("configuration"), "configuration_error"),
    ],
)
def test_typed_assistant_failure_pauses_all_model_work_until_exact_recovery(
    tmp_path, error, gate
):
    store, provider, _, assistant, clock, worker = make_worker(tmp_path)
    raising = RaisingAssistant(error)
    worker.assistant = raising

    clock.advance(minutes=1)
    provider.add_audio("q1", "Assistant, First", when=clock.value)
    assert worker.poll().enqueued == 1
    assert worker.run_once().detail == "question"
    report = worker.run_once()

    assert report.detail == "assistant_paused"
    assert store.assistant_gate() == gate
    status = store.assistant_gate_status()
    assert status is not None
    assert status["source_note_id"] == "q1"
    gated = store.get_job("q1")
    assert gated.state is JobState.FAILED
    assert gated.completed_at is None
    assert gated.resume_state is JobState.TRANSCRIBED

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    assert worker.poll().enqueued == 1
    if gate == "inference_ambiguous":
        assert worker.run_once().detail == "question"
        assert store.get_job("q2").state is JobState.TRANSCRIBED
    else:
        assert worker.run_once().detail == "assistant_gate"
    assert len(raising.calls) == 1

    with pytest.raises(ValueError, match="identifiers"):
        store.clear_assistant_gate(
            expected_gate=gate,
            expected_note_id="wrong",
            now=clock.value,
        )
    assert store.assistant_gate() == gate

    worker.assistant = assistant
    assert store.clear_assistant_gate(
        expected_gate=gate,
        expected_note_id="q1",
        now=clock.value,
    )
    worker.drain()
    assert store.get_job("q1").state is JobState.PUBLISHED
    assert store.get_job("q2").state is JobState.PUBLISHED
    assert len(assistant.calls) == 2


def test_daily_model_limit_pauses_before_second_assistant_call(tmp_path):
    config = WorkerConfig(
        max_model_calls_per_day=1,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    ask(provider, clock, worker, "q1", "First")
    assert len(assistant.calls) == 1

    clock.advance(minutes=1)
    provider.add_audio("q2", "Assistant, Second", when=clock.value)
    assert worker.poll().enqueued == 1
    reports = worker.drain()

    assert reports[-1].detail == "assistant_paused"
    assert store.assistant_gate() == "daily_model_limit"
    assert store.model_calls_for_day(now=clock.value) == 1
    assert len(assistant.calls) == 1

    clock.advance(days=1)
    resumed = worker.drain()
    assert resumed[-1].state is JobState.PUBLISHED
    assert store.assistant_gate() is None
    assert store.get_job("q2").state is JobState.PUBLISHED
    assert store.model_calls_for_day(now=clock.value) == 1
    assert len(assistant.calls) == 2


def test_pending_queue_cap_preserves_cloud_order_and_defers_download(tmp_path):
    config = WorkerConfig(max_pending_jobs=1, retry_delay=timedelta(seconds=1))
    store, provider, _, _, clock, worker = make_worker(tmp_path, config=config)
    clock.advance(minutes=1)
    provider.add_audio("first", "Assistant, First", when=clock.value)
    clock.advance(minutes=1)
    provider.add_audio("second", "Assistant, Second", when=clock.value)

    first_poll = worker.poll()
    assert first_poll.enqueued == 1
    assert first_poll.queue_limited is True
    assert provider.download_calls == 1
    assert store.get_job("first") is not None
    assert store.get_job("second") is None

    worker.drain()
    second_poll = worker.poll()
    assert second_poll.enqueued == 1
    assert second_poll.queue_limited is False
    assert provider.download_calls == 2
    worker.drain()
    assert store.get_job("second").state is JobState.PUBLISHED


def test_question_character_limit_fails_before_assistant_call(tmp_path):
    config = WorkerConfig(
        max_question_chars=5,
        max_context_chars=5,
        retry_delay=timedelta(seconds=1),
    )
    store, provider, _, assistant, clock, worker = make_worker(
        tmp_path, config=config
    )
    clock.advance(minutes=1)
    provider.add_text("long", "Assistant, sixxxx", when=clock.value)
    assert worker.poll().enqueued == 1
    worker.drain()

    job = store.get_job("long")
    assert job.state is JobState.FAILED
    assert job.failure_stage == "routing"
    assert job.retryable is False
    assert assistant.calls == []
