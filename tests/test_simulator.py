from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lightphone_gpt.contracts import LightProvider
from lightphone_gpt.models import MutationOutcome, NoteKind
from lightphone_gpt.simulator import (
    FaultPhase,
    FaultScript,
    SimulatedLightProvider,
    SimulatedProviderError,
)


def fixed_clock() -> datetime:
    return datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def test_simulator_persists_notes_across_provider_restarts(tmp_path: Path) -> None:
    database = tmp_path / "light-simulator.sqlite3"
    first = SimulatedLightProvider(database, clock=fixed_clock)
    assert isinstance(first, LightProvider)
    audio = first.seed_audio_note(b"audio-one", title="Recording")
    handset_text = first.seed_text_note("Assistant, text question", title="Question")
    created = first.create_text_note(title="AI READY", content="first answer")

    assert created.outcome is MutationOutcome.VERIFIED
    assert created.note is not None
    assert first.download_audio(audio.note_id) == b"audio-one"
    assert first.read_note_content(handset_text.note_id) == b"Assistant, text question"

    restarted = SimulatedLightProvider(database, clock=fixed_clock)
    assert restarted.get_note(audio.note_id) == audio
    assert restarted.get_note(handset_text.note_id) == handset_text
    assert restarted.read_note_content(created.note.note_id) == b"first answer"

    expected = hashlib.sha256(b"first answer").hexdigest()
    updated = restarted.update_text_note(
        created.note.note_id,
        title="AI READY",
        content="second answer",
        expected_title="AI READY",
        expected_content_sha256=expected,
    )
    assert updated.outcome is MutationOutcome.VERIFIED
    assert restarted.read_note_content(created.note.note_id) == b"second answer"


def test_update_detects_content_conflict_without_overwriting(tmp_path: Path) -> None:
    provider = SimulatedLightProvider(tmp_path / "sim.sqlite3")
    created = provider.create_text_note(title="AI READY", content="user edit")
    assert created.note is not None

    receipt = provider.update_text_note(
        created.note.note_id,
        title="AI READY",
        content="bridge answer",
        expected_title="AI READY",
        expected_content_sha256="0" * 64,
    )

    assert receipt.outcome is MutationOutcome.CONFLICT
    assert receipt.detail == "content_changed"
    assert provider.read_note_content(created.note.note_id) == b"user edit"


def test_after_create_failure_is_unknown_but_durable(tmp_path: Path) -> None:
    faults = FaultScript().fail(
        "create_text_note", phase=FaultPhase.AFTER, message="connection lost"
    )
    provider = SimulatedLightProvider(tmp_path / "sim.sqlite3", faults=faults)

    receipt = provider.create_text_note(title="AI READY", content="answer")

    assert receipt.outcome is MutationOutcome.UNKNOWN
    notes = provider.list_notes()
    assert len(notes) == 1
    assert notes[0].kind is NoteKind.TEXT
    assert provider.read_note_content(notes[0].note_id) == b"answer"


def test_after_update_failure_is_unknown_and_does_not_invite_blind_retry(
    tmp_path: Path,
) -> None:
    database = tmp_path / "sim.sqlite3"
    setup = SimulatedLightProvider(database)
    created = setup.create_text_note(title="AI READY", content="old")
    assert created.note is not None
    expected = hashlib.sha256(b"old").hexdigest()

    faults = FaultScript().fail("update_text_note", phase=FaultPhase.AFTER)
    provider = SimulatedLightProvider(database, faults=faults)
    receipt = provider.update_text_note(
        created.note.note_id,
        title="AI READY",
        content="new",
        expected_title="AI READY",
        expected_content_sha256=expected,
    )

    assert receipt.outcome is MutationOutcome.UNKNOWN
    assert provider.read_note_content(created.note.note_id) == b"new"


def test_verification_failure_returns_unknown(tmp_path: Path) -> None:
    faults = FaultScript().fail("get_note", call=1)
    provider = SimulatedLightProvider(tmp_path / "sim.sqlite3", faults=faults)

    receipt = provider.create_text_note(title="AI READY", content="answer")

    assert receipt.outcome is MutationOutcome.UNKNOWN
    assert len(provider.list_notes()) == 1


@pytest.mark.parametrize(
    ("operation", "invoke"),
    [
        ("list_notes", lambda provider, _note: provider.list_notes()),
        ("download_audio", lambda provider, note: provider.download_audio(note.note_id)),
        ("read_note_content", lambda provider, note: provider.read_note_content(note.note_id)),
    ],
)
def test_read_failures_are_scripted_on_exact_call(
    tmp_path: Path, operation: str, invoke
) -> None:
    database = tmp_path / f"{operation}.sqlite3"
    setup = SimulatedLightProvider(database)
    note = setup.seed_audio_note(b"audio")
    faults = FaultScript().fail(operation, call=2)
    provider = SimulatedLightProvider(database, faults=faults)

    invoke(provider, note)
    with pytest.raises(SimulatedProviderError):
        invoke(provider, note)
    invoke(provider, note)


def test_delete_fault_after_commit_models_lost_acknowledgement(tmp_path: Path) -> None:
    database = tmp_path / "delete.sqlite3"
    setup = SimulatedLightProvider(database)
    note = setup.seed_audio_note(b"audio")
    faults = FaultScript().fail("delete_note", phase=FaultPhase.AFTER)
    provider = SimulatedLightProvider(database, faults=faults)

    with pytest.raises(SimulatedProviderError):
        provider.delete_note(note.note_id)

    assert provider.get_note(note.note_id) is None


def test_source_delete_fault_before_mutation_proves_safe_retry(tmp_path: Path) -> None:
    database = tmp_path / "source-delete-before.sqlite3"
    setup = SimulatedLightProvider(database)
    note = setup.seed_audio_note(b"audio")
    faults = FaultScript().fail("delete_source_note", phase=FaultPhase.BEFORE)
    provider = SimulatedLightProvider(database, faults=faults)

    receipt = provider.delete_source_note(
        note.note_id,
        expected_kind=NoteKind.AUDIO,
        expected_content_sha256=hashlib.sha256(b"audio").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.UNKNOWN
    assert receipt.deletion_attempted is False
    assert provider.get_note(note.note_id) is not None


def test_source_delete_fault_after_commit_is_ambiguous_but_applied(tmp_path: Path) -> None:
    database = tmp_path / "source-delete-after.sqlite3"
    setup = SimulatedLightProvider(database)
    note = setup.seed_audio_note(b"audio")
    faults = FaultScript().fail("delete_source_note", phase=FaultPhase.AFTER)
    provider = SimulatedLightProvider(database, faults=faults)

    receipt = provider.delete_source_note(
        note.note_id,
        expected_kind=NoteKind.AUDIO,
        expected_content_sha256=hashlib.sha256(b"audio").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.UNKNOWN
    assert receipt.deletion_attempted is True
    assert provider.get_note(note.note_id) is None


def test_duplicate_list_events_keep_the_same_remote_identity(tmp_path: Path) -> None:
    provider = SimulatedLightProvider(tmp_path / "duplicates.sqlite3")
    note = provider.seed_audio_note(b"audio")
    provider.set_list_duplicates(note.note_id, extra_copies=2)

    listed = provider.list_notes()

    assert [item.note_id for item in listed] == [note.note_id] * 3
