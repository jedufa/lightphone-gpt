from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO
import hashlib
import json
from pathlib import Path

import pytest

from lightphone_gpt import cli
from lightphone_gpt.chatgpt_errors import (
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTTransientError,
)
from lightphone_gpt.e2e_command import E2ECommandError
from lightphone_gpt.models import JobState, NoteKind, RemoteNote
from lightphone_gpt.store import Store


def invoke(argv: list[str], env: dict[str, str]):
    stdout = StringIO()
    stderr = StringIO()
    code = cli.main(argv, environ=env, stdout=stdout, stderr=stderr)
    output = json.loads(stdout.getvalue()) if stdout.getvalue() else None
    error = json.loads(stderr.getvalue()) if stderr.getvalue() else None
    return code, output, error


def seed_ambiguous_create(state_dir: Path, *, source_note_id: str = "source") -> str:
    state_dir.mkdir(mode=0o700)
    now = datetime(2026, 10, 15, 8, 0, tzinfo=UTC)
    payload = "stored publication payload"
    with Store(state_dir / "bridge.sqlite3") as store:
        store.establish_baseline([], now=now)
        note = RemoteNote(
            source_note_id,
            NoteKind.AUDIO,
            "Recording",
            now,
            "audio/mp4",
        )
        store.enqueue_audio(note, b"audio", now=now)
        store.mark_question_transcribed(source_note_id, question="Question", now=now)
        store.mark_answered(
            source_note_id,
            answer="Answer",
            sections=("Answer",),
            response_id="response",
            now=now,
        )
        store.prepare_publication(
            source_note_id,
            content=payload,
            title="AI READY",
            output_mode="rolling",
            now=now,
        )
        store.increment_attempt(source_note_id, "publication", now=now)
        store.mark_publication_ambiguous(
            source_note_id,
            gate="create_ambiguous",
            error="unknown create result",
            now=now,
        )
    return hashlib.sha256(payload.encode()).hexdigest()


def seed_ambiguous_update(
    state_dir: Path,
    *,
    source_note_id: str = "update-source",
    readback_attempts: int = 2,
) -> tuple[str, str]:
    state_dir.mkdir(mode=0o700)
    now = datetime(2026, 10, 15, 8, 0, tzinfo=UTC)
    previous = "previous rolling reply"
    publication = "stored updated reply"
    previous_sha256 = hashlib.sha256(previous.encode()).hexdigest()
    with Store(state_dir / "bridge.sqlite3") as store:
        store.establish_baseline([], now=now)
        assert store.register_created_output(
            expected_epoch=0,
            note_id="rolling-output",
            content_sha256=previous_sha256,
            now=now,
        )
        note = RemoteNote(
            source_note_id,
            NoteKind.AUDIO,
            "Recording",
            now,
            "audio/mp4",
        )
        store.enqueue_audio(note, b"audio", now=now)
        store.mark_question_transcribed(source_note_id, question="Question", now=now)
        store.mark_answered(
            source_note_id,
            answer="Answer",
            sections=("Answer",),
            response_id="response",
            now=now,
        )
        store.prepare_publication(
            source_note_id,
            content=publication,
            title="AI READY",
            output_mode="rolling",
            now=now,
        )
        store.increment_attempt(source_note_id, "publication", now=now)
        store.mark_publication_ambiguous(
            source_note_id,
            gate="update_ambiguous",
            error="update outcome unknown",
            now=now,
        )
        for attempt in range(readback_attempts):
            store.resolve_unapplied_publication(
                output_note_id="rolling-output",
                observed_content_sha256=previous_sha256,
                maximum_readbacks=readback_attempts,
                now=now,
            )
    return hashlib.sha256(publication.encode()).hexdigest(), previous_sha256


def seed_assistant_gate(
    state_dir: Path,
    *,
    source_note_id: str = "assistant-source",
    gate: str = "authorization_required",
) -> None:
    state_dir.mkdir(mode=0o700)
    now = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
    with Store(state_dir / "bridge.sqlite3") as store:
        store.establish_baseline([], now=now)
        note = RemoteNote(
            source_note_id,
            NoteKind.TEXT,
            "Private title",
            now,
            "text/plain; charset=utf-8",
        )
        store.enqueue_text(note, b"Assistant, private question", now=now)
        store.mark_question_transcribed(
            source_note_id,
            question="private question",
            now=now,
        )
        store.pause_assistant(
            source_note_id,
            gate=gate,
            error="private dependency detail",
            now=now,
        )


def seed_recoverable_job(
    state_dir: Path,
    *,
    source_note_id: str = "recoverable-source",
) -> None:
    state_dir.mkdir(mode=0o700)
    now = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
    with Store(state_dir / "bridge.sqlite3") as store:
        store.establish_baseline([], now=now)
        note = RemoteNote(
            source_note_id,
            NoteKind.AUDIO,
            "Private title",
            now,
            "audio/mp4",
        )
        store.enqueue_audio(note, b"private audio payload", now=now)
        store.increment_attempt(source_note_id, "transcription", now=now)
        store.mark_failure(
            source_note_id,
            stage="transcription",
            resume_state=JobState.QUEUED,
            error="private dependency detail",
            retryable=False,
            manual_retry_allowed=True,
            preserve_audio=True,
            now=now,
        )


def seed_ignored_audio_job(
    state_dir: Path,
    *,
    source_note_id: str = "ignored-audio-source",
    payload: bytes = b"private ignored audio",
) -> str:
    state_dir.mkdir(mode=0o700)
    now = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
    with Store(state_dir / "bridge.sqlite3") as store:
        store.establish_baseline([], now=now)
        note = RemoteNote(
            source_note_id,
            NoteKind.AUDIO,
            "Private title",
            now,
            "audio/mp4",
        )
        store.enqueue_audio(note, payload, now=now)
        store.increment_attempt(source_note_id, "transcription", now=now)
        store.mark_ignored(
            source_note_id, reason="prefix_missing", now=now
        )
    return hashlib.sha256(payload).hexdigest()


def test_cli_full_simulator_workflow_is_durable_and_content_free(
    tmp_path: Path,
) -> None:
    env = {"LIGHTPHONE_GPT_STATE_DIR": str(tmp_path / "state")}
    private = "Assistant, a private synthetic question"

    code, initialized, _ = invoke(["init"], env)
    assert code == 0 and initialized["enrolled"] is True

    code, seeded, _ = invoke(
        ["simulator", "seed", "--transcript", private], env
    )
    assert code == 0
    source_note_id = seeded["note_id"]

    code, cycle, _ = invoke(["cycle"], env)
    assert code == 0
    assert cycle["jobs_published"] == 1
    assert cycle["work"]["steps"] == 3
    assert private not in json.dumps(cycle)

    code, status, _ = invoke(["status"], env)
    assert code == 0 and status["jobs_published"] == 1
    assert private not in json.dumps(status)

    code, listing, _ = invoke(["simulator", "list"], env)
    assert code == 0
    assert len(listing["notes"]) == 2
    assert private not in json.dumps(listing)

    code, deleted, _ = invoke(
        ["simulator", "delete", source_note_id], env
    )
    assert code == 0 and deleted == {"existed": True, "ok": True}

    code, retained, _ = invoke(["retention"], env)
    assert code == 0 and retained["ok"] is True


def test_cli_rejects_cloud_service_modes_even_when_credentials_are_present(
    tmp_path: Path,
) -> None:
    env = {
        "LIGHTPHONE_GPT_STATE_DIR": str(tmp_path / "state"),
        "LIGHTPHONE_GPT_PROVIDER": "light",
        "LIGHTPHONE_GPT_ASSISTANT": "chatgpt",
        "lightphone_username": "private-user",
        "lightphone_password": "private-password",
    }

    code, output, error = invoke(["status"], env)

    assert code == 2 and output is None
    serialized = json.dumps(error)
    assert error["error"] == "configuration_error"
    assert "private-user" not in serialized
    assert "private-password" not in serialized
    assert not (tmp_path / "state").exists()


def test_cli_simulated_text_note_skips_audio_setup(tmp_path: Path) -> None:
    env = {"LIGHTPHONE_GPT_STATE_DIR": str(tmp_path / "state")}
    private = "Assistant, a private typed question"
    assert invoke(["init"], env)[0] == 0

    code, seeded, error = invoke(
        ["simulator", "seed-text", "--content", private], env
    )
    assert code == 0 and error is None and seeded["note_id"]

    code, cycle, error = invoke(["cycle"], env)
    assert code == 0 and error is None
    assert cycle["jobs_published"] == 1
    assert cycle["work"]["steps"] == 3
    assert private not in json.dumps(cycle)


def test_light_e2e_requires_explicit_confirmation_without_building_runtime(
    monkeypatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("unconfirmed proof must have no side effects")

    monkeypatch.setattr(cli, "run_e2e_command", forbidden)
    monkeypatch.setattr(cli, "build_runtime", forbidden)

    code, output, error = invoke(["light-e2e"], {})

    assert code == 2 and output is None
    assert error == {
        "category": "confirmation_required",
        "mutation_attempted": False,
        "ok": False,
        "status": "failed_closed",
    }


def test_light_e2e_runs_protected_command_without_service_configuration(
    monkeypatch,
) -> None:
    private_env = {"private": "must-be-forwarded-without-being-emitted"}
    public = {
        "ok": True,
        "status": "verified",
        "operation_id": "a" * 32,
        "read_back_verified": True,
    }

    def proof(*, environ):
        assert environ is private_env
        return public

    def forbidden(*_args, **_kwargs):
        raise AssertionError("one-shot proof must not build the service runtime")

    monkeypatch.setattr(cli, "run_e2e_command", proof)
    monkeypatch.setattr(cli, "build_runtime", forbidden)

    code, output, error = invoke(
        ["light-e2e", "--confirm-run"], private_env
    )

    assert code == 0 and error is None and output == public
    assert private_env["private"] not in json.dumps(output)


def test_light_e2e_preserves_sanitized_typed_failure(monkeypatch) -> None:
    def proof(*, environ):
        assert environ == {}
        raise E2ECommandError(
            "publication_outcome_unknown",
            operation_id="b" * 32,
            mutation_attempted=True,
        )

    monkeypatch.setattr(cli, "run_e2e_command", proof)

    code, output, error = invoke(["light-e2e", "--confirm-run"], {})

    assert code == 1 and output is None
    assert error == {
        "category": "publication_outcome_unknown",
        "mutation_attempted": True,
        "ok": False,
        "operation_id": "b" * 32,
        "status": "failed_closed",
    }


def test_job_recovery_discovery_and_exact_retry_are_local_and_content_free(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    seed_recoverable_job(state_dir)
    env = {"LIGHTPHONE_GPT_STATE_DIR": str(state_dir)}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("job recovery must not construct adapters")

    monkeypatch.setattr(cli, "build_runtime", forbidden)

    code, listing, error = invoke(["job-recovery", "list"], env)
    assert code == 0 and error is None
    assert listing == {
        "ok": True,
        "candidates": [
            {
                "source_note_id": "recoverable-source",
                "failure_stage": "transcription",
                "attempts": 1,
                "failed_at": "2026-10-15T09:00:00+00:00",
            }
        ],
        "total": 1,
        "truncated": False,
    }

    code, inspected, error = invoke(
        [
            "job-recovery",
            "inspect",
            "--source-note-id",
            "recoverable-source",
        ],
        env,
    )
    assert code == 0 and error is None
    assert inspected["active"] is True
    assert inspected["retry_allowed"] is True
    assert "private" not in json.dumps(inspected)

    retry = [
        "job-recovery",
        "retry",
        "--source-note-id",
        "recoverable-source",
        "--expected-stage",
        "transcription",
        "--expected-attempts",
        "1",
    ]
    code, output, error = invoke(retry, env)
    assert code == 2 and output is None
    assert error == {
        "error": "job_recovery_confirmation_required",
        "ok": False,
    }

    code, output, error = invoke(
        [*retry[:-1], "2", "--confirm"], env
    )
    assert code == 1 and output is None
    assert error == {"error": "job_recovery_candidate_mismatch", "ok": False}

    code, recovered, error = invoke([*retry, "--confirm"], env)
    assert code == 0 and error is None
    assert recovered["retried"] is True
    assert recovered["source_note_id"] == "recoverable-source"
    assert recovered["failure_stage"] == "transcription"
    assert recovered["prior_attempts"] == 1
    assert recovered["resumed_state"] == "queued"
    assert "private" not in json.dumps(recovered)

    with Store(state_dir / "bridge.sqlite3") as store:
        job = store.get_job("recoverable-source")
        assert job is not None and job.state is JobState.QUEUED
        assert job.audio == b"private audio payload"
        assert job.transcription_attempts == 0
        assert store.connection.execute(
            "SELECT COUNT(*) FROM job_recovery_audit"
        ).fetchone()[0] == 1


def test_ignored_audio_recovery_redownloads_read_only_and_restages_exact_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    private_audio = b"private ignored audio"
    digest = seed_ignored_audio_job(
        state_dir, payload=private_audio
    )
    env = {
        "LIGHTPHONE_GPT_STATE_DIR": str(state_dir),
        "lightphone_username": "private-user",
        "lightphone_password": "private-password",
        "lightphone_device_id": "selected-device",
    }
    calls = {"download": 0, "close": 0}

    class ReadOnlyProvider:
        def __init__(self, **kwargs):
            assert kwargs["write_policy"] is cli.LightWritePolicy.READ_ONLY
            assert kwargs["device_id"] == "selected-device"

        def download_audio_with_metadata(self, note_id):
            calls["download"] += 1
            return (
                RemoteNote(
                    note_id,
                    NoteKind.AUDIO,
                    "Private title",
                    datetime(2026, 10, 15, 9, 1, tzinfo=UTC),
                    "audio/mp4",
                ),
                private_audio,
            )

        def create_text_note(self, **_kwargs):
            raise AssertionError("ignored audio recovery must not create")

        def update_text_note(self, *_args, **_kwargs):
            raise AssertionError("ignored audio recovery must not update")

        def close(self):
            calls["close"] += 1

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ignored audio recovery must not build runtime")

    monkeypatch.setattr(cli, "GaradoLightProvider", ReadOnlyProvider)
    monkeypatch.setattr(cli, "build_runtime", forbidden)

    code, inspected, error = invoke(
        [
            "ignored-audio-recovery",
            "inspect",
            "--source-note-id",
            "ignored-audio-source",
        ],
        env,
    )
    assert code == 0 and error is None
    assert inspected == {
        "active": True,
        "ignored_at": "2026-10-15T09:00:00+00:00",
        "ignored_reason": "prefix_missing",
        "model_attempts": 0,
        "ok": True,
        "publication_attempts": 0,
        "queue_seq": 1,
        "retained_sha256": digest,
        "source_kind": "audio",
        "source_note_id": "ignored-audio-source",
        "state": "ignored",
        "stt_attempts": 1,
    }
    assert calls == {"download": 0, "close": 0}

    command = [
        "ignored-audio-recovery",
        "restage",
        "--source-note-id",
        "ignored-audio-source",
        "--expected-audio-sha256",
        digest,
        "--expected-state",
        "ignored",
        "--expected-source-kind",
        "audio",
        "--expected-ignored-reason",
        "prefix_missing",
        "--expected-transcription-attempts",
        "1",
        "--expected-model-attempts",
        "0",
        "--expected-publication-attempts",
        "0",
    ]
    code, output, error = invoke(command, env)
    assert code == 2 and output is None
    assert error == {
        "error": "ignored_audio_recovery_confirmation_required",
        "ok": False,
    }

    code, recovered, error = invoke([*command, "--confirm"], env)
    assert code == 0 and error is None
    assert recovered["action"] == "restaged"
    assert recovered["audit_id"] == 1
    assert recovered["prior_stt_attempts"] == 1
    assert recovered["stt_attempts"] == 0
    assert recovered["retained_sha256"] == digest
    assert recovered["queue_seq"] == 1
    assert "private" not in json.dumps(recovered)

    code, repeated, error = invoke([*command, "--confirm"], env)
    assert code == 0 and error is None
    assert repeated["action"] == "already_restaged"
    assert repeated["audit_id"] == recovered["audit_id"]
    assert repeated["restaged_at"] == recovered["restaged_at"]
    assert calls == {"download": 2, "close": 2}

    with Store(state_dir / "bridge.sqlite3") as store:
        job = store.get_job("ignored-audio-source")
        assert job is not None and job.state is JobState.QUEUED
        assert job.audio == private_audio
        assert job.transcription_attempts == 0
        assert job.model_attempts == 0
        assert job.publication_attempts == 0
        assert store.connection.execute(
            "SELECT COUNT(*) FROM job_recovery_audit"
        ).fetchone()[0] == 1


def test_ignored_audio_recovery_fails_before_auth_on_local_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    seed_ignored_audio_job(state_dir)
    constructed = 0

    class ForbiddenProvider:
        def __init__(self, **_kwargs):
            nonlocal constructed
            constructed += 1
            raise AssertionError("local mismatch must fail before authentication")

    monkeypatch.setattr(cli, "GaradoLightProvider", ForbiddenProvider)
    code, output, error = invoke(
        [
            "ignored-audio-recovery",
            "restage",
            "--source-note-id",
            "ignored-audio-source",
            "--expected-audio-sha256",
            "f" * 64,
            "--expected-state",
            "ignored",
            "--expected-source-kind",
            "audio",
            "--expected-ignored-reason",
            "prefix_missing",
            "--expected-transcription-attempts",
            "1",
            "--expected-model-attempts",
            "0",
            "--expected-publication-attempts",
            "0",
            "--confirm",
        ],
        {
            "LIGHTPHONE_GPT_STATE_DIR": str(state_dir),
            "lightphone_username": "private-user",
            "lightphone_password": "private-password",
        },
    )
    assert code == 1 and output is None
    assert error == {
        "error": "ignored_audio_recovery_candidate_mismatch",
        "ok": False,
    }
    assert constructed == 0
    with Store(state_dir / "bridge.sqlite3") as store:
        job = store.get_job("ignored-audio-source")
        assert job is not None and job.state is JobState.IGNORED
        assert store.connection.execute(
            "SELECT COUNT(*) FROM job_recovery_audit"
        ).fetchone()[0] == 0


@pytest.mark.parametrize(
    ("remote_note_id", "remote_kind", "remote_payload", "expected_error"),
    (
        (
            "different-source",
            NoteKind.AUDIO,
            b"private ignored audio",
            "ignored_audio_recovery_remote_identity_mismatch",
        ),
        (
            "ignored-audio-source",
            NoteKind.TEXT,
            b"private ignored audio",
            "ignored_audio_recovery_remote_kind_mismatch",
        ),
        (
            "ignored-audio-source",
            NoteKind.AUDIO,
            b"changed private audio",
            "ignored_audio_recovery_remote_hash_mismatch",
        ),
    ),
)
def test_ignored_audio_recovery_rejects_changed_remote_identity(
    tmp_path: Path,
    monkeypatch,
    remote_note_id: str,
    remote_kind: NoteKind,
    remote_payload: bytes,
    expected_error: str,
) -> None:
    state_dir = tmp_path / "state"
    digest = seed_ignored_audio_job(state_dir)

    class ReadOnlyProvider:
        def __init__(self, **kwargs):
            assert kwargs["write_policy"] is cli.LightWritePolicy.READ_ONLY

        def download_audio_with_metadata(self, _note_id):
            return (
                RemoteNote(
                    remote_note_id,
                    remote_kind,
                    "Private title",
                    datetime(2026, 10, 15, 9, 1, tzinfo=UTC),
                    "audio/mp4",
                ),
                remote_payload,
            )

        def close(self):
            return None

    monkeypatch.setattr(cli, "GaradoLightProvider", ReadOnlyProvider)
    code, output, error = invoke(
        [
            "ignored-audio-recovery",
            "restage",
            "--source-note-id",
            "ignored-audio-source",
            "--expected-audio-sha256",
            digest,
            "--expected-state",
            "ignored",
            "--expected-source-kind",
            "audio",
            "--expected-ignored-reason",
            "prefix_missing",
            "--expected-transcription-attempts",
            "1",
            "--expected-model-attempts",
            "0",
            "--expected-publication-attempts",
            "0",
            "--confirm",
        ],
        {
            "LIGHTPHONE_GPT_STATE_DIR": str(state_dir),
            "lightphone_username": "private-user",
            "lightphone_password": "private-password",
        },
    )
    assert code == 1 and output is None
    assert error == {"error": expected_error, "ok": False}
    with Store(state_dir / "bridge.sqlite3") as store:
        job = store.get_job("ignored-audio-source")
        assert job is not None and job.state is JobState.IGNORED
        assert store.connection.execute(
            "SELECT COUNT(*) FROM job_recovery_audit"
        ).fetchone()[0] == 0


def test_light_probe_requires_an_explicit_command_and_reports_aggregates(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeProvider:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def list_notes(self):
            from datetime import UTC, datetime
            from lightphone_gpt.models import NoteKind, RemoteNote

            return [
                RemoteNote("private-id", NoteKind.AUDIO, "private title", datetime.now(UTC)),
                RemoteNote("reply-id", NoteKind.TEXT, "private reply", datetime.now(UTC)),
            ]

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(cli, "GaradoLightProvider", FakeProvider)
    code, output, error = invoke(
        ["light-probe"],
        {
            "lightphone_username": "private-user",
            "lightphone_password": "private-password",
            "lightphone_device_id": "selected-device",
        },
    )

    assert code == 0 and error is None
    assert output == {
        "audio_notes": 1,
        "notes_total": 2,
        "ok": True,
        "text_notes": 1,
        "write_policy": "read_only",
    }
    assert calls["write_policy"].value == "read_only"
    assert calls["device_id"] == "selected-device"
    assert calls["closed"] is True
    assert "private" not in json.dumps(output)


def test_cli_configuration_errors_do_not_echo_env_values(tmp_path: Path) -> None:
    secret = "credential-that-must-not-be-logged"
    code, _, error = invoke(
        ["status"],
        {
            "LIGHTPHONE_GPT_STATE_DIR": str(tmp_path / "state"),
            "LIGHTPHONE_GPT_PROVIDER": secret,
        },
    )

    assert code == 2
    assert secret not in json.dumps(error)


def test_chatgpt_auth_status_does_not_build_service_runtime(tmp_path: Path) -> None:
    credential = tmp_path / "chatgpt" / "credentials.json"
    code, output, error = invoke(
        ["chatgpt-auth", "status", "--credential-file", str(credential)],
        {"LIGHTPHONE_GPT_PROVIDER": "forbidden-provider"},
    )

    assert code == 0 and error is None
    assert output == {"authorized": False, "ok": True}
    assert not credential.exists()


@pytest.mark.parametrize(
    ("raised", "exit_code", "category"),
    [
        (ChatGPTAuthorizationRequiredError("private"), 1, "authorization_required"),
        (ChatGPTTransientError("private"), 1, "temporarily_unavailable"),
        (ChatGPTConfigurationError("private"), 2, "configuration_error"),
    ],
)
def test_chatgpt_auth_cli_errors_are_content_free(
    tmp_path: Path,
    monkeypatch,
    raised: Exception,
    exit_code: int,
    category: str,
) -> None:
    def fail(*_args, **_kwargs):
        raise raised

    monkeypatch.setattr(cli, "run_chatgpt_auth_command", fail)
    code, output, error = invoke(
        [
            "chatgpt-auth",
            "login",
            "--credential-file",
            str(tmp_path / "credentials.json"),
        ],
        {},
    )

    assert code == exit_code and output is None
    assert error == {"error": category, "ok": False}
    assert "private" not in json.dumps(error)


def test_publication_recovery_inspect_and_not_applied_are_local_and_audited(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    digest = seed_ambiguous_create(state_dir)
    env = {"LIGHTPHONE_GPT_STATE_DIR": str(state_dir)}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("recovery must not construct a provider runtime")

    monkeypatch.setattr(cli, "build_runtime", forbidden)
    monkeypatch.setattr(cli, "GaradoLightProvider", forbidden)

    code, inspected, error = invoke(["publication-recovery", "inspect"], env)
    assert code == 0 and error is None
    assert inspected == {
        "active": True,
        "gate": "create_ambiguous",
        "gated_at": "2026-10-15T08:00:00+00:00",
        "ok": True,
        "output_epoch": 0,
        "publication_attempts": 1,
        "publication_sha256": digest,
        "publication_title_sha256": hashlib.sha256(b"AI READY").hexdigest(),
        "provisional_output_note_id": None,
        "source_note_id": "source",
        "stored_output_mode": "rolling",
    }

    args = [
        "publication-recovery",
        "resolve-create-not-applied",
        "--source-note-id",
        "source",
        "--expected-publication-sha256",
        digest,
        "--expected-output-epoch",
        "0",
    ]
    code, output, error = invoke(args, env)
    assert code == 2 and output is None
    assert error["detail"] == "publication recovery requires --confirm"

    code, resolved, error = invoke([*args, "--confirm"], env)
    assert code == 0 and error is None
    assert resolved["decision"] == "not_applied"
    assert resolved["source_note_id"] == "source"
    assert resolved["publication_sha256"] == digest
    assert resolved["output_note_id"] is None
    assert resolved["audit_id"] == 1

    with Store(state_dir / "bridge.sqlite3") as store:
        job = store.get_job("source")
        assert job is not None and job.state is JobState.FAILED
        assert not job.retryable
        assert store.oldest_actionable(now=datetime.now(UTC)) is None
        assert store.output_gate() is None
        audit = store.connection.execute(
            "SELECT decision, source_note_id, publication_sha256 "
            "FROM publication_recovery_audit"
        ).fetchone()
        assert tuple(audit) == ("not_applied", "source", digest)


def test_publication_recovery_applied_requires_exact_remote_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    digest = seed_ambiguous_create(state_dir)
    env = {
        "LIGHTPHONE_GPT_STATE_DIR": str(state_dir),
        "lightphone_username": "private-user",
        "lightphone_password": "private-password",
        "lightphone_device_id": "selected-device",
    }
    calls: dict[str, object] = {"get": 0, "read": 0, "closed": 0}
    remote_content = {"value": b"wrong private remote content"}

    class ReadOnlyProvider:
        def __init__(self, **kwargs):
            calls["init"] = int(calls.get("init", 0)) + 1
            calls["write_policy"] = kwargs["write_policy"]
            calls["device_id"] = kwargs["device_id"]

        def get_note(self, note_id):
            calls["get"] = int(calls["get"]) + 1
            return RemoteNote(
                note_id,
                NoteKind.TEXT,
                "AI READY",
                datetime(2026, 10, 15, 8, 1, tzinfo=UTC),
                "text/plain; charset=utf-8",
            )

        def read_note_content(self, note_id):
            calls["read"] = int(calls["read"]) + 1
            return remote_content["value"]

        def create_text_note(self, **_kwargs):
            raise AssertionError("recovery must not mutate Light")

        def update_text_note(self, *_args, **_kwargs):
            raise AssertionError("recovery must not mutate Light")

        def close(self):
            calls["closed"] = int(calls["closed"]) + 1

    monkeypatch.setattr(cli, "GaradoLightProvider", ReadOnlyProvider)
    base = [
        "publication-recovery",
        "resolve-create-applied",
        "--source-note-id",
        "source",
        "--expected-publication-sha256",
        digest,
        "--expected-output-epoch",
        "0",
        "--output-note-id",
        "remote-output",
        "--confirm",
    ]

    code, output, error = invoke(base, env)
    assert code == 1 and output is None
    assert error == {"error": "recovery_remote_content_mismatch", "ok": False}
    assert "private" not in json.dumps(error)
    with Store(state_dir / "bridge.sqlite3") as store:
        assert store.output_gate() == "create_ambiguous"
        assert store.connection.execute(
            "SELECT COUNT(*) FROM publication_recovery_audit"
        ).fetchone()[0] == 0

    remote_content["value"] = b"stored publication payload"
    code, resolved, error = invoke(base, env)
    assert code == 0 and error is None
    assert resolved["decision"] == "applied"
    assert resolved["output_note_id"] == "remote-output"
    assert resolved["verification_method"] == "light_read_only_v1"
    assert calls["write_policy"] is cli.LightWritePolicy.READ_ONLY
    assert calls["device_id"] == "selected-device"
    assert calls["get"] == 3
    assert calls["read"] == 3
    assert calls["closed"] == 2
    with Store(state_dir / "bridge.sqlite3") as store:
        assert store.output_gate() is None
        assert store.get_job("source").state is JobState.PUBLISHED
        assert store.output_snapshot().note_id == "remote-output"
        audit = store.connection.execute(
            "SELECT verification_method, observed_kind, "
            "publication_title_sha256, observed_title_sha256, "
            "remote_updated_at FROM publication_recovery_audit"
        ).fetchone()
        assert tuple(audit) == (
            "light_read_only_v1",
            "text",
            hashlib.sha256(b"AI READY").hexdigest(),
            hashlib.sha256(b"AI READY").hexdigest(),
            "2026-10-15T08:01:00.000000+00:00",
        )


def test_publication_recovery_rejects_local_mismatch_before_authentication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    seed_ambiguous_create(state_dir)
    constructed = 0

    class ForbiddenProvider:
        def __init__(self, **_kwargs):
            nonlocal constructed
            constructed += 1
            raise AssertionError("local mismatch must fail before authentication")

    monkeypatch.setattr(cli, "GaradoLightProvider", ForbiddenProvider)
    code, output, error = invoke(
        [
            "publication-recovery",
            "resolve-create-applied",
            "--source-note-id",
            "source",
            "--expected-publication-sha256",
            "f" * 64,
            "--expected-output-epoch",
            "0",
            "--output-note-id",
            "remote-output",
            "--confirm",
        ],
        {"LIGHTPHONE_GPT_STATE_DIR": str(state_dir)},
    )

    assert code == 1 and output is None
    assert error == {"error": "recovery_candidate_mismatch", "ok": False}
    assert constructed == 0
    with Store(state_dir / "bridge.sqlite3") as store:
        assert store.output_gate() == "create_ambiguous"


def test_publication_recovery_requires_two_stable_remote_samples(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    digest = seed_ambiguous_create(state_dir)
    timestamps = iter(
        (
            datetime(2026, 10, 15, 8, 1, tzinfo=UTC),
            datetime(2026, 10, 15, 8, 2, tzinfo=UTC),
        )
    )

    class ChangingProvider:
        def __init__(self, **kwargs):
            assert kwargs["write_policy"] is cli.LightWritePolicy.READ_ONLY

        def get_note(self, note_id):
            return RemoteNote(
                note_id,
                NoteKind.TEXT,
                "AI READY",
                next(timestamps),
                "text/plain; charset=utf-8",
            )

        def read_note_content(self, _note_id):
            return b"stored publication payload"

        def close(self):
            return None

    monkeypatch.setattr(cli, "GaradoLightProvider", ChangingProvider)
    code, output, error = invoke(
        [
            "publication-recovery",
            "resolve-create-applied",
            "--source-note-id",
            "source",
            "--expected-publication-sha256",
            digest,
            "--expected-output-epoch",
            "0",
            "--output-note-id",
            "remote-output",
            "--confirm",
        ],
        {
            "LIGHTPHONE_GPT_STATE_DIR": str(state_dir),
            "lightphone_username": "private-user",
            "lightphone_password": "private-password",
        },
    )

    assert code == 1 and output is None
    assert error == {"error": "recovery_remote_state_unstable", "ok": False}
    with Store(state_dir / "bridge.sqlite3") as store:
        assert store.output_gate() == "create_ambiguous"
        assert store.connection.execute(
            "SELECT COUNT(*) FROM publication_recovery_audit"
        ).fetchone()[0] == 0


def test_update_recovery_inspect_and_resume_readback_are_local_and_audited(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    desired, previous = seed_ambiguous_update(state_dir)
    env = {
        "LIGHTPHONE_GPT_STATE_DIR": str(state_dir),
        "LIGHTPHONE_GPT_MAX_UPDATE_READBACK_ATTEMPTS": "2",
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("read-back resume must not construct a provider")

    monkeypatch.setattr(cli, "build_runtime", forbidden)
    monkeypatch.setattr(cli, "GaradoLightProvider", forbidden)

    code, inspected, error = invoke(["publication-recovery", "inspect"], env)
    assert code == 0 and error is None
    assert inspected == {
        "active": True,
        "gate": "update_ambiguous",
        "gated_at": "2026-10-15T08:00:00+00:00",
        "ok": True,
        "output_epoch": 0,
        "output_note_id": "rolling-output",
        "previous_content_sha256": previous,
        "publication_attempts": 1,
        "publication_sha256": desired,
        "publication_title_sha256": hashlib.sha256(b"AI READY").hexdigest(),
        "readback_attempts": 2,
        "readback_exhausted": True,
        "readback_limit": 2,
        "source_note_id": "update-source",
        "stored_output_mode": "rolling",
    }

    command = [
        "publication-recovery",
        "resume-update-readback",
        "--source-note-id",
        "update-source",
        "--expected-publication-sha256",
        desired,
        "--expected-output-epoch",
        "0",
        "--expected-output-note-id",
        "rolling-output",
        "--expected-previous-content-sha256",
        previous,
        "--expected-readback-attempts",
        "2",
    ]
    code, output, error = invoke(command, env)
    assert code == 2 and output is None
    assert error["detail"] == "publication recovery requires --confirm"

    code, resumed, error = invoke([*command, "--confirm"], env)
    assert code == 0 and error is None
    assert resumed["action"] == "resume_readback"
    assert resumed["prior_readback_attempts"] == 2
    assert resumed["verification_method"] is None

    code, output, error = invoke([*command[:-1], "0", "--confirm"], env)
    assert code == 1 and output is None
    assert error == {"error": "update_readback_not_exhausted", "ok": False}

    with Store(state_dir / "bridge.sqlite3") as store:
        assert store.output_gate() == "update_ambiguous"
        assert not store.update_readback_exhausted(maximum=2)
        job = store.get_job("update-source")
        assert job is not None and job.publication_attempts == 1
        assert job.state is JobState.FAILED
        assert store.connection.execute(
            "SELECT COUNT(*) FROM update_recovery_audit"
        ).fetchone()[0] == 1
        audit = store.connection.execute(
            "SELECT action, source_note_id, prior_readback_attempts, "
            "observed_content_sha256 FROM update_recovery_audit"
        ).fetchone()
        assert tuple(audit) == (
            "resume_readback",
            "update-source",
            2,
            None,
        )


def test_update_recovery_applied_uses_two_read_only_samples_and_never_rewrites(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    desired, previous = seed_ambiguous_update(state_dir)
    calls = {"get": 0, "read": 0, "create": 0, "update": 0, "close": 0}
    remote_content = {"value": b"unexpected remote content"}

    class ReadOnlyProvider:
        def __init__(self, **kwargs):
            assert kwargs["write_policy"] is cli.LightWritePolicy.READ_ONLY

        def get_note(self, note_id):
            calls["get"] += 1
            return RemoteNote(
                note_id,
                NoteKind.TEXT,
                "AI READY",
                datetime(2026, 10, 15, 8, 1, tzinfo=UTC),
                "text/plain; charset=utf-8",
            )

        def read_note_content(self, _note_id):
            calls["read"] += 1
            return remote_content["value"]

        def create_text_note(self, **_kwargs):
            calls["create"] += 1
            raise AssertionError("update recovery must not create")

        def update_text_note(self, *_args, **_kwargs):
            calls["update"] += 1
            raise AssertionError("update recovery must not update")

        def close(self):
            calls["close"] += 1

    monkeypatch.setattr(cli, "GaradoLightProvider", ReadOnlyProvider)
    command = [
        "publication-recovery",
        "resolve-update-applied",
        "--source-note-id",
        "update-source",
        "--expected-publication-sha256",
        desired,
        "--expected-output-epoch",
        "0",
        "--expected-output-note-id",
        "rolling-output",
        "--expected-previous-content-sha256",
        previous,
        "--expected-readback-attempts",
        "2",
        "--confirm",
    ]
    env = {
        "LIGHTPHONE_GPT_STATE_DIR": str(state_dir),
        "lightphone_username": "private-user",
        "lightphone_password": "private-password",
    }

    code, output, error = invoke(command, env)

    assert code == 1 and output is None
    assert error == {"error": "recovery_remote_content_mismatch", "ok": False}
    with Store(state_dir / "bridge.sqlite3") as store:
        assert store.output_gate() == "update_ambiguous"
        assert store.connection.execute(
            "SELECT COUNT(*) FROM update_recovery_audit"
        ).fetchone()[0] == 0

    remote_content["value"] = b"stored updated reply"
    code, resolved, error = invoke(command, env)

    assert code == 0 and error is None
    assert resolved["action"] == "applied"
    assert resolved["verification_method"] == "light_read_only_v1"
    assert calls == {"get": 3, "read": 3, "create": 0, "update": 0, "close": 2}
    with Store(state_dir / "bridge.sqlite3") as store:
        assert store.output_gate() is None
        assert store.get_job("update-source").state is JobState.PUBLISHED
        assert store.output_snapshot().content_sha256 == desired
        audit = store.connection.execute(
            "SELECT action, output_note_id, previous_content_sha256, "
            "observed_content_sha256, verification_method "
            "FROM update_recovery_audit"
        ).fetchone()
        assert tuple(audit) == (
            "applied",
            "rolling-output",
            previous,
            desired,
            "light_read_only_v1",
        )


def test_assistant_recovery_inspect_is_local_and_content_free(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    seed_assistant_gate(state_dir)
    env = {"LIGHTPHONE_GPT_STATE_DIR": str(state_dir)}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("assistant recovery must not construct adapters")

    monkeypatch.setattr(cli, "build_runtime", forbidden)
    monkeypatch.setattr(cli, "GaradoLightProvider", forbidden)

    code, output, error = invoke(["assistant-recovery", "inspect"], env)

    assert code == 0 and error is None
    assert output == {
        "active": True,
        "gate": "authorization_required",
        "gated_at": "2026-10-15T09:00:00+00:00",
        "ok": True,
        "source_note_id": "assistant-source",
    }
    serialized = json.dumps(output)
    assert "private question" not in serialized
    assert "private dependency detail" not in serialized


def test_assistant_recovery_resume_requires_confirmation_and_exact_identity(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    seed_assistant_gate(
        state_dir,
        source_note_id="exact-source",
        gate="credential_ambiguous",
    )
    env = {"LIGHTPHONE_GPT_STATE_DIR": str(state_dir)}
    base = [
        "assistant-recovery",
        "resume",
        "--expected-gate",
        "credential_ambiguous",
        "--source-note-id",
        "exact-source",
    ]

    code, output, error = invoke(base, env)
    assert code == 1 and output is None
    assert error == {
        "error": "assistant_recovery_confirmation_required",
        "ok": False,
    }

    code, output, error = invoke(
        [
            "assistant-recovery",
            "resume",
            "--expected-gate",
            "authorization_required",
            "--source-note-id",
            "exact-source",
            "--confirm",
        ],
        env,
    )
    assert code == 1 and output is None
    assert error == {
        "error": "assistant_recovery_candidate_mismatch",
        "ok": False,
    }

    code, output, error = invoke(
        [
            "assistant-recovery",
            "resume",
            "--expected-gate",
            "credential_ambiguous",
            "--source-note-id",
            "wrong-source",
            "--confirm",
        ],
        env,
    )
    assert code == 1 and output is None
    assert error == {
        "error": "assistant_recovery_candidate_mismatch",
        "ok": False,
    }

    with Store(state_dir / "bridge.sqlite3") as store:
        assert store.assistant_gate() == "credential_ambiguous"
        job = store.get_job("exact-source")
        assert job is not None and job.state is JobState.FAILED


def test_assistant_recovery_resume_clears_gate_durably_and_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    seed_assistant_gate(state_dir, source_note_id="exact-source")
    env = {"LIGHTPHONE_GPT_STATE_DIR": str(state_dir)}
    original_checkpoint = Store.checkpoint
    checkpoints = 0

    def record_checkpoint(store: Store) -> None:
        nonlocal checkpoints
        checkpoints += 1
        original_checkpoint(store)

    monkeypatch.setattr(Store, "checkpoint", record_checkpoint)
    code, output, error = invoke(
        [
            "assistant-recovery",
            "resume",
            "--expected-gate",
            "authorization_required",
            "--source-note-id",
            "exact-source",
            "--confirm",
        ],
        env,
    )

    assert code == 0 and error is None
    assert output == {
        "gate": "authorization_required",
        "ok": True,
        "resumed": True,
        "source_note_id": "exact-source",
    }
    # One explicit durability checkpoint plus Store.close's checkpoint.
    assert checkpoints == 2
    with Store(state_dir / "bridge.sqlite3") as store:
        assert store.assistant_gate() is None
        job = store.get_job("exact-source")
        assert job is not None and job.state is JobState.TRANSCRIBED


def test_assistant_recovery_reports_inactive_and_fails_closed_on_resume(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    with Store(state_dir / "bridge.sqlite3") as store:
        store.establish_baseline([], now=datetime(2026, 10, 15, tzinfo=UTC))
    env = {"LIGHTPHONE_GPT_STATE_DIR": str(state_dir)}

    code, output, error = invoke(["assistant-recovery", "inspect"], env)
    assert code == 0 and error is None
    assert output == {"active": False, "ok": True}

    code, output, error = invoke(
        [
            "assistant-recovery",
            "resume",
            "--expected-gate",
            "authorization_required",
            "--source-note-id",
            "absent",
            "--confirm",
        ],
        env,
    )
    assert code == 1 and output is None
    assert error == {
        "error": "assistant_recovery_gate_missing",
        "ok": False,
    }
