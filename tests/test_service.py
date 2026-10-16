from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import Event, Thread
import time
from types import SimpleNamespace

import pytest

import lightphone_gpt.service as service_module

from lightphone_gpt.adapters.simulated_envelope import (
    SIMULATED_TRANSCRIPT_MIME,
    SimulatedEnvelopeTranscriber,
    encode_simulated_transcript,
)
from lightphone_gpt.config import ServiceConfig
from lightphone_gpt.contracts import PermanentDependencyError
from lightphone_gpt.models import (
    AppliedCreateVerification,
    JobState,
    MutationOutcome,
    MutationReceipt,
    NoteKind,
)
from lightphone_gpt.output_titles import is_output_title
from lightphone_gpt.runtime_lock import RuntimeLockError
from lightphone_gpt.service import (
    CycleReport,
    Pre0GuardError,
    build_runtime,
    run_cycle,
    runtime_status,
    serve,
)
from lightphone_gpt.store import Store
from lightphone_gpt.worker import PollReport, WorkReport


def config(tmp_path: Path, **changes: object) -> ServiceConfig:
    base = ServiceConfig(state_dir=tmp_path / "state")
    return replace(base, **changes)


def test_simulated_envelope_is_strict_and_content_free_on_error() -> None:
    private = "Assistant, my private simulated question"
    encoded = encode_simulated_transcript(private)
    adapter = SimulatedEnvelopeTranscriber()

    assert adapter.transcribe(encoded, mime_type=SIMULATED_TRANSCRIPT_MIME) == private
    with pytest.raises(PermanentDependencyError) as captured:
        adapter.transcribe(private.encode(), mime_type="audio/mp4")
    assert private not in str(captured.value)


def test_cycle_persists_across_runtime_restart_and_status_has_no_content(
    tmp_path: Path,
) -> None:
    settings = config(tmp_path)
    private = "Assistant, private question that status must not expose"
    with build_runtime(settings) as first:
        assert first.worker.poll().baselined == 0
        first.provider.seed_audio_note(
            encode_simulated_transcript(private),
            mime_type=SIMULATED_TRANSCRIPT_MIME,
        )
        cycle = run_cycle(first)
        assert cycle.ok
        assert runtime_status(first)["jobs_published"] == 1

    with build_runtime(settings) as restarted:
        status = runtime_status(restarted)
        assert status["jobs_published"] == 1
        assert status["jobs_actionable"] == 0
        serialized = json.dumps(status)
        assert "private question" not in serialized
        assert run_cycle(restarted).poll.enqueued == 0

    assert settings.state_dir.stat().st_mode & 0o777 == 0o700
    assert settings.bridge_database.stat().st_mode & 0o777 == 0o600
    assert settings.simulator_database.stat().st_mode & 0o777 == 0o600


def test_cycle_reports_and_completes_enabled_source_cleanup(tmp_path: Path) -> None:
    settings = config(tmp_path, delete_source_after_publish=True)
    with build_runtime(settings) as runtime:
        assert runtime.worker.poll().baselined == 0
        source = runtime.provider.seed_audio_note(
            encode_simulated_transcript("Assistant, remove this source"),
            mime_type=SIMULATED_TRANSCRIPT_MIME,
        )

        first_cycle = run_cycle(runtime)
        assert first_cycle.ok is False
        cycle = run_cycle(runtime)
        status = runtime_status(runtime)

        assert cycle.ok is True
        assert runtime.provider.get_note(source.note_id) is None
        assert status["source_cleanup_enabled"] is True
        assert status["source_cleanups_confirmed"] == 1
        assert status["source_cleanups_unresolved"] == 0
        assert status["source_cleanup_blocked"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "light"},
        {"assistant": "chatgpt"},
        {"provider": "light", "assistant": "chatgpt"},
    ],
)
def test_continuous_service_refuses_real_integrations_before_state_creation(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    settings = config(tmp_path, **changes)
    with pytest.raises(Pre0GuardError):
        build_runtime(settings)
    assert not settings.state_dir.exists()


def test_service_stops_cleanly_via_event(tmp_path: Path) -> None:
    settings = config(
        tmp_path,
        poll_interval_seconds=0.02,
        retention_interval_seconds=0.02,
    )
    stop = Event()
    lifecycle: list[str] = []

    class FakeHealthServer:
        def __init__(self, host, port, state):
            assert host == "127.0.0.1" and port == 8824
            self.state = state

        def start(self):
            lifecycle.append("started")

        def close(self):
            lifecycle.append("closed")

    thread = Thread(
        target=serve,
        args=(settings,),
        kwargs={
            "stop_event": stop,
            "install_signal_handlers": False,
            "health_server_factory": FakeHealthServer,
        },
    )
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not settings.bridge_database.exists():
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert settings.bridge_database.exists()
    assert lifecycle == ["started", "closed"]


def test_service_runs_notifier_on_separate_port_and_closes_it_cleanly(
    tmp_path: Path,
) -> None:
    token = "test_notifier_token_0123456789_ABCDEF"
    settings = config(
        tmp_path,
        poll_interval_seconds=0.02,
        notifier_port=9912,
        notifier_wait_seconds=24,
        notifier_token=token,
    )
    stop = Event()
    lifecycle: list[str] = []
    captured_ready_states: list[object] = []

    class FakeHealthServer:
        def __init__(self, host, port, state):
            assert (host, port) == ("127.0.0.1", 8824)

        def start(self):
            lifecycle.append("health-started")

        def close(self):
            lifecycle.append("health-closed")

    class FakeNotifierServer:
        def __init__(self, host, port, state, supplied_token, wait_seconds):
            assert (host, port) == ("127.0.0.1", 9912)
            assert supplied_token == token
            assert wait_seconds == 24
            captured_ready_states.append(state)

        def start(self):
            lifecycle.append("notifier-started")

        def close(self):
            lifecycle.append("notifier-closed")

    thread = Thread(
        target=serve,
        args=(settings,),
        kwargs={
            "stop_event": stop,
            "install_signal_handlers": False,
            "health_server_factory": FakeHealthServer,
            "notifier_server_factory": FakeNotifierServer,
        },
    )
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not captured_ready_states:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert lifecycle == [
        "health-started",
        "notifier-started",
        "notifier-closed",
        "health-closed",
    ]
    assert captured_ready_states[0].wait_for_newer(0, 0.01) is None


def test_service_releases_runtime_when_notifier_setup_fails(tmp_path: Path) -> None:
    settings = config(
        tmp_path,
        notifier_token="test_notifier_token_0123456789_ABCDEF",
    )

    def fail_notifier(*_args):
        raise OSError("notifier address unavailable")

    class FakeHealthServer:
        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def close(self):
            pass

    with pytest.raises(OSError, match="notifier address unavailable"):
        serve(
            settings,
            install_signal_handlers=False,
            health_server_factory=FakeHealthServer,
            notifier_server_factory=fail_notifier,
        )

    with build_runtime(settings) as restarted:
        assert runtime_status(restarted)["database_ok"] is True


def test_service_closes_database_if_health_setup_fails(tmp_path: Path) -> None:
    settings = config(tmp_path)

    def fail_health(*_args):
        raise OSError("address unavailable")

    with pytest.raises(OSError, match="address unavailable"):
        serve(
            settings,
            install_signal_handlers=False,
            health_server_factory=fail_health,
        )

    # A fresh runtime can immediately take ownership of the same database.
    with build_runtime(settings) as restarted:
        assert runtime_status(restarted)["database_ok"] is True


def test_second_mutating_runtime_is_refused_until_owner_closes(tmp_path: Path) -> None:
    settings = config(tmp_path)
    with build_runtime(settings):
        with pytest.raises(RuntimeLockError, match="another lightphone-gpt runtime"):
            build_runtime(settings)
        assert (settings.state_dir / "runtime.lock").stat().st_mode & 0o777 == 0o600

    with build_runtime(settings) as restarted:
        assert runtime_status(restarted)["database_ok"] is True


def test_cycle_failure_includes_failed_work_stage() -> None:
    assert CycleReport(PollReport(), ()).ok
    assert not CycleReport(PollReport(queue_limited=True), ()).ok
    assert not CycleReport(PollReport(error="poll failed"), ()).ok
    assert not CycleReport(
        PollReport(output_gate="create_ambiguous"), ()
    ).ok
    assert not CycleReport(PollReport(), (), output_gate="create_ambiguous").ok
    assert not CycleReport(
        PollReport(),
        (WorkReport(True, note_id="opaque", state=JobState.FAILED),),
    ).ok
    assert not CycleReport(
        PollReport(), (), unresolved_job_failures=1
    ).ok
    assert not CycleReport(
        PollReport(), (), unresolved_source_cleanups=1
    ).ok
    assert not CycleReport(PollReport(source_cleanup_error="failed"), ()).ok


def test_cycle_does_not_drain_when_poll_failed(tmp_path: Path) -> None:
    with build_runtime(config(tmp_path)) as runtime:
        runtime.worker.poll = lambda: PollReport(error="cloud unavailable")  # type: ignore[method-assign]

        def unexpected_drain(*, max_steps: int):
            del max_steps
            raise AssertionError("drain must not run after a failed poll")

        runtime.worker.drain = unexpected_drain  # type: ignore[method-assign]
        report = run_cycle(runtime)

    assert report.poll.error == "cloud unavailable"
    assert report.work == ()
    assert not report.ok


def test_cycle_drains_recovery_work_when_poll_reports_output_gate(
    tmp_path: Path,
) -> None:
    with build_runtime(config(tmp_path)) as runtime:
        runtime.worker.poll = lambda: PollReport(  # type: ignore[method-assign]
            output_gate="create_ambiguous"
        )
        drained: list[int] = []

        def recovery_drain(*, max_steps: int):
            drained.append(max_steps)
            return []

        runtime.worker.drain = recovery_drain  # type: ignore[method-assign]
        report = run_cycle(runtime)

    assert drained
    assert report.poll.output_gate is None
    assert report.work == ()
    assert report.output_gate is None
    assert report.ok


def test_cycle_admits_later_input_while_uncertain_update_recovers_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = config(
        tmp_path,
        max_update_readback_attempts=2,
        confirm_output_deletion_after=2,
    )
    with build_runtime(settings) as runtime:
        assert runtime.worker.poll().baselined == 0

        first = runtime.provider.seed_audio_note(
            encode_simulated_transcript("Assistant, first request"),
            mime_type=SIMULATED_TRANSCRIPT_MIME,
        )
        assert run_cycle(runtime).ok
        first_job = runtime.store.get_job(first.note_id)
        assert first_job is not None
        assert first_job.state is JobState.PUBLISHED
        first_output = first_job.output_note_id
        assert first_output is not None

        update_calls = 0
        update = runtime.provider.update_text_note

        def uncertain_once(note_id: str, **kwargs: object) -> MutationReceipt:
            nonlocal update_calls
            update_calls += 1
            receipt = update(note_id, **kwargs)
            if update_calls == 1:
                assert receipt.outcome is MutationOutcome.VERIFIED
                return MutationReceipt(
                    MutationOutcome.UNKNOWN,
                    note=receipt.note,
                    observed_content_sha256=receipt.observed_content_sha256,
                    detail="injected uncertain update",
                )
            return receipt

        monkeypatch.setattr(runtime.provider, "update_text_note", uncertain_once)

        second = runtime.provider.seed_audio_note(
            encode_simulated_transcript("Assistant, second request"),
            mime_type=SIMULATED_TRANSCRIPT_MIME,
        )
        gated = run_cycle(runtime)
        second_job = runtime.store.get_job(second.note_id)
        assert second_job is not None
        assert second_job.state is JobState.FAILED
        assert second_job.response_id is None
        assert gated.output_gate == "update_ambiguous"
        assert update_calls == 1
        assert len(runtime.assistant.request_digests) == 2

        runtime.provider.delete_note(first_output)
        third = runtime.provider.seed_audio_note(
            encode_simulated_transcript("Assistant, third request"),
            mime_type=SIMULATED_TRANSCRIPT_MIME,
        )

        intake = run_cycle(runtime)
        assert intake.poll.output_gate is None
        assert intake.poll.enqueued == 1
        assert [report.note_id for report in intake.work] == [
            third.note_id,
            third.note_id,
            third.note_id,
        ]
        third_job = runtime.store.get_job(third.note_id)
        assert third_job is not None
        assert third_job.state is JobState.PUBLISHED
        assert update_calls == 1
        assert len(runtime.assistant.request_digests) == 3

        second_job = runtime.store.get_job(second.note_id)
        third_job = runtime.store.get_job(third.note_id)
        assert second_job is not None and third_job is not None
        assert second_job.state is JobState.FAILED
        assert third_job.state is JobState.PUBLISHED
        assert second_job.output_note_id is None
        assert third_job.output_note_id != first_output
        assert update_calls == 1
        assert len(runtime.assistant.request_digests) == 3


def test_cycle_reports_final_cleared_inference_gate_after_assistant_reset(
    tmp_path: Path,
) -> None:
    with build_runtime(config(tmp_path)) as runtime:
        assert runtime.worker.poll().baselined == 0
        stalled = runtime.provider.seed_audio_note(
            encode_simulated_transcript("Assistant, uncertain model request"),
            mime_type=SIMULATED_TRANSCRIPT_MIME,
        )
        assert runtime.worker.poll().enqueued == 1
        assert runtime.worker.run_once().state is JobState.TRANSCRIBED
        runtime.store.pause_assistant(
            stalled.note_id,
            gate="inference_ambiguous",
            error="injected uncertain model outcome",
            now=datetime.now(timezone.utc),
        )
        reset = runtime.provider.seed_audio_note(
            encode_simulated_transcript("Assistant, reset"),
            mime_type=SIMULATED_TRANSCRIPT_MIME,
        )

        report = run_cycle(runtime)

        assert report.assistant_gate is None
        assert runtime.store.assistant_gate() is None
        assert runtime.store.get_job(stalled.note_id).state is JobState.FAILED
        assert runtime.store.get_job(reset.note_id).state is JobState.PUBLISHED
        assert report.ok

def test_restart_gates_interrupted_create_before_poll_can_ingest_own_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = config(tmp_path)
    output_note_id: str
    output_updated_at: datetime

    with build_runtime(settings) as runtime:
        assert runtime.worker.poll().baselined == 0
        runtime.provider.seed_audio_note(
            encode_simulated_transcript("Assistant, preserve this answer"),
            mime_type=SIMULATED_TRANSCRIPT_MIME,
        )

        def fail_after_remote_create(*_args, **_kwargs):
            raise sqlite3.OperationalError("injected local persistence failure")

        monkeypatch.setattr(
            runtime.store, "remember_output_note", fail_after_remote_create
        )
        with pytest.raises(sqlite3.OperationalError, match="persistence failure"):
            run_cycle(runtime)

        outputs = [
            note
            for note in runtime.provider.list_notes()
            if note.kind is NoteKind.TEXT
            and is_output_title(note.title, "AI READY")
        ]
        assert len(outputs) == 1
        output_note_id = outputs[0].note_id
        output_updated_at = outputs[0].updated_at or datetime.now(timezone.utc)
        assert runtime.store.get_job(output_note_id) is None

    with build_runtime(settings) as restarted:
        report = run_cycle(restarted)

        assert report.poll.output_gate == "create_ambiguous"
        assert report.work == ()
        assert restarted.store.get_job(output_note_id) is None
        assert restarted.store.connection.execute(
            "SELECT 1 FROM request_ledger WHERE note_id = ?", (output_note_id,)
        ).fetchone() is None
        candidate = restarted.store.ambiguous_create_candidate()
        assert candidate is not None
        assert candidate.provisional_output_note_id is None

        observed_content = restarted.provider.read_note_content(output_note_id)
        assert hashlib.sha256(observed_content).hexdigest() == (
            candidate.publication_sha256
        )
        started = datetime.now(timezone.utc)
        result = restarted.store.resolve_ambiguous_create(
            source_note_id=candidate.source_note_id,
            expected_publication_sha256=candidate.publication_sha256,
            expected_output_epoch=candidate.output_epoch,
            decision="applied",
            verification=AppliedCreateVerification(
                method="light_read_only_v1",
                output_note_id=output_note_id,
                observed_content_sha256=candidate.publication_sha256,
                observed_title_sha256=hashlib.sha256(
                    outputs[0].title.encode("utf-8")
                ).hexdigest(),
                observed_kind=NoteKind.TEXT,
                remote_updated_at=output_updated_at,
                started_at=started,
                completed_at=started + timedelta(seconds=1),
            ),
            now=started + timedelta(seconds=2),
        )

        assert result.output_note_id == output_note_id
        assert restarted.store.output_snapshot().note_id == output_note_id
        recovered_job = restarted.store.get_job(candidate.source_note_id)
        assert recovered_job is not None
        assert recovered_job.state is JobState.PUBLISHED
        ledger = restarted.store.connection.execute(
            "SELECT disposition, source_kind FROM request_ledger WHERE note_id = ?",
            (output_note_id,),
        ).fetchone()
        assert tuple(ledger) == ("bridge_output", "text")


def test_cycle_does_not_poll_or_drain_while_assistant_is_blocked(
    tmp_path: Path,
) -> None:
    with build_runtime(config(tmp_path)) as runtime:
        runtime.store.assistant_gate = lambda: "authorization_required"  # type: ignore[method-assign]

        def forbidden(*_args, **_kwargs):
            raise AssertionError("cloud and model work must pause behind the gate")

        runtime.worker.poll = forbidden  # type: ignore[method-assign]
        runtime.worker.drain = forbidden  # type: ignore[method-assign]
        report = run_cycle(runtime)
        status = runtime_status(runtime)

    assert report.assistant_gate == "authorization_required"
    assert report.work == ()
    assert not report.ok
    assert status["assistant_blocked"] is True


def test_runtime_status_reports_content_free_queue_limit(tmp_path: Path) -> None:
    with build_runtime(config(tmp_path, max_pending_jobs=1)) as runtime:
        runtime.store.pending_job_count = lambda: 1  # type: ignore[method-assign]
        status = runtime_status(runtime)

    assert status["queue_limited"] is True


def _production_config(tmp_path: Path) -> ServiceConfig:
    authorization = tmp_path / "authorization"
    return ServiceConfig(
        state_dir=tmp_path / "state",
        provider="light",
        assistant="chatgpt",
        transcriber="faster-whisper",
        output_mode="rolling",
        output_title="AI READY",
        faster_whisper_model="local-base",
        faster_whisper_model_cache=tmp_path / "models",
        production_activated=True,
        chatgpt_credential_file=authorization / "credentials.json",
        chatgpt_host_id_file=authorization / "host-id",
        chatgpt_model="gpt-test-exact",
    )


def test_production_factory_wires_bound_adapters_and_does_not_create_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _production_config(tmp_path)
    monkeypatch.setenv("lightphone_username", "private-light-user")
    monkeypatch.setenv("lightphone_password", "private-light-password")
    settings.state_dir.mkdir(parents=True)
    prior = Store(settings.bridge_database)
    prior.connection.execute(
        "INSERT INTO output_surfaces(note_id, epoch, content_sha256, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("durable-output-id", 0, "0" * 64, "2026-10-15T00:00:00+00:00"),
    )
    prior.close()
    observed: dict[str, object] = {}

    class FakeProvider:
        closed = False

        @classmethod
        def from_environment(cls, **kwargs):
            observed["provider"] = kwargs
            return cls()

        def close(self):
            self.closed = True

    class FakeTranscriber:
        def __init__(self, **kwargs):
            observed["transcriber"] = kwargs

        def transcribe(self, audio, *, mime_type=None):
            del audio, mime_type
            return "Assistant, test"

    class FakeCredentialStore:
        def __init__(self, path):
            observed["credential_path"] = path

        def load(self):
            return SimpleNamespace(ext_agent_host_id="urn:uuid:bound-host")

    class FakeHostIdStore:
        def __init__(self, path):
            observed["host_id_path"] = path

        def load(self):
            return "urn:uuid:bound-host"

    class FakeRefreshManager:
        def __init__(self, store, **kwargs):
            observed["refresh"] = (store, kwargs)

    class FakeAssistant:
        def __init__(self, **kwargs):
            observed["assistant"] = kwargs

        def respond(self, messages):
            raise AssertionError(messages)

    monkeypatch.setattr(service_module, "GaradoLightProvider", FakeProvider)
    monkeypatch.setattr(service_module, "FasterWhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr(service_module, "CredentialStore", FakeCredentialStore)
    monkeypatch.setattr(service_module, "HostIdStore", FakeHostIdStore)
    monkeypatch.setattr(service_module, "RefreshManager", FakeRefreshManager)
    monkeypatch.setattr(service_module, "ChatGPTPlanAssistant", FakeAssistant)

    with build_runtime(settings) as runtime:
        status = runtime_status(runtime)
        provider = runtime.provider

    provider_options = observed["provider"]
    assert provider_options["write_policy"] == "bridge_outputs"
    assert provider_options["output_title"] == "AI READY"
    assert provider_options["owned_output_note_ids"] == ("durable-output-id",)
    assert observed["credential_path"] == settings.chatgpt_credential_file
    assert observed["host_id_path"] == settings.chatgpt_host_id_file
    assert observed["assistant"]["model"] == "gpt-test-exact"
    assert observed["transcriber"]["local_files_only"] is True
    assert provider.closed is True
    assert not settings.simulator_database.exists()
    assert not (settings.state_dir / "bridge.sqlite3").exists()
    serialized = json.dumps(status, sort_keys=True)
    assert "private-light-user" not in serialized
    assert "private-light-password" not in serialized
    assert str(settings.chatgpt_credential_file) not in serialized
    assert status["mode"] == "production"


def test_production_factory_rejects_host_mismatch_without_exposing_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _production_config(tmp_path)
    monkeypatch.setenv("lightphone_username", "private-light-user")
    monkeypatch.setenv("lightphone_password", "private-light-password")

    class FakeCredentialStore:
        def __init__(self, _path):
            pass

        def load(self):
            return SimpleNamespace(ext_agent_host_id="urn:uuid:credential-secret")

    class FakeHostIdStore:
        def __init__(self, _path):
            pass

        def load(self):
            return "urn:uuid:host-secret"

    monkeypatch.setattr(service_module, "CredentialStore", FakeCredentialStore)
    monkeypatch.setattr(service_module, "HostIdStore", FakeHostIdStore)

    with pytest.raises(Pre0GuardError, match="different host") as captured:
        build_runtime(settings)

    message = str(captured.value)
    assert "credential-secret" not in message
    assert "host-secret" not in message
    assert not settings.simulator_database.exists()


def test_state_directory_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-state"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(Pre0GuardError, match="symbolic link"):
        build_runtime(ServiceConfig(state_dir=link))
