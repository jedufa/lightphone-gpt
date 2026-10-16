"""Pre-0 service assembly and sequential poll/drain loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hmac
import os
from pathlib import Path
import signal
from threading import Event
from time import monotonic
from types import FrameType
from typing import Any, Callable

from .adapters.assistant import MockAssistant
from .adapters.chatgpt_plan import ChatGPTPlanAssistant
from .adapters.light_garado import GaradoLightProvider, LightWritePolicy
from .adapters.simulated_envelope import SimulatedEnvelopeTranscriber
from .adapters.transcriber import FasterWhisperTranscriber, WhisperCppTranscriber
from .chatgpt_auth import HostIdStore
from .chatgpt_oauth import CredentialStore, RefreshManager
from .config import ServiceConfig
from .contracts import Assistant, LightProvider, Transcriber
from .health import HealthServer, HealthSnapshot, HealthState
from .models import JobState
from .notifier import NotifierServer, ReadyEventState
from .retention import apply_retention
from .runtime_lock import RuntimeLock
from .simulator import SimulatedLightProvider
from .store import Store
from .worker import PollReport, WorkReport, Worker


class Pre0GuardError(RuntimeError):
    """A continuously running integration failed its activation boundary."""


@dataclass(frozen=True, slots=True)
class CycleReport:
    poll: PollReport
    work: tuple[WorkReport, ...]
    output_gate: str | None = None
    assistant_gate: str | None = None
    unresolved_job_failures: int = 0
    unresolved_source_cleanups: int = 0

    @property
    def ok(self) -> bool:
        return (
            self.poll.error is None
            and self.poll.source_cleanup_error is None
            and self.poll.output_gate is None
            and not self.poll.queue_limited
            and self.output_gate is None
            and self.assistant_gate is None
            and self.unresolved_job_failures == 0
            and self.unresolved_source_cleanups == 0
            and all(report.state != JobState.FAILED for report in self.work)
        )


@dataclass(slots=True)
class Runtime:
    config: ServiceConfig
    store: Store
    provider: LightProvider
    transcriber: Transcriber
    assistant: Assistant
    worker: Worker
    runtime_lock: RuntimeLock

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        try:
            if callable(close):
                close()
        finally:
            try:
                self.store.close()
            finally:
                self.runtime_lock.close()

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def prepare_state_directory(config: ServiceConfig) -> None:
    if config.state_dir.is_symlink():
        raise Pre0GuardError("state directory must not be a symbolic link")
    config.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    config.state_dir.chmod(0o700)
    config.transcription_temp_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    config.transcription_temp_dir.chmod(0o700)
    if config.faster_whisper_model_cache is not None:
        config.faster_whisper_model_cache.mkdir(
            parents=True, exist_ok=True, mode=0o700
        )
        config.faster_whisper_model_cache.chmod(0o700)


def _private_database_placeholder(path: Path) -> None:
    if path.is_symlink():
        raise Pre0GuardError("runtime database must not be a symbolic link")
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    os.close(descriptor)
    path.chmod(0o600)


def _secure_database_files(config: ServiceConfig) -> None:
    databases = [config.bridge_database]
    if config.provider == "simulator":
        databases.append(config.simulator_database)
    for base in databases:
        for path in base.parent.glob(f"{base.name}*"):
            if path.is_file():
                path.chmod(0o600)


def build_runtime(config: ServiceConfig) -> Runtime:
    """Build the selected runtime after its explicit fail-closed gate."""

    production_requested = (
        config.provider != "simulator"
        or config.assistant != "mock"
        or config.production_activated
    )
    if production_requested:
        try:
            config.validate_production_environment()
        except ValueError:
            raise Pre0GuardError(
                "production runtime configuration is incomplete"
            ) from None

    prepare_state_directory(config)
    runtime_lock = RuntimeLock(config.state_dir / "runtime.lock").acquire()
    store: Store | None = None
    provider: LightProvider | None = None
    try:
        _private_database_placeholder(config.bridge_database)
        store = Store(config.bridge_database)
        if config.provider == "simulator":
            _private_database_placeholder(config.simulator_database)
            provider = SimulatedLightProvider(config.simulator_database)
        else:
            provider = GaradoLightProvider.from_environment(
                write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
                output_title=config.output_title,
                owned_output_note_ids=store.known_output_note_ids(),
            )
        if config.transcriber == "simulated-envelope":
            transcriber: Transcriber = SimulatedEnvelopeTranscriber()
        elif config.transcriber == "whisper-cpp":
            assert config.whisper_executable is not None
            assert config.whisper_model is not None
            transcriber = WhisperCppTranscriber(
                executable=config.whisper_executable,
                model=config.whisper_model,
                temp_root=config.transcription_temp_dir,
            )
        elif config.transcriber == "faster-whisper":
            transcriber = FasterWhisperTranscriber(
                model=config.faster_whisper_model,
                model_cache=config.faster_whisper_model_cache,
                local_files_only=True,
                temp_root=config.transcription_temp_dir,
            )
        else:  # ServiceConfig validates this, retained as a fail-closed boundary.
            raise Pre0GuardError("unsupported transcriber")
        if config.assistant == "mock":
            assistant: Assistant = MockAssistant()
        else:
            assert config.chatgpt_credential_file is not None
            assert config.chatgpt_host_id_file is not None
            assert config.chatgpt_model is not None
            credential_store = CredentialStore(config.chatgpt_credential_file)
            profile = credential_store.load()
            host_id = HostIdStore(config.chatgpt_host_id_file).load()
            if not hmac.compare_digest(profile.ext_agent_host_id, host_id):
                raise Pre0GuardError(
                    "ChatGPT authorization is bound to a different host"
                )
            assistant = ChatGPTPlanAssistant(
                tokens=RefreshManager(credential_store, timeout_seconds=15.0),
                model=config.chatgpt_model,
                timeout_seconds=60.0,
            )
        worker = Worker(
            store=store,
            provider=provider,
            transcriber=transcriber,  # type: ignore[arg-type]
            assistant=assistant,
            config=config.worker_config(),
        )
        _secure_database_files(config)
        return Runtime(
            config,
            store,
            provider,
            transcriber,
            assistant,
            worker,
            runtime_lock,
        )
    except BaseException:
        if provider is not None:
            close = getattr(provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        try:
            if store is not None:
                store.close()
        finally:
            runtime_lock.close()
        raise


def run_cycle(runtime: Runtime) -> CycleReport:
    runtime.worker.recover_time_based_gates()
    runtime.worker.recover_interrupted_work()
    initial_assistant_gate = runtime.store.assistant_gate()
    poll = (
        PollReport()
        if initial_assistant_gate not in {None, "inference_ambiguous"}
        else runtime.worker.poll()
    )
    work = (
        ()
        if (
            initial_assistant_gate not in {None, "inference_ambiguous"}
            or poll.error is not None
        )
        else tuple(runtime.worker.drain(max_steps=runtime.config.drain_max_steps))
    )
    output_gate = runtime.store.output_gate()
    # A best-effort recovery can clear the gate during drain. Report the final
    # durable state so health does not stay degraded for one extra poll cycle.
    poll = replace(poll, output_gate=output_gate)
    assistant_gate = runtime.store.assistant_gate()
    unresolved_job_failures = int(
        runtime.store.connection.execute(
            """
            SELECT COUNT(*) FROM jobs
             WHERE state = 'failed' AND (
                retryable = 1 OR manual_retry_allowed = 1
                OR completed_at IS NULL
             )
            """
        ).fetchone()[0]
    )
    cleanup_status = runtime.store.source_cleanup_status()
    runtime.store.checkpoint()
    _secure_database_files(runtime.config)
    return CycleReport(
        poll=poll,
        work=work,
        output_gate=output_gate,
        assistant_gate=assistant_gate,
        unresolved_job_failures=unresolved_job_failures,
        unresolved_source_cleanups=int(cleanup_status["unresolved"]),
    )


def database_status(store: Store) -> dict[str, int | bool]:
    """Return aggregates only; no identifiers, titles, transcript, or answer."""

    connection = store.connection
    database_ok = connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    rows = connection.execute(
        "SELECT state, COUNT(*) AS count FROM jobs GROUP BY state"
    ).fetchall()
    counts = {str(row["state"]): int(row["count"]) for row in rows}
    actionable = connection.execute(
        """
        SELECT COUNT(*) FROM jobs
        WHERE state IN ('queued','transcribed','answered')
           OR (state = 'failed' AND retryable = 1)
        """
    ).fetchone()[0]
    failed = connection.execute(
        """
        SELECT
            SUM(CASE WHEN retryable = 1 THEN 1 ELSE 0 END) AS retry_waiting,
            SUM(CASE WHEN retryable = 0 THEN 1 ELSE 0 END) AS terminal,
            SUM(CASE WHEN manual_retry_allowed = 1 THEN 1 ELSE 0 END) AS recoverable,
            SUM(CASE WHEN retryable = 1 OR manual_retry_allowed = 1
                          OR completed_at IS NULL THEN 1 ELSE 0 END) AS unresolved
          FROM jobs WHERE state = 'failed'
        """
    ).fetchone()
    cleanup = store.source_cleanup_status()
    return {
        "database_ok": database_ok,
        "enrolled": store.is_enrolled(),
        "output_blocked": store.output_gate() is not None,
        "assistant_blocked": store.assistant_gate() is not None,
        "jobs_total": sum(counts.values()),
        "jobs_actionable": int(actionable),
        "jobs_ignored": counts.get("ignored", 0),
        "jobs_published": counts.get("published", 0),
        "jobs_failed": counts.get("failed", 0),
        "jobs_retry_waiting": int(failed["retry_waiting"] or 0),
        "jobs_terminal_failed": int(failed["terminal"] or 0),
        "jobs_recovery_available": int(failed["recoverable"] or 0),
        "jobs_unresolved_failed": int(failed["unresolved"] or 0),
        "jobs_queued": counts.get("queued", 0),
        "jobs_transcribed": counts.get("transcribed", 0),
        "jobs_answered": counts.get("answered", 0),
        "source_cleanup_enabled": bool(cleanup["enabled"]),
        "source_cleanups_pending": int(cleanup["pending"]),
        "source_cleanups_uncertain": int(cleanup["uncertain"]),
        "source_cleanups_conflicts": int(cleanup["conflicts"]),
        "source_cleanups_confirmed": int(cleanup["confirmed"]),
        "source_cleanups_unresolved": int(cleanup["unresolved"]),
        "source_cleanup_blocked": bool(cleanup["blocked"]),
    }


def runtime_status(runtime: Runtime) -> dict[str, object]:
    status: dict[str, object] = {
        "service": "lightphone-gpt",
        "mode": "production" if runtime.config.production_activated else "pre-0",
        "provider": runtime.config.provider,
        "assistant": runtime.config.assistant,
        "transcriber": runtime.config.transcriber,
    }
    status.update(database_status(runtime.store))
    status["queue_limited"] = (
        runtime.store.pending_job_count() >= runtime.config.max_pending_jobs
    )
    return status


def _refresh_health(
    state: HealthState, runtime: Runtime, *, cycle_ok: bool
) -> None:
    status = database_status(runtime.store)
    state.record_cycle(
        ok=cycle_ok,
        database_ok=bool(status["database_ok"]),
        jobs_total=int(status["jobs_total"]),
        jobs_actionable=int(status["jobs_actionable"]),
        jobs_failed=int(status["jobs_failed"]),
        jobs_unresolved_failed=int(status["jobs_unresolved_failed"]),
        output_blocked=bool(status["output_blocked"]),
        assistant_blocked=bool(status["assistant_blocked"]),
        jobs_retry_waiting=int(status["jobs_retry_waiting"]),
        jobs_terminal_failed=int(status["jobs_terminal_failed"]),
        source_cleanups_unresolved=int(status["source_cleanups_unresolved"]),
        source_cleanup_blocked=bool(status["source_cleanup_blocked"]),
    )


def _refresh_ready_event(state: ReadyEventState, runtime: Runtime) -> None:
    state.advance(runtime.store.latest_published_sequence())


def serve(
    config: ServiceConfig,
    *,
    stop_event: Event | None = None,
    install_signal_handlers: bool = True,
    health_server_factory: Callable[[str, int, HealthState], Any] = HealthServer,
    notifier_server_factory: Callable[
        [str, int, ReadyEventState, str, float], Any
    ] = NotifierServer,
) -> None:
    """Run until SIGTERM/SIGINT or the supplied event is set."""

    stop = stop_event or Event()
    runtime = build_runtime(config)
    health_state = HealthState(
        HealthSnapshot(
            provider=config.provider,
            assistant=config.assistant,
            transcriber=config.transcriber,
        )
    )
    ready_state = ReadyEventState(runtime.store.latest_published_sequence())
    health: Any | None = None
    notifier: Any | None = None
    old_handlers: dict[signal.Signals, Any] = {}

    def request_stop(_number: int, _frame: FrameType | None) -> None:
        stop.set()

    try:
        health = health_server_factory(
            config.health_host, config.health_port, health_state
        )
        if config.notifier_enabled:
            assert config.notifier_token is not None
            notifier = notifier_server_factory(
                config.notifier_host,
                config.notifier_port,
                ready_state,
                config.notifier_token,
                config.notifier_wait_seconds,
            )
        if install_signal_handlers:
            for item in (signal.SIGTERM, signal.SIGINT):
                old_handlers[item] = signal.signal(item, request_stop)
        health.start()
        if notifier is not None:
            notifier.start()
        next_retention = monotonic()
        while not stop.is_set():
            try:
                report = run_cycle(runtime)
                _refresh_health(health_state, runtime, cycle_ok=report.ok)
                _refresh_ready_event(ready_state, runtime)
                now = monotonic()
                if now >= next_retention:
                    apply_retention(
                        runtime.store,
                        now=datetime.now(UTC),
                        policy=config.retention_policy(),
                    )
                    runtime.store.checkpoint()
                    next_retention = now + config.retention_interval_seconds
            except Exception:
                # The service stays available for later retries. Error text is
                # intentionally excluded from health and stdout/stderr.
                try:
                    _refresh_health(health_state, runtime, cycle_ok=False)
                except Exception:
                    health_state.update(
                        phase="degraded", ready=False, database_ok=False
                    )
                try:
                    _refresh_ready_event(ready_state, runtime)
                except Exception:
                    pass
            stop.wait(config.poll_interval_seconds)
    finally:
        health_state.update(phase="stopping", ready=False)
        ready_state.close()
        if notifier is not None:
            notifier.close()
        if health is not None:
            health.close()
        runtime.close()
        for item, handler in old_handlers.items():
            signal.signal(item, handler)
