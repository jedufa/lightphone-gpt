"""Loopback-only health server with a content-free status payload."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
from threading import Lock, Thread
from .config import require_loopback_host


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    service: str = "lightphone-gpt"
    phase: str = "starting"
    ready: bool = False
    database_ok: bool = False
    provider: str = "simulator"
    assistant: str = "mock"
    transcriber: str = "simulated-envelope"
    cycles: int = 0
    last_cycle_ok: bool | None = None
    last_cycle_at: str | None = None
    jobs_total: int = 0
    jobs_actionable: int = 0
    jobs_failed: int = 0
    jobs_unresolved_failed: int = 0
    jobs_retry_waiting: int = 0
    jobs_terminal_failed: int = 0
    output_blocked: bool = False
    assistant_blocked: bool = False
    source_cleanups_unresolved: int = 0
    source_cleanup_blocked: bool = False

    def payload(self) -> dict[str, object]:
        return asdict(self)


class HealthState:
    """Thread-safe immutable snapshots shared with the HTTP thread."""

    def __init__(self, initial: HealthSnapshot | None = None) -> None:
        self._snapshot = initial or HealthSnapshot()
        self._lock = Lock()

    def get(self) -> HealthSnapshot:
        with self._lock:
            return self._snapshot

    def update(self, **changes: object) -> HealthSnapshot:
        with self._lock:
            self._snapshot = replace(self._snapshot, **changes)
            return self._snapshot

    def record_cycle(
        self,
        *,
        ok: bool,
        database_ok: bool,
        jobs_total: int,
        jobs_actionable: int,
        jobs_failed: int,
        jobs_unresolved_failed: int = 0,
        output_blocked: bool = False,
        assistant_blocked: bool = False,
        jobs_retry_waiting: int = 0,
        jobs_terminal_failed: int = 0,
        source_cleanups_unresolved: int = 0,
        source_cleanup_blocked: bool = False,
        at: datetime | None = None,
    ) -> HealthSnapshot:
        timestamp = (at or datetime.now(UTC)).astimezone(UTC).isoformat(
            timespec="seconds"
        )
        current = self.get()
        ready = bool(
            ok
            and database_ok
            and jobs_unresolved_failed == 0
            and not output_blocked
            and not assistant_blocked
            and source_cleanups_unresolved == 0
            and not source_cleanup_blocked
        )
        return self.update(
            phase="running" if ready else "degraded",
            ready=ready,
            database_ok=database_ok,
            cycles=current.cycles + 1,
            last_cycle_ok=ok,
            last_cycle_at=timestamp,
            jobs_total=jobs_total,
            jobs_actionable=jobs_actionable,
            jobs_failed=jobs_failed,
            jobs_unresolved_failed=jobs_unresolved_failed,
            jobs_retry_waiting=jobs_retry_waiting,
            jobs_terminal_failed=jobs_terminal_failed,
            output_blocked=output_blocked,
            assistant_blocked=assistant_blocked,
            source_cleanups_unresolved=source_cleanups_unresolved,
            source_cleanup_blocked=source_cleanup_blocked,
        )


def health_response(
    state: HealthState, path: str
) -> tuple[int, dict[str, object]]:
    """Build the HTTP result without touching sockets, logs, or application data."""

    if path != "/health":
        return 404, {"error": "not_found"}
    try:
        current = state.get()
        status = 200 if current.ready and current.database_ok else 503
        return status, current.payload()
    except Exception:
        return 503, {"service": "lightphone-gpt", "ready": False}


class _LoopbackHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


class _LoopbackHTTPServerV6(_LoopbackHTTPServer):
    address_family = socket.AF_INET6


class HealthServer:
    """A minimal server exposing only ``GET /health`` on loopback."""

    def __init__(self, host: str, port: int, state: HealthState) -> None:
        require_loopback_host(host)
        if not 0 <= port <= 65535:
            raise ValueError("health port is outside the allowed range")
        self._state = state
        snapshot = state.get

        class Handler(BaseHTTPRequestHandler):
            server_version = "LightphoneGPTHealth/1"
            sys_version = ""

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                status, payload = health_response(_SnapshotState(snapshot), self.path)
                self._reply(status, payload)

            def _reply(self, status: int, payload: dict[str, object]) -> None:
                encoded = json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        server_type: type[_LoopbackHTTPServer] = (
            _LoopbackHTTPServerV6 if ":" in host else _LoopbackHTTPServer
        )
        self._server = server_type((host, port), Handler)
        self._thread: Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("health server is already running")
        self._thread = Thread(
            target=self._server.serve_forever,
            name="lightphone-gpt-health",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._server.server_close()

    def __enter__(self) -> "HealthServer":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _SnapshotState:
    """Small adapter keeping the request handler closure content-free."""

    def __init__(self, getter: object) -> None:
        self._getter = getter

    def get(self) -> HealthSnapshot:
        return self._getter()  # type: ignore[operator]
