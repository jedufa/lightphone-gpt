from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import socket
import socketserver
import sqlite3
import struct
from threading import Thread
from types import MethodType
import time

import pytest

import lightphone_gpt.notifier as notifier_module
from lightphone_gpt.notifier import (
    NotifierServer,
    ReadyEventState,
    decode_cursor,
    encode_cursor,
    ready_response,
)
from lightphone_gpt.models import EnqueueOutcome, NoteKind, RemoteNote
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
)


TOKEN = "test_notifier_token_0123456789_ABCDEF"
NOW = "2026-10-16T09:00:00+00:00"
NOW_DT = datetime.fromisoformat(NOW)


def _publish(store: Store, note_id: str, *, now: datetime = NOW_DT) -> None:
    note = RemoteNote(
        note_id,
        NoteKind.AUDIO,
        "Recording",
        now,
        "audio/mp4",
    )
    assert (
        store.enqueue_audio(note, note_id.encode(), now=now)
        is EnqueueOutcome.ENQUEUED
    )
    store.mark_question_transcribed(note_id, question="Question", now=now)
    store.mark_answered(
        note_id,
        answer="Answer",
        sections=("Answer",),
        response_id=None,
        now=now,
    )
    store.mark_published(note_id, output_note_id="reply", now=now)


def _read_response(connection: socket.socket) -> bytes:
    connection.settimeout(2)
    chunks: list[bytes] = []
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _wait_for_connections(server: NotifierServer, expected: int) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if server.active_connection_count == expected:
            return
        time.sleep(0.01)
    assert server.active_connection_count == expected


def test_empty_cursor_returns_content_free_baseline_without_waiting() -> None:
    empty = ReadyEventState()
    existing = ReadyEventState(17)

    assert ready_response(empty, "", wait_seconds=1) == (
        200,
        b"v1:0\nready\n",
    )
    assert ready_response(existing, "", wait_seconds=1) == (
        200,
        b"v1:17\nready\n",
    )


def test_cursor_waits_for_strictly_newer_publication() -> None:
    state = ReadyEventState(7)
    result: list[tuple[int, bytes]] = []
    waiter = Thread(
        target=lambda: result.append(
            ready_response(state, "v1:7", wait_seconds=1)
        )
    )
    waiter.start()
    time.sleep(0.03)
    assert state.advance(6) is False
    assert state.advance(7) is False
    assert state.advance(8) is True
    waiter.join(timeout=1)

    assert result == [(200, b"v1:8\nready\n")]


def test_cursor_ahead_receives_explicit_silent_reset_contract() -> None:
    state = ReadyEventState(4)
    started = time.monotonic()
    status, body = ready_response(state, "v1:9", wait_seconds=0.03)

    assert status == 409
    assert body == b"v1:4\nreset\n"
    assert time.monotonic() - started < 0.5


@pytest.mark.parametrize(
    "cursor",
    [
        "v2:1",
        "v1:-1",
        "v1:01",
        "v1:abc",
        "1",
        "v1:9223372036854775808",
        "v1:10000000000000000000",
    ],
)
def test_cursor_parser_is_strict(cursor: str) -> None:
    with pytest.raises(ValueError, match="cursor"):
        decode_cursor(cursor)


def test_cursor_encoder_is_bounded_to_non_negative_integers() -> None:
    assert encode_cursor(0) == "v1:0"
    assert encode_cursor(123) == "v1:123"
    with pytest.raises(ValueError):
        encode_cursor(-1)
    with pytest.raises(ValueError):
        encode_cursor(9223372036854775808)
    with pytest.raises(ValueError):
        ReadyEventState(9223372036854775808)


def test_authentication_always_uses_fixed_size_constant_time_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bytes, bytes]] = []

    def compared(left: bytes, right: bytes) -> bool:
        calls.append((left, right))
        return left == right

    monkeypatch.setattr(notifier_module.hmac, "compare_digest", compared)
    expected = notifier_module.hashlib.sha256(TOKEN.encode()).digest()

    assert notifier_module._authorized(None, expected) is False
    assert notifier_module._authorized(["Basic irrelevant"], expected) is False
    assert notifier_module._authorized([f"Bearer {TOKEN}"], expected) is True
    assert len(calls) == 3
    assert all(len(left) == len(right) == 32 for left, right in calls)


def test_http_contract_is_authenticated_and_content_free() -> None:
    state = ReadyEventState(23)
    digest = notifier_module._token_digest(TOKEN)
    authorized = [f"Bearer {TOKEN}"]
    respond = lambda target, headers=authorized: notifier_module.notifier_response(
        state,
        target,
        headers,
        expected_digest=digest,
        wait_seconds=0.03,
    )

    unauthorized = respond("/v1/ready?cursor=", None)
    baseline = respond("/v1/ready?cursor=")
    quiet = respond("/v1/ready?cursor=v1%3A23")
    reset = respond("/v1/ready?cursor=v1%3A24")
    invalid = respond("/v1/ready?cursor=private-note-id")
    wrong_path = respond("/health?cursor=")

    assert unauthorized == (401, b"unauthorized\n")
    assert baseline == (200, b"v1:23\nready\n")
    assert quiet == (204, b"")
    assert reset == (409, b"v1:23\nreset\n")
    assert invalid[0] == 400
    assert wrong_path[0] == 400
    serialized = json.dumps(
        {
            "unauthorized": unauthorized[1].decode(),
            "baseline": baseline[1].decode(),
            "quiet": quiet[1].decode(),
            "reset": reset[1].decode(),
        }
    ).lower()
    for forbidden in ("note", "hash", "transcript", "answer", TOKEN.lower()):
        assert forbidden not in serialized


def test_server_refuses_non_loopback_bind() -> None:
    with pytest.raises(ValueError, match="loopback"):
        NotifierServer("0.0.0.0", 0, ReadyEventState(), TOKEN, 1)


@pytest.mark.parametrize("failure_point", ["constructor", "start"])
def test_server_start_failure_releases_bound_socket_without_shutdown_wait(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    state = ReadyEventState()
    server = NotifierServer("127.0.0.1", 0, state, TOKEN, 1)
    address = server.address

    if failure_point == "constructor":
        def fail_thread(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected thread construction failure")

        monkeypatch.setattr(notifier_module, "Thread", fail_thread)
    else:
        class StartFailure:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def start(self) -> None:
                raise RuntimeError("injected thread start failure")

        monkeypatch.setattr(notifier_module, "Thread", StartFailure)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="injected thread"):
        server.start()
    server.close()
    assert time.monotonic() - started < 1
    with pytest.raises(RuntimeError, match="closed"):
        server.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(address)


@pytest.mark.parametrize("failure_point", ["constructor", "start"])
def test_request_worker_startup_failure_is_fixed_bounded_and_recoverable(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    failure_point: str,
) -> None:
    server = NotifierServer("127.0.0.1", 0, ReadyEventState(5), TOKEN, 1)
    address = server.address
    real_thread = socketserver.threading.Thread
    injected = False

    def request_thread(*args: object, **kwargs: object) -> Thread:
        nonlocal injected
        target = kwargs.get("target")
        is_request_worker = (
            getattr(target, "__self__", None) is server._server
            and getattr(target, "__name__", "") == "process_request_thread"
        )
        if not is_request_worker:
            return real_thread(*args, **kwargs)
        assert injected is False
        injected = True
        if failure_point == "constructor":
            raise RuntimeError("private injected request constructor failure")
        thread = real_thread(*args, **kwargs)

        def fail_start(_self: Thread) -> None:
            raise RuntimeError("private injected request start failure")

        thread.start = MethodType(fail_start, thread)  # type: ignore[method-assign]
        return thread

    server.start()
    monkeypatch.setattr(socketserver.threading, "Thread", request_thread)
    try:
        with socket.create_connection(address, timeout=1) as connection:
            connection.sendall(
                b"GET /v1/ready?cursor= HTTP/1.1\r\n"
                b"Host: localhost\r\n\r\n"
            )
            response = _read_response(connection)

        assert injected is True
        assert response == notifier_module._BUSY_RESPONSE
        _wait_for_connections(server, 0)
        tracked_workers = tuple(vars(server._server).get("_threads", ()))
        assert all(worker.ident is not None for worker in tracked_workers)

        monkeypatch.setattr(socketserver.threading, "Thread", real_thread)
        with socket.create_connection(address, timeout=1) as connection:
            connection.sendall(
                b"GET /v1/ready?cursor= HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                + f"Authorization: Bearer {TOKEN}\r\n".encode()
                + b"\r\n"
            )
            recovered = _read_response(connection)
        assert recovered.endswith(b"\r\n\r\nv1:5\nready\n")
    finally:
        monkeypatch.setattr(socketserver.threading, "Thread", real_thread)
        server.close()
        server.close()

    captured = capfd.readouterr()
    assert "Traceback" not in captured.err
    assert "127.0.0.1" not in captured.err
    assert "private injected" not in captured.err
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(address)


def test_bound_server_serves_exact_android_path_and_closes_connection() -> None:
    server = NotifierServer("127.0.0.1", 0, ReadyEventState(5), TOKEN, 1)
    with server:
        with socket.create_connection(server.address, timeout=1) as connection:
            connection.sendall(
                b"GET /v1/ready?cursor= HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                + f"Authorization: Bearer {TOKEN}\r\n".encode()
                + b"\r\n"
            )
            response = _read_response(connection)

    headers, body = response.split(b"\r\n\r\n", 1)
    assert b" 200 " in headers
    assert b"Cache-Control: no-store" in headers
    assert b"Connection: close" in headers
    assert body == b"v1:5\nready\n"


def test_bound_server_ahead_cursor_uses_authenticated_reset_contract() -> None:
    server = NotifierServer("127.0.0.1", 0, ReadyEventState(5), TOKEN, 1)
    with server:
        with socket.create_connection(server.address, timeout=1) as connection:
            connection.sendall(
                b"GET /v1/ready?cursor=v1%3A8 HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                + f"Authorization: Bearer {TOKEN}\r\n".encode()
                + b"\r\n"
            )
            response = _read_response(connection)

    headers, body = response.split(b"\r\n\r\n", 1)
    assert b" 409 " in headers
    assert b"Content-Type: text/plain" in headers
    assert b"Cache-Control: no-store" in headers
    assert body == b"v1:5\nreset\n"


def test_bound_server_errors_are_fixed_content_free_and_not_reflected() -> None:
    server = NotifierServer("127.0.0.1", 0, ReadyEventState(), TOKEN, 1)
    with server:
        with socket.create_connection(server.address, timeout=1) as connection:
            connection.sendall(
                b"TRACE /private-attacker-value HTTP/1.1\r\n"
                b"Host: localhost\r\n\r\n"
            )
            response = _read_response(connection)

    headers, body = response.split(b"\r\n\r\n", 1)
    assert b" 501 " in headers
    assert b"Cache-Control: no-store" in headers
    assert b"Connection: close" in headers
    assert body == b"error\n"
    assert b"private-attacker-value" not in response
    assert b"<!DOCTYPE" not in response

    inherited = NotifierServer("127.0.0.1", 0, ReadyEventState(), TOKEN, 1)
    with inherited:
        with socket.create_connection(inherited.address, timeout=1) as connection:
            connection.sendall(
                b"GET /another-private-value HTTP/9.9\r\n"
                b"Host: localhost\r\n\r\n"
            )
            malformed = _read_response(connection)
    malformed_headers, malformed_body = malformed.split(b"\r\n\r\n", 1)
    assert b" 505 " in malformed_headers
    assert b"Cache-Control: no-store" in malformed_headers
    assert b"Connection: close" in malformed_headers
    assert malformed_body == b"error\n"
    assert b"another-private-value" not in malformed
    assert b"<!DOCTYPE" not in malformed


def test_http_09_request_is_fixed_http_11_error_without_server_noise(
    capfd: pytest.CaptureFixture[str],
) -> None:
    server = NotifierServer("127.0.0.1", 0, ReadyEventState(), TOKEN, 1)
    with server:
        with socket.create_connection(server.address, timeout=1) as connection:
            connection.sendall(b"GET /v1/ready?cursor=\r\n")
            response = _read_response(connection)

    headers, body = response.split(b"\r\n\r\n", 1)
    assert headers.startswith(b"HTTP/1.1 400 Bad Request\r\n")
    assert b"Content-Type: text/plain" in headers
    assert b"Content-Length: 6" in headers
    assert b"Cache-Control: no-store" in headers
    assert b"Connection: close" in headers
    assert body == b"error\n"
    assert b"/v1/ready" not in response
    captured = capfd.readouterr()
    assert "Traceback" not in captured.err
    assert "127.0.0.1" not in captured.err


def test_bound_server_limits_concurrency_and_times_out_partial_headers() -> None:
    server = NotifierServer(
        "127.0.0.1",
        0,
        ReadyEventState(),
        TOKEN,
        1,
        maximum_connections=1,
        header_timeout_seconds=0.2,
    )
    with server:
        partial = socket.create_connection(server.address, timeout=1)
        partial.sendall(b"GET /v1/ready?cursor= HTTP/1.1\r\nHost: localhost\r\n")
        _wait_for_connections(server, 1)

        with socket.create_connection(server.address, timeout=1) as overflow:
            overflow.sendall(
                b"GET /v1/ready?cursor= HTTP/1.1\r\nHost: localhost\r\n\r\n"
            )
            response = _read_response(overflow)
        assert response.startswith(b"HTTP/1.1 503 Service Unavailable\r\n")
        assert response.endswith(b"\r\n\r\nbusy\n")
        assert b"Cache-Control: no-store" in response

        partial.settimeout(1)
        assert partial.recv(1) == b""
        partial.close()
        _wait_for_connections(server, 0)


def test_close_drains_partial_header_connections_and_worker_threads() -> None:
    server = NotifierServer(
        "127.0.0.1",
        0,
        ReadyEventState(),
        TOKEN,
        1,
        maximum_connections=2,
        header_timeout_seconds=10,
    )
    server.start()
    partial = socket.create_connection(server.address, timeout=1)
    partial.sendall(b"GET /v1/ready?cursor= HTTP/1.1\r\n")
    _wait_for_connections(server, 1)

    started = time.monotonic()
    server.close()

    assert time.monotonic() - started < 2
    assert server.active_connection_count == 0
    partial.settimeout(1)
    assert partial.recv(1) == b""
    partial.close()


def test_client_reset_during_long_poll_does_not_emit_traceback(
    capfd: pytest.CaptureFixture[str],
) -> None:
    state = ReadyEventState()
    server = NotifierServer("127.0.0.1", 0, state, TOKEN, 1)
    with server:
        connection = socket.create_connection(server.address, timeout=1)
        connection.sendall(
            b"GET /v1/ready?cursor=v1%3A0 HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            + f"Authorization: Bearer {TOKEN}\r\n".encode()
            + b"\r\n"
        )
        _wait_for_connections(server, 1)
        connection.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_LINGER,
            struct.pack("ii", 1, 0),
        )
        connection.close()
        state.advance(1)
        _wait_for_connections(server, 0)

    captured = capfd.readouterr()
    assert "Traceback" not in captured.err
    assert "private" not in captured.err.lower()


def test_state_close_interrupts_a_waiting_request() -> None:
    state = ReadyEventState(2)
    result: list[tuple[int, bytes]] = []
    request = Thread(
        target=lambda: result.append(
            ready_response(state, "v1:2", wait_seconds=30)
        )
    )
    request.start()
    time.sleep(0.05)
    started = time.monotonic()
    state.close()
    request.join(timeout=1)

    assert time.monotonic() - started < 1
    assert result == [(204, b"")]


def test_schema_v11_preserves_legacy_publication_baseline(
    tmp_path: Path,
) -> None:
    database = tmp_path / "bridge.sqlite3"
    connection = sqlite3.connect(database)
    for schema in (
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
    ):
        connection.executescript(schema)
    for note_id, sequence, state in (
        ("published-a", 4, "published"),
        ("queued-private", 8, "queued"),
        ("published-b", 7, "published"),
    ):
        connection.execute(
            """
            INSERT INTO jobs (
                note_id, queue_seq, state, accepted_output_epoch,
                discovered_at, updated_at, completed_at
            ) VALUES (?, ?, ?, 0, ?, ?, ?)
            """,
            (
                note_id,
                sequence,
                state,
                NOW,
                NOW,
                NOW if state == "published" else None,
            ),
        )
    connection.execute("PRAGMA user_version = 10")
    connection.commit()
    connection.close()

    with Store(database) as migrated:
        assert migrated.latest_published_sequence() == 7
        assert migrated.connection.execute("PRAGMA user_version").fetchone()[0] == _LATEST_SCHEMA_VERSION
        events = migrated.connection.execute(
            "SELECT note_id, notifier_event_sequence FROM jobs "
            "WHERE notifier_event_sequence IS NOT NULL ORDER BY note_id"
        ).fetchall()
        assert [tuple(row) for row in events] == [
            ("published-a", 4),
            ("published-b", 7),
        ]


def test_publication_sequence_survives_retention_restart_and_queue_reuse(
    tmp_path: Path,
) -> None:
    database = tmp_path / "bridge.sqlite3"
    with Store(database) as store:
        store.establish_baseline([], now=NOW_DT)
        _publish(store, "first")
        assert store.latest_published_sequence() == 1
        assert store.get_job("first").queue_seq == 1  # type: ignore[union-attr]
        store.connection.execute(
            "UPDATE conversations SET closed_at = ?", (NOW,)
        )
        store.connection.execute(
            "UPDATE bridge_state SET active_conversation_id = NULL WHERE singleton = 1"
        )
        deleted = store.delete_expired_metadata(
            completed_before=datetime(2026, 10, 17, tzinfo=UTC)
        )
        assert deleted == 1
        assert store.get_job("first") is None
        assert store.latest_published_sequence() == 1

        _publish(store, "second")
        assert store.get_job("second").queue_seq == 1  # type: ignore[union-attr]
        assert store.latest_published_sequence() == 2

    with Store(database) as restarted:
        assert restarted.latest_published_sequence() == 2


def test_repeated_publication_and_recovery_do_not_repeat_ready_event(
    tmp_path: Path,
) -> None:
    with Store(tmp_path / "bridge.sqlite3") as store:
        store.establish_baseline([], now=NOW_DT)
        _publish(store, "stable")
        assert store.latest_published_sequence() == 1

        store.mark_published("stable", output_note_id="reply", now=NOW_DT)
        assert store.latest_published_sequence() == 1
        event = store.connection.execute(
            "SELECT notifier_event_sequence FROM jobs WHERE note_id = 'stable'"
        ).fetchone()[0]
        assert event == 1
