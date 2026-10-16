"""Content-free, authenticated long polling for answer-ready events."""

from __future__ import annotations

import errno
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import re
import socket
import sys
from threading import BoundedSemaphore, Condition, Lock, Thread
from time import monotonic
from urllib.parse import parse_qsl, urlsplit

from .config import require_loopback_host, require_notifier_token


_CURSOR = re.compile(r"v1:(0|[1-9][0-9]{0,18})\Z")
_MAX_SEQUENCE = 9223372036854775807
_MAX_REQUEST_TARGET_BYTES = 512
_DEFAULT_MAXIMUM_CONNECTIONS = 8
_DEFAULT_HEADER_TIMEOUT_SECONDS = 5.0
_EXPECTED_DISCONNECT_ERRNOS = {
    errno.ECONNABORTED,
    errno.ECONNRESET,
    errno.ENOTCONN,
    errno.EPIPE,
}
_BUSY_RESPONSE = (
    b"HTTP/1.1 503 Service Unavailable\r\n"
    b"Content-Type: text/plain\r\n"
    b"Content-Length: 5\r\n"
    b"Cache-Control: no-store\r\n"
    b"X-Content-Type-Options: nosniff\r\n"
    b"Connection: close\r\n"
    b"\r\n"
    b"busy\n"
)


class ReadyEventState:
    """Thread-safe monotonic view of the durable publication high-water mark."""

    def __init__(self, published_sequence: int = 0) -> None:
        if (
            not isinstance(published_sequence, int)
            or not 0 <= published_sequence <= _MAX_SEQUENCE
        ):
            raise ValueError("published sequence must be a non-negative integer")
        self._published_sequence = published_sequence
        self._closed = False
        self._condition = Condition()

    def current(self) -> int:
        with self._condition:
            return self._published_sequence

    def advance(self, published_sequence: int) -> bool:
        """Publish a newer durable sequence and ignore stale observations."""

        if (
            not isinstance(published_sequence, int)
            or not 0 <= published_sequence <= _MAX_SEQUENCE
        ):
            raise ValueError("published sequence must be a non-negative integer")
        with self._condition:
            if self._closed or published_sequence <= self._published_sequence:
                return False
            self._published_sequence = published_sequence
            self._condition.notify_all()
            return True

    def wait_for_newer(self, after: int, timeout_seconds: float) -> int | None:
        """Return a newer sequence, or ``None`` at timeout/service shutdown."""

        if after < 0:
            raise ValueError("cursor sequence must be non-negative")
        if not 0 < timeout_seconds <= 30:
            raise ValueError("long-poll timeout is outside the allowed range")
        deadline = monotonic() + timeout_seconds
        with self._condition:
            while not self._closed and self._published_sequence <= after:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._closed:
                return None
            return self._published_sequence

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


def encode_cursor(sequence: int) -> str:
    if not isinstance(sequence, int) or not 0 <= sequence <= _MAX_SEQUENCE:
        raise ValueError("published sequence must be a non-negative integer")
    return f"v1:{sequence}"


def decode_cursor(cursor: str) -> int:
    match = _CURSOR.fullmatch(cursor)
    if match is None:
        raise ValueError("invalid notifier cursor")
    sequence = int(match.group(1))
    if sequence > _MAX_SEQUENCE:
        raise ValueError("invalid notifier cursor")
    return sequence


def ready_response(
    state: ReadyEventState, cursor: str, *, wait_seconds: float
) -> tuple[int, bytes]:
    """Return only a public cursor and event kind; never application content."""

    if cursor == "":
        # A new client stores this durable high-water mark without alerting.
        sequence = state.current()
    else:
        after = decode_cursor(cursor)
        current = state.current()
        if after > current:
            # A restored or recreated database can legitimately be behind a
            # cursor already stored on the phone. Explicitly ask a compatible
            # client to adopt the current durable baseline without alerting.
            return 409, f"{encode_cursor(current)}\nreset\n".encode("ascii")
        sequence = (
            current
            if current > after
            else state.wait_for_newer(after, wait_seconds)
        )
        if sequence is None:
            return 204, b""
    return 200, f"{encode_cursor(sequence)}\nready\n".encode("ascii")


def _authorized(values: list[str] | None, expected_digest: bytes) -> bool:
    """Hash then compare fixed-size values so every request checks in constant time."""

    single = len(values) == 1 if values is not None else False
    header = values[0] if single and values is not None else ""
    scheme_ok = header.startswith("Bearer ")
    candidate = header[7:] if scheme_ok else ""
    candidate_digest = hashlib.sha256(candidate.encode("utf-8")).digest()
    digest_ok = hmac.compare_digest(candidate_digest, expected_digest)
    return bool(single and scheme_ok and candidate and digest_ok)


def _token_digest(token: str) -> bytes:
    require_notifier_token(token)
    return hashlib.sha256(token.encode("utf-8")).digest()


def _request_cursor(target: str) -> str:
    if len(target.encode("utf-8")) > _MAX_REQUEST_TARGET_BYTES:
        raise ValueError("request target is too long")
    parsed = urlsplit(target)
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.fragment
        or parsed.path != "/v1/ready"
    ):
        raise ValueError("invalid notifier path")
    try:
        parameters = parse_qsl(
            parsed.query,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=1,
        )
    except ValueError:
        raise ValueError("invalid notifier query") from None
    if len(parameters) != 1 or parameters[0][0] != "cursor":
        raise ValueError("invalid notifier query")
    cursor = parameters[0][1]
    if cursor:
        decode_cursor(cursor)
    return cursor


def notifier_response(
    state: ReadyEventState,
    target: str,
    authorization_values: list[str] | None,
    *,
    expected_digest: bytes,
    wait_seconds: float,
) -> tuple[int, bytes]:
    """Apply the complete authenticated HTTP contract without socket state."""

    if not _authorized(authorization_values, expected_digest):
        return 401, b"unauthorized\n"
    try:
        cursor = _request_cursor(target)
        return ready_response(state, cursor, wait_seconds=wait_seconds)
    except ValueError:
        return 400, b"bad_request\n"


def _is_expected_disconnect(error: BaseException | None) -> bool:
    if isinstance(
        error,
        (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, TimeoutError),
    ):
        return True
    return isinstance(error, OSError) and error.errno in _EXPECTED_DISCONNECT_ERRNOS


class _LoopbackHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server with a hard connection and lifetime bound."""

    daemon_threads = False
    block_on_close = True
    request_queue_size = _DEFAULT_MAXIMUM_CONNECTIONS

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        maximum_connections: int,
        header_timeout_seconds: float,
    ) -> None:
        self._slots = BoundedSemaphore(maximum_connections)
        self._connections: set[socket.socket] = set()
        self._connections_lock = Lock()
        self._header_timeout_seconds = header_timeout_seconds
        super().__init__(server_address, handler)

    def process_request(
        self, request: socket.socket, client_address: tuple[object, ...]
    ) -> None:
        request.settimeout(self._header_timeout_seconds)
        if not self._slots.acquire(blocking=False):
            try:
                request.sendall(_BUSY_RESPONSE)
            except BaseException as error:
                if not _is_expected_disconnect(error):
                    raise
            finally:
                self.shutdown_request(request)
            return
        with self._connections_lock:
            self._connections.add(request)
        tracked_threads = tuple(
            id(thread) for thread in vars(self).get("_threads", ())
        )
        try:
            super().process_request(request, client_address)
        except BaseException:
            if not self._discard_unstarted_request_threads(tracked_threads):
                # A pathological Thread.start() implementation may raise after
                # the worker began running. In that case the worker owns the
                # request and slot; replying or releasing either here could
                # race the real response.
                return
            try:
                request.sendall(_BUSY_RESPONSE)
            except BaseException:
                # The fixed response is best effort. Never let a peer reset or
                # an allocation failure reach BaseServer.handle_error(), whose
                # default output includes the client address and a traceback.
                pass
            with self._connections_lock:
                self._connections.discard(request)
            self._slots.release()
            try:
                self.shutdown_request(request)
            except BaseException:
                try:
                    request.close()
                except BaseException:
                    pass

    def _discard_unstarted_request_threads(
        self, tracked_threads: tuple[int, ...]
    ) -> bool:
        """Remove only new workers which failed before they could start.

        ``ThreadingMixIn`` appends a worker to its private close-time registry
        before invoking ``Thread.start``. Joining that object after a start
        failure raises and prevents a clean server close. A thread with an
        identity may already own the request, so leave it registered and let
        its normal finally path release the socket and connection slot.
        """

        threads = vars(self).get("_threads")
        if threads is None:
            return True
        tracked = set(tracked_threads)
        safe_to_release_request = True
        for thread in tuple(threads):
            if id(thread) in tracked:
                continue
            try:
                started = thread.ident is not None or thread.is_alive()
            except BaseException:
                started = True
            if started:
                safe_to_release_request = False
                continue
            try:
                threads.remove(thread)
            except ValueError:
                pass
        return safe_to_release_request

    def process_request_thread(
        self, request: socket.socket, client_address: tuple[object, ...]
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._connections_lock:
                self._connections.discard(request)
            self._slots.release()

    def handle_error(
        self, request: socket.socket, client_address: tuple[object, ...]
    ) -> None:
        if _is_expected_disconnect(sys.exc_info()[1]):
            return
        super().handle_error(request, client_address)

    def close_active_connections(self) -> None:
        with self._connections_lock:
            connections = tuple(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass

    def active_connection_count(self) -> int:
        with self._connections_lock:
            return len(self._connections)


class _LoopbackHTTPServerV6(_LoopbackHTTPServer):
    address_family = socket.AF_INET6


class NotifierServer:
    """Serve ``GET /v1/ready?cursor=...`` on a dedicated loopback port."""

    def __init__(
        self,
        host: str,
        port: int,
        state: ReadyEventState,
        token: str,
        wait_seconds: float,
        *,
        maximum_connections: int = _DEFAULT_MAXIMUM_CONNECTIONS,
        header_timeout_seconds: float = _DEFAULT_HEADER_TIMEOUT_SECONDS,
    ) -> None:
        require_loopback_host(host, setting="LIGHTPHONE_GPT_NOTIFIER_HOST")
        if not 0 <= port <= 65535:
            raise ValueError("notifier port is outside the allowed range")
        if not 0 < wait_seconds <= 30:
            raise ValueError("notifier wait is outside the allowed range")
        if not 1 <= maximum_connections <= 64:
            raise ValueError("notifier connection limit is outside the allowed range")
        if not 0.1 <= header_timeout_seconds <= 10:
            raise ValueError("notifier header timeout is outside the allowed range")
        expected_digest = _token_digest(token)

        class Handler(BaseHTTPRequestHandler):
            server_version = "LightphoneGPTNotifier/1"
            sys_version = ""
            protocol_version = "HTTP/1.1"

            def version_string(self) -> str:
                return self.server_version

            def handle(self) -> None:
                try:
                    super().handle()
                except BaseException as error:
                    if not _is_expected_disconnect(error):
                        raise
                    self.close_connection = True

            def parse_request(self) -> bool:
                # Python 3.11 and 3.12 try to parse headers after accepting a
                # two-token request line as HTTP/0.9, while 3.13+ returns
                # immediately. Reject it before the version-specific parser so
                # an incomplete legacy request always receives a bounded error.
                if len(self.raw_requestline.split()) == 2:
                    self.requestline = ""
                    self.close_connection = True
                    self.send_error(400)
                    return False
                parsed = super().parse_request()
                if parsed and self.request_version == "HTTP/0.9":
                    self.send_error(400)
                    return False
                return parsed

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                status, body = notifier_response(
                    state,
                    self.path,
                    self.headers.get_all("Authorization"),
                    expected_digest=expected_digest,
                    wait_seconds=wait_seconds,
                )
                self._reply(status, body)

            def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._reply(405, b"")

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._reply(405, b"method_not_allowed\n")

            def _reply(self, status: int, body: bytes) -> None:
                self.close_connection = True
                self.send_response(status)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Connection", "close")
                if status == 401:
                    self.send_header("WWW-Authenticate", "Bearer")
                if status == 405:
                    self.send_header("Allow", "GET")
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def send_error(
                self,
                code: int,
                message: str | None = None,
                explain: str | None = None,
            ) -> None:
                del message, explain
                safe_status = code if 400 <= code <= 599 else 500
                self.request_version = "HTTP/1.1"
                self._reply(safe_status, b"error\n")

            def log_message(self, _format: str, *_args: object) -> None:
                return

        server_type: type[_LoopbackHTTPServer] = (
            _LoopbackHTTPServerV6 if ":" in host else _LoopbackHTTPServer
        )
        self._state = state
        self._server = server_type(
            (host, port),
            Handler,
            maximum_connections=maximum_connections,
            header_timeout_seconds=header_timeout_seconds,
        )
        self._thread: Thread | None = None
        self._closed = False

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    @property
    def active_connection_count(self) -> int:
        return self._server.active_connection_count()

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("notifier server is closed")
        if self._thread is not None:
            raise RuntimeError("notifier server is already running")
        try:
            thread = Thread(
                target=self._server.serve_forever,
                name="lightphone-gpt-notifier",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        except BaseException:
            self._thread = None
            self._closed = True
            self._state.close()
            self._server.close_active_connections()
            # shutdown() waits for serve_forever() and must not be called when
            # the serving thread never started.
            self._server.server_close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._state.close()
        stop_error: RuntimeError | None = None
        if self._thread is not None:
            self._server.shutdown()
            self._server.close_active_connections()
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                stop_error = RuntimeError("notifier server did not stop")
            self._thread = None
        self._server.server_close()
        if stop_error is not None:
            raise stop_error

    def __enter__(self) -> "NotifierServer":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
