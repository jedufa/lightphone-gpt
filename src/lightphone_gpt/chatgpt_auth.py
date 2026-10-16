"""Interactive, local-only Sign in with ChatGPT administration.

This module owns the browser and loopback callback portion of authorization.
It deliberately returns content-free summaries and never logs authorization
URLs, callbacks, identity claims, or token endpoint responses.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import socket
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Protocol

from .chatgpt_errors import (
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTTransientError,
)
from .chatgpt_oauth import (
    DYNAMIC_CLIENT_ID,
    RESOURCE,
    TOKEN_ENDPOINT,
    AuthorizationAttempt,
    AuthorizationGrant,
    ChatGPTCredentialProfile,
    CredentialStore,
    OAuthEndpointError,
    OidcValidator,
    RefreshManager,
    TokenRequester,
    _post_form_json,
    activate_token_response,
    generate_host_id,
    new_authorization_attempt,
)


_CALLBACK_PATH = "/auth/callback"
_MAX_REQUEST_TARGET_BYTES = 8 * 1024
_MAX_QUERY_BYTES = 4 * 1024
_MAX_QUERY_FIELDS = 16
_MAX_QUERY_COMPONENT_BYTES = 2 * 1024
_MAX_HOST_ID_BYTES = 256
_MAX_LOGIN_SECONDS = 10 * 60
_MAX_MODELS_SECONDS = 60.0
_MAX_MODELS_RESPONSE_BYTES = 1024 * 1024
_MODELS_ENDPOINT = f"{RESOURCE}/models"
_SUCCESS_PAGE = (
    b"<!doctype html><html><head><meta charset=utf-8>"
    b"<title>ChatGPT authorization received</title></head>"
    b"<body><h1>Authorization received</h1>"
    b"<p>You may close this window and return to Lightphone GPT.</p>"
    b"</body></html>"
)
_ERROR_PAGE = (
    b"<!doctype html><html><head><meta charset=utf-8>"
    b"<title>ChatGPT authorization not accepted</title></head>"
    b"<body><h1>Authorization not accepted</h1>"
    b"<p>Return to Lightphone GPT and try again.</p></body></html>"
)


@dataclass(frozen=True, slots=True)
class ChatGPTAuthPaths:
    credential_file: Path
    host_id_file: Path


def resolve_auth_paths(
    environ: Mapping[str, str], *, credential_file: Path | None = None
) -> ChatGPTAuthPaths:
    """Resolve the local credential location without touching the filesystem."""

    configured = credential_file
    if configured is None:
        raw_credential = environ.get("LIGHTPHONE_GPT_CHATGPT_CREDENTIAL_FILE", "")
        configured = Path(raw_credential) if raw_credential.strip() else None
    if configured is None:
        raw_config = environ.get("XDG_CONFIG_HOME", "")
        if raw_config.strip():
            config_root = Path(raw_config)
        else:
            raw_home = environ.get("HOME", "")
            config_root = (
                Path(raw_home).expanduser() / ".config"
                if raw_home.strip()
                else Path.home() / ".config"
            )
        configured = config_root / "lightphone-gpt" / "chatgpt" / "credentials.json"
    configured = configured.expanduser()
    if not configured.is_absolute():
        raise ChatGPTConfigurationError(
            "ChatGPT credential path must be absolute"
        )
    if not configured.name:
        raise ChatGPTConfigurationError("ChatGPT credential path is invalid")
    return ChatGPTAuthPaths(
        credential_file=configured,
        host_id_file=configured.with_name("host-id"),
    )


def _validate_host_id(value: str) -> str:
    value = value.strip()
    if not value.startswith("urn:uuid:"):
        raise ChatGPTConfigurationError("ChatGPT host identifier is invalid")
    try:
        parsed = uuid.UUID(value.removeprefix("urn:uuid:"))
    except ValueError:
        raise ChatGPTConfigurationError(
            "ChatGPT host identifier is invalid"
        ) from None
    if str(parsed) != value.removeprefix("urn:uuid:").lower():
        raise ChatGPTConfigurationError("ChatGPT host identifier is invalid")
    return value


class HostIdStore:
    """Persist one non-secret, owner-only host registration identifier."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def _prepare_parent(self) -> None:
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = self.path.parent.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise OSError
            os.chmod(self.path.parent, 0o700)
        except OSError:
            raise ChatGPTConfigurationError(
                "ChatGPT authorization directory could not be secured"
            ) from None

    def load(self) -> str:
        with self._lock:
            return self._load()

    def _load(self) -> str:
        try:
            metadata = self.path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
                or metadata.st_size > _MAX_HOST_ID_BYTES
            ):
                raise OSError
            descriptor = os.open(
                self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                opened = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or stat.S_IMODE(opened.st_mode) != 0o600
                    or (hasattr(os, "geteuid") and opened.st_uid != os.geteuid())
                ):
                    raise OSError
                raw = stream.read(_MAX_HOST_ID_BYTES + 1)
            if len(raw) > _MAX_HOST_ID_BYTES:
                raise OSError
            return _validate_host_id(raw.decode("ascii"))
        except FileNotFoundError:
            raise ChatGPTAuthorizationRequiredError(
                "ChatGPT host identifier is not configured"
            ) from None
        except (OSError, UnicodeDecodeError):
            raise ChatGPTConfigurationError(
                "ChatGPT host identifier could not be read securely"
            ) from None

    def load_or_create(self, *, preferred: str | None = None) -> str:
        with self._lock:
            self._prepare_parent()
            try:
                existing = self._load()
            except ChatGPTAuthorizationRequiredError:
                existing = None
            if existing is not None:
                if preferred is not None and not hmac.compare_digest(
                    existing, preferred
                ):
                    raise ChatGPTConfigurationError(
                        "ChatGPT host identifier did not match saved authorization"
                    )
                return existing

            value = _validate_host_id(preferred or generate_host_id())
            descriptor = -1
            temporary = ""
            published = False
            try:
                descriptor, temporary = tempfile.mkstemp(
                    prefix=f".{self.path.name}.import.", dir=self.path.parent
                )
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb", closefd=True) as stream:
                    descriptor = -1
                    stream.write(f"{value}\n".encode("ascii"))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, self.path, follow_symlinks=False)
                published = True
                try:
                    os.unlink(temporary)
                    temporary = ""
                except OSError:
                    # The complete final identity is already authoritative.
                    pass
                try:
                    directory_fd = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except OSError:
                    # The exclusive link is the commit point. Never report a
                    # contradictory failure after a complete ID is published.
                    pass
                return value
            except FileExistsError:
                existing = self._load()
                if preferred is not None and not hmac.compare_digest(
                    existing, value
                ):
                    raise ChatGPTConfigurationError(
                        "ChatGPT host identifier did not match saved authorization"
                    )
                return existing
            except OSError:
                if published:  # pragma: no cover - post-publish work is guarded
                    return value
                raise ChatGPTConfigurationError(
                    "ChatGPT host identifier could not be stored securely"
                ) from None
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                if temporary:
                    try:
                        os.unlink(temporary)
                    except OSError:
                        pass


@dataclass(slots=True)
class _CallbackOutcome:
    grant: AuthorizationGrant | None = None
    error: Exception | None = None
    finished: bool = False


def _consume_callback_target(
    attempt: AuthorizationAttempt, redirect_uri: str, request_target: str
) -> AuthorizationGrant:
    try:
        encoded_target = request_target.encode("ascii")
        if len(encoded_target) > _MAX_REQUEST_TARGET_BYTES:
            raise ValueError
        parsed = urllib.parse.urlsplit(request_target)
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.path != _CALLBACK_PATH
            or parsed.fragment
            or len(parsed.query.encode("ascii")) > _MAX_QUERY_BYTES
        ):
            raise ValueError
        fields = urllib.parse.parse_qsl(
            parsed.query,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=_MAX_QUERY_FIELDS,
        )
        if any(
            len(name.encode("utf-8")) > _MAX_QUERY_COMPONENT_BYTES
            or len(value.encode("utf-8")) > _MAX_QUERY_COMPONENT_BYTES
            for name, value in fields
        ):
            raise ValueError
    except (UnicodeError, ValueError):
        raise ChatGPTConfigurationError(
            "ChatGPT authorization callback was invalid"
        ) from None
    callback_url = f"{redirect_uri}?{parsed.query}"
    return attempt.consume_callback(callback_url)


class CallbackListener(Protocol):
    redirect_uri: str

    def __enter__(self) -> "CallbackListener": ...

    def __exit__(self, *args: object) -> None: ...

    def wait_for_grant(
        self, attempt: AuthorizationAttempt, *, timeout_seconds: float
    ) -> AuthorizationGrant: ...


class LoopbackCallbackListener:
    """A single-use callback receiver bound only to IPv4 loopback."""

    def __init__(self) -> None:
        self._attempt: AuthorizationAttempt | None = None
        self._outcome = _CallbackOutcome()
        self._server = HTTPServer(("127.0.0.1", 0), self._handler_type())
        host, port = self._server.server_address[:2]
        if host != "127.0.0.1":  # pragma: no cover - defensive OS invariant
            self._server.server_close()
            raise ChatGPTConfigurationError(
                "ChatGPT callback listener was not bound to loopback"
            )
        self.redirect_uri = f"http://127.0.0.1:{port}{_CALLBACK_PATH}"

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        listener = self

        class CallbackHandler(BaseHTTPRequestHandler):
            server_version = "LightphoneGPT"
            sys_version = ""

            def log_message(self, _format: str, *args: object) -> None:
                return

            def _reply(self, status: int, page: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'none'")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(page)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                attempt = listener._attempt
                if attempt is None or listener._outcome.finished:
                    self._reply(409, _ERROR_PAGE)
                    return
                try:
                    grant = _consume_callback_target(
                        attempt, listener.redirect_uri, self.path
                    )
                except ChatGPTAuthorizationRequiredError as error:
                    listener._outcome.error = error
                    listener._outcome.finished = True
                    self._reply(400, _ERROR_PAGE)
                    return
                except (ChatGPTConfigurationError, UnicodeError, ValueError):
                    self._reply(400, _ERROR_PAGE)
                    return
                listener._outcome.grant = grant
                listener._outcome.finished = True
                self._reply(200, _SUCCESS_PAGE)

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._reply(405, _ERROR_PAGE)

        return CallbackHandler

    def __enter__(self) -> "LoopbackCallbackListener":
        return self

    def __exit__(self, *args: object) -> None:
        self._server.server_close()

    def wait_for_grant(
        self, attempt: AuthorizationAttempt, *, timeout_seconds: float
    ) -> AuthorizationGrant:
        if (
            not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds > _MAX_LOGIN_SECONDS
        ):
            raise ChatGPTConfigurationError(
                "ChatGPT authorization timeout is invalid"
            )
        if self._attempt is not None:
            raise ChatGPTConfigurationError(
                "ChatGPT callback listener is single-use"
            )
        self._attempt = attempt
        deadline = time.monotonic() + timeout_seconds
        while not self._outcome.finished:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ChatGPTAuthorizationRequiredError(
                    "ChatGPT authorization timed out"
                )
            self._server.timeout = min(0.25, remaining)
            self._server.handle_request()
        if self._outcome.error is not None:
            raise self._outcome.error
        if self._outcome.grant is None:  # pragma: no cover - internal invariant
            raise ChatGPTConfigurationError(
                "ChatGPT authorization callback was not accepted"
            )
        return self._outcome.grant


BrowserOpener = Callable[[str], bool]
ListenerFactory = Callable[[], CallbackListener]
ModelsRequester = Callable[[str, float], Mapping[str, object]]


class _ModelsEndpointError(RuntimeError):
    """Content-free HTTP failure from the model catalog endpoint."""

    def __init__(self, status_code: int) -> None:
        super().__init__("ChatGPT model discovery request failed")
        self.status_code = status_code


class _NoModelsRedirects(urllib.request.HTTPRedirectHandler):
    """Keep an OAuth bearer token on the single documented origin."""

    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


def _open_models_request(
    request: urllib.request.Request, timeout_seconds: float
) -> object:
    return urllib.request.build_opener(_NoModelsRedirects()).open(
        request, timeout=timeout_seconds
    )


def _get_models_json(
    access_token: str, timeout_seconds: float
) -> Mapping[str, object]:
    """Fetch one bounded model catalog without exposing the bearer token."""

    request = urllib.request.Request(
        _MODELS_ENDPOINT,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        },
        method="GET",
    )
    try:
        with _open_models_request(request, timeout_seconds) as response:
            raw = response.read(_MAX_MODELS_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        status_code = error.code
        try:
            error.close()
        except OSError:
            pass
        raise _ModelsEndpointError(status_code) from None
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
        raise ChatGPTTransientError(
            "ChatGPT model discovery is temporarily unavailable"
        ) from None

    if len(raw) > _MAX_MODELS_RESPONSE_BYTES:
        raise ChatGPTTransientError(
            "ChatGPT model discovery returned an oversized response"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ChatGPTTransientError(
            "ChatGPT model discovery returned an invalid response"
        ) from None
    if not isinstance(payload, Mapping):
        raise ChatGPTTransientError(
            "ChatGPT model discovery returned an invalid response"
        )
    return payload


def _raise_models_endpoint_error(error: _ModelsEndpointError) -> None:
    if error.status_code == 401:
        raise ChatGPTAuthorizationRequiredError(
            "ChatGPT authorization must be renewed"
        ) from None
    if error.status_code in {400, 403, 404}:
        raise ChatGPTConfigurationError(
            "ChatGPT model discovery was rejected"
        ) from None
    raise ChatGPTTransientError(
        "ChatGPT model discovery is temporarily unavailable"
    ) from None


def discover_models(
    store: CredentialStore,
    *,
    timeout_seconds: float = 15.0,
    requester: ModelsRequester = _get_models_json,
) -> dict[str, object]:
    """Return only account-visible model labels from the supported catalog."""

    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or timeout_seconds > _MAX_MODELS_SECONDS
    ):
        raise ChatGPTConfigurationError("ChatGPT model discovery timeout is invalid")

    tokens = RefreshManager(store, timeout_seconds=timeout_seconds)
    access_token = tokens.get_access_token()
    try:
        payload = requester(access_token, timeout_seconds)
    except _ModelsEndpointError as error:
        if error.status_code != 401:
            _raise_models_endpoint_error(error)
        access_token = tokens.get_access_token(force_refresh=True)
        try:
            payload = requester(access_token, timeout_seconds)
        except _ModelsEndpointError as retry_error:
            _raise_models_endpoint_error(retry_error)
        except (
            ChatGPTAuthorizationRequiredError,
            ChatGPTConfigurationError,
            ChatGPTTransientError,
        ):
            raise
        except Exception:
            raise ChatGPTTransientError(
                "ChatGPT model discovery is temporarily unavailable"
            ) from None
    except (
        ChatGPTAuthorizationRequiredError,
        ChatGPTConfigurationError,
        ChatGPTTransientError,
    ):
        raise
    except Exception:
        raise ChatGPTTransientError(
            "ChatGPT model discovery is temporarily unavailable"
        ) from None

    if not isinstance(payload, Mapping):
        raise ChatGPTTransientError(
            "ChatGPT model discovery returned an invalid response"
        )
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ChatGPTTransientError(
            "ChatGPT model discovery returned an invalid response"
        )
    models: list[dict[str, str]] = []
    for raw_model in raw_models:
        if (
            not isinstance(raw_model, Mapping)
            or raw_model.get("visibility") != "list"
        ):
            continue
        slug = raw_model.get("slug")
        display_name = raw_model.get("display_name")
        if not isinstance(slug, str) or not isinstance(display_name, str):
            continue
        slug = slug.strip()
        display_name = display_name.strip()
        if not slug or not display_name:
            continue
        models.append({"slug": slug, "display_name": display_name})
    return {"ok": True, "models": models}


def _open_browser(url: str) -> bool:
    try:
        return bool(webbrowser.open_new_tab(url))
    except Exception:
        return False


def _load_previous(store: CredentialStore) -> ChatGPTCredentialProfile | None:
    try:
        return store.load()
    except ChatGPTAuthorizationRequiredError:
        return None


def _account_binding(profile: ChatGPTCredentialProfile) -> str:
    material = f"{profile.issuer}\0{profile.subject}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def profile_status(
    store: CredentialStore, *, now: datetime | None = None
) -> dict[str, object]:
    try:
        profile = store.load()
    except ChatGPTAuthorizationRequiredError:
        return {"ok": True, "authorized": False}
    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("current time must include a timezone")
    current = current.astimezone(UTC)
    refresh_available_at = profile.earliest_refresh_at
    return {
        "ok": True,
        "authorized": True,
        "account_binding": _account_binding(profile),
        "access_token_state": (
            "current" if current < profile.access_expires_at else "expired"
        ),
        "access_expires_at": profile.access_expires_at.isoformat().replace(
            "+00:00", "Z"
        ),
        "refresh_state": (
            "available"
            if refresh_available_at is None or current >= refresh_available_at
            else "deferred"
        ),
        "refresh_available_at": (
            refresh_available_at.isoformat().replace("+00:00", "Z")
            if refresh_available_at is not None
            else None
        ),
        "plan_use_scope": True,
    }


def login(
    paths: ChatGPTAuthPaths,
    *,
    timeout_seconds: float = _MAX_LOGIN_SECONDS,
    browser_opener: BrowserOpener = _open_browser,
    listener_factory: ListenerFactory = LoopbackCallbackListener,
    token_requester: TokenRequester = _post_form_json,
    validator: OidcValidator | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Complete one interactive authorization and durably activate it."""

    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or timeout_seconds > _MAX_LOGIN_SECONDS
    ):
        raise ChatGPTConfigurationError("ChatGPT authorization timeout is invalid")
    clock = now or (lambda: datetime.now(UTC))
    store = CredentialStore(paths.credential_file)
    previous = _load_previous(store)
    host_id = HostIdStore(paths.host_id_file).load_or_create(
        preferred=previous.ext_agent_host_id if previous else None
    )

    with listener_factory() as listener:
        attempt = new_authorization_attempt(
            ext_agent_host_id=host_id,
            redirect_uri=listener.redirect_uri,
            client_id=previous.client_id if previous else DYNAMIC_CLIENT_ID,
            id_token_hint=previous.id_token if previous else None,
            login_hint=previous.email if previous else None,
            now=clock(),
        )
        try:
            opened = browser_opener(attempt.authorization_url)
        except Exception:
            opened = False
        if not opened:
            raise ChatGPTConfigurationError(
                "ChatGPT authorization browser could not be opened"
            )
        grant = listener.wait_for_grant(
            attempt, timeout_seconds=timeout_seconds
        )

    try:
        response = token_requester(
            TOKEN_ENDPOINT, attempt.token_exchange_form(grant), 15.0
        )
    except OAuthEndpointError as error:
        if error.code == "invalid_client":
            raise ChatGPTConfigurationError(
                "ChatGPT client registration is invalid"
            ) from None
        if error.status_code in {400, 401, 403}:
            raise ChatGPTAuthorizationRequiredError(
                "ChatGPT authorization could not be completed"
            ) from None
        raise ChatGPTTransientError(
            "ChatGPT token exchange is temporarily unavailable"
        ) from None
    except (ChatGPTAuthorizationRequiredError, ChatGPTConfigurationError, ChatGPTTransientError):
        raise
    except Exception:
        raise ChatGPTTransientError(
            "ChatGPT token exchange is temporarily unavailable"
        ) from None
    profile = activate_token_response(
        response,
        attempt=attempt,
        grant=grant,
        validator=validator or OidcValidator(now=clock),
        previous_profile=previous,
        now=clock(),
    )
    store.save(profile)
    return {"authorization": "completed", **profile_status(store, now=clock())}


def run_command(
    args: object, environ: Mapping[str, str]
) -> dict[str, object]:
    credential_file = getattr(args, "credential_file", None)
    paths = resolve_auth_paths(environ, credential_file=credential_file)
    command = getattr(args, "chatgpt_auth_command", "")
    if command == "status":
        return profile_status(CredentialStore(paths.credential_file))
    if command == "login":
        return login(paths, timeout_seconds=float(getattr(args, "timeout_seconds")))
    if command == "models":
        return discover_models(
            CredentialStore(paths.credential_file),
            timeout_seconds=float(getattr(args, "timeout_seconds")),
        )
    raise ChatGPTConfigurationError("Unknown ChatGPT authorization command")

