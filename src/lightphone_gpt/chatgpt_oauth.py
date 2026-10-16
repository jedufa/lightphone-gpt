"""Supported Sign in with ChatGPT primitives for a self-hosted bridge.

The module deliberately separates pure authorization construction and validation
from browser and network orchestration. Importing it never opens a browser or
makes a request. The production HTTP and PyJWT helpers are lazy defaults; tests
and administrative tooling can inject deterministic seams.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import socket
import stat
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Protocol

from .chatgpt_errors import (
    ChatGPTAmbiguousOutcomeError,
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTCredentialAmbiguousError,
    ChatGPTTransientError,
)


OIDC_ISSUER = "https://auth.openai.com"
OIDC_DISCOVERY_URL = f"{OIDC_ISSUER}/.well-known/openid-configuration"
AUTHORIZATION_ENDPOINT = f"{OIDC_ISSUER}/api/accounts/authorize"
TOKEN_ENDPOINT = f"{OIDC_ISSUER}/api/accounts/oauth/token"
JWKS_URI = f"{OIDC_ISSUER}/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
DIRECT_USE_SCOPE = "chatgpt.tokens.use.direct"
REQUESTED_SCOPES = (
    "openid",
    "profile",
    "email",
    "offline_access",
    "resource.invoke",
    DIRECT_USE_SCOPE,
)
_SAFE_JWT_ALGORITHM_PREFIXES = ("RS", "PS", "ES")
_MAX_CREDENTIAL_BYTES = 256 * 1024
_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def _shared_path_lock(path: Path) -> threading.RLock:
    key = os.path.abspath(path)
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value.astimezone(UTC)


def _format_time(value: datetime) -> str:
    return _aware_utc(value, field_name="timestamp").isoformat().replace(
        "+00:00", "Z"
    )


def _parse_time(value: object, *, field_name: str) -> datetime:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} is invalid")
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            raise ValueError(f"{field_name} is invalid") from None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is invalid")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return _aware_utc(datetime.fromisoformat(normalized), field_name=field_name)
    except ValueError:
        raise ValueError(f"{field_name} is invalid") from None


def _required_text(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"credential field {name} is invalid")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ChatGPTCredentialProfile:
    """One validated ChatGPT account registration and rotating token set."""

    issuer: str
    subject: str
    client_id: str
    ext_agent_host_id: str
    id_token: str = field(repr=False)
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    scopes: tuple[str, ...]
    expires_in: int
    saved_at: datetime
    token_type: str = "Bearer"
    earliest_refresh_at: datetime | None = None
    email: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported credential schema")
        if self.issuer != OIDC_ISSUER:
            raise ValueError("credential issuer is invalid")
        for name in (
            "subject",
            "client_id",
            "ext_agent_host_id",
            "id_token",
            "access_token",
            "refresh_token",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"credential field {name} is invalid")
        if not self.ext_agent_host_id.startswith(
            ("urn:uuid:", "urn:ietf:params:oauth:jwk-thumbprint:", "did:key:")
        ):
            raise ValueError("credential host identifier is invalid")
        if self.token_type.lower() != "bearer":
            raise ValueError("credential token type is invalid")
        if isinstance(self.expires_in, bool) or self.expires_in <= 0:
            raise ValueError("credential expiry is invalid")
        if DIRECT_USE_SCOPE not in self.scopes:
            raise ValueError("ChatGPT plan-use scope is missing")
        if any(not isinstance(scope, str) or not scope for scope in self.scopes):
            raise ValueError("credential scopes are invalid")
        object.__setattr__(
            self, "saved_at", _aware_utc(self.saved_at, field_name="saved_at")
        )
        if self.earliest_refresh_at is not None:
            object.__setattr__(
                self,
                "earliest_refresh_at",
                _aware_utc(
                    self.earliest_refresh_at, field_name="earliest_refresh_at"
                ),
            )

    @property
    def access_expires_at(self) -> datetime:
        return self.saved_at + timedelta(seconds=self.expires_in)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "issuer": self.issuer,
            "subject": self.subject,
            "email": self.email,
            "client_id": self.client_id,
            "ext_agent_host_id": self.ext_agent_host_id,
            "id_token": self.id_token,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "scopes": list(self.scopes),
            "expires_in": self.expires_in,
            "saved_at": _format_time(self.saved_at),
            "access_expires_at": _format_time(self.access_expires_at),
            "earliest_refresh_at": (
                _format_time(self.earliest_refresh_at)
                if self.earliest_refresh_at is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ChatGPTCredentialProfile":
        schema_version = payload.get("schema_version")
        if schema_version != 1:
            raise ValueError("unsupported credential schema")
        raw_scopes = payload.get("scopes")
        if not isinstance(raw_scopes, list) or not all(
            isinstance(item, str) and item for item in raw_scopes
        ):
            raise ValueError("credential scopes are invalid")
        expires_in = payload.get("expires_in")
        if isinstance(expires_in, bool) or not isinstance(expires_in, int):
            raise ValueError("credential expiry is invalid")
        saved_at = _parse_time(payload.get("saved_at"), field_name="saved_at")
        earliest_value = payload.get("earliest_refresh_at")
        earliest = (
            None
            if earliest_value is None
            else _parse_time(earliest_value, field_name="earliest_refresh_at")
        )
        email = payload.get("email")
        if email is not None and not isinstance(email, str):
            raise ValueError("credential email is invalid")
        profile = cls(
            issuer=_required_text(payload, "issuer"),
            subject=_required_text(payload, "subject"),
            client_id=_required_text(payload, "client_id"),
            ext_agent_host_id=_required_text(payload, "ext_agent_host_id"),
            id_token=_required_text(payload, "id_token"),
            access_token=_required_text(payload, "access_token"),
            refresh_token=_required_text(payload, "refresh_token"),
            token_type=_required_text(payload, "token_type"),
            scopes=tuple(raw_scopes),
            expires_in=expires_in,
            saved_at=saved_at,
            earliest_refresh_at=earliest,
            email=email,
            schema_version=1,
        )
        serialized_expiry = payload.get("access_expires_at")
        if serialized_expiry is not None:
            recorded_expiry = _parse_time(
                serialized_expiry, field_name="access_expires_at"
            )
            if recorded_expiry != profile.access_expires_at:
                raise ValueError("credential expiry fields disagree")
        return profile


class CredentialStore:
    """Owner-only credential storage with durable atomic replacement."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.name:
            raise ValueError("credential path must name a file")
        self._lock = threading.RLock()
        self._renewal_thread_lock = _shared_path_lock(self.path)

    def _prepare_parent(self) -> None:
        parent = self.path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise ChatGPTConfigurationError(
                "ChatGPT credential directory is not a secure directory"
            )
        os.chmod(parent, 0o700)

    def save(self, profile: ChatGPTCredentialProfile) -> None:
        encoded = (
            json.dumps(
                profile.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            + b"\n"
        )
        if len(encoded) > _MAX_CREDENTIAL_BYTES:
            raise ChatGPTConfigurationError("ChatGPT credential record is too large")

        with self._lock:
            self._prepare_parent()
            if self.path.is_symlink():
                raise ChatGPTConfigurationError(
                    "ChatGPT credential file must not be a symbolic link"
                )
            descriptor = -1
            temporary = ""
            try:
                descriptor, temporary = tempfile.mkstemp(
                    prefix=f".{self.path.name}.", dir=self.path.parent
                )
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb", closefd=True) as stream:
                    descriptor = -1
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
                temporary = ""
                os.chmod(self.path, 0o600)
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except ChatGPTConfigurationError:
                raise
            except OSError:
                raise ChatGPTConfigurationError(
                    "ChatGPT credentials could not be stored"
                ) from None
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                if temporary:
                    try:
                        os.unlink(temporary)
                    except OSError:
                        pass

    def create(self, profile: ChatGPTCredentialProfile) -> None:
        """Create a credential record exactly once without replacing a path.

        Imports use this stricter operation so two operators cannot both pass
        a separate existence check and silently replace rotating credentials.
        Normal token refreshes continue to use :meth:`save`.
        """

        encoded = (
            json.dumps(
                profile.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            + b"\n"
        )
        if len(encoded) > _MAX_CREDENTIAL_BYTES:
            raise ChatGPTConfigurationError("ChatGPT credential record is too large")

        with self._lock, self._renewal_thread_lock:
            self._prepare_parent()
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
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, self.path, follow_symlinks=False)
                published = True
                os.unlink(temporary)
                temporary = ""
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except FileExistsError:
                raise ChatGPTConfigurationError(
                    "ChatGPT authorization is already configured"
                ) from None
            except ChatGPTConfigurationError:
                raise
            except OSError:
                if published:
                    raise ChatGPTAmbiguousOutcomeError(
                        "ChatGPT credential import outcome requires review"
                    ) from None
                raise ChatGPTConfigurationError(
                    "ChatGPT credentials could not be stored"
                ) from None
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                if temporary:
                    try:
                        os.unlink(temporary)
                    except OSError:
                        pass

    def load(self) -> ChatGPTCredentialProfile:
        with self._lock:
            parent = self.path.parent
            try:
                parent_metadata = parent.lstat()
            except FileNotFoundError:
                raise ChatGPTAuthorizationRequiredError(
                    "ChatGPT authorization is not configured"
                ) from None
            if (
                stat.S_ISLNK(parent_metadata.st_mode)
                or not stat.S_ISDIR(parent_metadata.st_mode)
                or stat.S_IMODE(parent_metadata.st_mode) != 0o700
            ):
                raise ChatGPTConfigurationError(
                    "ChatGPT credential directory permissions must be 0700"
                )
            try:
                metadata = self.path.lstat()
            except FileNotFoundError:
                raise ChatGPTAuthorizationRequiredError(
                    "ChatGPT authorization is not configured"
                ) from None
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ChatGPTConfigurationError(
                    "ChatGPT credential file is not a regular file"
                )
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise ChatGPTConfigurationError(
                    "ChatGPT credential file permissions must be 0600"
                )
            if metadata.st_size > _MAX_CREDENTIAL_BYTES:
                raise ChatGPTConfigurationError(
                    "ChatGPT credential record is too large"
                )
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(self.path, flags)
                with os.fdopen(descriptor, "rb", closefd=True) as stream:
                    opened_metadata = os.fstat(stream.fileno())
                    if (
                        not stat.S_ISREG(opened_metadata.st_mode)
                        or stat.S_IMODE(opened_metadata.st_mode) != 0o600
                        or opened_metadata.st_nlink != 1
                        or (
                            hasattr(os, "geteuid")
                            and opened_metadata.st_uid != os.geteuid()
                        )
                    ):
                        raise ChatGPTConfigurationError(
                            "ChatGPT credential file ownership is invalid"
                        )
                    raw = stream.read(_MAX_CREDENTIAL_BYTES + 1)
            except ChatGPTConfigurationError:
                raise
            except OSError:
                raise ChatGPTConfigurationError(
                    "ChatGPT credentials could not be read"
                ) from None
            if len(raw) > _MAX_CREDENTIAL_BYTES:
                raise ChatGPTConfigurationError(
                    "ChatGPT credential record is too large"
                )
            try:
                payload = json.loads(raw)
                if not isinstance(payload, Mapping):
                    raise ValueError
                return ChatGPTCredentialProfile.from_dict(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                raise ChatGPTConfigurationError(
                    "ChatGPT credential record is invalid"
                ) from None

    def delete(self) -> bool:
        """Atomically unlink the token-bearing record under the renewal lock.

        The stable host identifier is stored separately and is deliberately not
        touched.  The current credential schema cannot represent a token-free
        account/client mapping, so sign-out removes the complete credential
        record rather than writing empty token fields.
        """

        with self.renewal_lock():
            return self._delete_under_renewal_lock()

    def _delete_under_renewal_lock(self) -> bool:
        """Unlink the credential record while ``renewal_lock`` is held."""

        with self._lock:
            parent = self.path.parent
            directory_descriptor = -1
            credential_descriptor = -1
            unlinked = False
            try:
                directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                directory_flags |= getattr(os, "O_NOFOLLOW", 0)
                directory_descriptor = os.open(parent, directory_flags)
                directory_metadata = os.fstat(directory_descriptor)
                if (
                    not stat.S_ISDIR(directory_metadata.st_mode)
                    or stat.S_IMODE(directory_metadata.st_mode) != 0o700
                    or (
                        hasattr(os, "geteuid")
                        and directory_metadata.st_uid != os.geteuid()
                    )
                ):
                    raise ChatGPTConfigurationError(
                        "ChatGPT credential directory ownership is invalid"
                    )

                try:
                    listed_metadata = os.stat(
                        self.path.name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    return False
                if (
                    not stat.S_ISREG(listed_metadata.st_mode)
                    or stat.S_ISLNK(listed_metadata.st_mode)
                    or stat.S_IMODE(listed_metadata.st_mode) != 0o600
                    or listed_metadata.st_nlink != 1
                    or (
                        hasattr(os, "geteuid")
                        and listed_metadata.st_uid != os.geteuid()
                    )
                ):
                    raise ChatGPTConfigurationError(
                        "ChatGPT credential file ownership is invalid"
                    )

                credential_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                credential_descriptor = os.open(
                    self.path.name,
                    credential_flags,
                    dir_fd=directory_descriptor,
                )
                opened_metadata = os.fstat(credential_descriptor)
                if (
                    opened_metadata.st_dev != listed_metadata.st_dev
                    or opened_metadata.st_ino != listed_metadata.st_ino
                    or not stat.S_ISREG(opened_metadata.st_mode)
                    or stat.S_IMODE(opened_metadata.st_mode) != 0o600
                    or opened_metadata.st_nlink != 1
                    or (
                        hasattr(os, "geteuid")
                        and opened_metadata.st_uid != os.geteuid()
                    )
                ):
                    raise ChatGPTConfigurationError(
                        "ChatGPT credential file ownership is invalid"
                    )
                os.close(credential_descriptor)
                credential_descriptor = -1

                os.unlink(self.path.name, dir_fd=directory_descriptor)
                unlinked = True
                os.fsync(directory_descriptor)
                return True
            except (ChatGPTConfigurationError, ChatGPTAmbiguousOutcomeError):
                raise
            except OSError:
                if unlinked:
                    raise ChatGPTAmbiguousOutcomeError(
                        "ChatGPT credential deletion outcome requires review"
                    ) from None
                raise ChatGPTConfigurationError(
                    "ChatGPT credentials could not be deleted"
                ) from None
            finally:
                if credential_descriptor >= 0:
                    os.close(credential_descriptor)
                if directory_descriptor >= 0:
                    os.close(directory_descriptor)

    @contextmanager
    def renewal_lock(self) -> Iterator[None]:
        """Serialize refresh-token rotation across threads and Linux processes."""

        with self._renewal_thread_lock:
            self._prepare_parent()
            lock_path = self.path.with_name(f".{self.path.name}.refresh.lock")
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            descriptor = -1
            try:
                descriptor = os.open(lock_path, flags, 0o600)
                os.fchmod(descriptor, 0o600)
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise OSError
            except OSError:
                if descriptor >= 0:
                    os.close(descriptor)
                raise ChatGPTConfigurationError(
                    "ChatGPT renewal lock could not be secured"
                ) from None
            fcntl_module = None
            locked = False
            try:
                try:
                    import fcntl as fcntl_module
                except ImportError:  # pragma: no cover - deployed on Linux
                    pass
                if fcntl_module is not None:
                    try:
                        fcntl_module.flock(descriptor, fcntl_module.LOCK_EX)
                        locked = True
                    except OSError:
                        raise ChatGPTConfigurationError(
                            "ChatGPT renewal lock could not be acquired"
                        ) from None
                yield
            finally:
                if locked and fcntl_module is not None:
                    try:
                        fcntl_module.flock(descriptor, fcntl_module.LOCK_UN)
                    except OSError:
                        pass
                os.close(descriptor)


class OAuthEndpointError(RuntimeError):
    """Sanitized structured failure from the OAuth token endpoint."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(f"OAuth endpoint rejected the request with HTTP {status_code}")
        self.status_code = status_code
        self.code = code
        self.request_id = request_id


TokenRequester = Callable[[str, Mapping[str, str], float], Mapping[str, object]]


def _post_form_json(
    url: str, form: Mapping[str, str], timeout_seconds: float
) -> Mapping[str, object]:
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(form).encode("ascii"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(_MAX_CREDENTIAL_BYTES)
    except urllib.error.HTTPError as error:
        raw = error.read(_MAX_CREDENTIAL_BYTES)
        code: str | None = None
        try:
            body = json.loads(raw)
            if isinstance(body, Mapping) and isinstance(body.get("error"), str):
                code = body["error"]
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        raise OAuthEndpointError(
            status_code=error.code,
            code=code,
            request_id=error.headers.get("openai-request-id")
            or error.headers.get("x-request-id"),
        ) from None
    except (urllib.error.URLError, TimeoutError, socket.timeout):
        raise ChatGPTTransientError("ChatGPT token renewal is unavailable") from None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ChatGPTTransientError(
            "ChatGPT token endpoint returned an invalid response"
        ) from None
    if not isinstance(payload, Mapping):
        raise ChatGPTTransientError(
            "ChatGPT token endpoint returned an invalid response"
        )
    return payload


_TERMINAL_REFRESH_CODES = {
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
}


class AccessTokenProvider(Protocol):
    def get_access_token(self, *, force_refresh: bool = False) -> str: ...


class RefreshManager:
    """Serialize rotating refresh tokens and expose a current access token."""

    def __init__(
        self,
        store: CredentialStore,
        *,
        token_requester: TokenRequester = _post_form_json,
        now: Callable[[], datetime] | None = None,
        refresh_skew: timedelta = timedelta(minutes=5),
        timeout_seconds: float = 15.0,
    ) -> None:
        if refresh_skew < timedelta(0):
            raise ValueError("refresh_skew must not be negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._store = store
        self._token_requester = token_requester
        self._now = now or (lambda: datetime.now(UTC))
        self._refresh_skew = refresh_skew
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()

    def get_access_token(self, *, force_refresh: bool = False) -> str:
        observed = self._store.load()
        now = _aware_utc(self._now(), field_name="current time")
        if not force_refresh and now < observed.access_expires_at - self._refresh_skew:
            return observed.access_token

        observed_refresh_token = observed.refresh_token
        with self._lock, self._store.renewal_lock():
            current = self._store.load()
            now = _aware_utc(self._now(), field_name="current time")
            another_refresh_completed = (
                current.refresh_token != observed_refresh_token
            )
            if another_refresh_completed:
                return current.access_token
            if (
                not force_refresh
                and now < current.access_expires_at - self._refresh_skew
            ):
                return current.access_token
            if (
                current.earliest_refresh_at is not None
                and now < current.earliest_refresh_at
            ):
                if now < current.access_expires_at:
                    return current.access_token
                raise ChatGPTTransientError(
                    "ChatGPT token renewal is not available yet"
                )
            return self._refresh(current, now).access_token

    def _refresh(
        self, profile: ChatGPTCredentialProfile, now: datetime
    ) -> ChatGPTCredentialProfile:
        form = {
            "grant_type": "refresh_token",
            "client_id": profile.client_id,
            "refresh_token": profile.refresh_token,
            "resource": RESOURCE,
        }
        try:
            response = self._token_requester(
                TOKEN_ENDPOINT, form, self._timeout_seconds
            )
        except OAuthEndpointError as error:
            if error.code in _TERMINAL_REFRESH_CODES:
                raise ChatGPTAuthorizationRequiredError(
                    "ChatGPT authorization must be renewed"
                ) from None
            if error.code == "invalid_client":
                raise ChatGPTConfigurationError(
                    "ChatGPT client registration is invalid"
                ) from None
            if error.status_code == 429:
                raise ChatGPTTransientError(
                    "ChatGPT token renewal is temporarily unavailable"
                ) from None
            if error.status_code >= 500:
                # A server failure response does not prove that this
                # non-idempotent refresh grant was unapplied. The endpoint may
                # have rotated the token before its request failed.
                raise ChatGPTCredentialAmbiguousError(
                    "ChatGPT token renewal outcome requires review"
                ) from None
            if error.status_code in {400, 401, 403}:
                raise ChatGPTAuthorizationRequiredError(
                    "ChatGPT authorization must be renewed"
                ) from None
            raise ChatGPTTransientError(
                "ChatGPT token renewal failed"
            ) from None
        except (ChatGPTAuthorizationRequiredError, ChatGPTConfigurationError):
            raise
        except ChatGPTTransientError:
            # A transport or response-decoding failure can happen after the
            # token endpoint accepted and rotated the one-use refresh token.
            # Retrying the old record automatically could then invalidate the
            # only recoverable state.
            raise ChatGPTCredentialAmbiguousError(
                "ChatGPT token renewal outcome requires review"
            ) from None
        except Exception:
            raise ChatGPTCredentialAmbiguousError(
                "ChatGPT token renewal outcome requires review"
            ) from None

        try:
            access_token = _required_text(response, "access_token")
            refresh_token = _required_text(response, "refresh_token")
            raw_expires = response.get("expires_in")
            if isinstance(raw_expires, bool) or not isinstance(raw_expires, int):
                raise ValueError
            raw_scope = response.get("scope")
            if raw_scope is None:
                scopes = profile.scopes
            elif isinstance(raw_scope, str):
                scopes = tuple(item for item in raw_scope.split() if item)
            else:
                raise ValueError
            if DIRECT_USE_SCOPE not in scopes:
                raise ChatGPTAuthorizationRequiredError(
                    "ChatGPT plan use is no longer authorized"
                )
            earliest_value = response.get("earliest_refresh_at")
            earliest = (
                None
                if earliest_value is None
                else _parse_time(
                    earliest_value, field_name="earliest_refresh_at"
                )
            )
            token_type = response.get("token_type", profile.token_type)
            if not isinstance(token_type, str):
                raise ValueError
            updated = replace(
                profile,
                # The retained, already validated ID token is sufficient as a
                # future account hint. A replacement from refresh would need
                # its own identity validation before it could be trusted.
                id_token=profile.id_token,
                access_token=access_token,
                refresh_token=refresh_token,
                token_type=token_type,
                scopes=scopes,
                expires_in=raw_expires,
                saved_at=now,
                earliest_refresh_at=earliest,
            )
        except (ChatGPTAuthorizationRequiredError, TypeError, ValueError):
            # Even a malformed successful response may contain a rotated
            # refresh token that we cannot safely recover or activate.
            raise ChatGPTCredentialAmbiguousError(
                "ChatGPT token renewal outcome requires review"
            ) from None
        try:
            self._store.save(updated)
        except Exception:
            # The endpoint has already issued the replacement token set. A
            # local persistence failure therefore cannot be reported as a
            # routine configuration error or safely retried.
            raise ChatGPTCredentialAmbiguousError(
                "ChatGPT token renewal outcome requires review"
            ) from None
        return updated


def _validate_loopback_redirect(redirect_uri: str) -> None:
    parsed = urllib.parse.urlsplit(redirect_uri)
    try:
        port = parsed.port
    except ValueError:
        port = None
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or not 1 <= port <= 65535
        or parsed.path != "/auth/callback"
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "redirect_uri must be http://127.0.0.1:<port>/auth/callback"
        )


@dataclass(frozen=True, slots=True)
class AuthorizationGrant:
    code: str = field(repr=False)
    client_id: str
    callback_scopes: tuple[str, ...]


@dataclass(slots=True)
class AuthorizationAttempt:
    """One short-lived, single-use OAuth authorization transaction."""

    client_id: str
    ext_agent_host_id: str
    agent_name_hint: str
    redirect_uri: str
    state: str = field(repr=False)
    nonce: str = field(repr=False)
    code_verifier: str = field(repr=False)
    code_challenge: str = field(repr=False)
    created_at: datetime
    expires_at: datetime
    id_token_hint: str | None = field(default=None, repr=False)
    login_hint: str | None = field(default=None, repr=False)
    _consumed: bool = field(default=False, init=False, repr=False)

    @property
    def authorization_url(self) -> str:
        parameters = {
            "client_id": self.client_id,
            "ext_agent_host_id": self.ext_agent_host_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(REQUESTED_SCOPES),
            "resource": RESOURCE,
            "state": self.state,
            "nonce": self.nonce,
            "code_challenge_method": "S256",
            "code_challenge": self.code_challenge,
        }
        if self.client_id == DYNAMIC_CLIENT_ID:
            parameters["agent_name_hint"] = self.agent_name_hint
        if self.id_token_hint:
            parameters["id_token_hint"] = self.id_token_hint
        if self.login_hint:
            parameters["login_hint"] = self.login_hint
        return f"{AUTHORIZATION_ENDPOINT}?{urllib.parse.urlencode(parameters)}"

    def consume_callback(
        self, callback_url: str, *, now: datetime | None = None
    ) -> AuthorizationGrant:
        if self._consumed:
            raise ChatGPTConfigurationError(
                "ChatGPT authorization callback was already consumed"
            )
        current = _aware_utc(
            now or datetime.now(UTC), field_name="callback time"
        )
        if current > self.expires_at:
            raise ChatGPTAuthorizationRequiredError(
                "ChatGPT authorization attempt expired"
            )
        callback = urllib.parse.urlsplit(callback_url)
        expected = urllib.parse.urlsplit(self.redirect_uri)
        try:
            callback_origin = (
                callback.scheme,
                callback.hostname,
                callback.port,
                callback.path,
            )
            expected_origin = (
                expected.scheme,
                expected.hostname,
                expected.port,
                expected.path,
            )
        except ValueError:
            raise ChatGPTConfigurationError(
                "ChatGPT authorization callback URI did not match"
            ) from None
        if callback_origin != expected_origin or callback.fragment:
            raise ChatGPTConfigurationError(
                "ChatGPT authorization callback URI did not match"
            )
        try:
            query = urllib.parse.parse_qs(
                callback.query, keep_blank_values=True, strict_parsing=True
            )
        except ValueError:
            raise ChatGPTConfigurationError(
                "ChatGPT authorization callback was invalid"
            ) from None
        for name in ("state", "code", "error", "client_id", "scope"):
            if len(query.get(name, ())) > 1:
                raise ChatGPTConfigurationError(
                    "ChatGPT authorization callback was ambiguous"
                )
        state = query.get("state", [""])[0]
        if not hmac.compare_digest(state, self.state):
            raise ChatGPTConfigurationError(
                "ChatGPT authorization state did not match"
            )
        callback_error = query.get("error", [""])[0]
        code = query.get("code", [""])[0]
        if not callback_error and not code:
            raise ChatGPTAuthorizationRequiredError(
                "ChatGPT authorization code was missing"
            )
        self._consumed = True
        if callback_error:
            raise ChatGPTAuthorizationRequiredError(
                "ChatGPT authorization was not granted"
            )
        returned_client = query.get("client_id", [""])[0]
        if self.client_id == DYNAMIC_CLIENT_ID:
            if not returned_client.startswith("oaiapp_"):
                raise ChatGPTConfigurationError(
                    "ChatGPT did not issue a client registration"
                )
            issued_client_id = returned_client
        else:
            if returned_client and returned_client != self.client_id:
                raise ChatGPTConfigurationError(
                    "ChatGPT authorization client did not match"
                )
            issued_client_id = self.client_id
        raw_scope = query.get("scope", [""])[0]
        return AuthorizationGrant(
            code=code,
            client_id=issued_client_id,
            callback_scopes=tuple(item for item in raw_scope.split() if item),
        )

    def token_exchange_form(self, grant: AuthorizationGrant) -> dict[str, str]:
        return {
            "grant_type": "authorization_code",
            "client_id": grant.client_id,
            "code": grant.code,
            "code_verifier": self.code_verifier,
            "redirect_uri": self.redirect_uri,
            "resource": RESOURCE,
        }


def new_authorization_attempt(
    *,
    ext_agent_host_id: str,
    redirect_uri: str,
    agent_name_hint: str = "Lightphone GPT",
    client_id: str = DYNAMIC_CLIENT_ID,
    id_token_hint: str | None = None,
    login_hint: str | None = None,
    now: datetime | None = None,
    lifetime: timedelta = timedelta(minutes=10),
) -> AuthorizationAttempt:
    """Create authorization material without opening a browser or listener."""

    _validate_loopback_redirect(redirect_uri)
    if not ext_agent_host_id.startswith(
        ("urn:uuid:", "urn:ietf:params:oauth:jwk-thumbprint:", "did:key:")
    ):
        raise ValueError("ext_agent_host_id has an unsupported format")
    if not client_id or not agent_name_hint.strip():
        raise ValueError("client and agent identifiers must not be blank")
    if lifetime <= timedelta(0):
        raise ValueError("authorization lifetime must be positive")
    created_at = _aware_utc(now or datetime.now(UTC), field_name="created_at")
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    return AuthorizationAttempt(
        client_id=client_id,
        ext_agent_host_id=ext_agent_host_id,
        agent_name_hint=agent_name_hint.strip(),
        redirect_uri=redirect_uri,
        state=secrets.token_urlsafe(32),
        nonce=secrets.token_urlsafe(32),
        code_verifier=verifier,
        code_challenge=challenge,
        created_at=created_at,
        expires_at=created_at + lifetime,
        id_token_hint=id_token_hint,
        login_hint=login_hint,
    )


def generate_host_id() -> str:
    """Generate the documented UUID-form host identifier once per host."""

    return f"urn:uuid:{uuid.uuid4()}"


JsonFetcher = Callable[[str, float], Mapping[str, object]]


class JwtVerifier(Protocol):
    def __call__(
        self,
        token: str,
        *,
        jwks_uri: str,
        issuer: str,
        audience: str,
        algorithms: tuple[str, ...],
        clock_tolerance_seconds: float,
        timeout_seconds: float,
    ) -> Mapping[str, object]: ...


def _get_json(url: str, timeout_seconds: float) -> Mapping[str, object]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(_MAX_CREDENTIAL_BYTES)
    except (urllib.error.URLError, TimeoutError, socket.timeout):
        raise ChatGPTTransientError(
            "OpenAI identity metadata is temporarily unavailable"
        ) from None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ChatGPTTransientError(
            "OpenAI identity metadata is invalid"
        ) from None
    if not isinstance(payload, Mapping):
        raise ChatGPTTransientError("OpenAI identity metadata is invalid")
    return payload


def _pyjwt_verify(
    token: str,
    *,
    jwks_uri: str,
    issuer: str,
    audience: str,
    algorithms: tuple[str, ...],
    clock_tolerance_seconds: float,
    timeout_seconds: float,
) -> Mapping[str, object]:
    try:
        import jwt
    except ImportError:
        raise ChatGPTConfigurationError(
            "Install the chatgpt optional dependency to validate authorization"
        ) from None
    try:
        jwks = jwt.PyJWKClient(
            jwks_uri,
            cache_keys=True,
            timeout=timeout_seconds,
        )
        signing_key = jwks.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=list(algorithms),
            audience=audience,
            issuer=issuer,
            leeway=clock_tolerance_seconds,
            options={"require": ["sub", "exp", "iat"]},
        )
    except (
        jwt.exceptions.PyJWKClientConnectionError,
        urllib.error.URLError,
        TimeoutError,
        socket.timeout,
    ):
        raise ChatGPTTransientError(
            "OpenAI identity signing keys are temporarily unavailable"
        ) from None
    except Exception:
        raise ChatGPTAuthorizationRequiredError(
            "ChatGPT identity token validation failed"
        ) from None
    if not isinstance(claims, Mapping):
        raise ChatGPTAuthorizationRequiredError(
            "ChatGPT identity token validation failed"
        )
    return claims


@dataclass(frozen=True, slots=True)
class ValidatedIdentity:
    issuer: str
    subject: str
    client_id: str
    email: str | None = None


class OidcValidator:
    """Validate discovery metadata and a returned OpenAI identity token."""

    def __init__(
        self,
        *,
        json_fetcher: JsonFetcher = _get_json,
        jwt_verifier: JwtVerifier = _pyjwt_verify,
        now: Callable[[], datetime] | None = None,
        timeout_seconds: float = 10.0,
        clock_tolerance_seconds: float = 5.0,
    ) -> None:
        if timeout_seconds <= 0 or clock_tolerance_seconds < 0:
            raise ValueError("OIDC timeouts and tolerances are invalid")
        self._json_fetcher = json_fetcher
        self._jwt_verifier = jwt_verifier
        self._now = now or (lambda: datetime.now(UTC))
        self._timeout_seconds = timeout_seconds
        self._clock_tolerance_seconds = clock_tolerance_seconds

    def validate(
        self,
        *,
        id_token: str,
        issued_client_id: str,
        expected_nonce: str,
    ) -> ValidatedIdentity:
        try:
            discovery = self._json_fetcher(
                OIDC_DISCOVERY_URL, self._timeout_seconds
            )
        except (ChatGPTConfigurationError, ChatGPTTransientError):
            raise
        except Exception:
            raise ChatGPTTransientError(
                "OpenAI identity metadata is temporarily unavailable"
            ) from None

        issuer = discovery.get("issuer")
        authorization_endpoint = discovery.get("authorization_endpoint")
        token_endpoint = discovery.get("token_endpoint")
        jwks_uri = discovery.get("jwks_uri")
        if (
            issuer != OIDC_ISSUER
            or authorization_endpoint != AUTHORIZATION_ENDPOINT
            or token_endpoint != TOKEN_ENDPOINT
            or jwks_uri != JWKS_URI
        ):
            raise ChatGPTConfigurationError(
                "OpenAI identity discovery metadata did not match pinned endpoints"
            )
        raw_algorithms = discovery.get("id_token_signing_alg_values_supported")
        if not isinstance(raw_algorithms, list):
            raise ChatGPTConfigurationError(
                "OpenAI identity discovery metadata omitted signing algorithms"
            )
        algorithms = tuple(
            algorithm
            for algorithm in raw_algorithms
            if isinstance(algorithm, str)
            and (
                algorithm.startswith(_SAFE_JWT_ALGORITHM_PREFIXES)
                or algorithm == "EdDSA"
            )
        )
        if not algorithms:
            raise ChatGPTConfigurationError(
                "OpenAI identity discovery metadata has no supported algorithm"
            )
        try:
            claims = self._jwt_verifier(
                id_token,
                jwks_uri=jwks_uri,
                issuer=issuer,
                audience=issued_client_id,
                algorithms=algorithms,
                clock_tolerance_seconds=self._clock_tolerance_seconds,
                timeout_seconds=self._timeout_seconds,
            )
        except (
            ChatGPTAuthorizationRequiredError,
            ChatGPTConfigurationError,
            ChatGPTTransientError,
        ):
            raise
        except Exception:
            raise ChatGPTAuthorizationRequiredError(
                "ChatGPT identity token validation failed"
            ) from None

        current = _aware_utc(self._now(), field_name="current time").timestamp()
        tolerance = self._clock_tolerance_seconds
        subject = claims.get("sub")
        expiration = claims.get("exp")
        issued_at = claims.get("iat")
        audience = claims.get("aud")
        authorized_party = claims.get("azp")
        nonce = claims.get("nonce")
        audience_matches = audience == issued_client_id or (
            isinstance(audience, list) and issued_client_id in audience
        )
        authorized_party_matches = (
            authorized_party is None
            and not (isinstance(audience, list) and len(audience) > 1)
        ) or (
            isinstance(authorized_party, str)
            and hmac.compare_digest(authorized_party, issued_client_id)
        )
        if (
            claims.get("iss") != issuer
            or not audience_matches
            or not authorized_party_matches
            or not isinstance(subject, str)
            or not subject
            or isinstance(expiration, bool)
            or not isinstance(expiration, (int, float))
            or expiration < current - tolerance
            or isinstance(issued_at, bool)
            or not isinstance(issued_at, (int, float))
            or issued_at > current + tolerance
            or not isinstance(nonce, str)
            or not hmac.compare_digest(nonce, expected_nonce)
        ):
            raise ChatGPTAuthorizationRequiredError(
                "ChatGPT identity token validation failed"
            )
        email = claims.get("email")
        return ValidatedIdentity(
            issuer=issuer,
            subject=subject,
            client_id=issued_client_id,
            email=email if isinstance(email, str) else None,
        )


def activate_token_response(
    response: Mapping[str, object],
    *,
    attempt: AuthorizationAttempt,
    grant: AuthorizationGrant,
    validator: OidcValidator,
    previous_profile: ChatGPTCredentialProfile | None = None,
    now: datetime | None = None,
) -> ChatGPTCredentialProfile:
    """Validate and bind the exact token response before activating it."""

    if attempt.client_id == DYNAMIC_CLIENT_ID:
        client_matches = grant.client_id.startswith("oaiapp_")
    else:
        client_matches = hmac.compare_digest(grant.client_id, attempt.client_id)
    if not client_matches:
        raise ChatGPTConfigurationError(
            "ChatGPT authorization grant did not match the authorization attempt"
        )
    if previous_profile is not None and (
        not hmac.compare_digest(previous_profile.client_id, attempt.client_id)
        or not hmac.compare_digest(previous_profile.client_id, grant.client_id)
        or not hmac.compare_digest(
            previous_profile.ext_agent_host_id, attempt.ext_agent_host_id
        )
    ):
        raise ChatGPTConfigurationError(
            "ChatGPT authorization did not match the saved registration"
        )

    try:
        id_token = _required_text(response, "id_token")
    except ValueError:
        raise ChatGPTConfigurationError("ChatGPT token response was invalid") from None
    identity = validator.validate(
        id_token=id_token,
        issued_client_id=grant.client_id,
        expected_nonce=attempt.nonce,
    )
    if not hmac.compare_digest(identity.client_id, grant.client_id):
        raise ChatGPTConfigurationError(
            "ChatGPT identity did not match the issued client"
        )
    if previous_profile is not None and not hmac.compare_digest(
        previous_profile.subject, identity.subject
    ):
        raise ChatGPTAuthorizationRequiredError(
            "A different ChatGPT account was selected"
        )

    raw_scope = response.get("scope")
    if not isinstance(raw_scope, str):
        raise ChatGPTAuthorizationRequiredError(
            "ChatGPT plan-use permission was not granted"
        )
    scopes = tuple(item for item in raw_scope.split() if item)
    if DIRECT_USE_SCOPE not in scopes:
        raise ChatGPTAuthorizationRequiredError(
            "ChatGPT plan-use permission was not granted"
        )
    raw_expires = response.get("expires_in")
    if isinstance(raw_expires, bool) or not isinstance(raw_expires, int):
        raise ChatGPTConfigurationError("ChatGPT token response was invalid")
    earliest_value = response.get("earliest_refresh_at")
    try:
        earliest = (
            None
            if earliest_value is None
            else _parse_time(
                earliest_value, field_name="earliest_refresh_at"
            )
        )
        return ChatGPTCredentialProfile(
            issuer=identity.issuer,
            subject=identity.subject,
            client_id=identity.client_id,
            ext_agent_host_id=attempt.ext_agent_host_id,
            id_token=id_token,
            access_token=_required_text(response, "access_token"),
            refresh_token=_required_text(response, "refresh_token"),
            token_type=str(response.get("token_type", "Bearer")),
            scopes=scopes,
            expires_in=raw_expires,
            saved_at=_aware_utc(
                now or datetime.now(UTC), field_name="saved_at"
            ),
            earliest_refresh_at=earliest,
            email=identity.email,
        )
    except ValueError:
        raise ChatGPTConfigurationError("ChatGPT token response was invalid") from None
