from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import urllib.parse
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import threading

import pytest

import lightphone_gpt.chatgpt_auth as chatgpt_auth

from lightphone_gpt.chatgpt_auth import (
    ChatGPTAuthPaths,
    HostIdStore,
    LoopbackCallbackListener,
    _consume_callback_target,
    login,
    profile_status,
    resolve_auth_paths,
)
from lightphone_gpt.chatgpt_errors import (
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
)
from lightphone_gpt.chatgpt_oauth import (
    AUTHORIZATION_ENDPOINT,
    DIRECT_USE_SCOPE,
    OIDC_ISSUER,
    TOKEN_ENDPOINT,
    AuthorizationGrant,
    ChatGPTCredentialProfile,
    CredentialStore,
    ValidatedIdentity,
    new_authorization_attempt,
)


NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
HOST_ID = "urn:uuid:11111111-2222-4333-8444-555555555555"


class FakeValidator:
    def __init__(self, *, subject: str = "account-subject") -> None:
        self.subject = subject
        self.calls: list[tuple[str, str, str]] = []

    def validate(
        self, *, id_token: str, issued_client_id: str, expected_nonce: str
    ) -> ValidatedIdentity:
        self.calls.append((id_token, issued_client_id, expected_nonce))
        return ValidatedIdentity(
            issuer=OIDC_ISSUER,
            subject=self.subject,
            client_id=issued_client_id,
            email="person@example.test",
        )


class FakeListener:
    def __init__(self) -> None:
        self.redirect_uri = "http://127.0.0.1:45678/auth/callback"
        self.attempt = None
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_args: object) -> None:
        self.entered = False

    def wait_for_grant(self, attempt, *, timeout_seconds: float):
        assert self.entered
        assert timeout_seconds == 25
        self.attempt = attempt
        return AuthorizationGrant(
            code="authorization-code",
            client_id=(
                "oaiapp_issued"
                if attempt.client_id == "dynamic_agent_client"
                else attempt.client_id
            ),
            callback_scopes=("openid", DIRECT_USE_SCOPE),
        )


def token_response() -> dict[str, object]:
    return {
        "id_token": "signed-id-token",
        "access_token": "private-access-token",
        "refresh_token": "private-refresh-token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": f"openid offline_access {DIRECT_USE_SCOPE}",
    }


def saved_profile() -> ChatGPTCredentialProfile:
    return ChatGPTCredentialProfile(
        issuer=OIDC_ISSUER,
        subject="account-subject",
        client_id="oaiapp_issued",
        ext_agent_host_id=HOST_ID,
        id_token="old-id-token",
        access_token="old-access-token",
        refresh_token="old-refresh-token",
        scopes=("openid", "offline_access", DIRECT_USE_SCOPE),
        expires_in=3600,
        saved_at=NOW,
        email="person@example.test",
    )


def test_paths_default_to_xdg_and_require_absolute_override(tmp_path: Path) -> None:
    paths = resolve_auth_paths({"XDG_CONFIG_HOME": str(tmp_path)})
    assert paths == ChatGPTAuthPaths(
        credential_file=tmp_path
        / "lightphone-gpt"
        / "chatgpt"
        / "credentials.json",
        host_id_file=tmp_path / "lightphone-gpt" / "chatgpt" / "host-id",
    )

    overridden = resolve_auth_paths(
        {}, credential_file=tmp_path / "other" / "profile.json"
    )
    assert overridden.host_id_file == tmp_path / "other" / "host-id"
    with pytest.raises(ChatGPTConfigurationError, match="absolute"):
        resolve_auth_paths({}, credential_file=Path("relative.json"))


def test_host_id_is_stable_owner_only_and_rejects_mismatch(tmp_path: Path) -> None:
    location = tmp_path / "private" / "host-id"
    store = HostIdStore(location)

    assert store.load_or_create(preferred=HOST_ID) == HOST_ID
    assert store.load_or_create(preferred=HOST_ID) == HOST_ID
    assert location.parent.stat().st_mode & 0o777 == 0o700
    assert location.stat().st_mode & 0o777 == 0o600

    with pytest.raises(ChatGPTConfigurationError, match="did not match"):
        store.load_or_create(
            preferred="urn:uuid:aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        )


def test_host_id_create_fsync_failure_never_publishes_partial_file(
    tmp_path: Path, monkeypatch
) -> None:
    location = tmp_path / "private" / "host-id"
    store = HostIdStore(location)
    monkeypatch.setattr(
        chatgpt_auth.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("private failure")),
    )

    with pytest.raises(ChatGPTConfigurationError) as captured:
        store.load_or_create(preferred=HOST_ID)

    assert "private" not in str(captured.value)
    assert not location.exists()
    assert list(location.parent.glob(".host-id.import.*")) == []


def test_concurrent_host_id_creates_return_one_complete_winner(
    tmp_path: Path, monkeypatch
) -> None:
    location = tmp_path / "private" / "host-id"
    barrier = threading.Barrier(2)
    real_link = chatgpt_auth.os.link

    def racing_link(source, destination, *, follow_symlinks=True):
        barrier.wait(timeout=2)
        return real_link(
            source, destination, follow_symlinks=follow_symlinks
        )

    monkeypatch.setattr(chatgpt_auth.os, "link", racing_link)

    with ThreadPoolExecutor(max_workers=2) as executor:
        observed = list(
            executor.map(
                lambda _index: HostIdStore(location).load_or_create(),
                range(2),
            )
        )

    assert len(set(observed)) == 1
    assert HostIdStore(location).load() == observed[0]
    assert location.read_text(encoding="ascii") == f"{observed[0]}\n"
    assert location.stat().st_mode & 0o777 == 0o600
    assert list(location.parent.glob(".host-id.import.*")) == []


def test_callback_target_rejects_invalid_request_then_accepts_one_callback() -> None:
    redirect_uri = "http://127.0.0.1:45678/auth/callback"
    attempt = new_authorization_attempt(
        ext_agent_host_id=HOST_ID,
        redirect_uri=redirect_uri,
    )
    invalid_target = (
        "/auth/callback?"
        + urllib.parse.urlencode(
            {
                "code": "must-not-appear-in-page",
                "state": "wrong-state",
                "client_id": "oaiapp_issued",
            }
        )
    )
    with pytest.raises(ChatGPTConfigurationError, match="state"):
        _consume_callback_target(attempt, redirect_uri, invalid_target)

    valid_target = (
        "/auth/callback?"
        + urllib.parse.urlencode(
            {
                "code": "authorization-code",
                "state": attempt.state,
                "client_id": "oaiapp_issued",
                "scope": f"openid {DIRECT_USE_SCOPE}",
            }
        )
    )
    grant = _consume_callback_target(attempt, redirect_uri, valid_target)
    assert grant.code == "authorization-code"
    assert grant.client_id == "oaiapp_issued"
    with pytest.raises(ChatGPTConfigurationError, match="already consumed"):
        _consume_callback_target(attempt, redirect_uri, valid_target)
    pages = chatgpt_auth._SUCCESS_PAGE + chatgpt_auth._ERROR_PAGE
    assert b"authorization-code" not in pages
    assert attempt.state.encode() not in pages
    assert b"http://" not in pages and b"https://" not in pages


@pytest.mark.parametrize(
    "target",
    [
        "/wrong/path?state=x&code=x&client_id=oaiapp_x",
        "http://127.0.0.1:45678/auth/callback?state=x&code=x&client_id=oaiapp_x",
        "/auth/callback?" + "x=" + ("a" * 4097),
        "/auth/callback?" + "&".join(f"x{index}=a" for index in range(17)),
    ],
)
def test_callback_target_enforces_path_and_query_bounds(target: str) -> None:
    redirect_uri = "http://127.0.0.1:45678/auth/callback"
    attempt = new_authorization_attempt(
        ext_agent_host_id=HOST_ID,
        redirect_uri=redirect_uri,
    )
    with pytest.raises(ChatGPTConfigurationError, match="invalid"):
        _consume_callback_target(attempt, redirect_uri, target)


def test_listener_binds_ipv4_loopback_ephemeral_before_exposing_redirect(
    monkeypatch,
) -> None:
    created: list[object] = []

    class FakeServer:
        server_address = ("127.0.0.1", 32123)

        def __init__(self, address, handler):
            assert address == ("127.0.0.1", 0)
            assert issubclass(handler, chatgpt_auth.BaseHTTPRequestHandler)
            created.append(self)

        def server_close(self):
            return None

    monkeypatch.setattr(chatgpt_auth, "HTTPServer", FakeServer)
    with LoopbackCallbackListener() as listener:
        assert created
        assert listener.redirect_uri == "http://127.0.0.1:32123/auth/callback"


def test_listener_times_out_without_accepting_a_callback(monkeypatch) -> None:
    class FakeServer:
        server_address = ("127.0.0.1", 32123)
        timeout = None

        def __init__(self, _address, _handler):
            return None

        def handle_request(self):
            return None

        def server_close(self):
            return None

    moments = iter((0.0, 1.0))
    monkeypatch.setattr(chatgpt_auth, "HTTPServer", FakeServer)
    monkeypatch.setattr(chatgpt_auth.time, "monotonic", lambda: next(moments))
    with LoopbackCallbackListener() as listener:
        attempt = new_authorization_attempt(
            ext_agent_host_id=HOST_ID,
            redirect_uri=listener.redirect_uri,
        )
        with pytest.raises(ChatGPTAuthorizationRequiredError, match="timed out"):
            listener.wait_for_grant(attempt, timeout_seconds=0.5)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0.0, 601.0])
def test_login_rejects_invalid_timeout_before_touching_storage(
    tmp_path: Path, timeout: float
) -> None:
    paths = ChatGPTAuthPaths(
        tmp_path / "auth" / "credentials.json", tmp_path / "auth" / "host-id"
    )

    with pytest.raises(ChatGPTConfigurationError, match="timeout"):
        login(paths, timeout_seconds=timeout)

    assert not paths.host_id_file.exists()


def test_login_opens_browser_exchanges_validates_then_saves(tmp_path: Path) -> None:
    paths = ChatGPTAuthPaths(
        tmp_path / "auth" / "credentials.json", tmp_path / "auth" / "host-id"
    )
    listener = FakeListener()
    opened: list[str] = []
    exchanges: list[tuple[str, dict[str, str], float]] = []
    validator = FakeValidator()

    def exchange(url: str, form, timeout: float):
        exchanges.append((url, dict(form), timeout))
        return token_response()

    result = login(
        paths,
        timeout_seconds=25,
        browser_opener=lambda url: opened.append(url) is None or True,
        listener_factory=lambda: listener,
        token_requester=exchange,
        validator=validator,  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    assert result["authorization"] == "completed"
    assert result["authorized"] is True
    assert "private" not in json.dumps(result)
    authorization = urllib.parse.urlsplit(opened[0])
    assert f"{authorization.scheme}://{authorization.netloc}{authorization.path}" == AUTHORIZATION_ENDPOINT
    query = urllib.parse.parse_qs(authorization.query)
    assert query["redirect_uri"] == [listener.redirect_uri]
    assert query["ext_agent_host_id"] == [HostIdStore(paths.host_id_file).load()]
    assert exchanges[0][0] == TOKEN_ENDPOINT
    assert exchanges[0][1]["code"] == "authorization-code"
    stored = CredentialStore(paths.credential_file).load()
    assert stored.access_token == "private-access-token"
    assert paths.credential_file.stat().st_mode & 0o777 == 0o600


def test_reauthorization_uses_saved_registration_and_account_hint(
    tmp_path: Path,
) -> None:
    paths = ChatGPTAuthPaths(
        tmp_path / "auth" / "credentials.json", tmp_path / "auth" / "host-id"
    )
    original = saved_profile()
    CredentialStore(paths.credential_file).save(original)
    HostIdStore(paths.host_id_file).load_or_create(preferred=HOST_ID)
    listener = FakeListener()
    opened: list[str] = []

    login(
        paths,
        timeout_seconds=25,
        browser_opener=lambda url: opened.append(url) is None or True,
        listener_factory=lambda: listener,
        token_requester=lambda *_args: token_response(),
        validator=FakeValidator(),  # type: ignore[arg-type]
        now=lambda: NOW + timedelta(minutes=5),
    )

    query = urllib.parse.parse_qs(urllib.parse.urlsplit(opened[0]).query)
    assert query["client_id"] == [original.client_id]
    assert query["id_token_hint"] == [original.id_token]
    assert query["login_hint"] == [original.email]
    assert "agent_name_hint" not in query


def test_failed_validation_never_replaces_saved_credentials(tmp_path: Path) -> None:
    paths = ChatGPTAuthPaths(
        tmp_path / "auth" / "credentials.json", tmp_path / "auth" / "host-id"
    )
    CredentialStore(paths.credential_file).save(saved_profile())
    HostIdStore(paths.host_id_file).load_or_create(preferred=HOST_ID)
    original_bytes = paths.credential_file.read_bytes()

    with pytest.raises(ChatGPTAuthorizationRequiredError, match="different"):
        login(
            paths,
            timeout_seconds=25,
            browser_opener=lambda _url: True,
            listener_factory=FakeListener,
            token_requester=lambda *_args: token_response(),
            validator=FakeValidator(subject="different-account"),  # type: ignore[arg-type]
            now=lambda: NOW,
        )

    assert paths.credential_file.read_bytes() == original_bytes


def test_browser_failure_does_not_call_token_endpoint_or_create_credentials(
    tmp_path: Path,
) -> None:
    paths = ChatGPTAuthPaths(
        tmp_path / "auth" / "credentials.json", tmp_path / "auth" / "host-id"
    )

    def forbidden(*_args: object):
        raise AssertionError("token endpoint must not be called")

    with pytest.raises(ChatGPTConfigurationError, match="browser"):
        login(
            paths,
            browser_opener=lambda _url: False,
            listener_factory=FakeListener,
            token_requester=forbidden,
            validator=FakeValidator(),  # type: ignore[arg-type]
            now=lambda: NOW,
        )

    assert not paths.credential_file.exists()
    assert paths.host_id_file.exists()


def test_status_is_content_free_for_missing_current_and_expired_profiles(
    tmp_path: Path,
) -> None:
    store = CredentialStore(tmp_path / "auth" / "credentials.json")
    assert profile_status(store, now=NOW) == {"ok": True, "authorized": False}

    store.save(saved_profile())
    current = profile_status(store, now=NOW + timedelta(minutes=1))
    expired = profile_status(store, now=NOW + timedelta(hours=2))
    serialized = json.dumps([current, expired])
    assert current["access_token_state"] == "current"
    assert expired["access_token_state"] == "expired"
    assert current["refresh_state"] == "available"
    assert current["refresh_available_at"] is None
    for private in (
        "person@example.test",
        "account-subject",
        "old-id-token",
        "old-access-token",
        "old-refresh-token",
        "oaiapp_issued",
    ):
        assert private not in serialized


def test_status_reports_issuer_refresh_deferral_without_credentials(
    tmp_path: Path,
) -> None:
    store = CredentialStore(tmp_path / "auth" / "credentials.json")
    store.save(
        replace(
            saved_profile(),
            earliest_refresh_at=NOW + timedelta(minutes=30),
        )
    )

    deferred = profile_status(store, now=NOW + timedelta(minutes=1))
    available = profile_status(store, now=NOW + timedelta(minutes=30))

    assert deferred["refresh_state"] == "deferred"
    assert deferred["refresh_available_at"] == "2026-10-15T12:30:00Z"
    assert available["refresh_state"] == "available"

