from __future__ import annotations

import fcntl
import json
import os
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lightphone_gpt.adapters.errors import (
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTTransientError,
)
from lightphone_gpt.chatgpt_errors import ChatGPTCredentialAmbiguousError
from lightphone_gpt.chatgpt_oauth import (
    AUTHORIZATION_ENDPOINT,
    DIRECT_USE_SCOPE,
    DYNAMIC_CLIENT_ID,
    JWKS_URI,
    OIDC_DISCOVERY_URL,
    OIDC_ISSUER,
    RESOURCE,
    TOKEN_ENDPOINT,
    AuthorizationAttempt,
    AuthorizationGrant,
    ChatGPTCredentialProfile,
    CredentialStore,
    OAuthEndpointError,
    OidcValidator,
    RefreshManager,
    ValidatedIdentity,
    generate_host_id,
    new_authorization_attempt,
    activate_token_response,
)


NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
HOST_ID = "urn:uuid:11111111-2222-4333-8444-555555555555"


def profile(*, expired: bool = False, suffix: str = "old") -> ChatGPTCredentialProfile:
    saved_at = NOW - timedelta(hours=2) if expired else NOW
    return ChatGPTCredentialProfile(
        issuer=OIDC_ISSUER,
        subject="account-subject",
        client_id="oaiapp_client",
        ext_agent_host_id=HOST_ID,
        id_token=f"id-{suffix}",
        access_token=f"access-{suffix}",
        refresh_token=f"refresh-{suffix}",
        scopes=("openid", "offline_access", DIRECT_USE_SCOPE),
        expires_in=3600,
        saved_at=saved_at,
        email="person@example.test",
    )


def test_credential_store_round_trip_and_permissions(tmp_path: Path) -> None:
    location = tmp_path / "private" / "chatgpt-profile.json"
    store = CredentialStore(location)
    original = profile()

    store.save(original)

    assert store.load() == original
    assert location.parent.stat().st_mode & 0o777 == 0o700
    assert location.stat().st_mode & 0o777 == 0o600
    assert "access-old" not in repr(original)
    assert "refresh-old" not in repr(original)
    serialized = json.loads(location.read_text())
    assert serialized["access_expires_at"] == "2026-10-15T13:00:00Z"


def test_atomic_write_failure_preserves_complete_previous_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    location = tmp_path / "private" / "profile.json"
    store = CredentialStore(location)
    old = profile()
    store.save(old)
    real_replace = os.replace

    def fail_replace(source: object, destination: object) -> None:
        if Path(destination) == location:
            raise OSError("simulated crash boundary")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(ChatGPTConfigurationError):
        store.save(replace(old, access_token="access-new", refresh_token="refresh-new"))

    assert store.load() == old
    assert list(location.parent.glob(".profile.json.*")) == []


def test_store_rejects_group_readable_or_symlinked_credentials(tmp_path: Path) -> None:
    location = tmp_path / "profile.json"
    store = CredentialStore(location)
    store.save(profile())
    location.chmod(0o640)
    with pytest.raises(ChatGPTConfigurationError, match="0600"):
        store.load()

    target = tmp_path / "target.json"
    target.write_text("{}")
    target.chmod(0o600)
    location.unlink()
    location.symlink_to(target)
    with pytest.raises(ChatGPTConfigurationError, match="regular file"):
        store.load()


def test_refresh_rotates_atomically_and_is_single_flight(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials" / "profile.json")
    store.save(profile(expired=True))
    entered = threading.Event()
    release = threading.Event()
    calls: list[dict[str, str]] = []

    def requester(
        url: str, form: dict[str, str], timeout: float
    ) -> dict[str, object]:
        assert url == TOKEN_ENDPOINT
        assert timeout == 8.0
        calls.append(dict(form))
        entered.set()
        assert release.wait(timeout=3)
        return {
            "access_token": "access-new",
            "refresh_token": "refresh-new",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": f"openid offline_access {DIRECT_USE_SCOPE}",
            "earliest_refresh_at": (NOW + timedelta(minutes=30)).timestamp(),
        }

    manager_one = RefreshManager(
        store,
        token_requester=requester,
        now=lambda: NOW,
        timeout_seconds=8.0,
    )
    manager_two = RefreshManager(
        CredentialStore(store.path),
        token_requester=requester,
        now=lambda: NOW,
        timeout_seconds=8.0,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(manager_one.get_access_token)
        assert entered.wait(timeout=3)
        second = executor.submit(manager_two.get_access_token)
        time.sleep(0.05)
        release.set()
        assert first.result(timeout=3) == "access-new"
        assert second.result(timeout=3) == "access-new"

    assert len(calls) == 1
    assert calls[0] == {
        "grant_type": "refresh_token",
        "client_id": "oaiapp_client",
        "refresh_token": "refresh-old",
        "resource": RESOURCE,
    }
    rotated = store.load()
    assert rotated.access_token == "access-new"
    assert rotated.refresh_token == "refresh-new"
    assert rotated.id_token == "id-old"
    assert rotated.saved_at == NOW
    renewal_lock = store.path.with_name(f".{store.path.name}.refresh.lock")
    assert renewal_lock.stat().st_mode & 0o777 == 0o600


def test_refresh_lock_failure_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CredentialStore(tmp_path / "credentials" / "profile.json")
    store.save(profile(expired=True))

    def fail_lock(_descriptor: int, operation: int) -> None:
        if operation == fcntl.LOCK_EX:
            raise OSError("private operating-system detail")

    monkeypatch.setattr(fcntl, "flock", fail_lock)
    with pytest.raises(ChatGPTConfigurationError, match="could not be acquired"):
        with store.renewal_lock():
            pytest.fail("lock body must not run")


@pytest.mark.parametrize(
    ("endpoint_error", "expected"),
    [
        (
            OAuthEndpointError(status_code=400, code="refresh_token_reused"),
            ChatGPTAuthorizationRequiredError,
        ),
        (
            OAuthEndpointError(status_code=400, code="invalid_client"),
            ChatGPTConfigurationError,
        ),
        (OAuthEndpointError(status_code=429), ChatGPTTransientError),
        (OAuthEndpointError(status_code=503), ChatGPTCredentialAmbiguousError),
    ],
)
def test_refresh_error_taxonomy(
    tmp_path: Path, endpoint_error: Exception, expected: type[Exception]
) -> None:
    store = CredentialStore(tmp_path / "credentials" / "profile.json")
    store.save(profile(expired=True))

    def requester(*_args: object) -> dict[str, object]:
        raise endpoint_error

    manager = RefreshManager(store, token_requester=requester, now=lambda: NOW)
    with pytest.raises(expected):
        manager.get_access_token()
    assert store.load().refresh_token == "refresh-old"


def test_refresh_transport_uncertainty_requires_manual_review(
    tmp_path: Path,
) -> None:
    store = CredentialStore(tmp_path / "credentials" / "profile.json")
    original = profile(expired=True)
    store.save(original)
    secret = "private-upstream-transport-detail"

    def requester(*_args: object) -> dict[str, object]:
        raise ChatGPTTransientError(secret)

    manager = RefreshManager(store, token_requester=requester, now=lambda: NOW)
    with pytest.raises(
        ChatGPTCredentialAmbiguousError,
        match="outcome requires review",
    ) as captured:
        manager.get_access_token()

    assert secret not in str(captured.value)
    assert store.load() == original


def test_refresh_invalid_success_response_requires_manual_review(
    tmp_path: Path,
) -> None:
    store = CredentialStore(tmp_path / "credentials" / "profile.json")
    original = profile(expired=True)
    store.save(original)

    def requester(*_args: object) -> dict[str, object]:
        return {
            "access_token": "private-new-access",
            # The endpoint may already have rotated the omitted refresh token.
            "expires_in": 3600,
            "scope": f"openid offline_access {DIRECT_USE_SCOPE}",
        }

    manager = RefreshManager(store, token_requester=requester, now=lambda: NOW)
    with pytest.raises(
        ChatGPTCredentialAmbiguousError,
        match="outcome requires review",
    ):
        manager.get_access_token()

    assert store.load() == original


def test_refresh_persistence_failure_requires_manual_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CredentialStore(tmp_path / "credentials" / "profile.json")
    original = profile(expired=True)
    store.save(original)
    secret = "private-local-write-detail"

    def requester(*_args: object) -> dict[str, object]:
        return {
            "access_token": "private-new-access",
            "refresh_token": "private-new-refresh",
            "expires_in": 3600,
            "scope": f"openid offline_access {DIRECT_USE_SCOPE}",
        }

    def fail_save(_updated: ChatGPTCredentialProfile) -> None:
        raise ChatGPTConfigurationError(secret)

    monkeypatch.setattr(store, "save", fail_save)
    manager = RefreshManager(store, token_requester=requester, now=lambda: NOW)
    with pytest.raises(
        ChatGPTCredentialAmbiguousError,
        match="outcome requires review",
    ) as captured:
        manager.get_access_token()

    assert secret not in str(captured.value)
    assert store.load() == original


def make_attempt(*, client_id: str = DYNAMIC_CLIENT_ID) -> AuthorizationAttempt:
    return new_authorization_attempt(
        ext_agent_host_id=HOST_ID,
        redirect_uri="http://127.0.0.1:1455/auth/callback",
        client_id=client_id,
        now=NOW,
    )


def test_generated_host_id_has_supported_persistent_format() -> None:
    generated = generate_host_id()
    assert generated.startswith("urn:uuid:")
    assert len(generated) == len("urn:uuid:") + 36


def test_authorization_attempt_builds_pkce_url_and_consumes_callback_once() -> None:
    attempt = make_attempt()
    authorization = urllib.parse.urlsplit(attempt.authorization_url)
    query = urllib.parse.parse_qs(authorization.query)
    assert authorization.scheme == "https"
    authorization_endpoint = (
        f"{authorization.scheme}://{authorization.netloc}{authorization.path}"
    )
    assert authorization_endpoint == AUTHORIZATION_ENDPOINT
    assert query["client_id"] == [DYNAMIC_CLIENT_ID]
    assert query["ext_agent_host_id"] == [HOST_ID]
    assert query["resource"] == [RESOURCE]
    assert query["code_challenge_method"] == ["S256"]
    assert DIRECT_USE_SCOPE in query["scope"][0].split()
    assert "id_token_hint" not in query

    callback = (
        "http://127.0.0.1:1455/auth/callback?"
        + urllib.parse.urlencode(
            {
                "code": "authorization-code",
                "state": attempt.state,
                "client_id": "oaiapp_issued",
                "scope": f"openid {DIRECT_USE_SCOPE}",
            }
        )
    )
    grant = attempt.consume_callback(callback, now=NOW + timedelta(seconds=3))
    assert grant.client_id == "oaiapp_issued"
    assert grant.callback_scopes == ("openid", DIRECT_USE_SCOPE)
    assert attempt.token_exchange_form(grant) == {
        "grant_type": "authorization_code",
        "client_id": "oaiapp_issued",
        "code": "authorization-code",
        "code_verifier": attempt.code_verifier,
        "redirect_uri": "http://127.0.0.1:1455/auth/callback",
        "resource": RESOURCE,
    }
    with pytest.raises(ChatGPTConfigurationError, match="already consumed"):
        attempt.consume_callback(callback, now=NOW)


@pytest.mark.parametrize(
    "callback",
    [
        "http://127.0.0.1:1455/auth/callback?code=x&state=wrong&client_id=oaiapp_x",
        "http://127.0.0.1:9999/auth/callback?code=x&state=unused&client_id=oaiapp_x",
    ],
)
def test_callback_validation_fails_closed(callback: str) -> None:
    attempt = make_attempt()
    if "state=unused" in callback:
        callback = callback.replace("state=unused", f"state={attempt.state}")
    with pytest.raises(ChatGPTConfigurationError):
        attempt.consume_callback(callback, now=NOW)


def test_wrong_state_does_not_consume_legitimate_callback() -> None:
    attempt = make_attempt()
    wrong = (
        "http://127.0.0.1:1455/auth/callback?"
        + urllib.parse.urlencode(
            {"code": "wrong", "state": "wrong", "client_id": "oaiapp_issued"}
        )
    )
    with pytest.raises(ChatGPTConfigurationError, match="state"):
        attempt.consume_callback(wrong, now=NOW)

    valid = (
        "http://127.0.0.1:1455/auth/callback?"
        + urllib.parse.urlencode(
            {
                "code": "authorization-code",
                "state": attempt.state,
                "client_id": "oaiapp_issued",
            }
        )
    )
    assert attempt.consume_callback(valid, now=NOW).code == "authorization-code"


def discovery() -> dict[str, object]:
    return {
        "issuer": OIDC_ISSUER,
        "authorization_endpoint": AUTHORIZATION_ENDPOINT,
        "token_endpoint": TOKEN_ENDPOINT,
        "jwks_uri": JWKS_URI,
        "revocation_endpoint": f"{OIDC_ISSUER}/api/accounts/oauth/revoke",
        "id_token_signing_alg_values_supported": ["none", "HS256", "RS256"],
    }


def valid_claims() -> dict[str, object]:
    return {
        "iss": OIDC_ISSUER,
        "aud": "oaiapp_issued",
        "sub": "account-subject",
        "iat": NOW.timestamp(),
        "exp": (NOW + timedelta(hours=1)).timestamp(),
        "nonce": "expected-nonce",
        "email": "person@example.test",
    }


def test_oidc_validator_pins_discovery_and_validates_identity() -> None:
    calls: list[tuple[object, ...]] = []

    def fetcher(url: str, timeout: float) -> dict[str, object]:
        calls.append((url, timeout))
        return discovery()

    def verifier(token: str, **kwargs: object) -> dict[str, object]:
        calls.append((token, kwargs))
        return valid_claims()

    validator = OidcValidator(
        json_fetcher=fetcher,
        jwt_verifier=verifier,
        now=lambda: NOW,
    )
    identity = validator.validate(
        id_token="signed-id-token",
        issued_client_id="oaiapp_issued",
        expected_nonce="expected-nonce",
    )

    assert identity == ValidatedIdentity(
        OIDC_ISSUER,
        "account-subject",
        "oaiapp_issued",
        "person@example.test",
    )
    assert calls[0][0] == OIDC_DISCOVERY_URL
    verification = calls[1][1]
    assert isinstance(verification, dict)
    assert verification["algorithms"] == ("RS256",)
    assert verification["jwks_uri"] == JWKS_URI


@pytest.mark.parametrize(
    "change",
    [
        {"nonce": "wrong"},
        {"aud": "other-client"},
        {"aud": ["oaiapp_issued", "other-client"]},
        {"aud": ["oaiapp_issued", "other-client"], "azp": "other-client"},
        {"iss": "https://attacker.invalid"},
        {"exp": (NOW - timedelta(minutes=1)).timestamp()},
        {"iat": (NOW + timedelta(minutes=1)).timestamp()},
        {"sub": ""},
    ],
)
def test_oidc_claim_validation_failures(change: dict[str, object]) -> None:
    claims = valid_claims()
    claims.update(change)
    validator = OidcValidator(
        json_fetcher=lambda *_args: discovery(),
        jwt_verifier=lambda *_args, **_kwargs: claims,
        now=lambda: NOW,
    )
    with pytest.raises(ChatGPTAuthorizationRequiredError):
        validator.validate(
            id_token="signed-id-token",
            issued_client_id="oaiapp_issued",
            expected_nonce="expected-nonce",
        )


def test_oidc_accepts_multiple_audiences_with_matching_authorized_party() -> None:
    claims = valid_claims()
    claims["aud"] = ["oaiapp_issued", "other-client"]
    claims["azp"] = "oaiapp_issued"
    validator = OidcValidator(
        json_fetcher=lambda *_args: discovery(),
        jwt_verifier=lambda *_args, **_kwargs: claims,
        now=lambda: NOW,
    )

    identity = validator.validate(
        id_token="signed-id-token",
        issued_client_id="oaiapp_issued",
        expected_nonce="expected-nonce",
    )

    assert identity.subject == "account-subject"


def test_oidc_discovery_mismatch_fails_before_jwks_validation() -> None:
    invalid = discovery()
    invalid["token_endpoint"] = "https://attacker.invalid/token"
    verifier_called = False

    def verifier(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal verifier_called
        verifier_called = True
        return valid_claims()

    validator = OidcValidator(
        json_fetcher=lambda *_args: invalid,
        jwt_verifier=verifier,
        now=lambda: NOW,
    )
    with pytest.raises(ChatGPTConfigurationError, match="pinned endpoints"):
        validator.validate(
            id_token="signed-id-token",
            issued_client_id="oaiapp_issued",
            expected_nonce="expected-nonce",
        )
    assert not verifier_called


def test_oidc_jwks_network_failure_stays_transient() -> None:
    def unavailable_verifier(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ChatGPTTransientError("JWKS unavailable")

    validator = OidcValidator(
        json_fetcher=lambda *_args: discovery(),
        jwt_verifier=unavailable_verifier,
        now=lambda: NOW,
    )
    with pytest.raises(ChatGPTTransientError, match="JWKS unavailable"):
        validator.validate(
            id_token="signed-id-token",
            issued_client_id="oaiapp_issued",
            expected_nonce="expected-nonce",
        )


def test_profile_activation_requires_direct_plan_scope() -> None:
    attempt = make_attempt()
    grant = AuthorizationGrant(
        code="authorization-code",
        client_id="oaiapp_issued",
        callback_scopes=("openid", DIRECT_USE_SCOPE),
    )
    claims = valid_claims()
    claims["nonce"] = attempt.nonce
    validator = OidcValidator(
        json_fetcher=lambda *_args: discovery(),
        jwt_verifier=lambda token, **_kwargs: (
            claims if token == "id-token" else pytest.fail("wrong ID token")
        ),
        now=lambda: NOW,
    )
    response = {
        "id_token": "id-token",
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": "openid offline_access",
    }
    with pytest.raises(ChatGPTAuthorizationRequiredError, match="not granted"):
        activate_token_response(
            response,
            attempt=attempt,
            grant=grant,
            validator=validator,
            now=NOW,
        )

    response["scope"] = f"openid offline_access {DIRECT_USE_SCOPE}"
    activated = activate_token_response(
        response,
        attempt=attempt,
        grant=grant,
        validator=validator,
        now=NOW,
    )
    assert activated.subject == "account-subject"
    assert activated.access_token == "access-token"


def test_reauthorization_rejects_different_account_subject() -> None:
    attempt = make_attempt(client_id="oaiapp_issued")
    grant = AuthorizationGrant(
        code="authorization-code",
        client_id="oaiapp_issued",
        callback_scopes=("openid", DIRECT_USE_SCOPE),
    )
    claims = valid_claims()
    claims["sub"] = "different-account"
    claims["nonce"] = attempt.nonce
    validator = OidcValidator(
        json_fetcher=lambda *_args: discovery(),
        jwt_verifier=lambda *_args, **_kwargs: claims,
        now=lambda: NOW,
    )
    response = {
        "id_token": "exact-response-id-token",
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": f"openid offline_access {DIRECT_USE_SCOPE}",
    }

    with pytest.raises(ChatGPTAuthorizationRequiredError, match="different"):
        activate_token_response(
            response,
            attempt=attempt,
            grant=grant,
            validator=validator,
            previous_profile=replace(profile(), client_id="oaiapp_issued"),
            now=NOW,
        )


@pytest.mark.parametrize(
    "previous",
    [
        replace(profile(), client_id="oaiapp_other"),
        replace(
            profile(),
            client_id="oaiapp_issued",
            ext_agent_host_id="urn:uuid:aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        ),
    ],
)
def test_reauthorization_rejects_registration_mismatch(
    previous: ChatGPTCredentialProfile,
) -> None:
    attempt = make_attempt(client_id="oaiapp_issued")
    grant = AuthorizationGrant(
        code="authorization-code",
        client_id="oaiapp_issued",
        callback_scopes=("openid", DIRECT_USE_SCOPE),
    )
    validator = OidcValidator(
        json_fetcher=lambda *_args: discovery(),
        jwt_verifier=lambda *_args, **_kwargs: valid_claims(),
        now=lambda: NOW,
    )
    with pytest.raises(ChatGPTConfigurationError, match="saved registration"):
        activate_token_response(
            {
                "id_token": "id-token",
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "scope": DIRECT_USE_SCOPE,
            },
            attempt=attempt,
            grant=grant,
            validator=validator,
            previous_profile=previous,
            now=NOW,
        )
