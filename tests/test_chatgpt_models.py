from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO, StringIO
import json
import urllib.error

import pytest

from lightphone_gpt import chatgpt_auth, cli
from lightphone_gpt.chatgpt_auth import discover_models
from lightphone_gpt.chatgpt_errors import (
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTTransientError,
)
from lightphone_gpt.chatgpt_oauth import (
    DIRECT_USE_SCOPE,
    OIDC_ISSUER,
    ChatGPTCredentialProfile,
    CredentialStore,
)


def saved_store(tmp_path) -> CredentialStore:
    store = CredentialStore(tmp_path / "auth" / "credentials.json")
    store.save(
        ChatGPTCredentialProfile(
            issuer=OIDC_ISSUER,
            subject="private-subject",
            client_id="private-client-id",
            ext_agent_host_id="urn:uuid:11111111-2222-4333-8444-555555555555",
            id_token="private-id-token",
            access_token="private-access-token",
            refresh_token="private-refresh-token",
            scopes=("openid", "offline_access", DIRECT_USE_SCOPE),
            expires_in=3600,
            saved_at=datetime.now(UTC),
            email="private@example.test",
        )
    )
    return store


def test_discovery_filters_to_listed_models_and_preserves_server_order(
    tmp_path,
) -> None:
    store = saved_store(tmp_path)
    calls: list[tuple[str, float]] = []

    def request(token: str, timeout: float):
        calls.append((token, timeout))
        return {
            "models": [
                {
                    "slug": "first-model",
                    "display_name": "First Model",
                    "visibility": "list",
                    "private_claim": "must-not-escape",
                },
                {
                    "slug": "hidden-model",
                    "display_name": "Hidden Model",
                    "visibility": "hide",
                },
                {"slug": "", "display_name": "No slug", "visibility": "list"},
                {
                    "slug": "second-model",
                    "display_name": "Second Model",
                    "visibility": "list",
                },
                "malformed",
            ],
            "account": "private-account-claim",
        }

    result = discover_models(store, timeout_seconds=7.5, requester=request)

    assert calls == [("private-access-token", 7.5)]
    assert result == {
        "ok": True,
        "models": [
            {"slug": "first-model", "display_name": "First Model"},
            {"slug": "second-model", "display_name": "Second Model"},
        ],
    }
    serialized = json.dumps(result)
    for private in (
        "private-access-token",
        "private-client-id",
        "private-subject",
        "private-account-claim",
        "must-not-escape",
    ):
        assert private not in serialized


def test_discovery_refreshes_once_after_unauthorized_response(
    tmp_path, monkeypatch
) -> None:
    store = saved_store(tmp_path)

    class FakeTokens:
        def __init__(self, observed_store, *, timeout_seconds: float) -> None:
            assert observed_store is store
            assert timeout_seconds == 9.0
            self.calls: list[bool] = []
            instances.append(self)

        def get_access_token(self, *, force_refresh: bool = False) -> str:
            self.calls.append(force_refresh)
            return "refreshed-token" if force_refresh else "stale-token"

    instances: list[FakeTokens] = []
    monkeypatch.setattr(chatgpt_auth, "RefreshManager", FakeTokens)
    observed_tokens: list[str] = []

    def request(token: str, _timeout: float):
        observed_tokens.append(token)
        if token == "stale-token":
            raise chatgpt_auth._ModelsEndpointError(401)
        return {"models": []}

    assert discover_models(store, timeout_seconds=9.0, requester=request) == {
        "ok": True,
        "models": [],
    }
    assert observed_tokens == ["stale-token", "refreshed-token"]
    assert instances[0].calls == [False, True]


def test_second_unauthorized_response_requires_new_authorization(
    tmp_path, monkeypatch
) -> None:
    store = saved_store(tmp_path)

    class FakeTokens:
        def __init__(self, *_args, **_kwargs) -> None:
            return None

        def get_access_token(self, *, force_refresh: bool = False) -> str:
            return "private-refreshed-token" if force_refresh else "private-stale-token"

    monkeypatch.setattr(chatgpt_auth, "RefreshManager", FakeTokens)

    with pytest.raises(ChatGPTAuthorizationRequiredError) as raised:
        discover_models(
            store,
            requester=lambda *_args: (_ for _ in ()).throw(
                chatgpt_auth._ModelsEndpointError(401)
            ),
        )
    assert "private" not in str(raised.value)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (403, ChatGPTConfigurationError),
        (429, ChatGPTTransientError),
        (500, ChatGPTTransientError),
    ],
)
def test_discovery_maps_http_status_without_upstream_content(
    tmp_path, status_code: int, expected: type[Exception]
) -> None:
    with pytest.raises(expected) as raised:
        discover_models(
            saved_store(tmp_path),
            requester=lambda *_args: (_ for _ in ()).throw(
                chatgpt_auth._ModelsEndpointError(status_code)
            ),
        )
    assert "private" not in str(raised.value)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0.0, 60.1])
def test_discovery_rejects_invalid_timeout_before_reading_credentials(
    tmp_path, timeout: float
) -> None:
    store = CredentialStore(tmp_path / "missing" / "credentials.json")
    with pytest.raises(ChatGPTConfigurationError, match="timeout"):
        discover_models(store, timeout_seconds=timeout)


@pytest.mark.parametrize(
    "payload",
    [[], {}, {"data": []}, {"models": "not-a-list"}],
)
def test_discovery_rejects_invalid_catalog_schema(tmp_path, payload) -> None:
    with pytest.raises(ChatGPTTransientError, match="invalid response"):
        discover_models(saved_store(tmp_path), requester=lambda *_args: payload)


def test_discovery_sanitizes_unexpected_requester_failure(tmp_path) -> None:
    secret = "private-upstream-error-and-token"
    with pytest.raises(ChatGPTTransientError) as raised:
        discover_models(
            saved_store(tmp_path),
            requester=lambda *_args: (_ for _ in ()).throw(RuntimeError(secret)),
        )
    assert secret not in str(raised.value)


class FakeHttpResponse:
    def __init__(self, body: bytes) -> None:
        self._body = BytesIO(body)
        self.read_sizes: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self._body.read(size)


def test_http_request_is_get_bearer_and_reads_one_bounded_body(monkeypatch) -> None:
    response = FakeHttpResponse(b'{"models":[]}')
    observed: dict[str, object] = {}

    def open_request(request, timeout: float):
        observed["request"] = request
        observed["timeout"] = timeout
        return response

    monkeypatch.setattr(chatgpt_auth, "_open_models_request", open_request)
    assert chatgpt_auth._get_models_json("private-bearer", 11.0) == {"models": []}

    request = observed["request"]
    assert request.full_url == "https://api.openai.com/v1/models"
    assert request.get_method() == "GET"
    assert request.get_header("Authorization") == "Bearer private-bearer"
    assert request.get_header("Accept") == "application/json"
    assert observed["timeout"] == 11.0
    assert response.read_sizes == [chatgpt_auth._MAX_MODELS_RESPONSE_BYTES + 1]


def test_http_request_rejects_oversized_body_without_echoing_it(monkeypatch) -> None:
    monkeypatch.setattr(chatgpt_auth, "_MAX_MODELS_RESPONSE_BYTES", 8)
    response = FakeHttpResponse(b"private-response-body")
    monkeypatch.setattr(
        chatgpt_auth,
        "_open_models_request",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(ChatGPTTransientError) as raised:
        chatgpt_auth._get_models_json("private-token", 2.0)
    assert "private" not in str(raised.value)
    assert response.read_sizes == [9]


def test_http_error_exposes_only_status(monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "https://private.example/private-token",
        403,
        "private-reason",
        {},
        BytesIO(b"private-response"),
    )
    monkeypatch.setattr(
        chatgpt_auth,
        "_open_models_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(chatgpt_auth._ModelsEndpointError) as raised:
        chatgpt_auth._get_models_json("private-token", 2.0)
    assert raised.value.status_code == 403
    assert "private" not in str(raised.value)


def test_model_catalog_redirects_are_not_followed() -> None:
    handler = chatgpt_auth._NoModelsRedirects()
    result = handler.redirect_request(
        object(), object(), 302, "moved", {}, "https://elsewhere"
    )
    assert result is None


def test_cli_models_dispatch_is_isolated_from_service_runtime(
    tmp_path, monkeypatch
) -> None:
    observed: dict[str, object] = {}

    def run(args, environ):
        observed["command"] = args.chatgpt_auth_command
        observed["credential_file"] = args.credential_file
        observed["timeout_seconds"] = args.timeout_seconds
        observed["environment"] = environ
        return {"ok": True, "models": []}

    monkeypatch.setattr(cli, "run_chatgpt_auth_command", run)
    monkeypatch.setattr(
        cli,
        "build_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("service runtime must not be built")
        ),
    )
    output = StringIO()
    error = StringIO()
    credential = tmp_path / "credentials.json"
    code = cli.main(
        [
            "chatgpt-auth",
            "models",
            "--credential-file",
            str(credential),
            "--timeout-seconds",
            "8",
        ],
        environ={"LIGHTPHONE_GPT_PROVIDER": "forbidden-provider"},
        stdout=output,
        stderr=error,
    )

    assert code == 0 and error.getvalue() == ""
    assert json.loads(output.getvalue()) == {"models": [], "ok": True}
    assert observed == {
        "command": "models",
        "credential_file": credential,
        "timeout_seconds": 8.0,
        "environment": {"LIGHTPHONE_GPT_PROVIDER": "forbidden-provider"},
    }
