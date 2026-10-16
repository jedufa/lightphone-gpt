from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from io import BytesIO, StringIO
import json
import os
from pathlib import Path

import pytest

import lightphone_gpt.chatgpt_operator as operator
import lightphone_gpt.chatgpt_oauth as oauth
from lightphone_gpt.chatgpt_errors import (
    ChatGPTConfigurationError,
    ChatGPTCredentialAmbiguousError,
    ChatGPTTransientError,
)
from lightphone_gpt.chatgpt_oauth import (
    DIRECT_USE_SCOPE,
    OIDC_ISSUER,
    ChatGPTCredentialProfile,
    CredentialStore,
)
from lightphone_gpt.models import AssistantResponse


SOURCE_SHA = "a" * 40
LOCAL_HOST = "urn:uuid:11111111-2222-4333-8444-555555555555"
SERVER_HOST = "urn:uuid:aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


def profile(*, host_id: str = LOCAL_HOST) -> ChatGPTCredentialProfile:
    return ChatGPTCredentialProfile(
        issuer=OIDC_ISSUER,
        subject="private-subject",
        client_id="private-client-id",
        ext_agent_host_id=host_id,
        id_token="private-id-token",
        access_token="private-access-token",
        refresh_token="private-refresh-token",
        scopes=("openid", "offline_access", DIRECT_USE_SCOPE),
        expires_in=3600,
        saved_at=datetime(2026, 10, 15, 12, 0, tzinfo=UTC),
        email="private@example.test",
    )


def operator_paths(tmp_path: Path) -> operator.OperatorPaths:
    secrets = tmp_path / "secrets"
    receipts = tmp_path / "receipts"
    secrets.mkdir(mode=0o700)
    receipts.mkdir(mode=0o700)
    return operator.OperatorPaths(
        credential_file=secrets / "credentials.json",
        host_id_file=secrets / "host-id",
        receipt_dir=receipts,
        source_sha=SOURCE_SHA,
    )


def environ(paths: operator.OperatorPaths) -> dict[str, str]:
    return {
        "LIGHTPHONE_GPT_CHATGPT_OPERATOR_CREDENTIAL_FILE": str(
            paths.credential_file
        ),
        "LIGHTPHONE_GPT_CHATGPT_OPERATOR_HOST_ID_FILE": str(paths.host_id_file),
        "LIGHTPHONE_GPT_CHATGPT_OPERATOR_RECEIPT_DIR": str(paths.receipt_dir),
        "LIGHTPHONE_GPT_CHATGPT_OPERATOR_SOURCE_SHA": paths.source_sha,
    }


def invoke(
    argv: list[str],
    paths: operator.OperatorPaths,
    *,
    stdin: bytes = b"",
) -> tuple[int, dict[str, object] | None, dict[str, object] | None]:
    stdout = StringIO()
    stderr = StringIO()
    code = operator.main(
        argv,
        environ=environ(paths),
        stdin=BytesIO(stdin),
        stdout=stdout,
        stderr=stderr,
        root_checker=lambda: True,
    )
    return (
        code,
        json.loads(stdout.getvalue()) if stdout.getvalue() else None,
        json.loads(stderr.getvalue()) if stderr.getvalue() else None,
    )


def test_prepare_host_is_stable_private_and_content_free(tmp_path: Path) -> None:
    paths = operator_paths(tmp_path)

    first = operator.prepare_host(paths)
    saved = paths.host_id_file.read_text(encoding="ascii")
    second = operator.prepare_host(paths)

    assert first == second == {
        "ok": True,
        "host_prepared": True,
        "authorized": False,
    }
    assert saved.startswith("urn:uuid:")
    assert paths.host_id_file.read_text(encoding="ascii") == saved
    assert paths.host_id_file.stat().st_mode & 0o777 == 0o600
    assert saved not in json.dumps(first)


def test_import_rewrites_only_host_and_refuses_overwrite(tmp_path: Path) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    source = profile()
    raw = json.dumps(source.to_dict()).encode("utf-8")

    result = operator.import_stdin(paths, BytesIO(raw))
    imported = CredentialStore(paths.credential_file).load()

    assert result["imported"] is True
    assert result["authorized"] is True
    assert imported.ext_agent_host_id == SERVER_HOST
    assert replace(imported, ext_agent_host_id=LOCAL_HOST) == source
    assert paths.credential_file.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ChatGPTConfigurationError, match="already configured"):
        operator.import_stdin(paths, BytesIO(raw))
    assert CredentialStore(paths.credential_file).load() == imported


def test_import_refuses_same_host_profile(tmp_path: Path) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    same_host = profile(host_id=SERVER_HOST)

    with pytest.raises(ChatGPTConfigurationError, match="another host"):
        operator.import_stdin(
            paths,
            BytesIO(json.dumps(same_host.to_dict()).encode("utf-8")),
        )

    assert not paths.credential_file.exists()


def test_import_rejects_oversized_or_invalid_input_before_creating_file(
    tmp_path: Path,
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)

    for raw in (b"{", b"x" * (operator._MAX_IMPORT_BYTES + 1)):
        with pytest.raises(ChatGPTConfigurationError):
            operator.import_stdin(paths, BytesIO(raw))
        assert not paths.credential_file.exists()


def test_import_requires_canonical_schema_and_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    payload = profile().to_dict()
    payload["unknown_private_extension"] = "must-not-be-silently-dropped"

    with pytest.raises(ChatGPTConfigurationError, match="invalid"):
        operator.import_stdin(
            paths, BytesIO(json.dumps(payload).encode("utf-8"))
        )

    assert not paths.credential_file.exists()


def test_main_import_output_and_receipt_never_contain_credentials(
    tmp_path: Path,
) -> None:
    paths = operator_paths(tmp_path)
    assert invoke(["prepare-host"], paths)[0] == 0
    private = profile()

    code, output, error = invoke(
        ["import-stdin"],
        paths,
        stdin=json.dumps(private.to_dict()).encode("utf-8"),
    )

    assert code == 0 and error is None
    assert output is not None and output["authorized"] is True
    receipts = sorted(paths.receipt_dir.glob("*.json"))
    assert len(receipts) == 2
    public_material = json.dumps(output) + "".join(
        path.read_text(encoding="utf-8") for path in receipts
    )
    for secret in (
        private.subject,
        private.client_id,
        private.id_token,
        private.access_token,
        private.refresh_token,
        private.email,
        LOCAL_HOST,
    ):
        assert secret not in public_material
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in receipts)


def test_failed_import_has_only_sanitized_error_and_receipt(tmp_path: Path) -> None:
    paths = operator_paths(tmp_path)
    assert invoke(["prepare-host"], paths)[0] == 0
    secret = "private-invalid-credential-material"

    code, output, error = invoke(
        ["import-stdin"], paths, stdin=secret.encode("utf-8")
    )

    assert code == 2 and output is None
    assert error is not None and error["error"] == "configuration_error"
    material = json.dumps(error) + "".join(
        path.read_text(encoding="utf-8")
        for path in paths.receipt_dir.glob("*.json")
    )
    assert secret not in material
    assert not paths.credential_file.exists()


def test_status_rejects_credentials_without_host_identity(tmp_path: Path) -> None:
    paths = operator_paths(tmp_path)
    CredentialStore(paths.credential_file).save(profile())

    with pytest.raises(ChatGPTConfigurationError, match="without a host"):
        operator.status(paths)


def test_prepare_host_refuses_existing_credentials_without_creating_host(
    tmp_path: Path,
) -> None:
    paths = operator_paths(tmp_path)
    CredentialStore(paths.credential_file).save(profile())

    with pytest.raises(ChatGPTConfigurationError):
        operator.prepare_host(paths)

    assert not paths.host_id_file.exists()


def test_models_uses_bound_store_and_adds_only_account_binding(
    tmp_path: Path,
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    CredentialStore(paths.credential_file).save(profile(host_id=SERVER_HOST))
    observed: dict[str, object] = {}

    def discoverer(store, *, timeout_seconds: float):
        observed["path"] = store.path
        observed["timeout"] = timeout_seconds
        return {
            "ok": True,
            "models": [{"slug": "gpt-test", "display_name": "GPT Test"}],
        }

    result = operator.models(paths, discoverer=discoverer)

    assert observed == {
        "path": paths.credential_file,
        "timeout": 15.0,
    }
    assert result["models"] == [
        {"slug": "gpt-test", "display_name": "GPT Test"}
    ]
    assert result["account_binding"]
    assert "private" not in json.dumps(result)


def test_refresh_forces_rotation_and_reports_only_safe_comparisons(
    tmp_path: Path,
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    before = profile(host_id=SERVER_HOST)
    CredentialStore(paths.credential_file).save(before)
    observed: dict[str, object] = {}

    class RotatingManager:
        def __init__(self, store, *, timeout_seconds: float) -> None:
            observed["path"] = store.path
            observed["timeout"] = timeout_seconds
            self.store = store

        def get_access_token(self, *, force_refresh: bool = False) -> str:
            observed["force_refresh"] = force_refresh
            current = self.store.load()
            self.store.save(
                replace(
                    current,
                    access_token="new-private-access-token",
                    refresh_token="new-private-refresh-token",
                    saved_at=current.saved_at + timedelta(minutes=15),
                    expires_in=7200,
                )
            )
            return "new-private-access-token"

    result = operator.refresh(paths, manager_factory=RotatingManager)

    assert observed == {
        "path": paths.credential_file,
        "timeout": 15.0,
        "force_refresh": True,
    }
    assert result["ok"] is True
    assert result["refresh_completed"] is True
    assert result["credential_record_changed"] is True
    assert result["access_token_changed"] is True
    assert result["refresh_token_rotated"] is True
    assert result["id_token_changed"] is False
    assert result["saved_at_advanced"] is True
    assert result["access_expiry_advanced"] is True
    assert result["saved_at"] == "2026-10-15T12:15:00Z"
    assert result["access_expires_at"] == "2026-10-15T14:15:00Z"
    serialized = json.dumps(result)
    for private in (
        before.subject,
        before.client_id,
        before.id_token,
        before.access_token,
        before.refresh_token,
        before.email,
        SERVER_HOST,
        "new-private-access-token",
        "new-private-refresh-token",
    ):
        assert private not in serialized


def test_refresh_without_credential_change_is_retryable_non_success(
    tmp_path: Path, monkeypatch
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    CredentialStore(paths.credential_file).save(profile(host_id=SERVER_HOST))

    class NoChangeManager:
        def __init__(self, store, *, timeout_seconds: float) -> None:
            self.store = store

        def get_access_token(self, *, force_refresh: bool = False) -> str:
            assert force_refresh is True
            return self.store.load().access_token

    monkeypatch.setattr(operator, "RefreshManager", NoChangeManager)
    code, output, error = invoke(["refresh"], paths)

    assert code == 1 and output is None
    assert error is not None and error["error"] == "temporarily_unavailable"
    receipts = list(paths.receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["command"] == "refresh"
    assert receipt["ok"] is False


def test_refresh_before_issuer_threshold_is_deferred_without_request(
    tmp_path: Path, monkeypatch
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    before = replace(
        profile(host_id=SERVER_HOST),
        earliest_refresh_at=datetime(2026, 10, 15, 12, 30, tzinfo=UTC),
    )
    CredentialStore(paths.credential_file).save(before)

    class ForbiddenManager:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("refresh manager must not be constructed")

    monkeypatch.setattr(operator, "RefreshManager", ForbiddenManager)
    real_refresh = operator.refresh
    monkeypatch.setattr(
        operator,
        "refresh",
        lambda paths: real_refresh(
            paths, now=datetime(2026, 10, 15, 12, 20, tzinfo=UTC)
        ),
    )

    code, output, error = invoke(["refresh"], paths)

    assert code == 1 and output is None
    assert error is not None
    assert error["error"] == "refresh_deferred"
    assert error["retry_at"] == "2026-10-15T12:30:00Z"
    receipts = list(paths.receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["ok"] is False
    assert receipt["result"] == {
        "error": "refresh_deferred",
        "retry_at": "2026-10-15T12:30:00Z",
    }
    material = json.dumps(error) + json.dumps(receipt)
    for private in (
        before.subject,
        before.client_id,
        before.id_token,
        before.access_token,
        before.refresh_token,
        before.email,
        SERVER_HOST,
    ):
        assert private not in material


def test_refresh_main_success_receipt_is_content_free(
    tmp_path: Path, monkeypatch
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    before = profile(host_id=SERVER_HOST)
    CredentialStore(paths.credential_file).save(before)

    class RotatingManager:
        def __init__(self, store, *, timeout_seconds: float) -> None:
            self.store = store

        def get_access_token(self, *, force_refresh: bool = False) -> str:
            assert force_refresh is True
            current = self.store.load()
            self.store.save(
                replace(
                    current,
                    access_token="receipt-private-access-token",
                    refresh_token="receipt-private-refresh-token",
                    saved_at=current.saved_at + timedelta(minutes=5),
                )
            )
            return "receipt-private-access-token"

    monkeypatch.setattr(operator, "RefreshManager", RotatingManager)
    code, output, error = invoke(["refresh"], paths)

    assert code == 0 and error is None
    assert output is not None and output["refresh_completed"] is True
    receipts = list(paths.receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    material = json.dumps(output) + receipts[0].read_text(encoding="utf-8")
    for private in (
        before.subject,
        before.client_id,
        before.id_token,
        before.access_token,
        before.refresh_token,
        before.email,
        SERVER_HOST,
        "receipt-private-access-token",
        "receipt-private-refresh-token",
    ):
        assert private not in material
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["command"] == "refresh"
    assert receipt["ok"] is True


def test_refresh_main_error_is_retryable_and_receipt_is_sanitized(
    tmp_path: Path, monkeypatch
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    before = profile(host_id=SERVER_HOST)
    CredentialStore(paths.credential_file).save(before)
    secret = "private-refresh-upstream-error"

    class FailingManager:
        def __init__(self, _store, *, timeout_seconds: float) -> None:
            return None

        def get_access_token(self, *, force_refresh: bool = False) -> str:
            assert force_refresh is True
            raise ChatGPTTransientError(secret)

    monkeypatch.setattr(operator, "RefreshManager", FailingManager)
    code, output, error = invoke(["refresh"], paths)

    assert code == 1 and output is None
    assert error is not None and error["error"] == "temporarily_unavailable"
    receipts = list(paths.receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    material = json.dumps(error) + receipts[0].read_text(encoding="utf-8")
    assert secret not in material
    for private in (
        before.subject,
        before.client_id,
        before.id_token,
        before.access_token,
        before.refresh_token,
        before.email,
        SERVER_HOST,
    ):
        assert private not in material
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["command"] == "refresh"
    assert receipt["ok"] is False


def test_refresh_credential_ambiguity_requires_review_and_receipt_is_sanitized(
    tmp_path: Path, monkeypatch
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    before = profile(host_id=SERVER_HOST)
    CredentialStore(paths.credential_file).save(before)
    secret = "private-rotated-refresh-token-detail"

    class FailingManager:
        def __init__(self, _store, *, timeout_seconds: float) -> None:
            return None

        def get_access_token(self, *, force_refresh: bool = False) -> str:
            assert force_refresh is True
            raise ChatGPTCredentialAmbiguousError(secret)

    monkeypatch.setattr(operator, "RefreshManager", FailingManager)
    code, output, error = invoke(["refresh"], paths)

    assert code == 1 and output is None
    assert error is not None
    assert error["error"] == "credential_ambiguous"
    assert error["ok"] is False
    assert isinstance(error.get("receipt_id"), str)
    receipts = list(paths.receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    material = json.dumps(error) + receipts[0].read_text(encoding="utf-8")
    assert secret not in material
    for private in (
        before.subject,
        before.client_id,
        before.id_token,
        before.access_token,
        before.refresh_token,
        before.email,
        SERVER_HOST,
    ):
        assert private not in material
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["command"] == "refresh"
    assert receipt["ok"] is False


def test_smoke_uses_one_fixed_prompt_and_does_not_persist_output(
    tmp_path: Path,
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    CredentialStore(paths.credential_file).save(profile(host_id=SERVER_HOST))
    observed: dict[str, object] = {"calls": 0}
    private_output = operator.SMOKE_EXPECTED_TEXT

    class FakeAssistant:
        def respond(self, messages):
            observed["calls"] = int(observed["calls"]) + 1
            observed["messages"] = messages
            return AssistantResponse(private_output, "private-response-id")

    def factory(**kwargs):
        observed["model"] = kwargs["model"]
        observed["timeout"] = kwargs["timeout_seconds"]
        observed["limit"] = kwargs["max_stream_chars"]
        return FakeAssistant()

    result = operator.smoke(paths, assistant_factory=factory)

    assert observed["calls"] == 1
    assert observed["messages"][0].content == operator.SMOKE_PROMPT
    assert observed["model"] == operator.SMOKE_MODEL
    assert result["response_completed"] is True
    assert result["response_id_present"] is True
    assert result["expected_text_match"] is True
    assert result["output_characters"] == len(private_output)
    assert private_output not in json.dumps(result)
    assert "private-response-id" not in json.dumps(result)


def test_smoke_mismatch_is_a_sanitized_failure(tmp_path: Path) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    CredentialStore(paths.credential_file).save(profile(host_id=SERVER_HOST))
    private_output = "private unexpected model output"

    class FakeAssistant:
        def respond(self, _messages):
            return AssistantResponse(private_output, "private-response-id")

    with pytest.raises(operator.ChatGPTTransientError) as captured:
        operator.smoke(
            paths, assistant_factory=lambda **_kwargs: FakeAssistant()
        )

    assert private_output not in str(captured.value)
    assert "private-response-id" not in str(captured.value)


def test_operator_paths_require_fixed_names_absolute_paths_and_source_sha(
    tmp_path: Path,
) -> None:
    paths = operator_paths(tmp_path)
    valid = environ(paths)
    assert operator.OperatorPaths.from_environ(valid) == paths

    for key, value in (
        ("LIGHTPHONE_GPT_CHATGPT_OPERATOR_CREDENTIAL_FILE", "relative.json"),
        ("LIGHTPHONE_GPT_CHATGPT_OPERATOR_HOST_ID_FILE", str(tmp_path / "other")),
        ("LIGHTPHONE_GPT_CHATGPT_OPERATOR_SOURCE_SHA", "not-a-sha"),
    ):
        changed = dict(valid)
        changed[key] = value
        with pytest.raises(ChatGPTConfigurationError):
            operator.OperatorPaths.from_environ(changed)


def test_main_refuses_non_root_before_reading_paths() -> None:
    stdout = StringIO()
    stderr = StringIO()

    code = operator.main(
        ["status"],
        environ={},
        stdout=stdout,
        stderr=stderr,
        root_checker=lambda: False,
    )

    assert code == 77
    assert stdout.getvalue() == ""
    assert json.loads(stderr.getvalue()) == {
        "error": "root_required",
        "ok": False,
    }


def test_revocation_uses_discovered_endpoint_and_documented_form() -> None:
    private = profile(host_id=SERVER_HOST)
    observed: dict[str, object] = {}

    def discovery_fetcher(url: str, timeout_seconds: float):
        observed["discovery"] = (url, timeout_seconds)
        return {
            "issuer": OIDC_ISSUER,
            "revocation_endpoint": (
                "https://auth.openai.com/api/accounts/oauth/revoke"
            ),
        }

    def requester(url: str, form: dict[str, str], timeout_seconds: float):
        observed["request"] = (url, dict(form), timeout_seconds)

    operator.revoke_refresh_session(
        private,
        discovery_fetcher=discovery_fetcher,
        requester=requester,
        timeout_seconds=7.0,
    )

    assert observed == {
        "discovery": (operator.OIDC_DISCOVERY_URL, 7.0),
        "request": (
            "https://auth.openai.com/api/accounts/oauth/revoke",
            {
                "token": private.refresh_token,
                "token_type_hint": "refresh_token",
                "client_id": private.client_id,
            },
            7.0,
        ),
    }


def test_sign_out_confirms_remote_revocation_and_clears_local_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    private = profile(host_id=SERVER_HOST)
    CredentialStore(paths.credential_file).save(private)
    calls: list[ChatGPTCredentialProfile] = []

    def revoker(candidate: ChatGPTCredentialProfile) -> None:
        calls.append(candidate)

    monkeypatch.setattr(operator, "revoke_refresh_session", revoker)
    code, output, error = invoke(["sign-out"], paths)

    assert code == 0 and error is None
    assert calls == [private]
    assert output is not None
    assert output["remote_revocation"] == "confirmed"
    assert output["remote_revocation_attempts"] == 1
    assert output["credential_record_deleted"] is True
    assert output["local_tokens_cleared"] is True
    assert output["host_id_retained"] is True
    assert output["account_client_mapping_retained"] is False
    assert not paths.credential_file.exists()
    assert paths.host_id_file.read_text(encoding="ascii") == SERVER_HOST
    receipts = list(paths.receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    material = json.dumps(output) + receipts[0].read_text(encoding="utf-8")
    for secret in (
        private.subject,
        private.client_id,
        private.id_token,
        private.access_token,
        private.refresh_token,
        private.email,
        SERVER_HOST,
    ):
        assert secret not in material


def test_sign_out_network_failure_is_unconfirmed_but_clears_local_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    private = profile(host_id=SERVER_HOST)
    CredentialStore(paths.credential_file).save(private)
    private_error = "private-upstream-network-detail"
    delays: list[float] = []
    calls = 0

    def revoker(_candidate: ChatGPTCredentialProfile) -> None:
        nonlocal calls
        calls += 1
        raise ChatGPTTransientError(private_error)

    real_sign_out = operator.sign_out
    monkeypatch.setattr(
        operator,
        "sign_out",
        lambda selected_paths: real_sign_out(
            selected_paths, revoker=revoker, sleeper=delays.append
        ),
    )
    code, output, error = invoke(["sign-out"], paths)

    assert code == 0 and error is None
    assert calls == 3
    assert delays == list(operator._REVOCATION_BACKOFF_SECONDS)
    assert output is not None
    assert output["remote_revocation"] == "unconfirmed"
    assert output["remote_revocation_attempts"] == 3
    assert output["credential_record_deleted"] is True
    assert output["local_tokens_cleared"] is True
    assert not paths.credential_file.exists()
    receipts = list(paths.receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    material = json.dumps(output) + receipts[0].read_text(encoding="utf-8")
    assert private_error not in material
    for secret in (
        private.subject,
        private.client_id,
        private.id_token,
        private.access_token,
        private.refresh_token,
        private.email,
        SERVER_HOST,
    ):
        assert secret not in material


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink"])
def test_sign_out_refuses_unsafe_credential_paths_without_touching_target(
    tmp_path: Path, monkeypatch, unsafe_kind: str
) -> None:
    paths = operator_paths(tmp_path)
    paths.host_id_file.write_text(SERVER_HOST, encoding="ascii")
    paths.host_id_file.chmod(0o600)
    private = profile(host_id=SERVER_HOST)
    target = tmp_path / "external-credentials.json"
    target.write_text(json.dumps(private.to_dict()), encoding="utf-8")
    target.chmod(0o600)
    if unsafe_kind == "symlink":
        paths.credential_file.symlink_to(target)
    else:
        os.link(target, paths.credential_file)
    original = target.read_bytes()
    called = False

    def forbidden_revoker(_candidate: ChatGPTCredentialProfile) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(operator, "revoke_refresh_session", forbidden_revoker)
    code, output, error = invoke(["sign-out"], paths)

    assert code == 2 and output is None
    assert error is not None and error["error"] == "configuration_error"
    assert called is False
    assert target.read_bytes() == original
    assert paths.credential_file.exists()


def test_sign_out_is_idempotent_when_credentials_are_absent(
    tmp_path: Path, monkeypatch
) -> None:
    paths = operator_paths(tmp_path)
    operator.prepare_host(paths)
    called = False

    def forbidden_revoker(_candidate: ChatGPTCredentialProfile) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(operator, "revoke_refresh_session", forbidden_revoker)
    first = invoke(["sign-out"], paths)
    second = invoke(["sign-out"], paths)

    for code, output, error in (first, second):
        assert code == 0 and error is None
        assert output is not None
        assert output["remote_revocation"] == "not_attempted"
        assert output["remote_revocation_attempts"] == 0
        assert output["credential_record_deleted"] is False
        assert output["local_tokens_cleared"] is True
        assert output["host_id_retained"] is True
    assert called is False
    assert not paths.credential_file.exists()


def test_sign_out_is_root_only_and_preserves_credentials(tmp_path: Path) -> None:
    paths = operator_paths(tmp_path)
    private = profile(host_id=SERVER_HOST)
    CredentialStore(paths.credential_file).save(private)
    stdout = StringIO()
    stderr = StringIO()

    code = operator.main(
        ["sign-out"],
        environ=environ(paths),
        stdout=stdout,
        stderr=stderr,
        root_checker=lambda: False,
    )

    assert code == 77
    assert stdout.getvalue() == ""
    assert json.loads(stderr.getvalue()) == {
        "error": "root_required",
        "ok": False,
    }
    assert CredentialStore(paths.credential_file).load() == private


def test_credential_create_is_exclusive_and_preserves_first_record(
    tmp_path: Path,
) -> None:
    store = CredentialStore(tmp_path / "credentials" / "credentials.json")
    first = profile()
    second = replace(first, access_token="second-private-access-token")

    store.create(first)
    with pytest.raises(ChatGPTConfigurationError, match="already configured"):
        store.create(second)

    assert store.load() == first
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_credential_create_publishes_only_a_complete_fsynced_record(
    tmp_path: Path, monkeypatch
) -> None:
    store = CredentialStore(tmp_path / "credentials" / "credentials.json")
    expected = profile()
    real_link = oauth.os.link
    observed: dict[str, object] = {}

    def inspect_link(source, destination, *, follow_symlinks=True):
        assert not Path(destination).exists()
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        assert ChatGPTCredentialProfile.from_dict(payload) == expected
        observed["complete"] = True
        return real_link(
            source, destination, follow_symlinks=follow_symlinks
        )

    monkeypatch.setattr(oauth.os, "link", inspect_link)
    store.create(expected)

    assert observed == {"complete": True}
    assert store.load() == expected


def test_credential_create_write_failure_never_publishes_partial_record(
    tmp_path: Path, monkeypatch
) -> None:
    store = CredentialStore(tmp_path / "credentials" / "credentials.json")
    monkeypatch.setattr(
        oauth.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("private failure")),
    )

    with pytest.raises(ChatGPTConfigurationError) as captured:
        store.create(profile())

    assert "private" not in str(captured.value)
    assert not store.path.exists()
    assert list(store.path.parent.glob(".credentials.json.import.*")) == []


def test_concurrent_credential_creates_have_exactly_one_winner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "credentials" / "credentials.json"
    first = profile()
    second = replace(first, access_token="second-private-access-token")

    def create(candidate: ChatGPTCredentialProfile) -> str:
        try:
            CredentialStore(path).create(candidate)
        except ChatGPTConfigurationError:
            return "refused"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(create, (first, second)))

    assert sorted(outcomes) == ["created", "refused"]
    assert CredentialStore(path).load() in (first, second)


def test_import_receipt_failure_reports_ambiguous_without_rollback(
    tmp_path: Path, monkeypatch
) -> None:
    paths = operator_paths(tmp_path)
    operator.prepare_host(paths)
    private = profile()

    def fail_receipt(*_args, **_kwargs):
        raise ChatGPTConfigurationError("private receipt failure")

    monkeypatch.setattr(operator.ReceiptStore, "write", fail_receipt)
    code, output, error = invoke(
        ["import-stdin"],
        paths,
        stdin=json.dumps(private.to_dict()).encode("utf-8"),
    )

    assert code == 1 and output is None
    assert error == {"error": "ambiguous_outcome", "ok": False}
    imported = CredentialStore(paths.credential_file).load()
    assert imported.access_token == private.access_token
    assert "private" not in json.dumps(error)


def test_stdout_failure_does_not_append_a_failure_receipt(tmp_path: Path) -> None:
    paths = operator_paths(tmp_path)
    operator.prepare_host(paths)

    class BrokenOutput(StringIO):
        def write(self, _value: str) -> int:
            raise BrokenPipeError

    stderr = StringIO()
    code = operator.main(
        ["status"],
        environ=environ(paths),
        stdout=BrokenOutput(),
        stderr=stderr,
        root_checker=lambda: True,
    )

    assert code == 1
    assert json.loads(stderr.getvalue()) == {
        "error": "result_delivery_failed",
        "ok": False,
    }
    receipts = list(paths.receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["command"] == "status"
    assert receipt["ok"] is True
