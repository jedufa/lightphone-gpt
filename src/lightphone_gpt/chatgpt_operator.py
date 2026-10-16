"""Root-only, one-shot ChatGPT administrative operator for a self-hosted VM.

The deployment wrapper supplies fixed protected paths. This module accepts no
credential paths, bearer tokens, account claims, prompt text, or model names on
the command line. Every result and receipt is deliberately content-free.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import secrets
import socket
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, TextIO

from .adapters.chatgpt_plan import ChatGPTPlanAssistant
from .chatgpt_auth import HostIdStore, discover_models, profile_status
from .chatgpt_errors import (
    ChatGPTAccountNotEligibleError,
    ChatGPTAmbiguousOutcomeError,
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTCredentialAmbiguousError,
    ChatGPTPlanUsageLimitError,
    ChatGPTTransientError,
)
from .chatgpt_oauth import (
    OIDC_DISCOVERY_URL,
    OIDC_ISSUER,
    ChatGPTCredentialProfile,
    CredentialStore,
    RefreshManager,
)
from .models import PromptMessage


_MAX_IMPORT_BYTES = 256 * 1024
_SOURCE_SHA = re.compile(r"[0-9a-f]{40}\Z")
_ENV_CREDENTIAL_FILE = "LIGHTPHONE_GPT_CHATGPT_OPERATOR_CREDENTIAL_FILE"
_ENV_HOST_ID_FILE = "LIGHTPHONE_GPT_CHATGPT_OPERATOR_HOST_ID_FILE"
_ENV_RECEIPT_DIR = "LIGHTPHONE_GPT_CHATGPT_OPERATOR_RECEIPT_DIR"
_ENV_SOURCE_SHA = "LIGHTPHONE_GPT_CHATGPT_OPERATOR_SOURCE_SHA"
SMOKE_MODEL = "gpt-6.1-sol"
SMOKE_EXPECTED_TEXT = "LIGHTPHONE GPT DISCOVERY OK"
SMOKE_PROMPT = (
    "Reply with exactly the following text and nothing else: "
    + SMOKE_EXPECTED_TEXT
)
_REVOCATION_ATTEMPTS = 3
_REVOCATION_BACKOFF_SECONDS = (0.25, 1.0)


class _RefreshDeferredError(ChatGPTTransientError):
    """The issuer has supplied a future lower bound for token renewal."""

    def __init__(self, retry_at: datetime) -> None:
        super().__init__("ChatGPT token renewal is not available yet")
        self.retry_at = retry_at


@dataclass(frozen=True, slots=True)
class OperatorPaths:
    credential_file: Path
    host_id_file: Path
    receipt_dir: Path
    source_sha: str

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "OperatorPaths":
        values = {
            "credential_file": environ.get(_ENV_CREDENTIAL_FILE, "").strip(),
            "host_id_file": environ.get(_ENV_HOST_ID_FILE, "").strip(),
            "receipt_dir": environ.get(_ENV_RECEIPT_DIR, "").strip(),
            "source_sha": environ.get(_ENV_SOURCE_SHA, "").strip(),
        }
        if not all(values.values()):
            raise ChatGPTConfigurationError(
                "ChatGPT operator paths are not configured"
            )
        credential_file = Path(values["credential_file"])
        host_id_file = Path(values["host_id_file"])
        receipt_dir = Path(values["receipt_dir"])
        if not all(path.is_absolute() for path in (credential_file, host_id_file, receipt_dir)):
            raise ChatGPTConfigurationError(
                "ChatGPT operator paths must be absolute"
            )
        if (
            credential_file.parent != host_id_file.parent
            or credential_file.name != "credentials.json"
            or host_id_file.name != "host-id"
        ):
            raise ChatGPTConfigurationError(
                "ChatGPT operator credential paths are invalid"
            )
        if not _SOURCE_SHA.fullmatch(values["source_sha"]):
            raise ChatGPTConfigurationError(
                "ChatGPT operator source identity is invalid"
            )
        return cls(
            credential_file=credential_file,
            host_id_file=host_id_file,
            receipt_dir=receipt_dir,
            source_sha=values["source_sha"],
        )

    def verify(self) -> None:
        _verify_private_directory(self.credential_file.parent)
        _verify_private_directory(self.receipt_dir)


def _verify_private_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise ChatGPTConfigurationError(
            "ChatGPT operator private directory is unavailable"
        ) from None
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise ChatGPTConfigurationError(
            "ChatGPT operator private directory is unsafe"
        )


def _read_import(stream: BinaryIO) -> ChatGPTCredentialProfile:
    raw = stream.read(_MAX_IMPORT_BYTES + 1)
    if not raw or len(raw) > _MAX_IMPORT_BYTES:
        raise ChatGPTConfigurationError(
            "ChatGPT credential import is empty or too large"
        )
    try:
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError
        profile = ChatGPTCredentialProfile.from_dict(payload)
        # CredentialStore emits this canonical schema. Reject unknown fields or
        # alternate encodings so the import can prove that it changes only the
        # host identifier below.
        if dict(payload) != profile.to_dict():
            raise ValueError
        return profile
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        raise ChatGPTConfigurationError(
            "ChatGPT credential import is invalid"
        ) from None


def _load_bound_profile(paths: OperatorPaths) -> ChatGPTCredentialProfile:
    profile = CredentialStore(paths.credential_file).load()
    try:
        host_id = HostIdStore(paths.host_id_file).load()
    except ChatGPTAuthorizationRequiredError:
        raise ChatGPTConfigurationError(
            "ChatGPT credentials exist without a host identifier"
        ) from None
    if not hmac.compare_digest(profile.ext_agent_host_id, host_id):
        raise ChatGPTConfigurationError(
            "ChatGPT credentials do not belong to this host"
        )
    return profile


def prepare_host(paths: OperatorPaths) -> dict[str, object]:
    store = CredentialStore(paths.credential_file)
    try:
        profile = store.load()
    except ChatGPTAuthorizationRequiredError:
        HostIdStore(paths.host_id_file).load_or_create()
        return {
            "ok": True,
            "host_prepared": True,
            "authorized": False,
        }
    try:
        host_id = HostIdStore(paths.host_id_file).load()
    except ChatGPTAuthorizationRequiredError:
        raise ChatGPTConfigurationError(
            "ChatGPT credentials exist without a host identifier"
        ) from None
    if not hmac.compare_digest(profile.ext_agent_host_id, host_id):
        raise ChatGPTConfigurationError(
            "ChatGPT credentials do not belong to this host"
        )
    return {"host_prepared": True, **profile_status(store)}


def import_stdin(paths: OperatorPaths, stream: BinaryIO) -> dict[str, object]:
    host_id = HostIdStore(paths.host_id_file).load()
    store = CredentialStore(paths.credential_file)
    try:
        store.load()
    except ChatGPTAuthorizationRequiredError:
        pass
    else:
        raise ChatGPTConfigurationError(
            "ChatGPT authorization is already configured"
        )
    source = _read_import(stream)
    if hmac.compare_digest(source.ext_agent_host_id, host_id):
        raise ChatGPTConfigurationError(
            "ChatGPT credential import must originate from another host"
        )
    imported = replace(source, ext_agent_host_id=host_id)
    expected = source.to_dict()
    expected["ext_agent_host_id"] = host_id
    if imported.to_dict() != expected:  # pragma: no cover - defensive invariant
        raise ChatGPTConfigurationError(
            "ChatGPT credential import changed unexpected fields"
        )
    store.create(imported)
    return {"imported": True, "host_prepared": True, **profile_status(store)}


def status(paths: OperatorPaths) -> dict[str, object]:
    try:
        host_id = HostIdStore(paths.host_id_file).load()
    except ChatGPTAuthorizationRequiredError:
        try:
            CredentialStore(paths.credential_file).load()
        except ChatGPTAuthorizationRequiredError:
            return {
                "ok": True,
                "host_prepared": False,
                "authorized": False,
            }
        raise ChatGPTConfigurationError(
            "ChatGPT credentials exist without a host identifier"
        ) from None
    store = CredentialStore(paths.credential_file)
    try:
        profile = store.load()
    except ChatGPTAuthorizationRequiredError:
        return {"ok": True, "host_prepared": True, "authorized": False}
    if not hmac.compare_digest(profile.ext_agent_host_id, host_id):
        raise ChatGPTConfigurationError(
            "ChatGPT credentials do not belong to this host"
        )
    return {"host_prepared": True, **profile_status(store)}


def _fetch_oidc_configuration(
    url: str, timeout_seconds: float
) -> Mapping[str, object]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(_MAX_IMPORT_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, socket.timeout):
        raise ChatGPTTransientError(
            "OpenAI identity metadata is temporarily unavailable"
        ) from None
    if len(raw) > _MAX_IMPORT_BYTES:
        raise ChatGPTConfigurationError(
            "OpenAI identity metadata is too large"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ChatGPTConfigurationError(
            "OpenAI identity metadata is invalid"
        ) from None
    if not isinstance(payload, Mapping):
        raise ChatGPTConfigurationError(
            "OpenAI identity metadata is invalid"
        )
    return payload


def _post_revocation(
    url: str, form: Mapping[str, str], timeout_seconds: float
) -> None:
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
            status_code = getattr(response, "status", response.getcode())
            raw = response.read(_MAX_IMPORT_BYTES + 1)
    except urllib.error.HTTPError as error:
        if 500 <= error.code <= 599:
            raise ChatGPTTransientError(
                "ChatGPT session revocation is temporarily unavailable"
            ) from None
        raise ChatGPTConfigurationError(
            "ChatGPT session revocation was rejected"
        ) from None
    except (urllib.error.URLError, TimeoutError, socket.timeout):
        raise ChatGPTTransientError(
            "ChatGPT session revocation is temporarily unavailable"
        ) from None
    if status_code != 200 or raw:
        raise ChatGPTConfigurationError(
            "ChatGPT session revocation returned an invalid response"
        )


def revoke_refresh_session(
    profile: ChatGPTCredentialProfile,
    *,
    discovery_fetcher: Callable[[str, float], Mapping[str, object]] = (
        _fetch_oidc_configuration
    ),
    requester: Callable[[str, Mapping[str, str], float], None] = _post_revocation,
    timeout_seconds: float = 15.0,
) -> None:
    """Revoke one renewable session using OpenAI OIDC discovery metadata."""

    discovery = discovery_fetcher(OIDC_DISCOVERY_URL, timeout_seconds)
    endpoint = discovery.get("revocation_endpoint")
    parsed = urllib.parse.urlsplit(endpoint if isinstance(endpoint, str) else "")
    if (
        discovery.get("issuer") != OIDC_ISSUER
        or parsed.scheme != "https"
        or parsed.hostname != "auth.openai.com"
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ChatGPTConfigurationError(
            "OpenAI identity metadata has an invalid revocation endpoint"
        )
    requester(
        endpoint,
        {
            "token": profile.refresh_token,
            "token_type_hint": "refresh_token",
            "client_id": profile.client_id,
        },
        timeout_seconds,
    )


def sign_out(
    paths: OperatorPaths,
    *,
    revoker: Callable[[ChatGPTCredentialProfile], None] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Attempt remote revocation, then always clear the local token record."""

    store = CredentialStore(paths.credential_file)
    with store.renewal_lock():
        try:
            profile = store.load()
        except ChatGPTAuthorizationRequiredError:
            try:
                HostIdStore(paths.host_id_file).load()
                host_id_retained = True
            except ChatGPTAuthorizationRequiredError:
                host_id_retained = False
            return {
                "ok": True,
                "signed_out": True,
                "credential_record_deleted": False,
                "local_tokens_cleared": True,
                "remote_revocation": "not_attempted",
                "remote_revocation_attempts": 0,
                "host_id_retained": host_id_retained,
                "account_client_mapping_retained": False,
            }

        try:
            host_id = HostIdStore(paths.host_id_file).load()
        except ChatGPTAuthorizationRequiredError:
            raise ChatGPTConfigurationError(
                "ChatGPT credentials exist without a host identifier"
            ) from None
        if not hmac.compare_digest(profile.ext_agent_host_id, host_id):
            raise ChatGPTConfigurationError(
                "ChatGPT credentials do not belong to this host"
            )

        operation = revoker or revoke_refresh_session
        remote_confirmed = False
        attempts = 0
        for attempt in range(_REVOCATION_ATTEMPTS):
            attempts += 1
            try:
                operation(profile)
            except ChatGPTTransientError:
                if attempt < _REVOCATION_ATTEMPTS - 1:
                    sleeper(_REVOCATION_BACKOFF_SECONDS[attempt])
                    continue
                break
            except ChatGPTConfigurationError:
                break
            except Exception:
                # Sign-out remains locally effective even when a revocation
                # implementation fails in an unclassified way. No exception
                # detail can leak token-bearing request state into receipts.
                break
            else:
                remote_confirmed = True
                break

        deleted = store._delete_under_renewal_lock()
        if not deleted:  # pragma: no cover - protected by the held renewal lock
            raise ChatGPTAmbiguousOutcomeError(
                "ChatGPT local sign-out outcome requires review"
            )
        return {
            "ok": True,
            "signed_out": True,
            "credential_record_deleted": True,
            "local_tokens_cleared": True,
            "remote_revocation": (
                "confirmed" if remote_confirmed else "unconfirmed"
            ),
            "remote_revocation_attempts": attempts,
            "host_id_retained": True,
            "account_client_mapping_retained": False,
        }


def models(
    paths: OperatorPaths,
    *,
    discoverer: Callable[..., dict[str, object]] = discover_models,
) -> dict[str, object]:
    _load_bound_profile(paths)
    store = CredentialStore(paths.credential_file)
    discovered = discoverer(store, timeout_seconds=15.0)
    binding = profile_status(store)["account_binding"]
    return {**discovered, "account_binding": binding}


def _safe_time(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ChatGPTConfigurationError(
            "ChatGPT credential timestamp is invalid"
        )
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def refresh(
    paths: OperatorPaths,
    *,
    manager_factory: Callable[..., RefreshManager] | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    before = _load_bound_profile(paths)
    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ChatGPTConfigurationError(
            "ChatGPT operator refresh time is invalid"
        )
    current = current.astimezone(UTC)
    if (
        before.earliest_refresh_at is not None
        and current < before.earliest_refresh_at
    ):
        raise _RefreshDeferredError(before.earliest_refresh_at)
    store = CredentialStore(paths.credential_file)
    binding_before = profile_status(store)["account_binding"]
    factory = manager_factory or RefreshManager
    manager = factory(store, timeout_seconds=15.0)
    manager.get_access_token(force_refresh=True)
    after = _load_bound_profile(paths)
    binding_after = profile_status(store)["account_binding"]

    if not hmac.compare_digest(str(binding_before), str(binding_after)):
        raise ChatGPTConfigurationError(
            "ChatGPT refresh changed the authorized account"
        )
    stable_fields = (
        "issuer",
        "subject",
        "client_id",
        "ext_agent_host_id",
        "email",
    )
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise ChatGPTConfigurationError(
            "ChatGPT refresh changed protected account metadata"
        )
    if after == before:
        raise ChatGPTTransientError(
            "ChatGPT refresh completed without updating the credential record"
        )

    return {
        "ok": True,
        "account_binding": binding_after,
        "refresh_completed": True,
        "credential_record_changed": True,
        "access_token_changed": not hmac.compare_digest(
            before.access_token, after.access_token
        ),
        "refresh_token_rotated": not hmac.compare_digest(
            before.refresh_token, after.refresh_token
        ),
        "id_token_changed": not hmac.compare_digest(
            before.id_token, after.id_token
        ),
        "saved_at_advanced": after.saved_at > before.saved_at,
        "access_expiry_advanced": (
            after.access_expires_at > before.access_expires_at
        ),
        "saved_at": _safe_time(after.saved_at),
        "access_expires_at": _safe_time(after.access_expires_at),
    }


def smoke(
    paths: OperatorPaths,
    *,
    assistant_factory: Callable[..., ChatGPTPlanAssistant] = ChatGPTPlanAssistant,
) -> dict[str, object]:
    _load_bound_profile(paths)
    store = CredentialStore(paths.credential_file)
    assistant = assistant_factory(
        tokens=RefreshManager(store, timeout_seconds=15.0),
        model=SMOKE_MODEL,
        timeout_seconds=60.0,
        max_stream_chars=2_048,
    )
    response = assistant.respond([PromptMessage("user", SMOKE_PROMPT)])
    if not response.response_id:
        raise ChatGPTTransientError(
            "ChatGPT smoke response did not include an identity"
        )
    if not hmac.compare_digest(response.text.strip(), SMOKE_EXPECTED_TEXT):
        raise ChatGPTTransientError(
            "ChatGPT smoke response did not match the fixed expectation"
        )
    return {
        "ok": True,
        "account_binding": profile_status(store)["account_binding"],
        "model": SMOKE_MODEL,
        "response_completed": True,
        "response_id_present": True,
        "output_characters": len(response.text),
        "expected_text_match": True,
    }


class ReceiptStore:
    def __init__(self, directory: Path, *, source_sha: str) -> None:
        self.directory = directory
        self.source_sha = source_sha

    def write(
        self,
        command: str,
        *,
        ok: bool,
        result: Mapping[str, object],
        now: datetime | None = None,
    ) -> str:
        _verify_private_directory(self.directory)
        timestamp = now or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ChatGPTConfigurationError(
                "ChatGPT operator receipt timestamp is invalid"
            )
        receipt_id = (
            timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ-")
            + secrets.token_hex(8)
        )
        payload = {
            "schema_version": 1,
            "receipt_id": receipt_id,
            "recorded_at": timestamp.astimezone(UTC).isoformat().replace(
                "+00:00", "Z"
            ),
            "source_sha": self.source_sha,
            "command": command,
            "ok": ok,
            "result": dict(result),
        }
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        descriptor = -1
        temporary = ""
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=".receipt.", dir=self.directory
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            destination = self.directory / f"{receipt_id}.json"
            os.link(temporary, destination, follow_symlinks=False)
            try:
                os.unlink(temporary)
                temporary = ""
            except OSError:
                # The committed receipt is authoritative. A private temporary
                # hard link may remain, but must not turn a successful command
                # into a contradictory failure outcome.
                pass
            try:
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Linking the complete file is the receipt commit point.
                pass
            return receipt_id
        except OSError:
            raise ChatGPTConfigurationError(
                "ChatGPT operator receipt could not be stored"
            ) from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass


def _error_category(error: BaseException) -> tuple[str, int]:
    if isinstance(error, ChatGPTAuthorizationRequiredError):
        return "authorization_required", 1
    if isinstance(error, ChatGPTAccountNotEligibleError):
        return "account_not_eligible", 1
    if isinstance(error, ChatGPTPlanUsageLimitError):
        return "plan_usage_limit", 1
    if isinstance(error, ChatGPTAmbiguousOutcomeError):
        return "ambiguous_outcome", 1
    if isinstance(error, ChatGPTCredentialAmbiguousError):
        return "credential_ambiguous", 1
    if isinstance(error, _RefreshDeferredError):
        return "refresh_deferred", 1
    if isinstance(error, ChatGPTTransientError):
        return "temporarily_unavailable", 1
    if isinstance(error, ChatGPTConfigurationError):
        return "configuration_error", 2
    return "internal_error", 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Root-only Lightphone GPT ChatGPT administrative operator."
    )
    parser.add_argument(
        "command",
        choices=(
            "prepare-host",
            "import-stdin",
            "status",
            "models",
            "refresh",
            "smoke",
            "sign-out",
        ),
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    stdin: BinaryIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    root_checker: Callable[[], bool] | None = None,
) -> int:
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    root_checker = root_checker or (lambda: os.geteuid() == 0)
    if not root_checker():
        print(
            json.dumps({"ok": False, "error": "root_required"}, sort_keys=True),
            file=stderr,
        )
        return 77
    args = parse_args(argv)
    command = str(args.command)
    paths: OperatorPaths | None = None
    try:
        paths = OperatorPaths.from_environ(
            os.environ if environ is None else environ
        )
        paths.verify()
        if command == "prepare-host":
            result = prepare_host(paths)
        elif command == "import-stdin":
            input_stream = stdin or sys.stdin.buffer
            result = import_stdin(paths, input_stream)
        elif command == "status":
            result = status(paths)
        elif command == "models":
            result = models(paths)
        elif command == "refresh":
            result = refresh(paths)
        elif command == "smoke":
            result = smoke(paths)
        elif command == "sign-out":
            result = sign_out(paths)
        else:  # pragma: no cover - argparse enforces the whitelist
            raise ChatGPTConfigurationError("Unknown ChatGPT operator command")
    except Exception as error:
        category, exit_code = _error_category(error)
        failure_result: dict[str, object] = {"error": category}
        if isinstance(error, _RefreshDeferredError):
            failure_result["retry_at"] = _safe_time(error.retry_at)
        failure: dict[str, object] = {"ok": False, **failure_result}
        if paths is not None:
            try:
                receipt_id = ReceiptStore(
                    paths.receipt_dir, source_sha=paths.source_sha
                ).write(command, ok=False, result=failure_result)
                failure["receipt_id"] = receipt_id
            except Exception:
                pass
        print(json.dumps(failure, sort_keys=True), file=stderr)
        return exit_code

    try:
        receipt_id = ReceiptStore(
            paths.receipt_dir, source_sha=paths.source_sha
        ).write(command, ok=True, result=result)
    except Exception:
        print(
            json.dumps(
                {"ok": False, "error": "ambiguous_outcome"}, sort_keys=True
            ),
            file=stderr,
        )
        return 1

    try:
        print(
            json.dumps({**result, "receipt_id": receipt_id}, sort_keys=True),
            file=stdout,
        )
    except OSError:
        try:
            print(
                json.dumps(
                    {"ok": False, "error": "result_delivery_failed"},
                    sort_keys=True,
                ),
                file=stderr,
            )
        except OSError:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
