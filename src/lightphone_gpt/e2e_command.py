"""Protected one-shot command assembly for a real Light/ChatGPT E2E proof.

All secrets and private paths come from the root-controlled environment.  The
command accepts no prompt, credential, note identity, digest, token, or model
as an argument.  It creates ``latest-e2e.json`` durably before touching Light
Cloud or ChatGPT and replaces that file atomically with a content-free result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import errno
import fcntl
import hmac
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Protocol
import uuid

from .adapters.chatgpt_plan import ChatGPTPlanAssistant
from .adapters.light_garado import GaradoLightProvider, LightWritePolicy
from .adapters.transcriber import FasterWhisperTranscriber
from .chatgpt_auth import HostIdStore
from .chatgpt_errors import (
    ChatGPTAccountNotEligibleError,
    ChatGPTAmbiguousOutcomeError,
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTCredentialAmbiguousError,
    ChatGPTPlanUsageLimitError,
    ChatGPTTransientError,
)
from .chatgpt_oauth import CredentialStore, RefreshManager
from .contracts import Transcriber
from .e2e_operator import (
    E2EInputBinding,
    E2EProofError,
    E2EProofLimits,
    E2EProofResult,
    run_e2e_proof,
)
from .light_input_audit import verify_input_audit_model
from .models import NoteKind


E2E_CHATGPT_MODEL = "gpt-6.1-sol"
E2E_RECEIPT_NAME = "latest-e2e.json"
E2E_RECEIPT_SCHEMA_VERSION = 1

_SOURCE_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_OPERATION_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_CATEGORY_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_MAX_RECEIPT_BYTES = 32 * 1024

_ENV_SOURCE_NOTE_ID = "LIGHTPHONE_GPT_E2E_SOURCE_NOTE_ID"
_ENV_SOURCE_KIND = "LIGHTPHONE_GPT_E2E_SOURCE_KIND"
_ENV_SOURCE_DIGEST = "LIGHTPHONE_GPT_E2E_SOURCE_SHA256"
_ENV_CREDENTIAL_PATH = "LIGHTPHONE_GPT_E2E_CHATGPT_CREDENTIAL_PATH"
_ENV_HOST_ID_PATH = "LIGHTPHONE_GPT_E2E_CHATGPT_HOST_ID_PATH"
_ENV_MODEL = "LIGHTPHONE_GPT_E2E_MODEL"
_ENV_RECEIPT_DIR = "LIGHTPHONE_GPT_E2E_RECEIPT_DIR"
_ENV_OPERATOR_SOURCE = "LIGHTPHONE_GPT_OPERATOR_SOURCE_SHA"
_ENV_STT_MODEL_DIR = "LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_DIR"
_ENV_STT_MODEL_MANIFEST = "LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_MANIFEST"
_ENV_STT_TEMP_DIR = "LIGHTPHONE_GPT_INPUT_AUDIT_TEMP_DIR"


class E2ECommandError(RuntimeError):
    """Stable, content-free command failure."""

    def __init__(
        self,
        category: str,
        *,
        operation_id: str | None = None,
        mutation_attempted: bool = False,
    ) -> None:
        super().__init__(category)
        self.category = category
        self.operation_id = operation_id
        self.mutation_attempted = mutation_attempted

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "ok": False,
            "status": "failed_closed",
            "category": self.category,
            "mutation_attempted": self.mutation_attempted,
        }
        if self.operation_id is not None:
            result["operation_id"] = self.operation_id
        return result


def _private_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise E2ECommandError("configuration_error") from None
    if (
        not path.is_absolute()
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_nlink < 1
        or resolved != path
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise E2ECommandError("configuration_error")


@dataclass(frozen=True, slots=True)
class E2ECommandConfig:
    username: str
    password: str
    device_id: str
    binding: E2EInputBinding
    credential_path: Path
    host_id_path: Path
    receipt_dir: Path
    operator_source_sha: str
    stt_model_dir: Path
    stt_model_manifest: Path
    stt_temp_dir: Path
    chatgpt_model: str = E2E_CHATGPT_MODEL

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "E2ECommandConfig":
        username = environ.get("lightphone_username", "").strip()
        password = environ.get("lightphone_password", "")
        device_id = environ.get("lightphone_device_id", "").strip()
        source_note_id = environ.get(_ENV_SOURCE_NOTE_ID, "").strip()
        source_kind_raw = environ.get(_ENV_SOURCE_KIND, "").strip().lower()
        source_digest = environ.get(_ENV_SOURCE_DIGEST, "").strip()
        credential_raw = environ.get(_ENV_CREDENTIAL_PATH, "").strip()
        host_id_raw = environ.get(_ENV_HOST_ID_PATH, "").strip()
        receipt_raw = environ.get(_ENV_RECEIPT_DIR, "").strip()
        operator_source = environ.get(_ENV_OPERATOR_SOURCE, "").strip()
        stt_model_raw = environ.get(_ENV_STT_MODEL_DIR, "").strip()
        stt_manifest_raw = environ.get(_ENV_STT_MODEL_MANIFEST, "").strip()
        stt_temp_raw = environ.get(_ENV_STT_TEMP_DIR, "").strip()
        model = environ.get(_ENV_MODEL, "").strip()
        if not all(
            (
                username,
                password,
                device_id,
                source_note_id,
                source_kind_raw,
                source_digest,
                credential_raw,
                host_id_raw,
                receipt_raw,
                operator_source,
                stt_model_raw,
                stt_manifest_raw,
                stt_temp_raw,
                model,
            )
        ):
            raise E2ECommandError("configuration_error")
        if model != E2E_CHATGPT_MODEL or _SOURCE_REVISION_RE.fullmatch(
            operator_source
        ) is None:
            raise E2ECommandError("configuration_error")
        try:
            source_kind = NoteKind(source_kind_raw)
            source_binding = E2EInputBinding(
                source_note_id, source_kind, source_digest
            )
        except ValueError:
            raise E2ECommandError("configuration_error") from None

        credential_path = Path(credential_raw)
        host_id_path = Path(host_id_raw)
        receipt_dir = Path(receipt_raw)
        stt_model_dir = Path(stt_model_raw)
        stt_model_manifest = Path(stt_manifest_raw)
        stt_temp_dir = Path(stt_temp_raw)
        paths = (
            credential_path,
            host_id_path,
            receipt_dir,
            stt_model_dir,
            stt_model_manifest,
            stt_temp_dir,
        )
        if not all(path.is_absolute() for path in paths):
            raise E2ECommandError("configuration_error")
        if (
            credential_path.parent != host_id_path.parent
            or credential_path.name != "credentials.json"
            or host_id_path.name != "host-id"
        ):
            raise E2ECommandError("configuration_error")
        for directory in (
            credential_path.parent,
            receipt_dir,
            stt_model_dir,
            stt_temp_dir,
        ):
            _private_directory(directory)
        return cls(
            username=username,
            password=password,
            device_id=device_id,
            binding=source_binding,
            credential_path=credential_path,
            host_id_path=host_id_path,
            receipt_dir=receipt_dir,
            operator_source_sha=operator_source,
            stt_model_dir=stt_model_dir,
            stt_model_manifest=stt_model_manifest,
            stt_temp_dir=stt_temp_dir,
            chatgpt_model=model,
        )


class _RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._descriptor: int | None = None

    def __enter__(self) -> "_RunLock":
        descriptor = -1
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise OSError
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if descriptor >= 0:
                os.close(descriptor)
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise E2ECommandError("operator_busy") from None
            raise E2ECommandError("configuration_error") from None
        self._descriptor = descriptor
        return self

    def __exit__(self, *_args: object) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _encode_receipt(payload: Mapping[str, object]) -> bytes:
    encoded = (
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_RECEIPT_BYTES:
        raise E2ECommandError("receipt_write_failed")
    return encoded


def _valid_final_result(value: object, *, operation_id: str) -> bool:
    if not isinstance(value, dict) or value.get("operation_id") != operation_id:
        return False
    if value.get("ok") is True:
        required = {
            "ok",
            "status",
            "operation_id",
            "source_kind",
            "source_bytes",
            "transcription_used",
            "transcript_characters",
            "request_characters",
            "answer_characters",
            "output_bytes",
            "response_id_present",
            "route_kind",
            "create_calls",
            "read_back_verified",
        }
        integer_fields = (
            "source_bytes",
            "transcript_characters",
            "request_characters",
            "answer_characters",
            "output_bytes",
        )
        return (
            set(value) == required
            and value.get("status") == "verified"
            and value.get("source_kind") in {"audio", "text"}
            and value.get("route_kind") == "question"
            and type(value.get("transcription_used")) is bool
            and value.get("transcription_used")
            is (value.get("source_kind") == "audio")
            and all(
                type(value.get(field)) is int and value[field] > 0
                for field in integer_fields
            )
            and value.get("read_back_verified") is True
            and value.get("create_calls") == 1
            and value.get("response_id_present") is True
        )
    required = {
        "ok",
        "status",
        "operation_id",
        "category",
        "mutation_attempted",
    }
    return (
        set(value) == required
        and value.get("ok") is False
        and value.get("status") == "failed_closed"
        and isinstance(value.get("category"), str)
        and _CATEGORY_RE.fullmatch(value["category"]) is not None
        and type(value.get("mutation_attempted")) is bool
    )


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _validate_receipt(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "operation",
        "operation_id",
        "operator_source_sha",
        "status",
        "started_at",
        "completed_at",
        "result",
    }:
        raise E2ECommandError("receipt_history_invalid")
    operation_id = payload.get("operation_id")
    if (
        payload.get("schema_version") != E2E_RECEIPT_SCHEMA_VERSION
        or payload.get("operation") != "light_e2e"
        or not isinstance(operation_id, str)
        or _OPERATION_ID_RE.fullmatch(operation_id) is None
        or not isinstance(payload.get("operator_source_sha"), str)
        or _SOURCE_REVISION_RE.fullmatch(payload["operator_source_sha"]) is None
        or not _valid_timestamp(payload.get("started_at"))
    ):
        raise E2ECommandError("receipt_history_invalid")
    status = payload.get("status")
    if status == "pending":
        if payload.get("completed_at") is not None or payload.get("result") is not None:
            raise E2ECommandError("receipt_history_invalid")
    elif status in {"verified", "failed_closed", "review_required"}:
        if not _valid_timestamp(payload.get("completed_at")) or not _valid_final_result(
            payload.get("result"), operation_id=operation_id
        ):
            raise E2ECommandError("receipt_history_invalid")
        mutation = payload["result"].get("mutation_attempted")
        if status == "review_required" and mutation is not True:
            raise E2ECommandError("receipt_history_invalid")
        if status == "failed_closed" and mutation is True:
            raise E2ECommandError("receipt_history_invalid")
        if status == "verified" and payload["result"].get("ok") is not True:
            raise E2ECommandError("receipt_history_invalid")
    else:
        raise E2ECommandError("receipt_history_invalid")
    return payload


class E2EReceiptStore:
    """Fixed latest receipt with content-free archival and atomic finalize."""

    def __init__(self, receipt_dir: Path) -> None:
        self.directory = Path(receipt_dir)
        self.path = self.directory / E2E_RECEIPT_NAME
        self._pending_encoded: bytes | None = None

    def _read(self) -> tuple[dict[str, object], bytes]:
        try:
            metadata = self.path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
                or metadata.st_size > _MAX_RECEIPT_BYTES
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise OSError
            descriptor = os.open(
                self.path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                opened = os.fstat(stream.fileno())
                if (
                    opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                    or not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                ):
                    raise OSError
                raw = stream.read(_MAX_RECEIPT_BYTES + 1)
            if len(raw) > _MAX_RECEIPT_BYTES:
                raise OSError
            payload = json.loads(raw)
        except FileNotFoundError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise E2ECommandError("receipt_history_invalid") from None
        return _validate_receipt(payload), raw

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _archive_complete_latest(self) -> None:
        try:
            existing, _raw = self._read()
        except FileNotFoundError:
            return
        if existing["status"] in {"pending", "review_required"}:
            raise E2ECommandError("unresolved_receipt_exists")
        archive = self.directory / f"e2e-{existing['operation_id']}.json"
        if archive.exists() or archive.is_symlink():
            raise E2ECommandError("receipt_history_invalid")
        try:
            os.rename(self.path, archive)
            os.chmod(archive, 0o600)
            self._fsync_directory()
        except OSError:
            raise E2ECommandError("receipt_write_failed") from None

    def create_pending(self, payload: Mapping[str, object]) -> None:
        _private_directory(self.directory)
        _validate_receipt(dict(payload))
        self._archive_complete_latest()
        encoded = _encode_receipt(payload)
        descriptor = -1
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._fsync_directory()
        except FileExistsError:
            raise E2ECommandError("unresolved_receipt_exists") from None
        except OSError:
            raise E2ECommandError("receipt_write_failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        self._pending_encoded = encoded

    def finalize(self, payload: Mapping[str, object]) -> None:
        _validate_receipt(dict(payload))
        pending = self._pending_encoded
        if pending is None:
            raise E2ECommandError("receipt_write_failed")
        _existing, observed = self._read()
        if not hmac.compare_digest(observed, pending):
            raise E2ECommandError("receipt_write_failed")
        encoded = _encode_receipt(payload)
        descriptor = -1
        temporary = ""
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{E2E_RECEIPT_NAME}.", dir=self.directory
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
            self._fsync_directory()
        except OSError:
            raise E2ECommandError("receipt_write_failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass


class _UnusedTranscriber:
    def transcribe(self, audio: bytes, *, mime_type: str | None = None) -> str:
        raise RuntimeError("text input unexpectedly requested transcription")


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise E2ECommandError("configuration_error")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _operation_id(factory: Callable[[], str]) -> str:
    value = factory()
    if not isinstance(value, str) or _OPERATION_ID_RE.fullmatch(value) is None:
        raise E2ECommandError("configuration_error")
    return value


def _chatgpt_category(error: Exception) -> str:
    if isinstance(error, ChatGPTCredentialAmbiguousError):
        return "chatgpt_credential_ambiguous"
    if isinstance(error, ChatGPTAuthorizationRequiredError):
        return "chatgpt_authorization_required"
    if isinstance(error, ChatGPTAccountNotEligibleError):
        return "chatgpt_account_not_eligible"
    if isinstance(error, ChatGPTPlanUsageLimitError):
        return "chatgpt_plan_usage_limit"
    if isinstance(error, ChatGPTConfigurationError):
        return "chatgpt_configuration_error"
    if isinstance(error, ChatGPTAmbiguousOutcomeError):
        return "chatgpt_response_ambiguous"
    if isinstance(error, ChatGPTTransientError):
        return "chatgpt_transient"
    return "e2e_failed"


def _pending_receipt(
    config: E2ECommandConfig, operation_id: str, started_at: str
) -> dict[str, object]:
    return {
        "schema_version": E2E_RECEIPT_SCHEMA_VERSION,
        "operation": "light_e2e",
        "operation_id": operation_id,
        "operator_source_sha": config.operator_source_sha,
        "status": "pending",
        "started_at": started_at,
        "completed_at": None,
        "result": None,
    }


def _final_receipt(
    pending: Mapping[str, object],
    *,
    status: str,
    result: Mapping[str, object],
    completed_at: str,
) -> dict[str, object]:
    return {
        **dict(pending),
        "status": status,
        "completed_at": completed_at,
        "result": dict(result),
    }


def run_e2e_command(
    *,
    environ: Mapping[str, str],
    limits: E2EProofLimits = E2EProofLimits(),
    provider_factory: Callable[..., Any] = GaradoLightProvider,
    transcriber_factory: Callable[..., Transcriber] = FasterWhisperTranscriber,
    credential_store_factory: Callable[[Path], Any] = CredentialStore,
    host_store_factory: Callable[[Path], Any] = HostIdStore,
    refresh_manager_factory: Callable[..., Any] = RefreshManager,
    assistant_factory: Callable[..., Any] = ChatGPTPlanAssistant,
    proof_runner: Callable[..., E2EProofResult] = run_e2e_proof,
    model_verifier: Callable[[Path, Path], None] = verify_input_audit_model,
    receipt_store_factory: Callable[[Path], E2EReceiptStore] = E2EReceiptStore,
    operation_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, object]:
    """Assemble and run one protected real E2E proof from environment only."""

    config = E2ECommandConfig.from_environ(environ)
    operation_id = _operation_id(operation_id_factory)
    lock_path = config.receipt_dir / ".light-e2e.lock"
    with _RunLock(lock_path):
        store = receipt_store_factory(config.receipt_dir)
        pending = _pending_receipt(config, operation_id, _timestamp(now()))
        store.create_pending(pending)
        provider: Any | None = None
        try:
            credential_store = credential_store_factory(config.credential_path)
            profile = credential_store.load()
            host_id = host_store_factory(config.host_id_path).load()
            if not hmac.compare_digest(profile.ext_agent_host_id, host_id):
                raise E2ECommandError(
                    "chatgpt_host_binding_mismatch",
                    operation_id=operation_id,
                )

            if config.binding.expected_kind is NoteKind.AUDIO:
                model_verifier(config.stt_model_dir, config.stt_model_manifest)
                transcriber: Transcriber = transcriber_factory(
                    model=os.fspath(config.stt_model_dir),
                    model_cache=None,
                    device="cpu",
                    compute_type="int8",
                    language=None,
                    local_files_only=True,
                    max_audio_bytes=limits.max_audio_bytes,
                    temp_root=config.stt_temp_dir,
                )
            else:
                transcriber = _UnusedTranscriber()

            tokens = refresh_manager_factory(
                credential_store, timeout_seconds=15.0
            )
            assistant = assistant_factory(
                tokens=tokens,
                model=config.chatgpt_model,
                timeout_seconds=60.0,
                max_stream_chars=limits.max_answer_chars,
            )
            provider = provider_factory(
                username=config.username,
                password=config.password,
                device_id=config.device_id,
                write_policy=LightWritePolicy.TEST_NOTES,
            )
            result = proof_runner(
                provider=provider,
                transcriber=transcriber,
                assistant=assistant,
                binding=config.binding,
                limits=limits,
                operation_id_factory=lambda: operation_id,
                now=now,
            )
            public = result.as_dict()
        except E2EProofError as error:
            command_error = E2ECommandError(
                error.category,
                operation_id=operation_id,
                mutation_attempted=error.mutation_attempted,
            )
            public_error = command_error.as_dict()
            status = (
                "review_required"
                if command_error.mutation_attempted
                else "failed_closed"
            )
            try:
                store.finalize(
                    _final_receipt(
                        pending,
                        status=status,
                        result=public_error,
                        completed_at=_timestamp(now()),
                    )
                )
            except Exception:
                raise E2ECommandError(
                    "receipt_finalize_failed",
                    operation_id=operation_id,
                    mutation_attempted=command_error.mutation_attempted,
                ) from None
            raise command_error from None
        except E2ECommandError as error:
            command_error = E2ECommandError(
                error.category,
                operation_id=operation_id,
                mutation_attempted=error.mutation_attempted,
            )
            try:
                store.finalize(
                    _final_receipt(
                        pending,
                        status=(
                            "review_required"
                            if command_error.mutation_attempted
                            else "failed_closed"
                        ),
                        result=command_error.as_dict(),
                        completed_at=_timestamp(now()),
                    )
                )
            except Exception:
                raise E2ECommandError(
                    "receipt_finalize_failed",
                    operation_id=operation_id,
                    mutation_attempted=command_error.mutation_attempted,
                ) from None
            raise command_error from None
        except Exception as error:
            command_error = E2ECommandError(
                _chatgpt_category(error), operation_id=operation_id
            )
            try:
                store.finalize(
                    _final_receipt(
                        pending,
                        status="failed_closed",
                        result=command_error.as_dict(),
                        completed_at=_timestamp(now()),
                    )
                )
            except Exception:
                raise E2ECommandError(
                    "receipt_finalize_failed", operation_id=operation_id
                ) from None
            raise command_error from None
        finally:
            if provider is not None:
                close = getattr(provider, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

        try:
            store.finalize(
                _final_receipt(
                    pending,
                    status="verified",
                    result=public,
                    completed_at=_timestamp(now()),
                )
            )
        except Exception:
            # The cloud output is already verified. Preserve the durable
            # pending record and require review rather than claiming failure
            # or starting a second proof.
            raise E2ECommandError(
                "receipt_finalize_failed",
                operation_id=operation_id,
                mutation_attempted=True,
            ) from None
        return public
