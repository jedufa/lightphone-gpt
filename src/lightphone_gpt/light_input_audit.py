"""Read-only, one-shot audit and binding tools for selected Light inputs.

The audit keeps source content and routed text in ephemeral memory, apart from
the transcriber's owner-only temporary audio file which is deleted before the
command returns. Its durable receipt contains sizes, routing metadata, and
timings, never a note identity, title, source content, routed text, or
credential value. The separate binding command persists opaque IDs only in a
private receipt and writes a second ID-free summary for shareable evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import errno
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
from typing import Any, Iterator
import uuid

from .adapters.light_garado import GaradoLightProvider, LightWritePolicy
from .adapters.transcriber import FasterWhisperTranscriber
from .models import NoteKind, RemoteNote, RouteKind
from .routing import route_spoken_transcript, route_transcript


_SCHEMA_VERSION = 1
_NOTE_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,256}\Z")
_SOURCE_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MIME_RE = re.compile(r"audio/[a-z0-9][a-z0-9.+-]{0,63}\Z")
_BINDING_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_MAX_AUDIO_BYTES = 32 * 1024 * 1024
_MAX_TEXT_BYTES = 64 * 1024
_MAX_BINDINGS_JSON_BYTES = 16 * 1024
_MAX_BINDINGS = 32
_TEMP_PREFIX = "lightphone-gpt-stt-"
_MODEL_MANIFEST_SCHEMA_VERSION = 1
_MODEL_MANIFEST_MAX_BYTES = 1024 * 1024
INPUT_AUDIT_MODEL_REPOSITORY = "Systran/faster-whisper-base"
INPUT_AUDIT_MODEL_REVISION = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"


class LightInputAuditError(RuntimeError):
    """A stable, content-free failure safe for operator output."""

    def __init__(self, category: str, *, operation_id: str | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.operation_id = operation_id


@dataclass(frozen=True, slots=True)
class _AuditConfig:
    username: str
    password: str
    device_id: str
    note_id: str
    source_kind: NoteKind
    source_sha: str
    state_dir: Path
    temp_dir: Path | None
    model_dir: Path | None
    model_manifest: Path | None


@dataclass(frozen=True, slots=True)
class _BindingDesignation:
    name: str
    kind: NoteKind
    title_sha256: str


@dataclass(frozen=True, slots=True)
class _BindingConfig:
    username: str
    password: str
    device_id: str
    source_sha: str
    state_dir: Path
    designations: tuple[_BindingDesignation, ...]


def _timestamp(value: datetime | None = None) -> str:
    current = value or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise LightInputAuditError("configuration_error")
    return current.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _private_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise LightInputAuditError("configuration_error") from None
    if (
        not path.is_absolute()
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or resolved != path
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise LightInputAuditError("configuration_error")


def _configuration(environ: Mapping[str, str]) -> _AuditConfig:
    username = environ.get("lightphone_username", "").strip()
    password = environ.get("lightphone_password", "")
    device_id = environ.get("lightphone_device_id", "").strip()
    note_id = environ.get("LIGHTPHONE_GPT_INPUT_AUDIT_NOTE_ID", "").strip()
    raw_kind = environ.get("LIGHTPHONE_GPT_INPUT_AUDIT_KIND", "audio").strip().lower()
    source_sha = environ.get("LIGHTPHONE_GPT_OPERATOR_SOURCE_SHA", "").strip()
    raw_state = environ.get("LIGHTPHONE_GPT_STATE_DIR", "").strip()
    raw_temp = environ.get("LIGHTPHONE_GPT_INPUT_AUDIT_TEMP_DIR", "").strip()
    raw_model = environ.get("LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_DIR", "").strip()
    raw_manifest = environ.get(
        "LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_MANIFEST", ""
    ).strip()
    try:
        source_kind = NoteKind(raw_kind)
    except ValueError:
        raise LightInputAuditError("configuration_error") from None
    if (
        not username
        or not password
        or not device_id
        or _NOTE_ID_RE.fullmatch(note_id) is None
    ):
        raise LightInputAuditError("configuration_error")
    if _SOURCE_SHA_RE.fullmatch(source_sha) is None or not raw_state:
        raise LightInputAuditError("configuration_error")
    state_dir = Path(raw_state)
    _private_directory(state_dir)
    temp_dir: Path | None = None
    model_dir: Path | None = None
    model_manifest: Path | None = None
    if source_kind is NoteKind.AUDIO:
        if not raw_temp or not raw_model or not raw_manifest:
            raise LightInputAuditError("configuration_error")
        temp_dir = Path(raw_temp)
        model_dir = Path(raw_model)
        model_manifest = Path(raw_manifest)
        for path in (temp_dir, model_dir):
            _private_directory(path)
        verify_input_audit_model(model_dir, model_manifest)
    return _AuditConfig(
        username=username,
        password=password,
        device_id=device_id,
        note_id=note_id,
        source_kind=source_kind,
        source_sha=source_sha,
        state_dir=state_dir,
        temp_dir=temp_dir,
        model_dir=model_dir,
        model_manifest=model_manifest,
    )


def _binding_configuration(environ: Mapping[str, str]) -> _BindingConfig:
    username = environ.get("lightphone_username", "").strip()
    password = environ.get("lightphone_password", "")
    device_id = environ.get("lightphone_device_id", "").strip()
    source_sha = environ.get("LIGHTPHONE_GPT_OPERATOR_SOURCE_SHA", "").strip()
    raw_state = environ.get("LIGHTPHONE_GPT_STATE_DIR", "").strip()
    raw_designations = environ.get(
        "LIGHTPHONE_GPT_INPUT_BINDINGS_JSON", ""
    )
    try:
        designations_bytes = raw_designations.encode("utf-8")
    except UnicodeEncodeError:
        raise LightInputAuditError("configuration_error") from None
    if (
        not username
        or not password
        or not device_id
        or _SOURCE_SHA_RE.fullmatch(source_sha) is None
        or not raw_state
        or not raw_designations
        or len(designations_bytes) > _MAX_BINDINGS_JSON_BYTES
    ):
        raise LightInputAuditError("configuration_error")
    state_dir = Path(raw_state)
    _private_directory(state_dir)
    try:
        payload = json.loads(raw_designations)
    except (UnicodeError, json.JSONDecodeError):
        raise LightInputAuditError("configuration_error") from None
    if not isinstance(payload, list) or not 1 <= len(payload) <= _MAX_BINDINGS:
        raise LightInputAuditError("configuration_error")

    designations: list[_BindingDesignation] = []
    names: set[str] = set()
    title_digests: set[str] = set()
    for item in payload:
        if not isinstance(item, dict) or set(item) != {
            "name",
            "kind",
            "title_sha256",
        }:
            raise LightInputAuditError("configuration_error")
        name = item.get("name")
        raw_item_kind = item.get("kind")
        title_sha256 = item.get("title_sha256")
        if (
            not isinstance(name, str)
            or _BINDING_NAME_RE.fullmatch(name) is None
            or not isinstance(raw_item_kind, str)
            or not isinstance(title_sha256, str)
            or _SHA256_RE.fullmatch(title_sha256) is None
            or name in names
            or title_sha256 in title_digests
        ):
            raise LightInputAuditError("configuration_error")
        try:
            kind = NoteKind(raw_item_kind)
        except ValueError:
            raise LightInputAuditError("configuration_error") from None
        names.add(name)
        title_digests.add(title_sha256)
        designations.append(_BindingDesignation(name, kind, title_sha256))
    return _BindingConfig(
        username=username,
        password=password,
        device_id=device_id,
        source_sha=source_sha,
        state_dir=state_dir,
        designations=tuple(designations),
    )


def _private_regular_file(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise LightInputAuditError("configuration_error") from None
    if (
        not path.is_absolute()
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or resolved != path
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise LightInputAuditError("configuration_error")
    return metadata


def _sha256_file(path: Path, expected: os.stat_result) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    digest = hashlib.sha256()
    size = 0
    try:
        descriptor = os.open(path, flags)
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_dev != expected.st_dev
            or observed.st_ino != expected.st_ino
            or (hasattr(os, "geteuid") and observed.st_uid != os.geteuid())
            or stat.S_IMODE(observed.st_mode) != 0o600
        ):
            raise OSError
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    except OSError:
        raise LightInputAuditError("configuration_error") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return size, digest.hexdigest()


def _read_private_file(
    path: Path, expected: os.stat_result, *, maximum_bytes: int
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    content = bytearray()
    try:
        descriptor = os.open(path, flags)
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_dev != expected.st_dev
            or observed.st_ino != expected.st_ino
            or observed.st_size < 1
            or observed.st_size > maximum_bytes
            or stat.S_IMODE(observed.st_mode) != 0o600
            or (hasattr(os, "geteuid") and observed.st_uid != os.geteuid())
        ):
            raise OSError
        while len(content) <= maximum_bytes:
            block = os.read(descriptor, min(64 * 1024, maximum_bytes + 1))
            if not block:
                break
            content.extend(block)
        if len(content) > maximum_bytes:
            raise OSError
    except OSError:
        raise LightInputAuditError("configuration_error") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return bytes(content)


def _model_file_entries(model_dir: Path) -> list[dict[str, object]]:
    _private_directory(model_dir)
    entries: list[dict[str, object]] = []
    directories = [model_dir]
    while directories:
        directory = directories.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            raise LightInputAuditError("configuration_error") from None
        for child in children:
            try:
                metadata = child.lstat()
            except OSError:
                raise LightInputAuditError("configuration_error") from None
            if (
                stat.S_ISLNK(metadata.st_mode)
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise LightInputAuditError("configuration_error")
            if stat.S_ISDIR(metadata.st_mode):
                directories.append(child)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise LightInputAuditError("configuration_error")
            relative = child.relative_to(model_dir).as_posix()
            size, digest = _sha256_file(child, metadata)
            entries.append(
                {"path": relative, "sha256": digest, "size": size}
            )
    entries.sort(key=lambda entry: str(entry["path"]))
    if not entries:
        raise LightInputAuditError("configuration_error")
    return entries


def _model_manifest(model_dir: Path) -> dict[str, object]:
    return {
        "schema_version": _MODEL_MANIFEST_SCHEMA_VERSION,
        "repository": INPUT_AUDIT_MODEL_REPOSITORY,
        "revision": INPUT_AUDIT_MODEL_REVISION,
        "files": _model_file_entries(model_dir),
    }


def _encode_model_manifest(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def write_input_audit_model_manifest(
    model_dir: Path, manifest_path: Path
) -> None:
    """Write the deterministic complete-file manifest for a new model copy."""

    payload = _model_manifest(model_dir)
    encoded = _encode_model_manifest(payload)
    if not manifest_path.is_absolute() or manifest_path.exists():
        raise LightInputAuditError("configuration_error")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(manifest_path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise LightInputAuditError("configuration_error") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_input_audit_model(model_dir: Path, manifest_path: Path) -> None:
    """Verify candidate-local model bytes against its canonical manifest."""

    metadata = _private_regular_file(manifest_path)
    try:
        encoded = _read_private_file(
            manifest_path,
            metadata,
            maximum_bytes=_MODEL_MANIFEST_MAX_BYTES,
        )
        payload = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise LightInputAuditError("configuration_error") from None
    if not isinstance(payload, dict) or encoded != _encode_model_manifest(payload):
        raise LightInputAuditError("configuration_error")
    if payload != _model_manifest(model_dir):
        raise LightInputAuditError("configuration_error")


def _copy_snapshot_tree(source: Path, destination: Path, cache: Path) -> None:
    try:
        source_metadata = source.lstat()
        cache_root = cache.resolve(strict=True)
        source_root = source.resolve(strict=True)
        source_root.relative_to(cache_root)
    except (FileNotFoundError, OSError, ValueError):
        raise LightInputAuditError("configuration_error") from None
    if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISDIR(
        source_metadata.st_mode
    ):
        raise LightInputAuditError("configuration_error")

    destination.mkdir(mode=0o700)
    pending = [(source, destination)]
    while pending:
        current_source, current_destination = pending.pop()
        try:
            children = sorted(
                current_source.iterdir(), key=lambda path: path.name
            )
        except OSError:
            raise LightInputAuditError("configuration_error") from None
        for child in children:
            target = current_destination / child.name
            try:
                metadata = child.lstat()
            except OSError:
                raise LightInputAuditError("configuration_error") from None
            if stat.S_ISDIR(metadata.st_mode):
                target.mkdir(mode=0o700)
                pending.append((child, target))
                continue
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    actual = child.resolve(strict=True)
                    actual.relative_to(cache_root)
                    actual_metadata = actual.lstat()
                except (FileNotFoundError, OSError, ValueError):
                    raise LightInputAuditError("configuration_error") from None
                if not stat.S_ISREG(actual_metadata.st_mode):
                    raise LightInputAuditError("configuration_error")
            elif stat.S_ISREG(metadata.st_mode):
                actual = child
            else:
                raise LightInputAuditError("configuration_error")
            try:
                with actual.open("rb") as source_stream:
                    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                    flags |= getattr(os, "O_CLOEXEC", 0)
                    descriptor = os.open(target, flags, 0o600)
                    try:
                        with os.fdopen(descriptor, "wb", closefd=True) as output:
                            descriptor = -1
                            shutil.copyfileobj(source_stream, output)
                            output.flush()
                            os.fsync(output.fileno())
                    finally:
                        if descriptor >= 0:
                            os.close(descriptor)
            except OSError:
                raise LightInputAuditError("configuration_error") from None


def prepare_input_audit_model(
    cache_dir: Path, destination: Path, manifest_path: Path
) -> None:
    """Fetch the pin into staging and create a symlink-free candidate copy."""

    _private_directory(cache_dir)
    if destination.exists() or manifest_path.exists():
        raise LightInputAuditError("configuration_error")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise LightInputAuditError("configuration_error") from None
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=INPUT_AUDIT_MODEL_REPOSITORY,
                revision=INPUT_AUDIT_MODEL_REVISION,
                cache_dir=os.fspath(cache_dir),
                local_files_only=False,
            )
        )
    except Exception:
        raise LightInputAuditError("configuration_error") from None
    if snapshot.name != INPUT_AUDIT_MODEL_REVISION:
        raise LightInputAuditError("configuration_error")
    try:
        _copy_snapshot_tree(snapshot, destination, cache_dir)
        write_input_audit_model_manifest(destination, manifest_path)
        verify_input_audit_model(destination, manifest_path)
    except Exception:
        if destination.exists() and not destination.is_symlink():
            shutil.rmtree(destination, ignore_errors=True)
        try:
            manifest_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _cleanup_stale_audio(temp_dir: Path) -> None:
    """Remove only this transcriber's private temporary directories."""

    _private_directory(temp_dir)
    try:
        candidates = list(temp_dir.iterdir())
    except OSError:
        raise LightInputAuditError("configuration_error") from None
    for candidate in candidates:
        if not candidate.name.startswith(_TEMP_PREFIX):
            continue
        try:
            metadata = candidate.lstat()
        except OSError:
            raise LightInputAuditError("configuration_error") from None
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise LightInputAuditError("configuration_error")
        try:
            shutil.rmtree(candidate)
        except OSError:
            raise LightInputAuditError("configuration_error") from None


class _ReceiptLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._descriptor: int | None = None

    def __enter__(self) -> "_ReceiptLock":
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        try:
            descriptor = os.open(self.path, flags, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise OSError
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if descriptor >= 0:
                os.close(descriptor)
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise LightInputAuditError("operator_busy") from None
            raise LightInputAuditError("configuration_error") from None
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


class _ReceiptStore:
    """Create one owner-only receipt and replace it atomically as stages finish."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_absolute() or not self.path.name:
            raise LightInputAuditError("configuration_error")

    @contextmanager
    def locked(self) -> Iterator["_ReceiptStore"]:
        _private_directory(self.path.parent)
        with _ReceiptLock(self.path.with_name(f".{self.path.name}.lock")):
            yield self

    def create(self, payload: Mapping[str, object]) -> None:
        encoded = self._encode(payload)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        try:
            descriptor = os.open(self.path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._fsync_parent()
        except FileExistsError:
            raise LightInputAuditError("receipt_already_exists") from None
        except OSError:
            raise LightInputAuditError("configuration_error") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def replace(self, payload: Mapping[str, object]) -> None:
        self._verify_existing()
        descriptor = -1
        temporary = ""
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{self.path.name}.", dir=self.path.parent
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(self._encode(payload))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = ""
            self._fsync_parent()
        except OSError:
            raise LightInputAuditError("receipt_write_failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _verify_existing(self) -> None:
        try:
            metadata = self.path.lstat()
        except OSError:
            raise LightInputAuditError("receipt_write_failed") from None
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise LightInputAuditError("receipt_write_failed")

    @staticmethod
    def _encode(payload: Mapping[str, object]) -> bytes:
        return (
            json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")

    def _fsync_parent(self) -> None:
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _elapsed_ms(start: int, end: int) -> int:
    return max(0, round((end - start) / 1_000_000))


def run_input_audit(
    *,
    receipt_path: Path,
    confirm_read: bool,
    environ: Mapping[str, str],
    provider_factory: Callable[..., Any] = GaradoLightProvider,
    transcriber_factory: Callable[..., Any] = FasterWhisperTranscriber,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, object]:
    """Download and route exactly one selected audio or text note read-only."""

    if not confirm_read:
        raise LightInputAuditError("confirmation_required")
    config = _configuration(environ)
    note_id = config.note_id
    receipt = _ReceiptStore(receipt_path)
    operation_id = uuid.uuid4().hex
    started_at = _timestamp(now())
    pending: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "operation_id": operation_id,
        "source_sha": config.source_sha,
        "status": "pending",
        "started_at": started_at,
        "completed_at": None,
        "result": None,
    }
    source_content: bytes | None = None
    routed_text: str | None = None
    stage = "prepared"
    total_start = monotonic_ns()

    with _ReceiptLock(config.state_dir / ".input-audit.lock"):
        if config.temp_dir is not None:
            _cleanup_stale_audio(config.temp_dir)
        with receipt.locked():
            receipt.create(pending)
            try:
                stage = "cloud_download"
                cloud_start = monotonic_ns()
                provider = provider_factory(
                    username=config.username,
                    password=config.password,
                    device_id=config.device_id,
                    write_policy=LightWritePolicy.READ_ONLY,
                )
                try:
                    if config.source_kind is NoteKind.AUDIO:
                        note, source_content = (
                            provider.download_audio_with_metadata(note_id)
                        )
                    else:
                        note, source_content = provider.read_text_with_metadata(
                            note_id
                        )
                finally:
                    provider.close()
                cloud_end = monotonic_ns()

                if (
                    note.note_id != note_id
                    or note.kind is not config.source_kind
                ):
                    raise LightInputAuditError("input_audit_failed")
                if not isinstance(source_content, bytes):
                    raise LightInputAuditError("input_audit_failed")

                if config.source_kind is NoteKind.AUDIO:
                    mime_type = note.mime_type
                    if (
                        not isinstance(mime_type, str)
                        or _MIME_RE.fullmatch(mime_type) is None
                        or not source_content
                        or len(source_content) > _MAX_AUDIO_BYTES
                    ):
                        raise LightInputAuditError("input_audit_failed")
                    audio_bytes = len(source_content)
                    stage = "transcription"
                    transcription_start = monotonic_ns()
                    if config.model_dir is None or config.temp_dir is None:
                        raise LightInputAuditError("input_audit_failed")
                    transcriber = transcriber_factory(
                        model=os.fspath(config.model_dir),
                        model_cache=None,
                        device="cpu",
                        compute_type="int8",
                        language=None,
                        local_files_only=True,
                        max_audio_bytes=_MAX_AUDIO_BYTES,
                        temp_root=config.temp_dir,
                    )
                    routed_text = transcriber.transcribe(
                        source_content, mime_type=mime_type
                    )
                    transcription_end = monotonic_ns()
                    source_content = None
                    source_metrics: dict[str, object] = {
                        "downloaded": True,
                        "audio_bytes": audio_bytes,
                        "mime_type": mime_type,
                        "transcribed": True,
                        "transcript_characters": len(routed_text),
                    }
                    stage_timings: dict[str, int] = {
                        "cloud_download": _elapsed_ms(
                            cloud_start, cloud_end
                        ),
                        "transcription": _elapsed_ms(
                            transcription_start, transcription_end
                        ),
                    }
                else:
                    stage = "source_validation"
                    if len(source_content) > _MAX_TEXT_BYTES:
                        raise LightInputAuditError("input_audit_failed")
                    stage = "text_decode"
                    decoding_start = monotonic_ns()
                    try:
                        routed_text = source_content.decode("utf-8")
                    except UnicodeDecodeError:
                        raise LightInputAuditError("input_audit_failed") from None
                    decoding_end = monotonic_ns()
                    content_bytes = len(source_content)
                    source_content = None
                    source_metrics = {
                        "downloaded": True,
                        "content_bytes": content_bytes,
                        "mime_type": "text/plain; charset=utf-8",
                        "transcribed": False,
                        "text_characters": len(routed_text),
                    }
                    stage_timings = {
                        "cloud_download": _elapsed_ms(
                            cloud_start, cloud_end
                        ),
                        "text_decode": _elapsed_ms(
                            decoding_start, decoding_end
                        ),
                    }

                stage = "routing"
                routing_start = monotonic_ns()
                route = (
                    route_spoken_transcript(routed_text)
                    if config.source_kind is NoteKind.AUDIO
                    else route_transcript(routed_text)
                )
                routing_end = monotonic_ns()
                route_kind = route.kind
                route = None
                routed_text = None
                if config.temp_dir is not None:
                    _cleanup_stale_audio(config.temp_dir)
                total_end = monotonic_ns()
                stage_timings.update(
                    {
                        "routing": _elapsed_ms(routing_start, routing_end),
                        "total": _elapsed_ms(total_start, total_end),
                    }
                )

                accepted = route_kind is not RouteKind.IGNORED
                result: dict[str, object] = {
                    "operation": "light_input_audit",
                    "source_sha": config.source_sha,
                    "source_kind": config.source_kind.value,
                    "write_policy": "read_only",
                    **source_metrics,
                    "prefix_recognized": accepted,
                    "accepted": accepted,
                    "route_kind": route_kind.value,
                    "timing_ms": stage_timings,
                }
                completed = {
                    **pending,
                    "status": "complete",
                    "completed_at": _timestamp(now()),
                    "result": result,
                }
                receipt.replace(completed)
                public = {
                    "ok": accepted,
                    "pipeline_completed": True,
                    "operation_id": operation_id,
                    **result,
                }
                if not accepted:
                    public["error"] = "prefix_not_recognized"
                return public
            except Exception as error:
                failure = {
                    **pending,
                    "status": "failed",
                    "completed_at": _timestamp(now()),
                    "result": {
                        "error": "input_audit_failed",
                        "failed_stage": stage,
                    },
                }
                try:
                    receipt.replace(failure)
                except Exception:
                    pass
                if isinstance(error, LightInputAuditError) and error.category in {
                    "receipt_write_failed",
                }:
                    raise
                raise LightInputAuditError(
                    "input_audit_failed", operation_id=operation_id
                ) from None
            finally:
                source_content = None
                routed_text = None
                if config.temp_dir is not None:
                    try:
                        _cleanup_stale_audio(config.temp_dir)
                    except Exception:
                        pass


class _BindingResolutionError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _resolve_bindings(
    notes: object,
    designations: tuple[_BindingDesignation, ...],
) -> list[dict[str, str]]:
    if not isinstance(notes, Sequence) or isinstance(notes, (str, bytes)):
        raise _BindingResolutionError("remote_metadata_invalid")
    indexed: list[tuple[str, RemoteNote]] = []
    for note in notes:
        if (
            not isinstance(note, RemoteNote)
            or _NOTE_ID_RE.fullmatch(note.note_id) is None
            or not isinstance(note.title, str)
        ):
            raise _BindingResolutionError("remote_metadata_invalid")
        try:
            title_sha256 = hashlib.sha256(note.title.encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            raise _BindingResolutionError("remote_metadata_invalid") from None
        indexed.append((title_sha256, note))

    resolved: list[dict[str, str]] = []
    for designation in designations:
        matches = [
            note
            for title_sha256, note in indexed
            if hmac.compare_digest(title_sha256, designation.title_sha256)
        ]
        if not matches:
            raise _BindingResolutionError("binding_not_found")
        if len(matches) != 1:
            raise _BindingResolutionError("binding_not_unique")
        note = matches[0]
        if note.kind is not designation.kind:
            raise _BindingResolutionError("binding_kind_mismatch")
        resolved.append(
            {
                "name": designation.name,
                "kind": designation.kind.value,
                "note_id": note.note_id,
            }
        )

    note_ids = [binding["note_id"] for binding in resolved]
    if len(set(note_ids)) != len(note_ids):
        raise _BindingResolutionError("duplicate_resolved_note_id")
    return resolved


def run_input_binding(
    *,
    receipt_path: Path,
    summary_path: Path,
    confirm_read: bool,
    environ: Mapping[str, str],
    provider_factory: Callable[..., Any] = GaradoLightProvider,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, object]:
    """Resolve protected title digests to exact IDs using metadata only.

    Exact IDs are written only to ``receipt_path``. The returned object and
    ``summary_path`` omit IDs, titles, title digests, content, and credentials.
    """

    if not confirm_read:
        raise LightInputAuditError("confirmation_required")
    config = _binding_configuration(environ)
    private_receipt = _ReceiptStore(receipt_path)
    summary_receipt = _ReceiptStore(summary_path)
    if private_receipt.path == summary_receipt.path:
        raise LightInputAuditError("configuration_error")

    operation_id = uuid.uuid4().hex
    started_at = _timestamp(now())
    pending: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "operation_id": operation_id,
        "source_sha": config.source_sha,
        "status": "pending",
        "started_at": started_at,
        "completed_at": None,
        "result": None,
    }
    stage = "prepared"
    total_start = monotonic_ns()
    private_complete = False

    with _ReceiptLock(config.state_dir / ".input-audit.lock"):
        with private_receipt.locked():
            with summary_receipt.locked():
                private_receipt.create(pending)
                try:
                    summary_receipt.create(pending)
                except Exception:
                    try:
                        private_receipt.replace(
                            {
                                **pending,
                                "status": "failed",
                                "completed_at": _timestamp(now()),
                                "result": {
                                    "error": "input_binding_failed",
                                    "failed_stage": "prepared",
                                    "failure_reason": "receipt_write_failed",
                                },
                            }
                        )
                    except Exception:
                        pass
                    raise
                try:
                    stage = "metadata_listing"
                    listing_start = monotonic_ns()
                    provider = provider_factory(
                        username=config.username,
                        password=config.password,
                        device_id=config.device_id,
                        write_policy=LightWritePolicy.READ_ONLY,
                    )
                    try:
                        notes = provider.list_notes()
                    finally:
                        provider.close()
                    listing_end = monotonic_ns()

                    stage = "matching"
                    matching_start = monotonic_ns()
                    bindings = _resolve_bindings(notes, config.designations)
                    matching_end = monotonic_ns()
                    total_end = monotonic_ns()
                    kinds = {
                        kind.value: sum(
                            1
                            for binding in bindings
                            if binding["kind"] == kind.value
                        )
                        for kind in NoteKind
                    }
                    summary_result: dict[str, object] = {
                        "operation": "light_input_binding_summary",
                        "source_sha": config.source_sha,
                        "write_policy": "read_only",
                        "notes_scanned": len(notes),
                        "bindings_requested": len(config.designations),
                        "bindings_resolved": len(bindings),
                        "resolved_kinds": kinds,
                        "all_unique": True,
                        "timing_ms": {
                            "metadata_listing": _elapsed_ms(
                                listing_start, listing_end
                            ),
                            "matching": _elapsed_ms(
                                matching_start, matching_end
                            ),
                            "total": _elapsed_ms(total_start, total_end),
                        },
                    }
                    completed_at = _timestamp(now())
                    private_receipt.replace(
                        {
                            **pending,
                            "status": "complete",
                            "completed_at": completed_at,
                            "result": {
                                "operation": "light_input_binding",
                                "source_sha": config.source_sha,
                                "write_policy": "read_only",
                                "bindings": bindings,
                                "summary": summary_result,
                            },
                        }
                    )
                    private_complete = True
                    summary_receipt.replace(
                        {
                            **pending,
                            "status": "complete",
                            "completed_at": completed_at,
                            "result": summary_result,
                        }
                    )
                    return {
                        "ok": True,
                        "operation_id": operation_id,
                        **summary_result,
                    }
                except Exception as error:
                    failure_reason = (
                        error.reason
                        if isinstance(error, _BindingResolutionError)
                        else "operation_failed"
                    )
                    failure = {
                        **pending,
                        "status": "failed",
                        "completed_at": _timestamp(now()),
                        "result": {
                            "error": "input_binding_failed",
                            "failed_stage": stage,
                            "failure_reason": failure_reason,
                        },
                    }
                    if not private_complete:
                        try:
                            private_receipt.replace(failure)
                        except Exception:
                            pass
                    try:
                        summary_receipt.replace(failure)
                    except Exception:
                        pass
                    if isinstance(error, LightInputAuditError) and error.category == (
                        "receipt_write_failed"
                    ):
                        raise
                    raise LightInputAuditError(
                        "input_binding_failed", operation_id=operation_id
                    ) from None


def run_command(args: Any, environ: Mapping[str, str]) -> dict[str, object]:
    return run_input_audit(
        receipt_path=args.receipt,
        confirm_read=args.confirm_read,
        environ=environ,
    )


def run_binding_command(
    args: Any, environ: Mapping[str, str]
) -> dict[str, object]:
    return run_input_binding(
        receipt_path=args.receipt,
        summary_path=args.summary,
        confirm_read=args.confirm_read,
        environ=environ,
    )
