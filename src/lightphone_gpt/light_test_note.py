"""Fail-closed, one-shot Light Cloud test-note operations.

This module is deliberately independent of the continuous bridge runtime.  It
owns only a synthetic Phase 0 note described by an owner-only local receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterator
import uuid

from .adapters.light_garado import GaradoLightProvider, LightWritePolicy
from .models import MutationOutcome, NoteKind, RemoteNote


TEST_TITLE_PREFIX = "LIGHTPHONE GPT TEST"
_SCHEMA_VERSION = 1
_MAX_RECEIPT_BYTES = 32 * 1024
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_NOTE_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,256}\Z")
_RECEIPT_FIELDS = {
    "schema_version",
    "account_binding",
    "run_id",
    "note_id",
    "revision",
    "expected_title",
    "expected_content_sha256",
    "status",
    "intent_sequence",
    "active_intent",
    "last_intent",
    "created_at",
    "updated_at",
}
_INTENT_FIELDS = {
    "sequence",
    "operation_id",
    "operation",
    "outcome",
    "started_at",
    "resolved_at",
    "previous_revision",
    "previous_title",
    "previous_content_sha256",
    "desired_revision",
    "desired_title",
    "desired_content_sha256",
}


class LightTestNoteError(ValueError):
    """A content-free operator or receipt validation error."""


class LightTestNoteLockError(LightTestNoteError):
    """Another operator command owns the receipt lock."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _digest(content: str | bytes) -> str:
    raw = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(raw).hexdigest()


def _title(run_id: str, revision: int) -> str:
    return f"{TEST_TITLE_PREFIX} {run_id} R{revision}"


def _body(run_id: str, revision: int) -> str:
    return (
        "Lightphone GPT Phase 0 sync test\n"
        f"Run: {run_id}\n"
        f"Revision: {revision}\n"
    )


def _binding(username: str, device_id: str) -> str:
    identity = f"{username.strip().casefold()}\0{device_id}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _credentials(environ: Mapping[str, str]) -> tuple[str, str, str, str]:
    username = environ.get("lightphone_username", "").strip()
    password = environ.get("lightphone_password", "")
    device_id = environ.get("lightphone_device_id", "").strip()
    if not username or not password:
        raise LightTestNoteError("Light credentials are required")
    if not device_id:
        raise LightTestNoteError(
            "an explicit Light device ID is required for test-note operations"
        )
    return username, password, device_id, _binding(username, device_id)


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def _valid_note_id(value: object) -> bool:
    return isinstance(value, str) and _NOTE_ID_RE.fullmatch(value) is not None


def _require_note_id(value: object) -> str:
    if not _valid_note_id(value):
        raise LightTestNoteError("Light returned an unsafe note identifier")
    return value


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
                raise LightTestNoteLockError(
                    "another test-note operator owns this receipt"
                ) from None
            raise LightTestNoteLockError(
                "the test-note receipt lock is unavailable or unsafe"
            ) from None
        self._descriptor = descriptor
        return self

    def __exit__(self, *_: object) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class ReceiptStore:
    """Owner-only, durable JSON receipt protected by an adjacent lock."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_absolute() or not self.path.name:
            raise LightTestNoteError("the receipt path must be an absolute file path")

    def _validate_parent(self) -> None:
        parent = self.path.parent
        try:
            metadata = parent.lstat()
            resolved = parent.resolve(strict=True)
        except (FileNotFoundError, OSError):
            raise LightTestNoteError(
                "the receipt directory must already exist with mode 0700"
            ) from None
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or resolved != parent
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise LightTestNoteError(
                "the receipt directory must be owner-controlled with mode 0700"
            )

    @contextmanager
    def locked(self) -> Iterator["ReceiptStore"]:
        self._validate_parent()
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with _ReceiptLock(lock_path):
            yield self

    def exists(self) -> bool:
        return self.path.exists() or self.path.is_symlink()

    def create(self, payload: dict[str, Any]) -> None:
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
            raise LightTestNoteError("the create receipt already exists") from None
        except OSError:
            raise LightTestNoteError("the test-note receipt could not be created") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def load(self) -> dict[str, Any]:
        raw = self._read()
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise LightTestNoteError("the test-note receipt is invalid") from None
        if not isinstance(payload, dict):
            raise LightTestNoteError("the test-note receipt is invalid")
        _validate_receipt(payload)
        if payload["status"] == "pending":
            active = dict(payload["active_intent"])
            active["outcome"] = "unknown"
            active["resolved_at"] = _timestamp()
            payload["active_intent"] = active
            payload["status"] = "unknown"
            payload["updated_at"] = active["resolved_at"]
            self.save(payload)
        return payload

    def save(self, payload: dict[str, Any]) -> None:
        self._verify_existing()
        encoded = self._encode(payload)
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
            self._fsync_parent()
        except LightTestNoteError:
            raise
        except OSError:
            raise LightTestNoteError("the test-note receipt could not be saved") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def _encode(self, payload: dict[str, Any]) -> bytes:
        _validate_receipt(payload)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        if len(encoded) > _MAX_RECEIPT_BYTES:
            raise LightTestNoteError("the test-note receipt is too large")
        return encoded

    def _read(self) -> bytes:
        self._verify_existing()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                metadata = os.fstat(stream.fileno())
                if not _secure_regular_file(metadata):
                    raise OSError
                raw = stream.read(_MAX_RECEIPT_BYTES + 1)
        except OSError:
            raise LightTestNoteError(
                "the test-note receipt is unavailable or unsafe"
            ) from None
        if len(raw) > _MAX_RECEIPT_BYTES:
            raise LightTestNoteError("the test-note receipt is too large")
        return raw

    def _verify_existing(self) -> None:
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            raise LightTestNoteError("the test-note receipt does not exist") from None
        if not _secure_regular_file(metadata):
            raise LightTestNoteError(
                "the test-note receipt must be owner-only with mode 0600"
            )

    def _fsync_parent(self) -> None:
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _secure_regular_file(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o600
        and metadata.st_nlink == 1
        and (not hasattr(os, "geteuid") or metadata.st_uid == os.geteuid())
    )


def _validate_intent(intent: object, *, run_id: str) -> None:
    if not isinstance(intent, dict):
        raise LightTestNoteError("the test-note receipt is invalid")
    if set(intent) != _INTENT_FIELDS:
        raise LightTestNoteError("the test-note receipt is invalid")
    operation = intent.get("operation")
    if operation not in {"create", "update", "delete"}:
        raise LightTestNoteError("the test-note receipt is invalid")
    if type(intent.get("sequence")) is not int or intent["sequence"] < 1:
        raise LightTestNoteError("the test-note receipt is invalid")
    if not _RUN_ID_RE.fullmatch(str(intent.get("operation_id", ""))):
        raise LightTestNoteError("the test-note receipt is invalid")
    if intent.get("outcome") not in {
        "pending",
        "unknown",
        "conflict",
        "verified",
        "resolved_applied",
        "resolved_not_applied",
    }:
        raise LightTestNoteError("the test-note receipt is invalid")
    if not isinstance(intent.get("started_at"), str):
        raise LightTestNoteError("the test-note receipt is invalid")
    if intent.get("resolved_at") is not None and not isinstance(
        intent.get("resolved_at"), str
    ):
        raise LightTestNoteError("the test-note receipt is invalid")

    previous_revision = intent.get("previous_revision")
    if operation == "create":
        if any(
            intent.get(field) is not None
            for field in (
                "previous_revision",
                "previous_title",
                "previous_content_sha256",
            )
        ):
            raise LightTestNoteError("the test-note receipt is invalid")
    else:
        if type(previous_revision) is not int or previous_revision < 0:
            raise LightTestNoteError("the test-note receipt is invalid")
        if intent.get("previous_title") != _title(run_id, previous_revision):
            raise LightTestNoteError("the test-note receipt is invalid")
        if intent.get("previous_content_sha256") != _digest(
            _body(run_id, previous_revision)
        ):
            raise LightTestNoteError("the test-note receipt is invalid")

    desired_revision = intent.get("desired_revision")
    if type(desired_revision) is not int or desired_revision < 0:
        raise LightTestNoteError("the test-note receipt is invalid")
    if intent.get("desired_title") != _title(run_id, desired_revision):
        raise LightTestNoteError("the test-note receipt is invalid")
    if intent.get("desired_content_sha256") != _digest(
        _body(run_id, desired_revision)
    ):
        raise LightTestNoteError("the test-note receipt is invalid")


def _validate_receipt(payload: dict[str, Any]) -> None:
    if set(payload) != _RECEIPT_FIELDS:
        raise LightTestNoteError("the test-note receipt is invalid")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise LightTestNoteError("the test-note receipt has an unsupported schema")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise LightTestNoteError("the test-note receipt is invalid")
    revision = payload.get("revision")
    if type(revision) is not int or revision < 0:
        raise LightTestNoteError("the test-note receipt is invalid")
    if payload.get("expected_title") != _title(run_id, revision):
        raise LightTestNoteError("the test-note receipt is invalid")
    if payload.get("expected_content_sha256") != _digest(_body(run_id, revision)):
        raise LightTestNoteError("the test-note receipt is invalid")
    if not _valid_digest(payload.get("account_binding")):
        raise LightTestNoteError("the test-note receipt is invalid")
    if (
        type(payload.get("intent_sequence")) is not int
        or payload["intent_sequence"] < 0
    ):
        raise LightTestNoteError("the test-note receipt is invalid")
    if not isinstance(payload.get("created_at"), str) or not isinstance(
        payload.get("updated_at"), str
    ):
        raise LightTestNoteError("the test-note receipt is invalid")
    status = payload.get("status")
    if status not in {"pending", "ready", "unknown", "conflict", "deleted"}:
        raise LightTestNoteError("the test-note receipt is invalid")
    note_id = payload.get("note_id")
    if note_id is not None and not _valid_note_id(note_id):
        raise LightTestNoteError("the test-note receipt is invalid")
    if status in {"ready", "deleted"} and note_id is None:
        raise LightTestNoteError("the test-note receipt is invalid")
    active = payload.get("active_intent")
    if status in {"pending", "unknown", "conflict"}:
        _validate_intent(active, run_id=run_id)
        if active["sequence"] != payload["intent_sequence"]:
            raise LightTestNoteError("the test-note receipt is invalid")
        expected_outcome = status
        if active.get("outcome") != expected_outcome:
            raise LightTestNoteError("the test-note receipt is invalid")
        if status == "pending" and active["resolved_at"] is not None:
            raise LightTestNoteError("the test-note receipt is invalid")
        if status in {"unknown", "conflict"} and not isinstance(
            active["resolved_at"], str
        ):
            raise LightTestNoteError("the test-note receipt is invalid")
        operation = active["operation"]
        if operation == "create":
            if revision != 0 or active["desired_revision"] != 0 or note_id is not None:
                raise LightTestNoteError("the test-note receipt is invalid")
        else:
            if note_id is None or active["previous_revision"] != revision:
                raise LightTestNoteError("the test-note receipt is invalid")
            if operation == "update" and active["desired_revision"] != revision + 1:
                raise LightTestNoteError("the test-note receipt is invalid")
            if operation == "delete" and active["desired_revision"] != revision:
                raise LightTestNoteError("the test-note receipt is invalid")
    elif active is not None:
        raise LightTestNoteError("the test-note receipt is invalid")
    last = payload.get("last_intent")
    if last is not None:
        _validate_intent(last, run_id=run_id)
        if last["outcome"] not in {
            "verified",
            "resolved_applied",
            "resolved_not_applied",
        }:
            raise LightTestNoteError("the test-note receipt is invalid")
        if active is None and last["sequence"] != payload["intent_sequence"]:
            raise LightTestNoteError("the test-note receipt is invalid")
        if active is not None and last["sequence"] >= payload["intent_sequence"]:
            raise LightTestNoteError("the test-note receipt is invalid")
    elif status == "ready":
        raise LightTestNoteError("the test-note receipt is invalid")
    if status == "deleted" and (
        last is None
        or last["operation"] != "delete"
        or last["outcome"] not in {"verified", "resolved_applied"}
        or last["previous_revision"] != revision
        or last["desired_revision"] != revision
    ):
        raise LightTestNoteError("the test-note receipt is invalid")
    if status == "ready" and last is not None and last["operation"] == "delete":
        raise LightTestNoteError("the test-note receipt is invalid")


def _new_intent(
    receipt: dict[str, Any],
    operation: str,
    *,
    desired_revision: int,
) -> dict[str, Any]:
    previous_revision = None if operation == "create" else receipt["revision"]
    return {
        "sequence": receipt["intent_sequence"] + 1,
        "operation_id": uuid.uuid4().hex,
        "operation": operation,
        "outcome": "pending",
        "started_at": _timestamp(),
        "resolved_at": None,
        "previous_revision": previous_revision,
        "previous_title": (
            None if previous_revision is None else _title(receipt["run_id"], previous_revision)
        ),
        "previous_content_sha256": (
            None
            if previous_revision is None
            else _digest(_body(receipt["run_id"], previous_revision))
        ),
        "desired_revision": desired_revision,
        "desired_title": _title(receipt["run_id"], desired_revision),
        "desired_content_sha256": _digest(
            _body(receipt["run_id"], desired_revision)
        ),
    }


def _set_active(
    store: ReceiptStore,
    receipt: dict[str, Any],
    intent: dict[str, Any],
    outcome: str,
) -> None:
    intent = dict(intent)
    intent["outcome"] = outcome
    if outcome != "pending":
        intent["resolved_at"] = _timestamp()
    receipt["intent_sequence"] = intent["sequence"]
    receipt["active_intent"] = intent
    receipt["status"] = outcome
    receipt["updated_at"] = _timestamp()
    store.save(receipt)


def _complete(
    store: ReceiptStore,
    receipt: dict[str, Any],
    intent: dict[str, Any],
    outcome: str,
    *,
    status: str,
    note_id: str | None = None,
    revision: int | None = None,
) -> None:
    completed = dict(intent)
    completed["outcome"] = outcome
    completed["resolved_at"] = _timestamp()
    receipt["intent_sequence"] = completed["sequence"]
    receipt["active_intent"] = None
    receipt["last_intent"] = completed
    receipt["status"] = status
    if note_id is not None:
        receipt["note_id"] = note_id
    if revision is not None:
        receipt["revision"] = revision
        receipt["expected_title"] = _title(receipt["run_id"], revision)
        receipt["expected_content_sha256"] = _digest(
            _body(receipt["run_id"], revision)
        )
    receipt["updated_at"] = completed["resolved_at"]
    store.save(receipt)


def _provider(
    credentials: tuple[str, str, str, str],
    *,
    write_policy: LightWritePolicy,
    owned_note_ids: tuple[str, ...] = (),
) -> GaradoLightProvider:
    username, password, device_id, _ = credentials
    return GaradoLightProvider(
        username=username,
        password=password,
        device_id=device_id,
        write_policy=write_policy,
        test_title_prefix=TEST_TITLE_PREFIX,
        owned_test_note_ids=owned_note_ids,
    )


@contextmanager
def _opened_provider(provider: GaradoLightProvider) -> Iterator[GaradoLightProvider]:
    try:
        yield provider
    finally:
        provider.close()


def _check_binding(receipt: dict[str, Any], binding: str) -> None:
    if receipt["account_binding"] != binding:
        raise LightTestNoteError(
            "the receipt does not belong to the configured Light account and device"
        )


def _observe(
    provider: GaradoLightProvider, note_id: str
) -> tuple[RemoteNote | None, str | None]:
    note = provider.get_note(note_id)
    if note is None or note.kind is not NoteKind.TEXT:
        return note, None
    return note, _digest(provider.read_note_content(note_id))


def _matches(
    note: RemoteNote | None,
    observed_hash: str | None,
    *,
    note_id: str,
    title: str,
    content_sha256: str,
) -> bool:
    return (
        note is not None
        and note.note_id == note_id
        and note.kind is NoteKind.TEXT
        and note.title == title
        and observed_hash == content_sha256
    )


def _candidate_observations(
    provider: GaradoLightProvider, *, title: str, desired_hash: str
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for note in provider.list_notes():
        if note.title != title:
            continue
        note_id = _require_note_id(note.note_id)
        observed_hash = None
        if note.kind is NoteKind.TEXT:
            observed_hash = _digest(provider.read_note_content(note_id))
        candidates.append(
            {
                "note_id": note_id,
                "content_matches": observed_hash == desired_hash,
                "text_note": note.kind is NoteKind.TEXT,
            }
        )
    return sorted(candidates, key=lambda value: str(value["note_id"]))


def _local_summary(receipt: dict[str, Any]) -> dict[str, object]:
    active = receipt.get("active_intent")
    result: dict[str, object] = {
        "ok": receipt["status"] not in {"unknown", "conflict"},
        "status": receipt["status"],
        "run_id": receipt["run_id"],
        "revision": receipt["revision"],
        "note_id": receipt["note_id"],
        "content_sha256": receipt["expected_content_sha256"],
    }
    if active is not None:
        result.update(
            {
                "operation": active["operation"],
                "operation_id": active["operation_id"],
                "operation_outcome": active["outcome"],
            }
        )
        if active["desired_content_sha256"] is not None:
            # This is the hash of generated synthetic content, never a hash of
            # user-edited remote content.
            result["desired_content_sha256"] = active[
                "desired_content_sha256"
            ]
    return result


def _delete_summary(receipt: dict[str, Any]) -> dict[str, object]:
    """Return deletion state without exposing the receipt's private identity."""

    deleted = receipt["status"] == "deleted"
    return {
        "ok": deleted,
        "status": receipt["status"],
        "source_note_deleted": deleted,
        "absence_observations": 2 if deleted else 0,
        "exact_identity_authority": True,
        "title_deletion_authority": False,
        "delete_retried": False,
    }


def create_test_note(
    *, receipt_path: Path, confirm: bool, environ: Mapping[str, str]
) -> dict[str, object]:
    if not confirm:
        raise LightTestNoteError("create requires --confirm-create")
    credentials = _credentials(environ)
    store = ReceiptStore(receipt_path)
    with store.locked():
        if store.exists():
            raise LightTestNoteError("the create receipt already exists")
        provider = _provider(credentials, write_policy=LightWritePolicy.TEST_NOTES)
        with _opened_provider(provider):
            # Authenticate and establish read compatibility before recording a
            # mutation intent. This call cannot mutate Light Cloud.
            provider.list_notes()
            run_id = uuid.uuid4().hex
            revision = 0
            receipt: dict[str, Any] = {
                "schema_version": _SCHEMA_VERSION,
                "account_binding": credentials[3],
                "run_id": run_id,
                "note_id": None,
                "revision": revision,
                "expected_title": _title(run_id, revision),
                "expected_content_sha256": _digest(_body(run_id, revision)),
                "status": "pending",
                "intent_sequence": 0,
                "active_intent": None,
                "last_intent": None,
                "created_at": _timestamp(),
                "updated_at": _timestamp(),
            }
            intent = _new_intent(receipt, "create", desired_revision=revision)
            receipt["intent_sequence"] = intent["sequence"]
            receipt["active_intent"] = intent
            store.create(receipt)
            try:
                result = provider.create_text_note(
                    title=intent["desired_title"],
                    content=_body(run_id, revision),
                )
            except BaseException:
                try:
                    _set_active(store, receipt, intent, "unknown")
                finally:
                    raise
            if (
                result.outcome is MutationOutcome.VERIFIED
                and result.note is not None
                and result.note.kind is NoteKind.TEXT
                and result.note.title == intent["desired_title"]
                and result.observed_content_sha256
                == intent["desired_content_sha256"]
            ):
                note_id = _require_note_id(result.note.note_id)
                _complete(
                    store,
                    receipt,
                    intent,
                    "verified",
                    status="ready",
                    note_id=note_id,
                    revision=revision,
                )
            else:
                _set_active(store, receipt, intent, "unknown")
        return _local_summary(receipt)


def _require_ready(receipt: dict[str, Any]) -> str:
    if receipt["status"] != "ready" or receipt["active_intent"] is not None:
        raise LightTestNoteError(
            "the receipt is not ready for another cloud mutation"
        )
    return _require_note_id(receipt["note_id"])


def _check_operator_expectations(
    receipt: dict[str, Any], *, note_id: str, content_sha256: str
) -> None:
    if note_id != receipt["note_id"]:
        raise LightTestNoteError("the expected note identifier does not match the receipt")
    if content_sha256 != receipt["expected_content_sha256"]:
        raise LightTestNoteError("the expected content hash does not match the receipt")


def update_test_note(
    *,
    receipt_path: Path,
    expected_note_id: str,
    expected_content_sha256: str,
    confirm: bool,
    environ: Mapping[str, str],
) -> dict[str, object]:
    if not confirm:
        raise LightTestNoteError("update requires --confirm-update")
    if not _valid_note_id(expected_note_id) or not _valid_digest(
        expected_content_sha256
    ):
        raise LightTestNoteError("update expectations are invalid")
    credentials = _credentials(environ)
    store = ReceiptStore(receipt_path)
    with store.locked():
        receipt = store.load()
        _check_binding(receipt, credentials[3])
        note_id = _require_ready(receipt)
        _check_operator_expectations(
            receipt,
            note_id=expected_note_id,
            content_sha256=expected_content_sha256,
        )
        desired_revision = receipt["revision"] + 1
        intent = _new_intent(receipt, "update", desired_revision=desired_revision)
        provider = _provider(
            credentials,
            write_policy=LightWritePolicy.TEST_NOTES,
            owned_note_ids=(note_id,),
        )
        with _opened_provider(provider):
            note, observed_hash = _observe(provider, note_id)
            if not _matches(
                note,
                observed_hash,
                note_id=note_id,
                title=intent["previous_title"],
                content_sha256=intent["previous_content_sha256"],
            ):
                _set_active(store, receipt, intent, "conflict")
                return _local_summary(receipt)
            _set_active(store, receipt, intent, "pending")
            try:
                result = provider.update_text_note(
                    note_id,
                    title=intent["desired_title"],
                    content=_body(receipt["run_id"], desired_revision),
                    expected_title=intent["previous_title"],
                    expected_content_sha256=intent["previous_content_sha256"],
                )
            except BaseException:
                try:
                    _set_active(store, receipt, intent, "unknown")
                finally:
                    raise
            if (
                result.outcome is MutationOutcome.VERIFIED
                and result.note is not None
                and result.note.note_id == note_id
                and result.note.kind is NoteKind.TEXT
                and result.note.title == intent["desired_title"]
                and result.observed_content_sha256
                == intent["desired_content_sha256"]
            ):
                _complete(
                    store,
                    receipt,
                    intent,
                    "verified",
                    status="ready",
                    revision=desired_revision,
                )
            elif result.outcome is MutationOutcome.CONFLICT and result.detail in {
                "note_missing",
                "note_is_not_text",
                "title_changed",
                "content_changed",
            }:
                _set_active(store, receipt, intent, "conflict")
            else:
                _set_active(store, receipt, intent, "unknown")
        return _local_summary(receipt)


def _reconcile_delete(
    provider: GaradoLightProvider,
    receipt: dict[str, Any],
    *,
    note_id: str,
) -> str:
    """Observe an attempted delete twice without crossing a mutation boundary."""

    for _ in range(2):
        note, observed_hash = _observe(provider, note_id)
        if note is None:
            continue
        if _matches(
            note,
            observed_hash,
            note_id=note_id,
            title=receipt["expected_title"],
            content_sha256=receipt["expected_content_sha256"],
        ):
            return "present"
        return "conflict"
    return "absent"


def delete_test_note(
    *,
    receipt_path: Path,
    expected_note_id: str,
    expected_content_sha256: str,
    confirm: bool,
    environ: Mapping[str, str],
) -> dict[str, object]:
    """Delete one exact receipt-owned test note with at-most-once mutation."""

    if not confirm:
        raise LightTestNoteError("delete requires --confirm-delete")
    if not _valid_note_id(expected_note_id) or not _valid_digest(
        expected_content_sha256
    ):
        raise LightTestNoteError("delete expectations are invalid")
    credentials = _credentials(environ)
    store = ReceiptStore(receipt_path)
    with store.locked():
        receipt = store.load()
        _check_binding(receipt, credentials[3])
        _check_operator_expectations(
            receipt,
            note_id=expected_note_id,
            content_sha256=expected_content_sha256,
        )
        note_id = _require_note_id(receipt["note_id"])
        if receipt["status"] == "deleted":
            return _delete_summary(receipt)

        active = receipt.get("active_intent")
        if active is not None:
            if active["operation"] != "delete" or receipt["status"] not in {
                "unknown",
                "conflict",
            }:
                raise LightTestNoteError(
                    "the receipt is not ready for test-note deletion"
                )
            provider = _provider(credentials, write_policy=LightWritePolicy.READ_ONLY)
            with _opened_provider(provider):
                observed = _reconcile_delete(provider, receipt, note_id=note_id)
                if observed == "absent":
                    _complete(
                        store,
                        receipt,
                        active,
                        "resolved_applied",
                        status="deleted",
                    )
                elif observed == "conflict" and receipt["status"] != "conflict":
                    _set_active(store, receipt, active, "conflict")
            return _delete_summary(receipt)

        note_id = _require_ready(receipt)
        intent = _new_intent(
            receipt,
            "delete",
            desired_revision=receipt["revision"],
        )
        provider = _provider(
            credentials,
            write_policy=LightWritePolicy.TEST_NOTES,
            owned_note_ids=(note_id,),
        )
        with _opened_provider(provider):
            note, observed_hash = _observe(provider, note_id)
            if note is not None and not _matches(
                note,
                observed_hash,
                note_id=note_id,
                title=receipt["expected_title"],
                content_sha256=receipt["expected_content_sha256"],
            ):
                _set_active(store, receipt, intent, "conflict")
                return _delete_summary(receipt)

            # Persist the at-most-once boundary before any DELETE could begin.
            _set_active(store, receipt, intent, "pending")
            if note is not None:
                try:
                    deletion = provider.delete_test_note(
                        note_id,
                        expected_title=receipt["expected_title"],
                        expected_content_sha256=receipt["expected_content_sha256"],
                    )
                except BaseException:
                    try:
                        _set_active(store, receipt, intent, "unknown")
                    finally:
                        raise
                if deletion.outcome is MutationOutcome.CONFLICT:
                    _set_active(store, receipt, intent, "conflict")
                    return _delete_summary(receipt)
            _set_active(store, receipt, intent, "unknown")
            observed = _reconcile_delete(provider, receipt, note_id=note_id)
            if observed == "absent":
                _complete(
                    store,
                    receipt,
                    intent,
                    "verified",
                    status="deleted",
                )
            elif observed == "conflict":
                _set_active(store, receipt, intent, "conflict")
        return _delete_summary(receipt)


def inspect_test_note(
    *, receipt_path: Path, environ: Mapping[str, str]
) -> dict[str, object]:
    credentials = _credentials(environ)
    store = ReceiptStore(receipt_path)
    with store.locked():
        receipt = store.load()
        _check_binding(receipt, credentials[3])
        if receipt["status"] == "deleted":
            return _delete_summary(receipt)
        provider = _provider(credentials, write_policy=LightWritePolicy.READ_ONLY)
        with _opened_provider(provider):
            result = _local_summary(receipt)
            active = receipt.get("active_intent")
            if receipt["note_id"] is None and active is not None:
                desired_hash = active["desired_content_sha256"]
                candidates = _candidate_observations(
                    provider,
                    title=active["desired_title"],
                    desired_hash=desired_hash,
                )
                result["candidate_count"] = len(candidates)
                result["candidates"] = candidates
                result["ok"] = False
                return result
            if receipt["note_id"] is None:
                result["remote_state"] = "not_owned"
                result["ok"] = False
                return result
            note_id = _require_note_id(receipt["note_id"])
            note, observed_hash = _observe(provider, note_id)
            if note is None:
                result["remote_state"] = "absent"
            elif note.kind is not NoteKind.TEXT:
                result["remote_state"] = "wrong_kind"
            else:
                result["remote_state"] = (
                    "expected"
                    if note.title == receipt["expected_title"]
                    and observed_hash == receipt["expected_content_sha256"]
                    else "changed"
                )
                result["expected_title_matches"] = note.title == receipt["expected_title"]
                result["expected_content_matches"] = (
                    observed_hash == receipt["expected_content_sha256"]
                )
                active = receipt.get("active_intent")
                if active is not None and active["operation"] == "update":
                    result["desired_state_matches"] = _matches(
                        note,
                        observed_hash,
                        note_id=note_id,
                        title=active["desired_title"],
                        content_sha256=active["desired_content_sha256"],
                    )
            result["ok"] = (
                receipt["status"] == "ready"
                and result["remote_state"] == "expected"
            )
            return result


def resolve_test_note(
    *,
    receipt_path: Path,
    decision: str,
    confirm_run_id: str,
    observed_note_id: str | None,
    observed_content_sha256: str | None,
    environ: Mapping[str, str],
) -> dict[str, object]:
    if decision not in {"applied", "not_applied"}:
        raise LightTestNoteError("the resolution decision is invalid")
    credentials = _credentials(environ)
    store = ReceiptStore(receipt_path)
    with store.locked():
        receipt = store.load()
        _check_binding(receipt, credentials[3])
        if confirm_run_id != receipt["run_id"]:
            raise LightTestNoteError("resolution requires the exact receipt run ID")
        if receipt["status"] not in {"unknown", "conflict"}:
            raise LightTestNoteError("the receipt has no unresolved mutation")
        intent = receipt["active_intent"]
        operation = intent["operation"]
        if operation == "create" and decision == "not_applied":
            # Cloud absence is not authoritative: a create may become visible
            # after an empty listing. Preserve the unresolved intent forever
            # rather than creating an unowned note.
            raise LightTestNoteError(
                "an uncertain create cannot be resolved as not applied"
            )
        provider = _provider(credentials, write_policy=LightWritePolicy.READ_ONLY)
        with _opened_provider(provider):
            if operation == "create":
                candidates = _candidate_observations(
                    provider,
                    title=intent["desired_title"],
                    desired_hash=intent["desired_content_sha256"],
                )
                exact = [value for value in candidates if value["content_matches"]]
                if decision == "applied":
                    if not _valid_note_id(observed_note_id) or not _valid_digest(
                        observed_content_sha256
                    ):
                        raise LightTestNoteError(
                            "applied create resolution requires observed identifiers"
                        )
                    if observed_content_sha256 != intent["desired_content_sha256"]:
                        raise LightTestNoteError(
                            "the observed content hash does not match the intent"
                        )
                    if len(candidates) != 1 or len(exact) != 1:
                        raise LightTestNoteError(
                            "the create result is not uniquely verified"
                        )
                    selected = exact[0]
                    if selected["note_id"] != observed_note_id:
                        raise LightTestNoteError(
                            "the selected create result was not verified"
                        )
                    _complete(
                        store,
                        receipt,
                        intent,
                        "resolved_applied",
                        status="ready",
                        note_id=observed_note_id,
                        revision=intent["desired_revision"],
                    )
                else:  # pragma: no cover - rejected before provider construction
                    raise AssertionError("unsupported create resolution")
            else:
                if operation != "update":
                    raise LightTestNoteError(
                        "the unresolved test-note operation is unsupported"
                    )
                note_id = _require_note_id(receipt["note_id"])
                note, observed_hash = _observe(provider, note_id)
                if observed_note_id is not None and observed_note_id != note_id:
                    raise LightTestNoteError(
                        "the observed note identifier does not match the receipt"
                    )
                if observed_content_sha256 is not None and (
                    not _valid_digest(observed_content_sha256)
                    or observed_content_sha256 != observed_hash
                ):
                    raise LightTestNoteError(
                        "the observed content hash does not match Light Cloud"
                    )
                if operation == "update" and decision == "applied":
                    if observed_note_id != note_id or not _valid_digest(
                        observed_content_sha256
                    ):
                        raise LightTestNoteError(
                            "applied update resolution requires observed identifiers"
                        )
                    if not _matches(
                        note,
                        observed_hash,
                        note_id=note_id,
                        title=intent["desired_title"],
                        content_sha256=intent["desired_content_sha256"],
                    ):
                        raise LightTestNoteError(
                            "the desired update state was not verified"
                        )
                    _complete(
                        store,
                        receipt,
                        intent,
                        "resolved_applied",
                        status="ready",
                        revision=intent["desired_revision"],
                    )
                elif decision == "not_applied":
                    if not _matches(
                        note,
                        observed_hash,
                        note_id=note_id,
                        title=intent["previous_title"],
                        content_sha256=intent["previous_content_sha256"],
                    ):
                        raise LightTestNoteError(
                            "the previous note state was not verified"
                        )
                    _complete(
                        store,
                        receipt,
                        intent,
                        "resolved_not_applied",
                        status="ready",
                    )
                else:  # pragma: no cover - parser and operation validation constrain this
                    raise LightTestNoteError("the resolution decision is incompatible")
        return _local_summary(receipt)


def run_command(args: Any, environ: Mapping[str, str]) -> dict[str, object]:
    """Dispatch an already parsed ``light-test-note`` command."""

    common = {"receipt_path": args.receipt, "environ": environ}
    if args.light_test_note_command == "create":
        return create_test_note(confirm=args.confirm_create, **common)
    if args.light_test_note_command == "inspect":
        return inspect_test_note(**common)
    if args.light_test_note_command == "update":
        return update_test_note(
            expected_note_id=args.expected_note_id,
            expected_content_sha256=args.expected_content_sha256,
            confirm=args.confirm_update,
            **common,
        )
    if args.light_test_note_command == "delete":
        return delete_test_note(
            expected_note_id=args.expected_note_id,
            expected_content_sha256=args.expected_content_sha256,
            confirm=args.confirm_delete,
            **common,
        )
    if args.light_test_note_command == "resolve":
        return resolve_test_note(
            decision=args.decision,
            confirm_run_id=args.confirm_run_id,
            observed_note_id=args.observed_note_id,
            observed_content_sha256=args.observed_content_sha256,
            **common,
        )
    raise AssertionError("unknown light-test-note command")
