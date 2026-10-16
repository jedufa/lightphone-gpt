"""Adapter for the unofficial ``garado/light`` Light Cloud client.

The dependency is imported lazily and all mutations are disabled unless the
caller selects an explicit write policy. The default is safe for cloud
inventory and audio-download compatibility checks.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import Lock
from typing import Any
from urllib.parse import quote

from lightphone_gpt.contracts import PermanentDependencyError
from lightphone_gpt.models import (
    DeletionReceipt,
    MutationOutcome,
    MutationReceipt,
    NoteKind,
    RemoteNote,
)
from lightphone_gpt.output_titles import is_output_title

from .errors import LightAdapterError, LightWriteDisabledError

GARADO_LIGHT_COMMIT = "06620db5ac1068ce7be36c4994f5e793b8b3ead0"
GARADO_LIGHT_VERSION = "0.6.0"


class _LightSessionExpired(RuntimeError):
    """Internal marker for a rejected authenticated Light API request."""


@dataclass(frozen=True, slots=True)
class _NativeNoteWithFileMetadata:
    """The pinned client's note fields plus file metadata it normally drops."""

    id: str
    file_id: str
    note_type: str
    title: str
    updated_at: str
    content_type: str | None
    storage_key: str | None


class LightWritePolicy(StrEnum):
    READ_ONLY = "read_only"
    TEST_NOTES = "test_notes"
    BRIDGE_OUTPUTS = "bridge_outputs"


class GaradoLightProvider:
    """Normalize garado/light notes and add conservative mutation receipts."""

    def __init__(
        self,
        *,
        username: str | None = None,
        password: str | None = None,
        device_id: str | None = None,
        write_policy: LightWritePolicy | str = LightWritePolicy.READ_ONLY,
        test_title_prefix: str = "LIGHTPHONE GPT TEST",
        owned_test_note_ids: Iterable[str] = (),
        output_title: str = "AI READY",
        owned_output_note_ids: Iterable[str] = (),
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if client is None and (not username or not password):
            raise ValueError("username and password are required without a client")
        if not test_title_prefix:
            raise ValueError("test_title_prefix must be non-empty")
        if (
            not output_title
            or output_title != output_title.strip()
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in output_title
            )
            or len(output_title) > 94
        ):
            raise ValueError("output_title must be non-empty and trimmed")
        self._username = username
        self._password = password
        self._device_id = device_id
        self._write_policy = LightWritePolicy(write_policy)
        self._test_title_prefix = test_title_prefix
        self._owned_test_note_ids = set(owned_test_note_ids)
        self._output_title = output_title
        self._owned_output_note_ids = set(owned_output_note_ids)
        self._client = client
        self._client_factory = client_factory
        self._owns_client = client is None
        self._entered = client is not None
        self._lock = Lock()

    @classmethod
    def from_environment(
        cls,
        *,
        username_variable: str = "lightphone_username",
        password_variable: str = "lightphone_password",
        device_id_variable: str = "lightphone_device_id",
        **kwargs: Any,
    ) -> "GaradoLightProvider":
        username = os.environ.get(username_variable)
        password = os.environ.get(password_variable)
        if not username or not password:
            raise PermanentDependencyError(
                "Light credentials are missing from the process environment"
            )
        device_id = os.environ.get(device_id_variable) or None
        return cls(
            username=username,
            password=password,
            device_id=device_id,
            **kwargs,
        )

    def list_notes(self) -> Sequence[RemoteNote]:
        try:
            native_notes = self._authenticated_read(self._native_notes_once)
        except (PermanentDependencyError, LightAdapterError):
            raise
        except Exception:
            raise LightAdapterError("Light note listing failed") from None
        try:
            return [self._normalize(note) for note in native_notes]
        except Exception:
            raise PermanentDependencyError(
                "Light returned unsupported note metadata"
            ) from None

    def get_note(self, note_id: str) -> RemoteNote | None:
        try:
            native = self._authenticated_read(
                lambda: self._find_native_note_once(note_id)
            )
        except (PermanentDependencyError, LightAdapterError):
            raise
        except Exception:
            raise LightAdapterError("Light note lookup failed") from None
        if native is None:
            return None
        try:
            return self._normalize(native)
        except Exception:
            raise PermanentDependencyError(
                "Light returned unsupported note metadata"
            ) from None

    def read_note_content(self, note_id: str) -> bytes:
        try:
            return self._authenticated_read(
                lambda: self._read_note_content_once(note_id)
            )
        except PermanentDependencyError:
            raise
        except Exception:
            raise LightAdapterError("Light note download failed") from None

    def read_text_with_metadata(
        self, note_id: str
    ) -> tuple[RemoteNote, bytes]:
        """Return one text payload with metadata from the same lookup."""

        try:
            note, content = self._authenticated_read(
                lambda: self._read_typed_note_once(note_id, NoteKind.TEXT)
            )
        except PermanentDependencyError:
            raise
        except Exception:
            raise LightAdapterError("Light text note download failed") from None
        return note, content

    def download_audio(self, note_id: str) -> bytes:
        _note, content = self.download_audio_with_metadata(note_id)
        return content

    def download_audio_with_metadata(
        self, note_id: str
    ) -> tuple[RemoteNote, bytes]:
        """Return one audio payload with metadata from the same lookup."""

        try:
            note, content = self._authenticated_read(
                lambda: self._read_typed_note_once(note_id, NoteKind.AUDIO)
            )
        except PermanentDependencyError:
            raise
        except Exception:
            raise LightAdapterError("Light audio download failed") from None
        return note, content

    def create_text_note(self, *, title: str, content: str) -> MutationReceipt:
        self._guard_create(title)
        notes = self._notes_api()
        native: Any | None = None
        try:
            native = notes.create_text_note(title, content)
        except Exception:
            return MutationReceipt(
                MutationOutcome.UNKNOWN,
                detail="create_outcome_unknown",
            )

        note = self._normalize_or_none(native)
        if note is None:
            return MutationReceipt(
                MutationOutcome.UNKNOWN,
                detail="create_returned_no_note",
            )
        receipt = self._verify_mutation(note.note_id, title=title, content=content)
        if (
            self._write_policy is LightWritePolicy.TEST_NOTES
            and receipt.outcome is MutationOutcome.VERIFIED
        ):
            self._owned_test_note_ids.add(note.note_id)
        if (
            self._write_policy is LightWritePolicy.BRIDGE_OUTPUTS
            and receipt.outcome is MutationOutcome.VERIFIED
        ):
            # A verified create is safe to update for the lifetime of this
            # process. The worker persists the identity before accepting later
            # work; a restart reconstructs this allowlist only from that
            # durable record.
            self._owned_output_note_ids.add(note.note_id)
        return receipt

    def update_text_note(
        self,
        note_id: str,
        *,
        title: str,
        content: str,
        expected_title: str,
        expected_content_sha256: str,
    ) -> MutationReceipt:
        _validate_digest(expected_content_sha256)
        native = self._find_native_note(note_id)
        if native is None:
            return MutationReceipt(
                MutationOutcome.CONFLICT,
                detail="note_missing",
            )
        current = self._normalize(native)
        if current.kind is not NoteKind.TEXT:
            return MutationReceipt(
                MutationOutcome.CONFLICT,
                note=current,
                detail="note_is_not_text",
            )
        self._guard_existing(
            current,
            expected_title=expected_title,
            desired_title=title,
        )
        if current.title != expected_title:
            return MutationReceipt(
                MutationOutcome.CONFLICT,
                note=current,
                detail="title_changed",
            )

        try:
            old_content = self._authenticated_read(
                lambda: bytes(self._notes_api().get_note_content(native))
            )
        except Exception:
            raise LightAdapterError("Light note precheck failed") from None
        observed = hashlib.sha256(old_content).hexdigest()
        if observed != expected_content_sha256:
            return MutationReceipt(
                MutationOutcome.CONFLICT,
                note=current,
                observed_content_sha256=observed,
                detail="content_changed",
            )

        notes = self._notes_api()
        try:
            notes.update_note_content(native, content.encode("utf-8"))
            if current.title != title:
                # Refresh so the title PATCH carries the latest metadata.
                refreshed = notes.get_note_metadata(note_id)
                notes.update_note_title(refreshed, title)
        except Exception:
            verified = self._verify_after_uncertain_write(
                note_id, title=title, content=content
            )
            if verified is not None:
                return verified
            return MutationReceipt(
                MutationOutcome.UNKNOWN,
                note=self._safe_get_note(note_id),
                detail="update_outcome_unknown",
            )

        return self._verify_update_mutation(
            note_id,
            title=title,
            content=content,
            expected_title=expected_title,
            expected_content_sha256=expected_content_sha256,
        )

    def delete_source_note(
        self,
        note_id: str,
        *,
        expected_kind: NoteKind,
        expected_content_sha256: str,
    ) -> DeletionReceipt:
        """Delete one exact accepted source without ever retrying DELETE.

        All metadata/content checks occur before the mutation. The receipt
        explicitly says whether the DELETE boundary was crossed so the worker
        may safely retry a completed precheck failure while treating every
        post-boundary failure as ambiguous.
        """

        _validate_digest(expected_content_sha256)
        self._guard_source_delete(note_id)
        try:
            native = self._find_native_note(note_id)
            if native is None:
                return DeletionReceipt(
                    MutationOutcome.VERIFIED,
                    deletion_attempted=False,
                    detail="source_already_missing",
                )
            note = self._normalize(native)
            if note.kind is not expected_kind:
                return DeletionReceipt(
                    MutationOutcome.CONFLICT,
                    deletion_attempted=False,
                    note=note,
                    detail="source_kind_changed",
                )
            content = self._authenticated_read(
                lambda: bytes(self._notes_api().get_note_content(native))
            )
        except (PermanentDependencyError, LightAdapterError):
            return DeletionReceipt(
                MutationOutcome.UNKNOWN,
                deletion_attempted=False,
                detail="source_delete_precheck_unavailable",
            )
        except Exception:
            return DeletionReceipt(
                MutationOutcome.UNKNOWN,
                deletion_attempted=False,
                detail="source_delete_precheck_failed",
            )

        observed = hashlib.sha256(content).hexdigest()
        if not hmac.compare_digest(observed, expected_content_sha256):
            return DeletionReceipt(
                MutationOutcome.CONFLICT,
                deletion_attempted=False,
                note=note,
                observed_content_sha256=observed,
                detail="source_content_changed",
            )

        try:
            http_client, delete_url = self._prepare_source_delete(note_id)
        except Exception:
            return DeletionReceipt(
                MutationOutcome.UNKNOWN,
                deletion_attempted=False,
                note=note,
                observed_content_sha256=observed,
                detail="source_delete_transport_unavailable",
            )

        try:
            # This mutation deliberately bypasses both _authenticated_read and
            # garado/light's LightNotes.delete_note. The latter goes through
            # Light.call_api, which automatically repeats DELETE after a 401.
            # One direct request keeps the durable attempt boundary truthful:
            # every failure after this call begins is ambiguous and must be
            # reconciled only by reads.
            self._delete_note_once(http_client, delete_url)
        except Exception:
            return self._verify_source_absent_after_delete(
                note_id,
                fallback_detail="source_delete_outcome_unknown",
            )
        return self._verify_source_absent_after_delete(
            note_id,
            fallback_detail="source_delete_verification_pending",
        )

    def delete_test_note(
        self,
        note_id: str,
        *,
        expected_title: str,
        expected_content_sha256: str,
    ) -> DeletionReceipt:
        """Delete one exact operator-owned synthetic test note at most once."""

        _validate_digest(expected_content_sha256)
        if (
            self._write_policy is not LightWritePolicy.TEST_NOTES
            or note_id not in self._owned_test_note_ids
            or not self._is_test_title(expected_title)
        ):
            raise LightWriteDisabledError(
                "test-note deletion requires an exact owned test-note identity"
            )
        try:
            native = self._find_native_note(note_id)
            if native is None:
                return DeletionReceipt(
                    MutationOutcome.VERIFIED,
                    deletion_attempted=False,
                    detail="test_note_already_missing",
                )
            note = self._normalize(native)
            if note.kind is not NoteKind.TEXT:
                return DeletionReceipt(
                    MutationOutcome.CONFLICT,
                    deletion_attempted=False,
                    note=note,
                    detail="test_note_kind_changed",
                )
            if note.title != expected_title:
                return DeletionReceipt(
                    MutationOutcome.CONFLICT,
                    deletion_attempted=False,
                    note=note,
                    detail="test_note_title_changed",
                )
            self._guard_existing(
                note,
                expected_title=expected_title,
                desired_title=expected_title,
            )
            content = self._authenticated_read(
                lambda: bytes(self._notes_api().get_note_content(native))
            )
        except LightWriteDisabledError:
            raise
        except (PermanentDependencyError, LightAdapterError):
            return DeletionReceipt(
                MutationOutcome.UNKNOWN,
                deletion_attempted=False,
                detail="test_note_delete_precheck_unavailable",
            )
        except Exception:
            return DeletionReceipt(
                MutationOutcome.UNKNOWN,
                deletion_attempted=False,
                detail="test_note_delete_precheck_failed",
            )

        observed = hashlib.sha256(content).hexdigest()
        if not hmac.compare_digest(observed, expected_content_sha256):
            return DeletionReceipt(
                MutationOutcome.CONFLICT,
                deletion_attempted=False,
                note=note,
                observed_content_sha256=observed,
                detail="test_note_content_changed",
            )

        try:
            http_client, delete_url = self._prepare_source_delete(note_id)
        except Exception:
            return DeletionReceipt(
                MutationOutcome.UNKNOWN,
                deletion_attempted=False,
                note=note,
                observed_content_sha256=observed,
                detail="test_note_delete_transport_unavailable",
            )

        try:
            self._delete_note_once(http_client, delete_url)
        except Exception:
            return self._verify_source_absent_after_delete(
                note_id,
                fallback_detail="test_note_delete_outcome_unknown",
            )
        return self._verify_source_absent_after_delete(
            note_id,
            fallback_detail="test_note_delete_verification_pending",
        )

    def _prepare_source_delete(self, note_id: str) -> tuple[Any, str]:
        """Resolve the transport and URL before crossing the DELETE boundary."""

        client = self._ensure_client()
        api_client = getattr(client, "_api_client", None)
        get_httpx_client = getattr(api_client, "get_httpx_client", None)
        if not callable(get_httpx_client):
            raise PermanentDependencyError(
                "Light client does not expose a single-attempt HTTP transport"
            )
        http_client = get_httpx_client()
        if not callable(getattr(http_client, "request", None)):
            raise PermanentDependencyError(
                "Light client HTTP transport cannot issue requests"
            )
        return http_client, f"/api/notes/{quote(note_id, safe='')}"

    @staticmethod
    def _delete_note_once(http_client: Any, delete_url: str) -> None:
        """Issue exactly one application-level DELETE through a prepared client."""

        response = http_client.request(method="DELETE", url=delete_url)
        status_code = int(getattr(response, "status_code", 0))
        if status_code not in {200, 204}:
            raise LightAdapterError(f"Light source deletion returned {status_code}")

    def close(self) -> None:
        with self._lock:
            if self._owns_client and self._client is not None and self._entered:
                try:
                    self._client.__exit__(None, None, None)
                except Exception:
                    pass
                self._entered = False

    def __enter__(self) -> "GaradoLightProvider":
        self._ensure_client()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_client(self) -> Any:
        with self._lock:
            if self._client is None:
                factory = self._client_factory
                if factory is None:
                    try:
                        from light_api.client import Light
                    except ImportError:
                        raise PermanentDependencyError(
                            "garado/light is not installed"
                        ) from None
                    factory = Light
                try:
                    self._client = factory(
                        email=self._username,
                        password=self._password,
                        device_id=self._device_id,
                        cache_enabled=False,
                    )
                except Exception:
                    raise PermanentDependencyError(
                        "Light client configuration failed"
                    ) from None
            if not self._entered:
                try:
                    _disable_upstream_auth_cache(self._client)
                    _start_notes_only_session(self._client)
                    observed_email = _authenticated_email(self._client)
                    if not self._username or not hmac.compare_digest(
                        observed_email.strip().casefold(),
                        self._username.strip().casefold(),
                    ):
                        raise RuntimeError("authenticated Light account mismatch")
                except Exception:
                    raise LightAdapterError("Light authentication failed") from None
                self._entered = True
            return self._client

    def _notes_api(self) -> Any:
        client = self._ensure_client()
        try:
            return client.notes
        except AttributeError:
            raise PermanentDependencyError(
                "Light client does not expose Notes"
            ) from None

    def _native_notes_once(self) -> list[Any]:
        client = self._ensure_client()
        try:
            if _supports_raw_note_listing(client):
                return _fetch_native_notes_with_file_metadata(client)
            return list(client.notes.get_notes())
        except (PermanentDependencyError, LightAdapterError):
            raise
        except Exception as error:
            if _is_authenticated_session_expiry(error):
                raise _LightSessionExpired from None
            raise LightAdapterError("Light note listing failed") from None

    def _find_native_note(self, note_id: str) -> Any | None:
        try:
            return self._authenticated_read(
                lambda: self._find_native_note_once(note_id)
            )
        except (PermanentDependencyError, LightAdapterError):
            raise
        except Exception:
            raise LightAdapterError("Light note lookup failed") from None

    def _find_native_note_once(self, note_id: str) -> Any | None:
        return next(
            (
                note
                for note in self._native_notes_once()
                if str(note.id) == note_id
            ),
            None,
        )

    def _read_note_content_once(self, note_id: str) -> bytes:
        native = self._find_native_note_once(note_id)
        if native is None:
            raise PermanentDependencyError("Light note does not exist")
        return bytes(self._notes_api().get_note_content(native))

    def _read_typed_note_once(
        self, note_id: str, expected_kind: NoteKind
    ) -> tuple[RemoteNote, bytes]:
        native = self._find_native_note_once(note_id)
        if native is None:
            raise PermanentDependencyError(
                f"Light {expected_kind.value} note does not exist"
            )
        try:
            note = self._normalize(native)
        except Exception:
            raise PermanentDependencyError(
                "Light returned unsupported note metadata"
            ) from None
        if note.kind is not expected_kind:
            raise PermanentDependencyError(
                f"requested Light note is not {expected_kind.value}"
            )
        return note, bytes(self._notes_api().get_note_content(native))

    def _authenticated_read(self, operation: Callable[[], Any]) -> Any:
        """Retry one rejected read with a newly authenticated client.

        Only providers that own a credential-backed client can replace it. The
        caller supplies an entire read operation so metadata and payload reads
        are both repeated against the same fresh session. Mutation calls never
        pass through this helper.
        """

        failed_client = self._ensure_client()
        try:
            return operation()
        except Exception as error:
            if not _is_authenticated_session_expiry(error):
                raise
        if not self._replace_expired_client(failed_client):
            raise _LightSessionExpired from None
        return operation()

    def _replace_expired_client(self, failed_client: Any) -> bool:
        if not self._owns_client or not self._username or not self._password:
            return False
        with self._lock:
            if self._client is not failed_client:
                return True
            stale_client = self._client
            stale_entered = self._entered
            self._client = None
            self._entered = False
            if stale_client is not None and stale_entered:
                try:
                    stale_client.__exit__(None, None, None)
                except Exception:
                    pass
        self._ensure_client()
        return True

    def _verify_mutation(
        self, note_id: str, *, title: str, content: str
    ) -> MutationReceipt:
        desired = hashlib.sha256(content.encode("utf-8")).hexdigest()
        try:
            note = self.get_note(note_id)
            if note is None:
                return MutationReceipt(
                    MutationOutcome.CONFLICT,
                    detail="note_missing_after_write",
                )
            observed = hashlib.sha256(self.read_note_content(note_id)).hexdigest()
        except (LightAdapterError, PermanentDependencyError):
            return MutationReceipt(
                MutationOutcome.UNKNOWN,
                note=self._safe_get_note(note_id),
                detail="write_verification_failed",
            )
        if note.title == title and observed == desired:
            return MutationReceipt(
                MutationOutcome.VERIFIED,
                note=note,
                observed_content_sha256=observed,
            )
        return MutationReceipt(
            MutationOutcome.CONFLICT,
            note=note,
            observed_content_sha256=observed,
            detail="written_note_differs",
        )

    def _verify_after_uncertain_write(
        self, note_id: str, *, title: str, content: str
    ) -> MutationReceipt | None:
        receipt = self._verify_mutation(note_id, title=title, content=content)
        return receipt if receipt.outcome is MutationOutcome.VERIFIED else None

    def _verify_update_mutation(
        self,
        note_id: str,
        *,
        title: str,
        content: str,
        expected_title: str,
        expected_content_sha256: str,
    ) -> MutationReceipt:
        """Verify one update without treating a stale read as a conflict.

        Light stores note bytes through a presigned object upload, while note
        metadata and subsequent object reads can converge later. Once the PUT
        has been attempted, seeing the exact pre-write value (or a temporarily
        absent note) cannot prove that the write was not applied. Return an
        unknown receipt so the durable worker can perform bounded read-only
        confirmation without ever repeating the mutation.
        """

        receipt = self._verify_mutation(note_id, title=title, content=content)
        if receipt.outcome is not MutationOutcome.CONFLICT:
            return receipt
        note = receipt.note
        if receipt.detail == "note_missing_after_write" or (
            receipt.observed_content_sha256 == expected_content_sha256
            and note is not None
            and note.note_id == note_id
            and note.kind is NoteKind.TEXT
            and note.title in {expected_title, title}
        ):
            return MutationReceipt(
                MutationOutcome.UNKNOWN,
                note=note,
                observed_content_sha256=receipt.observed_content_sha256,
                detail="update_verification_pending",
            )
        return receipt

    def _safe_get_note(self, note_id: str) -> RemoteNote | None:
        try:
            return self.get_note(note_id)
        except Exception:
            return None

    def _verify_source_absent_after_delete(
        self, note_id: str, *, fallback_detail: str
    ) -> DeletionReceipt:
        try:
            note = self.get_note(note_id)
        except Exception:
            return DeletionReceipt(
                MutationOutcome.UNKNOWN,
                deletion_attempted=True,
                detail=fallback_detail,
            )
        if note is None:
            return DeletionReceipt(
                MutationOutcome.VERIFIED,
                deletion_attempted=True,
                detail="source_delete_verified",
            )
        return DeletionReceipt(
            MutationOutcome.UNKNOWN,
            deletion_attempted=True,
            note=note,
            detail=fallback_detail,
        )

    def _guard_create(self, title: str) -> None:
        if self._write_policy is LightWritePolicy.READ_ONLY:
            raise LightWriteDisabledError("Light writes are disabled")
        if (
            self._write_policy is LightWritePolicy.TEST_NOTES
            and not self._is_test_title(title)
        ):
            raise LightWriteDisabledError(
                "test mode only permits designated test-note titles"
            )
        if (
            self._write_policy is LightWritePolicy.BRIDGE_OUTPUTS
            and not is_output_title(title, self._output_title)
        ):
            raise LightWriteDisabledError(
                "bridge mode only permits the configured output title"
            )

    def _guard_source_delete(self, note_id: str) -> None:
        if self._write_policy is not LightWritePolicy.BRIDGE_OUTPUTS:
            raise LightWriteDisabledError(
                "source deletion requires the bridge runtime write policy"
            )
        if note_id in self._owned_output_note_ids:
            raise LightWriteDisabledError("a bridge output cannot be deleted as input")

    def _guard_existing(
        self,
        note: RemoteNote,
        *,
        expected_title: str,
        desired_title: str,
    ) -> None:
        if self._write_policy is LightWritePolicy.READ_ONLY:
            raise LightWriteDisabledError("Light writes are disabled")
        if self._write_policy is LightWritePolicy.TEST_NOTES and (
            note.note_id not in self._owned_test_note_ids
            or not self._is_test_title(note.title)
            or not self._is_test_title(expected_title)
            or not self._is_test_title(desired_title)
        ):
            raise LightWriteDisabledError(
                "test mode only permits owned designated test notes"
            )
        if self._write_policy is LightWritePolicy.BRIDGE_OUTPUTS and (
            note.note_id not in self._owned_output_note_ids
            or not is_output_title(note.title, self._output_title)
            or not is_output_title(expected_title, self._output_title)
            or not is_output_title(desired_title, self._output_title)
        ):
            raise LightWriteDisabledError(
                "bridge mode only permits owned output notes"
            )

    def _is_test_title(self, title: str) -> bool:
        return title == self._test_title_prefix or title.startswith(
            f"{self._test_title_prefix} "
        )

    @staticmethod
    def _normalize(native: Any) -> RemoteNote:
        raw_kind = str(native.note_type).lower()
        if raw_kind == "audio":
            kind = NoteKind.AUDIO
            mime_type = _audio_mime_type(native)
        elif raw_kind == "text":
            kind = NoteKind.TEXT
            mime_type = "text/plain; charset=utf-8"
        else:
            raise ValueError("unsupported note kind")
        raw_updated = getattr(native, "updated_at", None)
        updated_at = _parse_datetime(raw_updated) if raw_updated else None
        return RemoteNote(
            note_id=str(native.id),
            kind=kind,
            title=str(native.title or ""),
            updated_at=updated_at,
            mime_type=mime_type,
        )

    @classmethod
    def _normalize_or_none(cls, native: Any | None) -> RemoteNote | None:
        if native is None:
            return None
        try:
            return cls._normalize(native)
        except Exception:
            return None


def _supports_raw_note_listing(client: Any) -> bool:
    return (
        getattr(client, "_api_client", None) is not None
        and isinstance(getattr(client, "_device_tool_ids", None), dict)
        and bool(client._device_tool_ids.get("notes"))
        and callable(getattr(client, "_ensure_ok", None))
    )


def _fetch_native_notes_with_file_metadata(
    client: Any,
    *,
    request: Callable[..., Any] | None = None,
) -> list[Any]:
    """List notes without losing their cloud file MIME type and storage suffix."""

    try:
        if request is None:
            from open_api_specification_client.api.default import get_api_notes

            request = get_api_notes.sync_detailed

        response = request(
            client=client._api_client,
            device_tool_id=client._device_tool_ids["notes"],
        )
        body = client._ensure_ok(response, "List notes", require_parsed=True)
        included = {
            str(item.id): item
            for item in body.included
            if str(getattr(item, "type_", "")) == "files"
        }
        result: list[Any] = []
        for item in body.data:
            attributes = item.attributes
            file_id = str(attributes.file_id)
            file_item = included.get(file_id)
            file_attributes = (
                None if file_item is None else getattr(file_item, "attributes", None)
            )
            result.append(
                _NativeNoteWithFileMetadata(
                    id=str(item.id),
                    file_id=file_id,
                    note_type=str(attributes.note_type),
                    title=str(attributes.title),
                    updated_at=str(attributes.updated_at),
                    content_type=(
                        None
                        if file_attributes is None
                        or getattr(file_attributes, "content_type", None) is None
                        else str(file_attributes.content_type)
                    ),
                    storage_key=(
                        None
                        if file_attributes is None
                        or getattr(file_attributes, "key", None) is None
                        else str(file_attributes.key)
                    ),
                )
            )
        return result
    except Exception as error:
        if _is_authenticated_session_expiry(error):
            raise _LightSessionExpired from None
        raise LightAdapterError("Light note listing failed") from None


def _is_authenticated_session_expiry(error: BaseException) -> bool:
    """Recognize only 401s from authenticated Light API operations.

    The pinned client reduces API errors to ``RuntimeError`` messages. Match
    its exact authenticated read actions; in particular, do not treat a 401
    from a presigned object-storage download as an expired Light session.
    """

    if isinstance(error, _LightSessionExpired):
        return True
    error_type = type(error)
    if (
        error_type.__module__ == "open_api_specification_client.errors"
        and error_type.__name__ == "UnexpectedStatus"
        and getattr(error, "status_code", None) == 401
    ):
        return True
    if error_type is not RuntimeError:
        return False
    message = str(error)
    if message == "List notes: 401":
        return True
    return (
        message.startswith("Presigned get URL for ")
        or message.startswith("Fetching note ")
    ) and message.endswith(": 401")


def _audio_mime_type(native: Any) -> str | None:
    declared = str(getattr(native, "content_type", "") or "")
    declared = declared.partition(";")[0].strip().lower()
    if declared.startswith("audio/"):
        return declared

    storage_key = str(getattr(native, "storage_key", "") or "").lower()
    suffix = storage_key.rsplit("/", 1)[-1].rsplit(".", 1)
    extension = suffix[1] if len(suffix) == 2 else ""
    return {
        "flac": "audio/flac",
        "m4a": "audio/mp4",
        "mp3": "audio/mpeg",
        "mp4": "audio/mp4",
        "oga": "audio/ogg",
        "ogg": "audio/ogg",
        "wav": "audio/wav",
    }.get(extension)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    return datetime.fromisoformat(text)


def _validate_digest(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PermanentDependencyError(
            "expected_content_sha256 must be a lowercase SHA-256 digest"
        )


def _disable_upstream_auth_cache(client: Any) -> None:
    """Disable garado/light's process-global keyring session cache.

    ``cache_enabled=False`` controls downloaded API data only. The pinned client
    otherwise reads and writes one global keyring entry which is not scoped to
    an email address and can therefore select the wrong account.
    """

    if not callable(getattr(client, "_load_auth_cache", None)) or not callable(
        getattr(client, "_save_auth_cache", None)
    ):
        raise RuntimeError("unsupported garado/light authentication contract")
    client._load_auth_cache = lambda: False
    client._save_auth_cache = lambda: None


def _start_notes_only_session(client: Any) -> None:
    """Authenticate the pinned client without initializing Music or Podcasts."""

    from light_api.client import API_BASE, API_HEADERS
    from light_api.notes import LightNotes
    from open_api_specification_client.client import AuthenticatedClient

    client.login()
    client._api_client = AuthenticatedClient(
        base_url=API_BASE,
        token=client._api_token,
        headers=API_HEADERS,
        timeout=30,
    )
    client._fetch_device_tool_ids()
    if not client._device_tool_ids.get("notes"):
        raise RuntimeError("Light Notes tool is not installed on the selected device")
    client.notes = LightNotes(client)


def _authenticated_email(client: Any) -> str:
    """Read the authenticated identity and return only the email for comparison."""

    from open_api_specification_client.api.default import get_api_users_current

    response = client.call_api(
        get_api_users_current.sync_detailed,
        client=client._api_client,
    )
    parsed = client._ensure_ok(
        response,
        "Could not verify current Light user",
        require_data=True,
    )
    return str(parsed.data.attributes.email)
