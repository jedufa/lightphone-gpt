from __future__ import annotations

from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightphone_gpt.adapters.light_garado import LightWritePolicy
from lightphone_gpt.e2e_command import (
    E2E_CHATGPT_MODEL,
    E2E_RECEIPT_NAME,
    E2ECommandError,
    E2EReceiptStore,
    run_e2e_command,
)
from lightphone_gpt.models import (
    AssistantResponse,
    MutationOutcome,
    MutationReceipt,
    NoteKind,
    RemoteNote,
)


OPERATION_ID = "a" * 32
SECOND_OPERATION_ID = "b" * 32
SOURCE_REVISION = "c" * 40
PRIVATE_SOURCE_ID = "private-source-note-id"
PRIVATE_OUTPUT_ID = "private-output-note-id"
PRIVATE_SOURCE_TITLE = "private source title"
PRIVATE_QUESTION = "private question"
PRIVATE_TEXT = f"Assistant, {PRIVATE_QUESTION}".encode()
PRIVATE_AUDIO = b"private-audio-content"
PRIVATE_ANSWER = "private answer"
PRIVATE_RESPONSE_ID = "private-response-id"
PRIVATE_HOST_ID = "urn:uuid:11111111-2222-4333-8444-555555555555"
PRIVATE_ACCESS_TOKEN = "private-access-token"


def private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def environment(
    tmp_path: Path,
    *,
    kind: NoteKind = NoteKind.TEXT,
    payload: bytes = PRIVATE_TEXT,
) -> dict[str, str]:
    credentials = private_directory(tmp_path / "chatgpt")
    receipts = private_directory(tmp_path / "receipts")
    model = private_directory(tmp_path / "model")
    temporary = private_directory(tmp_path / "temp")
    return {
        "lightphone_username": "private-user@example.test",
        "lightphone_password": "private-light-password",
        "lightphone_device_id": "private-device-id",
        "LIGHTPHONE_GPT_E2E_SOURCE_NOTE_ID": PRIVATE_SOURCE_ID,
        "LIGHTPHONE_GPT_E2E_SOURCE_KIND": kind.value,
        "LIGHTPHONE_GPT_E2E_SOURCE_SHA256": hashlib.sha256(payload).hexdigest(),
        "LIGHTPHONE_GPT_E2E_CHATGPT_CREDENTIAL_PATH": str(
            credentials / "credentials.json"
        ),
        "LIGHTPHONE_GPT_E2E_CHATGPT_HOST_ID_PATH": str(credentials / "host-id"),
        "LIGHTPHONE_GPT_E2E_MODEL": E2E_CHATGPT_MODEL,
        "LIGHTPHONE_GPT_E2E_RECEIPT_DIR": str(receipts),
        "LIGHTPHONE_GPT_OPERATOR_SOURCE_SHA": SOURCE_REVISION,
        "LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_DIR": str(model),
        "LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_MANIFEST": str(
            tmp_path / "model.manifest.json"
        ),
        "LIGHTPHONE_GPT_INPUT_AUDIT_TEMP_DIR": str(temporary),
    }


class FakeCredentialStore:
    def __init__(self, path: Path, observations: list[object]) -> None:
        self.path = path
        self.observations = observations

    def load(self):
        latest = Path(self.observations[0]) / E2E_RECEIPT_NAME
        self.observations.append(("credential_load", json.loads(latest.read_text())))
        return SimpleNamespace(
            ext_agent_host_id=PRIVATE_HOST_ID,
            access_token=PRIVATE_ACCESS_TOKEN,
        )


class FakeHostStore:
    def __init__(self, path: Path, host_id: str = PRIVATE_HOST_ID) -> None:
        self.path = path
        self.host_id = host_id

    def load(self) -> str:
        return self.host_id


class FakeRefreshManager:
    def __init__(self, store: object, *, timeout_seconds: float) -> None:
        self.store = store
        self.timeout_seconds = timeout_seconds

    def get_access_token(self, *, force_refresh: bool = False) -> str:
        return PRIVATE_ACCESS_TOKEN


class FakeAssistant:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def respond(self, messages: object) -> AssistantResponse:
        self.messages.append(messages)
        return AssistantResponse(PRIVATE_ANSWER, PRIVATE_RESPONSE_ID)


class FakeProvider:
    def __init__(
        self,
        *,
        source_kind: NoteKind,
        source: bytes,
        create_outcome: MutationOutcome = MutationOutcome.VERIFIED,
    ) -> None:
        self.source_kind = source_kind
        self.source = source
        self.create_outcome = create_outcome
        self.create_calls = 0
        self.close_calls = 0
        self.output_note: RemoteNote | None = None
        self.output_content: bytes | None = None

    def _source_note(self) -> RemoteNote:
        return RemoteNote(
            PRIVATE_SOURCE_ID,
            self.source_kind,
            PRIVATE_SOURCE_TITLE,
            mime_type=(
                "audio/mpeg"
                if self.source_kind is NoteKind.AUDIO
                else "text/plain; charset=utf-8"
            ),
        )

    def read_text_with_metadata(self, note_id: str):
        assert note_id == PRIVATE_SOURCE_ID
        return self._source_note(), self.source

    def download_audio_with_metadata(self, note_id: str):
        assert note_id == PRIVATE_SOURCE_ID
        return self._source_note(), self.source

    def get_note(self, note_id: str):
        return self.output_note if note_id == PRIVATE_OUTPUT_ID else None

    def read_note_content(self, note_id: str) -> bytes:
        assert note_id == PRIVATE_OUTPUT_ID
        assert self.output_content is not None
        return self.output_content

    def list_notes(self):
        raise AssertionError("one-shot must not poll")

    def download_audio(self, note_id: str) -> bytes:
        raise AssertionError("rich audio read should be used")

    def create_text_note(self, *, title: str, content: str) -> MutationReceipt:
        self.create_calls += 1
        if self.create_outcome is MutationOutcome.UNKNOWN:
            return MutationReceipt(MutationOutcome.UNKNOWN)
        self.output_note = RemoteNote(PRIVATE_OUTPUT_ID, NoteKind.TEXT, title)
        self.output_content = content.encode()
        return MutationReceipt(
            MutationOutcome.VERIFIED,
            self.output_note,
            hashlib.sha256(self.output_content).hexdigest(),
        )

    def update_text_note(self, *args: object, **kwargs: object):
        raise AssertionError("one-shot must not update")

    def close(self) -> None:
        self.close_calls += 1


class FakeTranscriber:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls: list[tuple[bytes, str | None]] = []

    def transcribe(self, audio: bytes, *, mime_type: str | None = None) -> str:
        self.calls.append((audio, mime_type))
        return self.transcript


def invoke(
    env: dict[str, str],
    provider: FakeProvider,
    *,
    operation_id: str = OPERATION_ID,
    host_id: str = PRIVATE_HOST_ID,
    transcriber_factory=None,
    model_verifier=lambda _model, _manifest: None,
    receipt_store_factory=E2EReceiptStore,
):
    receipt_dir = Path(env["LIGHTPHONE_GPT_E2E_RECEIPT_DIR"])
    observations: list[object] = [receipt_dir]
    assistant = FakeAssistant()
    provider_kwargs: dict[str, object] = {}
    assistant_kwargs: dict[str, object] = {}

    def credentials(path: Path):
        return FakeCredentialStore(path, observations)

    def hosts(path: Path):
        return FakeHostStore(path, host_id)

    def providers(**kwargs: object):
        provider_kwargs.update(kwargs)
        return provider

    def assistants(**kwargs: object):
        assistant_kwargs.update(kwargs)
        return assistant

    result = run_e2e_command(
        environ=env,
        provider_factory=providers,
        transcriber_factory=(
            transcriber_factory
            if transcriber_factory is not None
            else lambda **_kwargs: FakeTranscriber(f"Assistant, {PRIVATE_QUESTION}")
        ),
        credential_store_factory=credentials,
        host_store_factory=hosts,
        refresh_manager_factory=FakeRefreshManager,
        assistant_factory=assistants,
        model_verifier=model_verifier,
        receipt_store_factory=receipt_store_factory,
        operation_id_factory=lambda: operation_id,
        now=lambda: datetime(2026, 10, 15, 22, 30, tzinfo=UTC),
    )
    return result, observations, provider_kwargs, assistant_kwargs, assistant


def test_text_e2e_uses_pending_receipt_host_binding_and_bypasses_stt(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    provider = FakeProvider(source_kind=NoteKind.TEXT, source=PRIVATE_TEXT)

    def forbidden_transcriber(**_kwargs: object):
        raise AssertionError("text must not construct local STT")

    def forbidden_model_verify(_model: Path, _manifest: Path) -> None:
        raise AssertionError("text must not inspect the STT model")

    result, observations, provider_kwargs, assistant_kwargs, assistant = invoke(
        env,
        provider,
        transcriber_factory=forbidden_transcriber,
        model_verifier=forbidden_model_verify,
    )
    receipt_path = Path(env["LIGHTPHONE_GPT_E2E_RECEIPT_DIR"]) / E2E_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())

    assert result["ok"] is True
    assert result["response_id_present"] is True
    assert result["read_back_verified"] is True
    assert result["create_calls"] == 1
    assert receipt["schema_version"] == 1
    assert receipt["operation"] == "light_e2e"
    assert receipt["operation_id"] == OPERATION_ID
    assert receipt["operator_source_sha"] == SOURCE_REVISION
    assert receipt["status"] == "verified"
    assert receipt["result"] == result
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert observations[1][0] == "credential_load"
    assert observations[1][1]["status"] == "pending"
    assert provider_kwargs["write_policy"] is LightWritePolicy.TEST_NOTES
    assert provider_kwargs["username"] == env["lightphone_username"]
    assert provider_kwargs["password"] == env["lightphone_password"]
    assert assistant_kwargs["model"] == E2E_CHATGPT_MODEL
    assert assistant_kwargs["max_stream_chars"] == 4_000
    assert len(assistant.messages) == 1
    assert provider.create_calls == 1
    assert provider.close_calls == 1


def test_audio_uses_verified_local_faster_whisper_configuration(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path, kind=NoteKind.AUDIO, payload=PRIVATE_AUDIO)
    provider = FakeProvider(source_kind=NoteKind.AUDIO, source=PRIVATE_AUDIO)
    transcriber = FakeTranscriber(f"Assistant, {PRIVATE_QUESTION}")
    observed: dict[str, object] = {}

    def verifier(model: Path, manifest: Path) -> None:
        observed["verified"] = (model, manifest)

    def transcribers(**kwargs: object):
        observed["transcriber"] = kwargs
        return transcriber

    result, *_rest = invoke(
        env,
        provider,
        transcriber_factory=transcribers,
        model_verifier=verifier,
    )

    assert result["ok"] is True
    assert result["transcription_used"] is True
    assert observed["verified"] == (
        Path(env["LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_DIR"]),
        Path(env["LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_MANIFEST"]),
    )
    kwargs = observed["transcriber"]
    assert kwargs["model"] == env["LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_DIR"]
    assert kwargs["model_cache"] is None
    assert kwargs["local_files_only"] is True
    assert kwargs["temp_root"] == Path(
        env["LIGHTPHONE_GPT_INPUT_AUDIT_TEMP_DIR"]
    )
    assert transcriber.calls == [(PRIVATE_AUDIO, "audio/mpeg")]


def test_receipt_and_return_never_leak_private_material(tmp_path: Path) -> None:
    env = environment(tmp_path)
    provider = FakeProvider(source_kind=NoteKind.TEXT, source=PRIVATE_TEXT)

    result, *_rest = invoke(env, provider)
    receipt = (
        Path(env["LIGHTPHONE_GPT_E2E_RECEIPT_DIR"]) / E2E_RECEIPT_NAME
    ).read_text()
    public = json.dumps(result) + receipt

    for private in (
        PRIVATE_SOURCE_ID,
        PRIVATE_OUTPUT_ID,
        PRIVATE_SOURCE_TITLE,
        PRIVATE_QUESTION,
        PRIVATE_ANSWER,
        PRIVATE_RESPONSE_ID,
        PRIVATE_HOST_ID,
        PRIVATE_ACCESS_TOKEN,
        env["lightphone_username"],
        env["lightphone_password"],
        env["lightphone_device_id"],
        env["LIGHTPHONE_GPT_E2E_SOURCE_SHA256"],
    ):
        assert private not in public


def test_exact_host_binding_failure_is_sanitized_and_prevents_cloud_access(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    provider = FakeProvider(source_kind=NoteKind.TEXT, source=PRIVATE_TEXT)
    private_other_host = "urn:uuid:aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

    with pytest.raises(E2ECommandError) as caught:
        invoke(env, provider, host_id=private_other_host)

    assert caught.value.category == "chatgpt_host_binding_mismatch"
    assert provider.create_calls == 0
    receipt = json.loads(
        (
            Path(env["LIGHTPHONE_GPT_E2E_RECEIPT_DIR"]) / E2E_RECEIPT_NAME
        ).read_text()
    )
    assert receipt["status"] == "failed_closed"
    assert receipt["result"]["mutation_attempted"] is False
    material = json.dumps([caught.value.as_dict(), receipt])
    assert PRIVATE_HOST_ID not in material
    assert private_other_host not in material


def test_uncertain_create_is_never_retried_and_requires_review(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    provider = FakeProvider(
        source_kind=NoteKind.TEXT,
        source=PRIVATE_TEXT,
        create_outcome=MutationOutcome.UNKNOWN,
    )

    with pytest.raises(E2ECommandError) as caught:
        invoke(env, provider)

    assert caught.value.category == "publication_outcome_unknown"
    assert caught.value.mutation_attempted is True
    assert provider.create_calls == 1
    receipt = json.loads(
        (
            Path(env["LIGHTPHONE_GPT_E2E_RECEIPT_DIR"]) / E2E_RECEIPT_NAME
        ).read_text()
    )
    assert receipt["status"] == "review_required"
    assert receipt["result"] == caught.value.as_dict()

    with pytest.raises(E2ECommandError) as second:
        invoke(env, provider, operation_id=SECOND_OPERATION_ID)
    assert second.value.category == "unresolved_receipt_exists"
    assert provider.create_calls == 1


def test_pending_receipt_survives_unfinalizable_uncertain_create(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    provider = FakeProvider(
        source_kind=NoteKind.TEXT,
        source=PRIVATE_TEXT,
        create_outcome=MutationOutcome.UNKNOWN,
    )

    class CannotFinalize(E2EReceiptStore):
        def finalize(self, payload):
            raise E2ECommandError("private-finalize-detail")

    with pytest.raises(E2ECommandError) as caught:
        invoke(env, provider, receipt_store_factory=CannotFinalize)

    assert caught.value.category == "receipt_finalize_failed"
    assert caught.value.mutation_attempted is True
    assert provider.create_calls == 1
    path = Path(env["LIGHTPHONE_GPT_E2E_RECEIPT_DIR"]) / E2E_RECEIPT_NAME
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "pending"
    assert receipt["result"] is None
    assert path.stat().st_mode & 0o777 == 0o600


def test_completed_latest_is_archived_content_free_before_next_run(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    first = FakeProvider(source_kind=NoteKind.TEXT, source=PRIVATE_TEXT)
    first_result, *_rest = invoke(env, first)
    second = FakeProvider(source_kind=NoteKind.TEXT, source=PRIVATE_TEXT)

    second_result, *_rest = invoke(
        env, second, operation_id=SECOND_OPERATION_ID
    )

    directory = Path(env["LIGHTPHONE_GPT_E2E_RECEIPT_DIR"])
    archive = json.loads((directory / f"e2e-{OPERATION_ID}.json").read_text())
    latest = json.loads((directory / E2E_RECEIPT_NAME).read_text())
    assert archive["result"] == first_result
    assert archive["status"] == "verified"
    assert latest["result"] == second_result
    assert latest["operation_id"] == SECOND_OPERATION_ID
    assert (directory / f"e2e-{OPERATION_ID}.json").stat().st_mode & 0o777 == 0o600


def test_existing_receipt_must_be_strictly_content_free_before_archival(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    first = FakeProvider(source_kind=NoteKind.TEXT, source=PRIVATE_TEXT)
    invoke(env, first)
    directory = Path(env["LIGHTPHONE_GPT_E2E_RECEIPT_DIR"])
    latest = directory / E2E_RECEIPT_NAME
    payload = json.loads(latest.read_text())
    payload["result"]["source_bytes"] = "private injected content"
    latest.write_text(json.dumps(payload) + "\n")
    latest.chmod(0o600)
    second = FakeProvider(source_kind=NoteKind.TEXT, source=PRIVATE_TEXT)

    with pytest.raises(E2ECommandError) as caught:
        invoke(env, second, operation_id=SECOND_OPERATION_ID)

    assert caught.value.category == "receipt_history_invalid"
    assert second.create_calls == 0
    assert not (directory / f"e2e-{OPERATION_ID}.json").exists()


def test_module_lock_rejects_a_concurrent_run(tmp_path: Path) -> None:
    env = environment(tmp_path)
    directory = Path(env["LIGHTPHONE_GPT_E2E_RECEIPT_DIR"])
    lock_path = directory / ".light-e2e.lock"
    descriptor = os.open(
        lock_path, os.O_CREAT | os.O_RDWR, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provider = FakeProvider(source_kind=NoteKind.TEXT, source=PRIVATE_TEXT)
        with pytest.raises(E2ECommandError) as caught:
            invoke(env, provider)
        assert caught.value.category == "operator_busy"
        assert provider.create_calls == 0
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("LIGHTPHONE_GPT_E2E_MODEL", "wrong-model"),
        ("LIGHTPHONE_GPT_OPERATOR_SOURCE_SHA", "bad-revision"),
        ("LIGHTPHONE_GPT_E2E_SOURCE_SHA256", "bad-digest"),
        ("LIGHTPHONE_GPT_E2E_SOURCE_KIND", "unknown"),
        ("LIGHTPHONE_GPT_E2E_SOURCE_NOTE_ID", "bad id"),
    ],
)
def test_invalid_protected_configuration_creates_no_receipt(
    tmp_path: Path, key: str, value: str
) -> None:
    env = environment(tmp_path)
    env[key] = value
    provider = FakeProvider(source_kind=NoteKind.TEXT, source=PRIVATE_TEXT)

    with pytest.raises(E2ECommandError) as caught:
        invoke(env, provider)

    assert caught.value.category == "configuration_error"
    assert not (
        Path(env["LIGHTPHONE_GPT_E2E_RECEIPT_DIR"]) / E2E_RECEIPT_NAME
    ).exists()
    assert provider.create_calls == 0


def test_receipt_directory_must_be_owner_only(tmp_path: Path) -> None:
    env = environment(tmp_path)
    Path(env["LIGHTPHONE_GPT_E2E_RECEIPT_DIR"]).chmod(0o755)
    provider = FakeProvider(source_kind=NoteKind.TEXT, source=PRIVATE_TEXT)

    with pytest.raises(E2ECommandError) as caught:
        invoke(env, provider)

    assert caught.value.category == "configuration_error"
    assert provider.create_calls == 0
