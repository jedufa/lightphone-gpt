from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

import lightphone_gpt.adapters.light_garado as light_garado
from lightphone_gpt.adapters.errors import LightAdapterError, LightWriteDisabledError
from lightphone_gpt.adapters.light_garado import GaradoLightProvider, LightWritePolicy
from lightphone_gpt.contracts import (
    LightProvider,
    PermanentDependencyError,
    TransientDependencyError,
)
from lightphone_gpt.models import MutationOutcome, NoteKind


@dataclass
class NativeNote:
    id: str
    file_id: str
    note_type: str
    title: str
    updated_at: str = "2026-10-15T12:00:00Z"
    content_type: str | None = None
    storage_key: str | None = None


class FakeNotesApi:
    def __init__(self) -> None:
        self.notes = [
            NativeNote(
                "audio-1",
                "file-1",
                "audio",
                "Recording",
                content_type="audio/mp4",
                storage_key="files/file-1/recording.m4a",
            )
        ]
        self.content = {"audio-1": b"m4a-bytes"}
        self.mutations = 0
        self.raise_after_create = False
        self.raise_after_update = False
        self.delete_calls = 0
        self.raise_before_delete = False
        self.raise_after_delete = False

    def get_notes(self):
        return list(self.notes)

    def get_note_content(self, note):
        return self.content[note.id]

    def get_note_metadata(self, note_id):
        return next(note for note in self.notes if note.id == note_id)

    def create_text_note(self, title, content):
        self.mutations += 1
        note = NativeNote(f"text-{self.mutations}", "file-new", "text", title)
        self.notes.append(note)
        self.content[note.id] = content.encode()
        if self.raise_after_create:
            raise RuntimeError("lost response with sensitive upstream body")
        return note

    def update_note_content(self, note, content):
        self.mutations += 1
        self.content[note.id] = bytes(content)
        if self.raise_after_update:
            raise RuntimeError("lost response with sensitive upstream body")

    def update_note_title(self, note, title):
        self.mutations += 1
        note.title = title

    def delete_note(self, note_id):
        self.delete_calls += 1
        if self.raise_before_delete:
            raise RuntimeError("delete request failed with private upstream body")
        self.notes = [note for note in self.notes if note.id != note_id]
        self.content.pop(note_id, None)
        if self.raise_after_delete:
            raise RuntimeError("delete response was lost with private upstream body")


class FakeClient:
    def __init__(self, notes: FakeNotesApi) -> None:
        self.notes = notes
        self._api_client = self
        self.raw_delete_status = 204
        self.raw_delete_requests: list[tuple[str, str]] = []

    def get_httpx_client(self):
        return self

    def request(self, *, method, url):
        self.raw_delete_requests.append((method, url))
        if self.raw_delete_status not in {200, 204}:
            return SimpleNamespace(status_code=self.raw_delete_status)
        prefix = "/api/notes/"
        assert url.startswith(prefix)
        self.notes.delete_note(unquote(url.removeprefix(prefix)))
        return SimpleNamespace(status_code=self.raw_delete_status)


class ConstructedSessionClient(FakeClient):
    def __init__(self, notes: FakeNotesApi) -> None:
        super().__init__(notes)
        self.exit_calls = 0

    def _load_auth_cache(self):
        return False

    def _save_auth_cache(self):
        return None

    def __exit__(self, *_):
        self.exit_calls += 1
        return None


def credential_provider(
    monkeypatch: pytest.MonkeyPatch,
    clients: list[ConstructedSessionClient],
    *,
    write_policy: LightWritePolicy = LightWritePolicy.READ_ONLY,
    owned_test_note_ids: tuple[str, ...] = (),
) -> tuple[GaradoLightProvider, list[dict[str, object]]]:
    created: list[dict[str, object]] = []

    def factory(**kwargs):
        created.append(kwargs)
        return clients[len(created) - 1]

    monkeypatch.setattr(light_garado, "_start_notes_only_session", lambda _value: None)
    monkeypatch.setattr(
        light_garado,
        "_authenticated_email",
        lambda _value: "owner@example.test",
    )
    return (
        GaradoLightProvider(
            username="owner@example.test",
            password="secret",
            device_id="device-1",
            client_factory=factory,
            write_policy=write_policy,
            owned_test_note_ids=owned_test_note_ids,
        ),
        created,
    )


def test_read_only_adapter_normalizes_and_downloads_without_mutation() -> None:
    api = FakeNotesApi()
    provider = GaradoLightProvider(client=FakeClient(api))
    assert isinstance(provider, LightProvider)

    listed = provider.list_notes()

    assert len(listed) == 1
    assert listed[0].note_id == "audio-1"
    assert listed[0].kind is NoteKind.AUDIO
    assert listed[0].mime_type == "audio/mp4"
    assert provider.download_audio("audio-1") == b"m4a-bytes"
    with pytest.raises(LightWriteDisabledError):
        provider.create_text_note(title="AI READY", content="answer")
    assert api.mutations == 0


def test_exact_audio_download_returns_matching_metadata_and_one_payload() -> None:
    api = FakeNotesApi()
    payload_calls: list[str] = []
    original = api.get_note_content

    def get_note_content(note):
        payload_calls.append(note.id)
        return original(note)

    api.get_note_content = get_note_content
    provider = GaradoLightProvider(client=FakeClient(api))

    note, content = provider.download_audio_with_metadata("audio-1")

    assert note.note_id == "audio-1"
    assert note.kind is NoteKind.AUDIO
    assert note.mime_type == "audio/mp4"
    assert content == b"m4a-bytes"
    assert payload_calls == ["audio-1"]
    assert api.mutations == 0


def test_exact_audio_download_rejects_non_audio_before_payload_read() -> None:
    api = FakeNotesApi()
    api.notes[0].note_type = "text"
    payload_calls = 0

    def forbidden_content(_note):
        nonlocal payload_calls
        payload_calls += 1
        raise AssertionError("non-audio content must not be downloaded")

    api.get_note_content = forbidden_content
    provider = GaradoLightProvider(client=FakeClient(api))

    with pytest.raises(PermanentDependencyError, match="not audio"):
        provider.download_audio_with_metadata("audio-1")

    assert payload_calls == 0
    assert api.mutations == 0


def test_exact_text_read_returns_matching_metadata_and_one_payload() -> None:
    api = FakeNotesApi()
    api.notes[0].note_type = "text"
    api.content["audio-1"] = b"Assistant, typed request"
    payload_calls: list[str] = []
    original = api.get_note_content

    def get_note_content(note):
        payload_calls.append(note.id)
        return original(note)

    api.get_note_content = get_note_content
    provider = GaradoLightProvider(client=FakeClient(api))

    note, content = provider.read_text_with_metadata("audio-1")

    assert note.note_id == "audio-1"
    assert note.kind is NoteKind.TEXT
    assert note.mime_type == "text/plain; charset=utf-8"
    assert content == b"Assistant, typed request"
    assert payload_calls == ["audio-1"]
    assert api.mutations == 0


def test_exact_text_read_rejects_non_text_before_payload_read() -> None:
    api = FakeNotesApi()
    payload_calls = 0

    def forbidden_content(_note):
        nonlocal payload_calls
        payload_calls += 1
        raise AssertionError("non-text content must not be downloaded")

    api.get_note_content = forbidden_content
    provider = GaradoLightProvider(client=FakeClient(api))

    with pytest.raises(PermanentDependencyError, match="not text"):
        provider.read_text_with_metadata("audio-1")

    assert payload_calls == 0
    assert api.mutations == 0


def test_duplicate_titles_are_read_by_distinct_light_note_ids() -> None:
    api = FakeNotesApi()
    api.notes.extend(
        [
            NativeNote("text-a", "file-a", "text", "Synthetic duplicate title"),
            NativeNote("text-b", "file-b", "text", "Synthetic duplicate title"),
        ]
    )
    api.content.update({"text-a": b"first", "text-b": b"second"})
    provider = GaradoLightProvider(client=FakeClient(api))

    listed = provider.list_notes()

    assert [
        note.note_id
        for note in listed
        if note.title == "Synthetic duplicate title"
    ] == [
        "text-a",
        "text-b",
    ]
    assert provider.read_note_content("text-a") == b"first"
    assert provider.read_note_content("text-b") == b"second"


@pytest.mark.parametrize(
    ("content_type", "storage_key", "expected"),
    [
        ("audio/mpeg", "files/file-1/recording.mp3", "audio/mpeg"),
        ("audio/mp4", "files/file-1/recording.m4a", "audio/mp4"),
        ("binary/octet-stream", "files/file-1/recording.mp3", "audio/mpeg"),
        (None, "files/file-1/recording.wav", "audio/wav"),
        (None, "files/file-1/recording.unknown", None),
    ],
)
def test_audio_mime_uses_cloud_file_metadata_and_suffix_fallback(
    content_type: str | None,
    storage_key: str,
    expected: str | None,
) -> None:
    api = FakeNotesApi()
    api.notes[0].content_type = content_type
    api.notes[0].storage_key = storage_key
    provider = GaradoLightProvider(client=FakeClient(api))

    assert provider.list_notes()[0].mime_type == expected


def test_pinned_client_listing_preserves_live_style_file_metadata(monkeypatch) -> None:
    api = FakeNotesApi()
    client = FakeClient(api)
    client._api_client = object()
    client._device_tool_ids = {"notes": "notes-tool"}
    client._ensure_ok = lambda *_args, **_kwargs: None
    calls = 0

    def fetch(_client):
        nonlocal calls
        calls += 1
        assert _client is client
        return [
            NativeNote(
                "audio-live",
                "file-live",
                "audio",
                "Synthetic input note",
                content_type="audio/mpeg",
                storage_key="files/file-live/phone-recording.mp3",
            )
        ]

    monkeypatch.setattr(light_garado, "_fetch_native_notes_with_file_metadata", fetch)
    provider = GaradoLightProvider(client=client)

    listed = provider.list_notes()

    assert calls == 1
    assert listed[0].note_id == "audio-live"
    assert listed[0].mime_type == "audio/mpeg"


def test_raw_listing_joins_note_and_file_metadata_by_file_id() -> None:
    note = SimpleNamespace(
        id="audio-live",
        attributes=SimpleNamespace(
            file_id="file-live",
            note_type="audio",
            title="Synthetic input note",
            updated_at="2026-10-15T12:00:00Z",
        ),
    )
    unrelated_file = SimpleNamespace(
        id="file-other",
        type_="files",
        attributes=SimpleNamespace(
            content_type="audio/mp4",
            key="files/file-other/unrelated.m4a",
        ),
    )
    matching_file = SimpleNamespace(
        id="file-live",
        type_="files",
        attributes=SimpleNamespace(
            content_type="audio/mpeg",
            key="files/file-live/phone-recording.mp3",
        ),
    )
    body = SimpleNamespace(data=[note], included=[unrelated_file, matching_file])
    client = SimpleNamespace(
        _api_client=object(),
        _device_tool_ids={"notes": "notes-tool"},
        _ensure_ok=lambda *_args, **_kwargs: body,
    )
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        return object()

    native = light_garado._fetch_native_notes_with_file_metadata(
        client, request=request
    )

    assert calls == [
        {"client": client._api_client, "device_tool_id": "notes-tool"}
    ]
    assert len(native) == 1
    assert native[0].id == "audio-live"
    assert native[0].file_id == "file-live"
    assert native[0].content_type == "audio/mpeg"
    assert native[0].storage_key.endswith(".mp3")


def test_test_note_gate_blocks_personal_notes_before_mutation() -> None:
    api = FakeNotesApi()
    personal = NativeNote("text-personal", "file-2", "text", "My journal")
    api.notes.append(personal)
    api.content[personal.id] = b"private"
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.TEST_NOTES,
    )

    with pytest.raises(LightWriteDisabledError):
        provider.update_text_note(
            personal.id,
            title="LIGHTPHONE GPT TEST answer",
            content="replacement",
            expected_title="My journal",
            expected_content_sha256=hashlib.sha256(b"private").hexdigest(),
        )

    assert api.content[personal.id] == b"private"
    assert api.mutations == 0


def test_test_note_namespace_requires_an_exact_separator() -> None:
    api = FakeNotesApi()
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.TEST_NOTES,
    )

    with pytest.raises(LightWriteDisabledError):
        provider.create_text_note(
            title="LIGHTPHONE GPT TESTING is not the namespace",
            content="synthetic",
        )

    assert api.mutations == 0


def test_test_note_create_and_compare_before_replace_are_verified() -> None:
    api = FakeNotesApi()
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.TEST_NOTES,
    )
    created = provider.create_text_note(
        title="LIGHTPHONE GPT TEST reply",
        content="first",
    )
    assert created.outcome is MutationOutcome.VERIFIED
    assert created.note is not None

    updated = provider.update_text_note(
        created.note.note_id,
        title="LIGHTPHONE GPT TEST reply",
        content="second",
        expected_title="LIGHTPHONE GPT TEST reply",
        expected_content_sha256=hashlib.sha256(b"first").hexdigest(),
    )

    assert updated.outcome is MutationOutcome.VERIFIED
    assert updated.observed_content_sha256 == hashlib.sha256(b"second").hexdigest()
    assert api.content[created.note.note_id] == b"second"


def test_bridge_output_policy_requires_exact_title_and_owned_identity() -> None:
    api = FakeNotesApi()
    existing = NativeNote("reply-owned", "file-2", "text", "AI READY")
    personal = NativeNote("reply-personal", "file-3", "text", "AI READY")
    api.notes.extend([existing, personal])
    api.content.update({"reply-owned": b"old", "reply-personal": b"personal"})
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
        output_title="AI READY",
        owned_output_note_ids={"reply-owned"},
    )

    with pytest.raises(LightWriteDisabledError, match="configured output title"):
        provider.create_text_note(title="AI WORKING", content="answer")
    with pytest.raises(LightWriteDisabledError, match="owned output notes"):
        provider.update_text_note(
            "reply-personal",
            title="AI READY",
            content="replacement",
            expected_title="AI READY",
            expected_content_sha256=hashlib.sha256(b"personal").hexdigest(),
        )

    updated = provider.update_text_note(
        "reply-owned",
        title="AI READY",
        content="new",
        expected_title="AI READY",
        expected_content_sha256=hashlib.sha256(b"old").hexdigest(),
    )

    assert updated.outcome is MutationOutcome.VERIFIED
    assert api.content["reply-owned"] == b"new"
    assert api.content["reply-personal"] == b"personal"


def test_bridge_output_policy_accepts_strict_timestamp_titles_with_exact_cas() -> None:
    api = FakeNotesApi()
    existing = NativeNote("reply-owned", "file-2", "text", "AI READY")
    api.notes.append(existing)
    api.content["reply-owned"] = b"old"
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
        output_title="AI READY",
        owned_output_note_ids={"reply-owned"},
    )

    created = provider.create_text_note(title="AI READY 19:35", content="first")
    assert created.outcome is MutationOutcome.VERIFIED

    updated = provider.update_text_note(
        "reply-owned",
        title="AI READY 19:36",
        content="new",
        expected_title="AI READY",
        expected_content_sha256=hashlib.sha256(b"old").hexdigest(),
    )
    assert updated.outcome is MutationOutcome.VERIFIED
    assert existing.title == "AI READY 19:36"

    stale = provider.update_text_note(
        "reply-owned",
        title="AI READY 19:37",
        content="must not apply",
        expected_title="AI READY 19:35",
        expected_content_sha256=hashlib.sha256(b"new").hexdigest(),
    )
    assert stale.outcome is MutationOutcome.CONFLICT
    assert stale.detail == "title_changed"
    assert api.content["reply-owned"] == b"new"


@pytest.mark.parametrize(
    "title",
    ["AI READY 9:35", "AI READY 24:00", "AI READY 19:60", "AI READY later"],
)
def test_bridge_output_policy_rejects_noncanonical_timestamp_titles(
    title: str,
) -> None:
    provider = GaradoLightProvider(
        client=FakeClient(FakeNotesApi()),
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
        output_title="AI READY",
    )

    with pytest.raises(LightWriteDisabledError, match="configured output title"):
        provider.create_text_note(title=title, content="answer")


def test_bridge_output_prefix_reserves_space_for_timestamp_suffix() -> None:
    with pytest.raises(ValueError, match="output_title"):
        GaradoLightProvider(
            client=FakeClient(FakeNotesApi()),
            write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
            output_title="A" * 95,
        )


def test_bridge_output_policy_admits_only_its_verified_create() -> None:
    api = FakeNotesApi()
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
        output_title="ASSISTANT REPLY",
    )

    created = provider.create_text_note(
        title="ASSISTANT REPLY",
        content="first",
    )
    assert created.outcome is MutationOutcome.VERIFIED
    assert created.note is not None

    updated = provider.update_text_note(
        created.note.note_id,
        title="ASSISTANT REPLY",
        content="second",
        expected_title="ASSISTANT REPLY",
        expected_content_sha256=hashlib.sha256(b"first").hexdigest(),
    )

    assert updated.outcome is MutationOutcome.VERIFIED
    assert api.content[created.note.note_id] == b"second"


def test_bridge_output_policy_does_not_admit_an_ambiguous_create() -> None:
    api = FakeNotesApi()
    api.raise_after_create = True
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
        output_title="AI READY",
    )

    created = provider.create_text_note(title="AI READY", content="private")

    assert created.outcome is MutationOutcome.UNKNOWN
    candidate = api.notes[-1]
    with pytest.raises(LightWriteDisabledError, match="owned output notes"):
        provider.update_text_note(
            candidate.id,
            title="AI READY",
            content="second",
            expected_title="AI READY",
            expected_content_sha256=hashlib.sha256(b"private").hexdigest(),
        )
    assert api.content[candidate.id] == b"private"


def test_bridge_policy_deletes_exact_source_after_kind_and_hash_precheck() -> None:
    api = FakeNotesApi()
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
        output_title="AI READY",
    )
    expected = hashlib.sha256(b"m4a-bytes").hexdigest()

    receipt = provider.delete_source_note(
        "audio-1",
        expected_kind=NoteKind.AUDIO,
        expected_content_sha256=expected,
    )

    assert receipt.outcome is MutationOutcome.VERIFIED
    assert receipt.deletion_attempted is True
    assert api.delete_calls == 1
    assert not api.notes


def test_source_delete_ignores_title_but_rejects_content_change_without_mutation() -> None:
    api = FakeNotesApi()
    api.notes[0].title = "AI READY and still only a source title"
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
    )

    receipt = provider.delete_source_note(
        "audio-1",
        expected_kind=NoteKind.AUDIO,
        expected_content_sha256=hashlib.sha256(b"different").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.CONFLICT
    assert receipt.deletion_attempted is False
    assert api.delete_calls == 0
    assert api.notes[0].id == "audio-1"


def test_source_delete_is_disabled_outside_bridge_runtime_policy() -> None:
    api = FakeNotesApi()
    provider = GaradoLightProvider(client=FakeClient(api))

    with pytest.raises(LightWriteDisabledError, match="source deletion"):
        provider.delete_source_note(
            "audio-1",
            expected_kind=NoteKind.AUDIO,
            expected_content_sha256=hashlib.sha256(b"m4a-bytes").hexdigest(),
        )

    assert api.delete_calls == 0


def test_source_delete_lost_response_is_verified_by_exact_absence() -> None:
    api = FakeNotesApi()
    api.raise_after_delete = True
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
    )

    receipt = provider.delete_source_note(
        "audio-1",
        expected_kind=NoteKind.AUDIO,
        expected_content_sha256=hashlib.sha256(b"m4a-bytes").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.VERIFIED
    assert receipt.deletion_attempted is True
    assert api.delete_calls == 1


def test_source_delete_failure_after_call_is_unknown_and_not_retried() -> None:
    api = FakeNotesApi()
    api.raise_before_delete = True
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
    )

    receipt = provider.delete_source_note(
        "audio-1",
        expected_kind=NoteKind.AUDIO,
        expected_content_sha256=hashlib.sha256(b"m4a-bytes").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.UNKNOWN
    assert receipt.deletion_attempted is True
    assert api.delete_calls == 1
    assert api.notes[0].id == "audio-1"


def test_source_delete_401_is_not_reauthenticated_or_retried() -> None:
    api = FakeNotesApi()
    client = FakeClient(api)
    client.raw_delete_status = 401
    call_api_calls = 0

    def forbidden_call_api(*_args, **_kwargs):
        nonlocal call_api_calls
        call_api_calls += 1
        raise AssertionError("source DELETE must bypass garado call_api")

    client.call_api = forbidden_call_api
    provider = GaradoLightProvider(
        client=client,
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
    )

    receipt = provider.delete_source_note(
        "audio-1",
        expected_kind=NoteKind.AUDIO,
        expected_content_sha256=hashlib.sha256(b"m4a-bytes").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.UNKNOWN
    assert receipt.deletion_attempted is True
    assert client.raw_delete_requests == [("DELETE", "/api/notes/audio-1")]
    assert call_api_calls == 0
    assert api.delete_calls == 0
    assert api.notes[0].id == "audio-1"


def test_source_delete_transport_absence_is_proven_before_attempt() -> None:
    api = FakeNotesApi()
    client = FakeClient(api)
    client._api_client = SimpleNamespace()
    provider = GaradoLightProvider(
        client=client,
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
    )

    receipt = provider.delete_source_note(
        "audio-1",
        expected_kind=NoteKind.AUDIO,
        expected_content_sha256=hashlib.sha256(b"m4a-bytes").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.UNKNOWN
    assert receipt.deletion_attempted is False
    assert receipt.detail == "source_delete_transport_unavailable"
    assert client.raw_delete_requests == []
    assert api.delete_calls == 0


def test_source_delete_url_quotes_the_exact_note_id_once() -> None:
    api = FakeNotesApi()
    api.notes[0].id = "audio/one two"
    api.content["audio/one two"] = api.content.pop("audio-1")
    client = FakeClient(api)
    provider = GaradoLightProvider(
        client=client,
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
    )

    receipt = provider.delete_source_note(
        "audio/one two",
        expected_kind=NoteKind.AUDIO,
        expected_content_sha256=hashlib.sha256(b"m4a-bytes").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.VERIFIED
    assert client.raw_delete_requests == [
        ("DELETE", "/api/notes/audio%2Fone%20two")
    ]
    assert api.delete_calls == 1


def test_source_delete_never_targets_a_known_bridge_output() -> None:
    api = FakeNotesApi()
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.BRIDGE_OUTPUTS,
        owned_output_note_ids={"audio-1"},
    )

    with pytest.raises(LightWriteDisabledError, match="output"):
        provider.delete_source_note(
            "audio-1",
            expected_kind=NoteKind.AUDIO,
            expected_content_sha256=hashlib.sha256(b"m4a-bytes").hexdigest(),
        )

    assert api.delete_calls == 0


def test_test_note_delete_requires_exact_owned_title_content_and_identity() -> None:
    api = FakeNotesApi()
    title = "LIGHTPHONE GPT TEST receipt-owned R1"
    api.notes.append(NativeNote("text-1", "file-2", "text", title))
    api.content["text-1"] = b"synthetic fixture"
    client = FakeClient(api)
    provider = GaradoLightProvider(
        client=client,
        write_policy=LightWritePolicy.TEST_NOTES,
        owned_test_note_ids={"text-1"},
    )

    receipt = provider.delete_test_note(
        "text-1",
        expected_title=title,
        expected_content_sha256=hashlib.sha256(b"synthetic fixture").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.VERIFIED
    assert receipt.deletion_attempted is True
    assert client.raw_delete_requests == [("DELETE", "/api/notes/text-1")]
    assert api.delete_calls == 1
    assert [note.id for note in api.notes] == ["audio-1"]


@pytest.mark.parametrize("change", ["title", "content", "kind"])
def test_test_note_delete_rejects_changed_note_without_mutation(change: str) -> None:
    api = FakeNotesApi()
    title = "LIGHTPHONE GPT TEST receipt-owned R1"
    note = NativeNote("text-1", "file-2", "text", title)
    api.notes.append(note)
    api.content["text-1"] = b"synthetic fixture"
    if change == "title":
        note.title = "renamed by user"
    elif change == "content":
        api.content["text-1"] = b"changed by user"
    else:
        note.note_type = "audio"
    client = FakeClient(api)
    provider = GaradoLightProvider(
        client=client,
        write_policy=LightWritePolicy.TEST_NOTES,
        owned_test_note_ids={"text-1"},
    )

    receipt = provider.delete_test_note(
        "text-1",
        expected_title=title,
        expected_content_sha256=hashlib.sha256(b"synthetic fixture").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.CONFLICT
    assert receipt.deletion_attempted is False
    assert client.raw_delete_requests == []
    assert api.delete_calls == 0
    assert any(candidate.id == "text-1" for candidate in api.notes)


@pytest.mark.parametrize(
    ("write_policy", "owned"),
    [
        (LightWritePolicy.READ_ONLY, ("text-1",)),
        (LightWritePolicy.TEST_NOTES, ()),
        (LightWritePolicy.BRIDGE_OUTPUTS, ("text-1",)),
    ],
)
def test_test_note_delete_is_disabled_without_exact_test_ownership(
    write_policy: LightWritePolicy,
    owned: tuple[str, ...],
) -> None:
    api = FakeNotesApi()
    title = "LIGHTPHONE GPT TEST receipt-owned R1"
    api.notes.append(NativeNote("text-1", "file-2", "text", title))
    api.content["text-1"] = b"synthetic fixture"
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=write_policy,
        owned_test_note_ids=owned,
    )

    with pytest.raises(LightWriteDisabledError, match="exact owned"):
        provider.delete_test_note(
            "text-1",
            expected_title=title,
            expected_content_sha256=hashlib.sha256(b"synthetic fixture").hexdigest(),
        )

    assert api.delete_calls == 0
    assert any(candidate.id == "text-1" for candidate in api.notes)


def test_replace_conflict_never_writes() -> None:
    api = FakeNotesApi()
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.TEST_NOTES,
        owned_test_note_ids={"text-1"},
    )
    api.notes.append(
        NativeNote("text-1", "file-2", "text", "LIGHTPHONE GPT TEST reply")
    )
    api.content["text-1"] = b"unexpected user edit"

    receipt = provider.update_text_note(
        "text-1",
        title="LIGHTPHONE GPT TEST reply",
        content="bridge answer",
        expected_title="LIGHTPHONE GPT TEST reply",
        expected_content_sha256=hashlib.sha256(b"old answer").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.CONFLICT
    assert api.content["text-1"] == b"unexpected user edit"
    assert api.mutations == 0


def test_replace_requires_the_exact_expected_title() -> None:
    api = FakeNotesApi()
    api.notes.append(
        NativeNote("text-1", "file-2", "text", "LIGHTPHONE GPT TEST current")
    )
    api.content["text-1"] = b"current"
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.TEST_NOTES,
        owned_test_note_ids={"text-1"},
    )

    receipt = provider.update_text_note(
        "text-1",
        title="LIGHTPHONE GPT TEST next",
        content="next",
        expected_title="LIGHTPHONE GPT TEST stale",
        expected_content_sha256=hashlib.sha256(b"current").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.CONFLICT
    assert receipt.detail == "title_changed"
    assert api.mutations == 0


def test_ambiguous_create_returns_unknown_and_does_not_expose_error() -> None:
    api = FakeNotesApi()
    api.raise_after_create = True
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.TEST_NOTES,
    )

    receipt = provider.create_text_note(
        title="LIGHTPHONE GPT TEST reply",
        content="private answer",
    )

    assert receipt.outcome is MutationOutcome.UNKNOWN
    assert receipt.detail == "create_outcome_unknown"
    assert len(api.notes) == 2
    assert "private answer" not in repr(receipt)


def test_ambiguous_replace_can_be_verified_by_readback() -> None:
    api = FakeNotesApi()
    api.notes.append(
        NativeNote("text-1", "file-2", "text", "LIGHTPHONE GPT TEST reply")
    )
    api.content["text-1"] = b"old"
    api.raise_after_update = True
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.TEST_NOTES,
        owned_test_note_ids={"text-1"},
    )

    receipt = provider.update_text_note(
        "text-1",
        title="LIGHTPHONE GPT TEST reply",
        content="new",
        expected_title="LIGHTPHONE GPT TEST reply",
        expected_content_sha256=hashlib.sha256(b"old").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.VERIFIED
    assert api.content["text-1"] == b"new"


def test_post_update_old_readback_is_unknown_until_content_converges() -> None:
    api = FakeNotesApi()
    note = NativeNote("text-1", "file-2", "text", "LIGHTPHONE GPT TEST reply")
    api.notes.append(note)
    api.content["text-1"] = b"old"
    reads = 0

    def stale_then_current(native):
        nonlocal reads
        reads += 1
        if reads <= 2:
            return b"old"
        return api.content[native.id]

    api.get_note_content = stale_then_current
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.TEST_NOTES,
        owned_test_note_ids={"text-1"},
    )

    receipt = provider.update_text_note(
        "text-1",
        title="LIGHTPHONE GPT TEST reply",
        content="new",
        expected_title="LIGHTPHONE GPT TEST reply",
        expected_content_sha256=hashlib.sha256(b"old").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.UNKNOWN
    assert receipt.detail == "update_verification_pending"
    assert receipt.observed_content_sha256 == hashlib.sha256(b"old").hexdigest()
    assert api.mutations == 1
    assert note.updated_at == "2026-10-15T12:00:00Z"
    assert provider.read_note_content("text-1") == b"new"
    assert api.mutations == 1


def test_post_update_third_party_content_remains_a_conflict() -> None:
    api = FakeNotesApi()
    note = NativeNote("text-1", "file-2", "text", "LIGHTPHONE GPT TEST reply")
    api.notes.append(note)
    api.content["text-1"] = b"old"
    reads = 0

    def changed_after_precheck(_native):
        nonlocal reads
        reads += 1
        return b"old" if reads == 1 else b"unexpected user edit"

    api.get_note_content = changed_after_precheck
    provider = GaradoLightProvider(
        client=FakeClient(api),
        write_policy=LightWritePolicy.TEST_NOTES,
        owned_test_note_ids={"text-1"},
    )

    receipt = provider.update_text_note(
        "text-1",
        title="LIGHTPHONE GPT TEST reply",
        content="new",
        expected_title="LIGHTPHONE GPT TEST reply",
        expected_content_sha256=hashlib.sha256(b"old").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.CONFLICT
    assert receipt.observed_content_sha256 == hashlib.sha256(
        b"unexpected user edit"
    ).hexdigest()
    assert api.mutations == 1


def test_light_error_classification_separates_retryable_from_write_policy() -> None:
    assert isinstance(LightAdapterError("temporary"), TransientDependencyError)
    assert isinstance(LightWriteDisabledError("disabled"), PermanentDependencyError)


def test_constructed_client_uses_fresh_verified_notes_only_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeNotesApi()
    captured: dict[str, object] = {}

    class ConstructedClient:
        notes = api

        def _load_auth_cache(self):
            raise AssertionError("global auth cache must be disabled")

        def _save_auth_cache(self):
            raise AssertionError("global auth cache must be disabled")

        def __enter__(self):
            raise AssertionError("full Music/Podcast initialization is forbidden")

        def __exit__(self, *_):
            return None

    client = ConstructedClient()

    def factory(**kwargs):
        captured.update(kwargs)
        return client

    initialized: list[object] = []
    monkeypatch.setattr(
        light_garado,
        "_start_notes_only_session",
        lambda value: initialized.append(value),
    )
    monkeypatch.setattr(
        light_garado,
        "_authenticated_email",
        lambda _value: "owner@example.test",
    )
    provider = GaradoLightProvider(
        username="Owner@Example.Test",
        password="secret",
        device_id="device-1",
        client_factory=factory,
    )

    assert len(provider.list_notes()) == 1
    assert initialized == [client]
    assert captured == {
        "email": "Owner@Example.Test",
        "password": "secret",
        "device_id": "device-1",
        "cache_enabled": False,
    }
    assert client._load_auth_cache() is False
    assert client._save_auth_cache() is None


def test_constructed_client_rejects_authenticated_account_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConstructedClient:
        def _load_auth_cache(self):
            return True

        def _save_auth_cache(self):
            return None

    monkeypatch.setattr(light_garado, "_start_notes_only_session", lambda _value: None)
    monkeypatch.setattr(
        light_garado,
        "_authenticated_email",
        lambda _value: "someone-else@example.test",
    )
    provider = GaradoLightProvider(
        username="owner@example.test",
        password="secret",
        client_factory=lambda **_kwargs: ConstructedClient(),
    )

    with pytest.raises(LightAdapterError, match="Light authentication failed"):
        provider.list_notes()


def test_raw_note_listing_replaces_expired_session_and_retries_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = ConstructedSessionClient(FakeNotesApi())
    second = ConstructedSessionClient(FakeNotesApi())
    for client in (first, second):
        client._api_client = object()
        client._device_tool_ids = {"notes": "notes-tool"}
        client._ensure_ok = lambda *_args, **_kwargs: None
    provider, created = credential_provider(monkeypatch, [first, second])
    calls: list[ConstructedSessionClient] = []

    def fetch(client):
        calls.append(client)
        if client is first:
            raise RuntimeError("List notes: 401")
        return list(client.notes.notes)

    monkeypatch.setattr(
        light_garado,
        "_fetch_native_notes_with_file_metadata",
        fetch,
    )

    notes = provider.list_notes()

    assert [note.note_id for note in notes] == ["audio-1"]
    assert calls == [first, second]
    assert len(created) == 2
    assert created[0]["device_id"] == created[1]["device_id"] == "device-1"
    assert first.exit_calls == 1
    assert second.exit_calls == 0


def test_audio_download_repeats_metadata_and_payload_on_fresh_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_api = FakeNotesApi()
    second_api = FakeNotesApi()
    first_content_calls = 0
    second_content_calls = 0

    def expired_content(_note):
        nonlocal first_content_calls
        first_content_calls += 1
        raise RuntimeError("Presigned get URL for audio-1: 401")

    def fresh_content(note):
        nonlocal second_content_calls
        second_content_calls += 1
        return second_api.content[note.id]

    first_api.get_note_content = expired_content
    second_api.get_note_content = fresh_content
    first = ConstructedSessionClient(first_api)
    second = ConstructedSessionClient(second_api)
    provider, created = credential_provider(monkeypatch, [first, second])

    note, content = provider.download_audio_with_metadata("audio-1")

    assert note.note_id == "audio-1"
    assert content == b"m4a-bytes"
    assert first_content_calls == 1
    assert second_content_calls == 1
    assert len(created) == 2
    assert first.exit_calls == 1


def test_read_authentication_retry_is_bounded_to_one_fresh_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = [
        ConstructedSessionClient(FakeNotesApi()),
        ConstructedSessionClient(FakeNotesApi()),
    ]
    for client in clients:
        client.notes.get_notes = lambda: (_ for _ in ()).throw(
            RuntimeError("List notes: 401")
        )
    provider, created = credential_provider(monkeypatch, clients)

    with pytest.raises(LightAdapterError, match="Light note listing failed"):
        provider.list_notes()

    assert len(created) == 2
    assert clients[0].exit_calls == 1
    assert clients[1].exit_calls == 0


def test_injected_client_is_not_replaced_without_owned_credentials() -> None:
    api = FakeNotesApi()
    api.get_notes = lambda: (_ for _ in ()).throw(RuntimeError("List notes: 401"))
    client = ConstructedSessionClient(api)
    provider = GaradoLightProvider(client=client)

    with pytest.raises(LightAdapterError, match="Light note listing failed"):
        provider.list_notes()

    assert client.exit_calls == 0


@pytest.mark.parametrize(
    "message",
    [
        "List notes: 503",
        "Download note audio-1: 401",
        "unrelated service rejected request: 401",
    ],
)
def test_non_session_read_failures_do_not_trigger_login(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    first_api = FakeNotesApi()
    second_api = FakeNotesApi()
    if message.startswith("List notes"):
        first_api.get_notes = lambda: (_ for _ in ()).throw(RuntimeError(message))
        operation = lambda provider: provider.list_notes()
    else:
        first_api.get_note_content = lambda _note: (_ for _ in ()).throw(
            RuntimeError(message)
        )
        operation = lambda provider: provider.download_audio("audio-1")
    first = ConstructedSessionClient(first_api)
    second = ConstructedSessionClient(second_api)
    provider, created = credential_provider(monkeypatch, [first, second])

    with pytest.raises(LightAdapterError):
        operation(provider)

    assert len(created) == 1
    assert first.exit_calls == 0


def test_create_is_never_retried_after_authenticated_mutation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_api = FakeNotesApi()
    second_api = FakeNotesApi()

    def rejected_create(_title, _content):
        first_api.mutations += 1
        raise RuntimeError("Creating note: 401")

    first_api.create_text_note = rejected_create
    first = ConstructedSessionClient(first_api)
    second = ConstructedSessionClient(second_api)
    provider, created = credential_provider(
        monkeypatch,
        [first, second],
        write_policy=LightWritePolicy.TEST_NOTES,
    )

    receipt = provider.create_text_note(
        title="LIGHTPHONE GPT TEST session failure",
        content="answer",
    )

    assert receipt.outcome is MutationOutcome.UNKNOWN
    assert receipt.detail == "create_outcome_unknown"
    assert first_api.mutations == 1
    assert second_api.mutations == 0
    assert len(created) == 1


def test_safe_update_precheck_can_reauthenticate_before_one_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_api = FakeNotesApi()
    second_api = FakeNotesApi()
    for api in (first_api, second_api):
        api.notes[0].note_type = "text"
        api.notes[0].title = "LIGHTPHONE GPT TEST reply"
        api.content["audio-1"] = b"old"
    first_api.get_note_content = lambda _note: (_ for _ in ()).throw(
        RuntimeError("Presigned get URL for audio-1: 401")
    )
    first = ConstructedSessionClient(first_api)
    second = ConstructedSessionClient(second_api)
    provider, created = credential_provider(
        monkeypatch,
        [first, second],
        write_policy=LightWritePolicy.TEST_NOTES,
        owned_test_note_ids=("audio-1",),
    )

    receipt = provider.update_text_note(
        "audio-1",
        title="LIGHTPHONE GPT TEST reply",
        content="new",
        expected_title="LIGHTPHONE GPT TEST reply",
        expected_content_sha256=hashlib.sha256(b"old").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.VERIFIED
    assert first_api.mutations == 0
    assert second_api.mutations == 1
    assert len(created) == 2


def test_update_is_not_retried_when_reauthentication_is_needed_for_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExpiringUpdateApi(FakeNotesApi):
        expired = False

        def get_notes(self):
            if self.expired:
                raise RuntimeError("List notes: 401")
            return super().get_notes()

        def update_note_content(self, note, content):
            self.mutations += 1
            self.expired = True
            raise RuntimeError(f"Presigned put URL for {note.id}: 401")

    first_api = ExpiringUpdateApi()
    second_api = FakeNotesApi()
    for api in (first_api, second_api):
        api.notes[0].note_type = "text"
        api.notes[0].title = "LIGHTPHONE GPT TEST reply"
        api.content["audio-1"] = b"old"
    first = ConstructedSessionClient(first_api)
    second = ConstructedSessionClient(second_api)
    provider, created = credential_provider(
        monkeypatch,
        [first, second],
        write_policy=LightWritePolicy.TEST_NOTES,
        owned_test_note_ids=("audio-1",),
    )

    receipt = provider.update_text_note(
        "audio-1",
        title="LIGHTPHONE GPT TEST reply",
        content="new",
        expected_title="LIGHTPHONE GPT TEST reply",
        expected_content_sha256=hashlib.sha256(b"old").hexdigest(),
    )

    assert receipt.outcome is MutationOutcome.UNKNOWN
    assert receipt.detail == "update_outcome_unknown"
    assert first_api.mutations == 1
    assert second_api.mutations == 0
    assert len(created) == 2
