from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json

import pytest

from lightphone_gpt.chatgpt_errors import (
    ChatGPTAccountNotEligibleError,
    ChatGPTAmbiguousOutcomeError,
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTCredentialAmbiguousError,
    ChatGPTPlanUsageLimitError,
    ChatGPTTransientError,
)
from lightphone_gpt.conversation import DEFAULT_SYSTEM_PROMPT
from lightphone_gpt.e2e_operator import (
    DEFAULT_OUTPUT_TITLE_PREFIX,
    E2EInputBinding,
    E2EProofError,
    E2EProofLimits,
    run_e2e_proof,
)
from lightphone_gpt.models import (
    AssistantResponse,
    MutationOutcome,
    MutationReceipt,
    NoteKind,
    RemoteNote,
)


OPERATION_ID = "a" * 32
SOURCE_ID = "private-source-id"
OUTPUT_ID = "private-output-id"
PRIVATE_AUDIO = b"private-audio-payload"
PRIVATE_TRANSCRIPT = "Assistant, explain private topic"
PRIVATE_QUESTION = "explain private topic"
PRIVATE_ANSWER = "A private answer for the handset."


class FakeTranscriber:
    def __init__(self, transcript: str = PRIVATE_TRANSCRIPT) -> None:
        self.transcript = transcript
        self.calls: list[tuple[bytes, str | None]] = []

    def transcribe(self, audio: bytes, *, mime_type: str | None = None) -> str:
        self.calls.append((audio, mime_type))
        return self.transcript


class FakeAssistant:
    def __init__(self, answer: str = PRIVATE_ANSWER) -> None:
        self.answer = answer
        self.calls: list[object] = []

    def respond(self, messages: object) -> AssistantResponse:
        self.calls.append(messages)
        return AssistantResponse(self.answer, "private-response-id")


class FakeProvider:
    def __init__(
        self,
        *,
        source_kind: NoteKind = NoteKind.AUDIO,
        source: bytes = PRIVATE_AUDIO,
        create_outcome: MutationOutcome = MutationOutcome.VERIFIED,
    ) -> None:
        self.source_kind = source_kind
        self.source = source
        self.create_outcome = create_outcome
        self.created: list[tuple[str, str]] = []
        self.get_calls: list[str] = []
        self.read_calls: list[str] = []
        self.output_note: RemoteNote | None = None
        self.output_content: bytes | None = None
        self.raise_create = False
        self.raise_readback = False

    def _source_note(self) -> RemoteNote:
        return RemoteNote(
            note_id=SOURCE_ID,
            kind=self.source_kind,
            title="private source title",
            mime_type=(
                "audio/mpeg"
                if self.source_kind is NoteKind.AUDIO
                else "text/plain; charset=utf-8"
            ),
        )

    def download_audio_with_metadata(self, note_id: str):
        assert note_id == SOURCE_ID
        return self._source_note(), self.source

    def read_text_with_metadata(self, note_id: str):
        assert note_id == SOURCE_ID
        return self._source_note(), self.source

    def list_notes(self):
        raise AssertionError("the proof must not select notes by title")

    def get_note(self, note_id: str):
        self.get_calls.append(note_id)
        if note_id == SOURCE_ID:
            return self._source_note()
        if self.raise_readback:
            raise RuntimeError("private cloud failure")
        return self.output_note if note_id == OUTPUT_ID else None

    def read_note_content(self, note_id: str) -> bytes:
        self.read_calls.append(note_id)
        if self.raise_readback:
            raise RuntimeError("private cloud failure")
        assert note_id == OUTPUT_ID
        assert self.output_content is not None
        return self.output_content

    def download_audio(self, note_id: str) -> bytes:
        assert note_id == SOURCE_ID
        return self.source

    def create_text_note(self, *, title: str, content: str) -> MutationReceipt:
        self.created.append((title, content))
        if self.raise_create:
            raise RuntimeError("private uncertain failure")
        note = RemoteNote(OUTPUT_ID, NoteKind.TEXT, title)
        content_bytes = content.encode("utf-8")
        self.output_note = note
        self.output_content = content_bytes
        return MutationReceipt(
            self.create_outcome,
            note=(note if self.create_outcome is MutationOutcome.VERIFIED else None),
            observed_content_sha256=(
                hashlib.sha256(content_bytes).hexdigest()
                if self.create_outcome is MutationOutcome.VERIFIED
                else None
            ),
        )

    def update_text_note(self, *args: object, **kwargs: object):
        raise AssertionError("the one-shot proof must never update notes")


def binding(payload: bytes, kind: NoteKind = NoteKind.AUDIO) -> E2EInputBinding:
    return E2EInputBinding(
        note_id=SOURCE_ID,
        expected_kind=kind,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )


def run(
    provider: FakeProvider,
    *,
    source_binding: E2EInputBinding | None = None,
    transcriber: FakeTranscriber | None = None,
    assistant: FakeAssistant | None = None,
    limits: E2EProofLimits = E2EProofLimits(),
):
    return run_e2e_proof(
        provider=provider,
        transcriber=transcriber or FakeTranscriber(),
        assistant=assistant or FakeAssistant(),
        binding=source_binding or binding(provider.source, provider.source_kind),
        limits=limits,
        operation_id_factory=lambda: OPERATION_ID,
        now=lambda: datetime(2026, 10, 15, 21, 7, tzinfo=UTC),
    )


def test_audio_proof_uses_exact_source_and_returns_content_free_receipt() -> None:
    provider = FakeProvider()
    transcriber = FakeTranscriber()
    assistant = FakeAssistant()

    result = run(provider, transcriber=transcriber, assistant=assistant)
    public = result.as_dict()
    serialized = json.dumps(public, sort_keys=True)

    assert public == {
        "ok": True,
        "status": "verified",
        "operation_id": OPERATION_ID,
        "source_kind": "audio",
        "source_bytes": len(PRIVATE_AUDIO),
        "transcription_used": True,
        "transcript_characters": len(PRIVATE_TRANSCRIPT),
        "request_characters": len(PRIVATE_QUESTION),
        "answer_characters": len(PRIVATE_ANSWER),
        "output_bytes": len(provider.created[0][1].encode("utf-8")),
        "response_id_present": True,
        "route_kind": "question",
        "create_calls": 1,
        "read_back_verified": True,
    }
    assert transcriber.calls == [(PRIVATE_AUDIO, "audio/mpeg")]
    messages = assistant.calls[0]
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[0].content == DEFAULT_SYSTEM_PROMPT
    assert messages[1].role == "user"
    assert messages[1].content == PRIVATE_QUESTION
    assert len(provider.created) == 1
    title, output = provider.created[0]
    assert title == f"{DEFAULT_OUTPUT_TITLE_PREFIX} {OPERATION_ID}"
    assert PRIVATE_TRANSCRIPT not in output
    assert f"YOU SAID\n{PRIVATE_QUESTION}\n" in output
    assert PRIVATE_ANSWER in output
    assert "Completed 2026-10-15 21:07 UTC" in output
    assert provider.get_calls == [OUTPUT_ID, OUTPUT_ID]
    assert provider.read_calls == [OUTPUT_ID]
    for private in (
        SOURCE_ID,
        OUTPUT_ID,
        PRIVATE_AUDIO.decode(),
        PRIVATE_TRANSCRIPT,
        PRIVATE_QUESTION,
        PRIVATE_ANSWER,
        "private-response-id",
        hashlib.sha256(PRIVATE_AUDIO).hexdigest(),
    ):
        assert private not in serialized


def test_audio_proof_accepts_observed_vocative_and_reconstructs_question() -> None:
    provider = FakeProvider()
    assistant = FakeAssistant()
    transcript = "Do you, Assistant, remember the earlier answer?"

    result = run(
        provider,
        transcriber=FakeTranscriber(transcript),
        assistant=assistant,
    )

    assert result.as_dict()["route_kind"] == "question"
    messages = assistant.calls[0]
    assert messages[1].content == "do you remember the earlier answer?"
    assert "YOU SAID\ndo you remember the earlier answer?\n" in provider.created[0][1]


def test_text_proof_bypasses_transcription() -> None:
    private_text = "Assistant, typed private request \N{SNOWMAN}".encode()
    provider = FakeProvider(source_kind=NoteKind.TEXT, source=private_text)

    class ForbiddenTranscriber:
        def transcribe(self, *_args: object, **_kwargs: object) -> str:
            raise AssertionError("text must bypass transcription")

    result = run(
        provider,
        source_binding=binding(private_text, NoteKind.TEXT),
        transcriber=ForbiddenTranscriber(),  # type: ignore[arg-type]
    )

    assert result.source_kind is NoteKind.TEXT
    assert result.transcription_used is False
    assert len(provider.created) == 1


def test_text_proof_keeps_vocative_form_strictly_rejected() -> None:
    private_text = b"Do you, Assistant, process this typed note?"
    provider = FakeProvider(source_kind=NoteKind.TEXT, source=private_text)
    assistant = FakeAssistant()

    with pytest.raises(E2EProofError) as caught:
        run(
            provider,
            source_binding=binding(private_text, NoteKind.TEXT),
            assistant=assistant,
        )

    assert caught.value.category == "assistant_prefix_required"
    assert assistant.calls == []
    assert provider.created == []


@pytest.mark.parametrize(
    ("transcript", "category"),
    [
        ("a question without the prefix", "assistant_prefix_required"),
        ("Assistant, continue", "command_not_supported_in_one_shot"),
        ("Assistant, new conversation", "command_not_supported_in_one_shot"),
    ],
)
def test_non_assistant_and_commands_never_publish(
    transcript: str, category: str
) -> None:
    provider = FakeProvider()
    assistant = FakeAssistant()

    with pytest.raises(E2EProofError) as caught:
        run(provider, transcriber=FakeTranscriber(transcript), assistant=assistant)

    assert caught.value.category == category
    assert caught.value.mutation_attempted is False
    assert provider.created == []
    assert assistant.calls == []


def test_digest_mismatch_stops_before_transcription() -> None:
    provider = FakeProvider()
    transcriber = FakeTranscriber()
    wrong = E2EInputBinding(SOURCE_ID, NoteKind.AUDIO, "0" * 64)

    with pytest.raises(E2EProofError) as caught:
        run(provider, source_binding=wrong, transcriber=transcriber)

    assert caught.value.category == "source_digest_mismatch"
    assert transcriber.calls == []
    assert provider.created == []


def test_kind_mismatch_stops_before_transcription() -> None:
    provider = FakeProvider(source_kind=NoteKind.TEXT, source=PRIVATE_AUDIO)
    expected_audio = binding(PRIVATE_AUDIO, NoteKind.AUDIO)

    with pytest.raises(E2EProofError) as caught:
        run(provider, source_binding=expected_audio)

    assert caught.value.category == "source_kind_mismatch"
    assert provider.created == []


def test_input_size_and_invalid_utf8_are_rejected_before_assistant() -> None:
    oversized = FakeProvider(source=PRIVATE_AUDIO)
    assistant = FakeAssistant()
    with pytest.raises(E2EProofError) as caught:
        run(
            oversized,
            assistant=assistant,
            limits=E2EProofLimits(max_audio_bytes=len(PRIVATE_AUDIO) - 1),
        )
    assert caught.value.category == "source_size_rejected"
    assert assistant.calls == []

    invalid = b"Assistant, \xff"
    text_provider = FakeProvider(source_kind=NoteKind.TEXT, source=invalid)
    with pytest.raises(E2EProofError) as caught:
        run(
            text_provider,
            source_binding=binding(invalid, NoteKind.TEXT),
            assistant=assistant,
        )
    assert caught.value.category == "text_is_not_utf8"
    assert text_provider.created == []


def test_transcript_answer_and_output_bounds_prevent_publication() -> None:
    provider = FakeProvider()
    with pytest.raises(E2EProofError) as caught:
        run(
            provider,
            transcriber=FakeTranscriber("Assistant, " + "x" * 20),
            limits=E2EProofLimits(max_transcript_chars=10),
        )
    assert caught.value.category == "transcript_size_rejected"

    with pytest.raises(E2EProofError) as caught:
        run(
            provider,
            assistant=FakeAssistant("x" * 20),
            limits=E2EProofLimits(max_answer_chars=10),
        )
    assert caught.value.category == "answer_size_rejected"

    with pytest.raises(E2EProofError) as caught:
        run(provider, limits=E2EProofLimits(max_output_bytes=10))
    assert caught.value.category == "output_size_rejected"
    assert provider.created == []


@pytest.mark.parametrize("raised", [False, True])
def test_ambiguous_create_stops_after_one_attempt(raised: bool) -> None:
    provider = FakeProvider(create_outcome=MutationOutcome.UNKNOWN)
    provider.raise_create = raised

    with pytest.raises(E2EProofError) as caught:
        run(provider)

    assert caught.value.category == "publication_outcome_unknown"
    assert caught.value.mutation_attempted is True
    assert caught.value.as_dict() == {
        "ok": False,
        "status": "failed_closed",
        "operation_id": OPERATION_ID,
        "category": "publication_outcome_unknown",
        "mutation_attempted": True,
    }
    assert len(provider.created) == 1
    assert provider.get_calls == []


def test_conflict_and_receipt_mismatch_stop_without_second_create() -> None:
    conflict = FakeProvider(create_outcome=MutationOutcome.CONFLICT)
    with pytest.raises(E2EProofError) as caught:
        run(conflict)
    assert caught.value.category == "publication_conflict"
    assert len(conflict.created) == 1

    mismatch = FakeProvider()

    def bad_create(*, title: str, content: str) -> MutationReceipt:
        mismatch.created.append((title, content))
        note = RemoteNote(OUTPUT_ID, NoteKind.TEXT, "wrong title")
        return MutationReceipt(
            MutationOutcome.VERIFIED,
            note,
            hashlib.sha256(content.encode()).hexdigest(),
        )

    mismatch.create_text_note = bad_create  # type: ignore[method-assign]
    with pytest.raises(E2EProofError) as caught:
        run(mismatch)
    assert caught.value.category == "publication_receipt_mismatch"
    assert len(mismatch.created) == 1
    assert mismatch.get_calls == []


def test_readback_failure_or_mismatch_never_creates_again() -> None:
    unavailable = FakeProvider()
    unavailable.raise_readback = True
    with pytest.raises(E2EProofError) as caught:
        run(unavailable)
    assert caught.value.category == "publication_readback_unknown"
    assert len(unavailable.created) == 1

    mismatch = FakeProvider()

    def wrong_read(_note_id: str) -> bytes:
        return b"different output"

    mismatch.read_note_content = wrong_read  # type: ignore[method-assign]
    with pytest.raises(E2EProofError) as caught:
        run(mismatch)
    assert caught.value.category == "publication_readback_mismatch"
    assert len(mismatch.created) == 1

    changed = FakeProvider()
    original_get = changed.get_note
    output_reads = 0

    def changed_get(note_id: str):
        nonlocal output_reads
        note = original_get(note_id)
        if note_id == OUTPUT_ID:
            output_reads += 1
            if output_reads == 2:
                return RemoteNote(OUTPUT_ID, NoteKind.TEXT, "changed title")
        return note

    changed.get_note = changed_get  # type: ignore[method-assign]
    with pytest.raises(E2EProofError) as caught:
        run(changed)
    assert caught.value.category == "publication_readback_mismatch"
    assert len(changed.created) == 1


def test_duplicate_titles_are_irrelevant_because_returned_id_is_verified() -> None:
    provider = FakeProvider()
    result = run(provider)

    assert result.as_dict()["read_back_verified"] is True
    assert provider.get_calls == [OUTPUT_ID, OUTPUT_ID]
    assert provider.read_calls == [OUTPUT_ID]


def test_assistant_failure_is_sanitized_and_does_not_publish() -> None:
    provider = FakeProvider()

    class FailingAssistant:
        def respond(self, _messages: object) -> AssistantResponse:
            raise RuntimeError("private token and request detail")

    with pytest.raises(E2EProofError) as caught:
        run(provider, assistant=FailingAssistant())  # type: ignore[arg-type]

    assert str(caught.value) == "assistant_failed"
    assert "private" not in json.dumps(caught.value.as_dict())
    assert provider.created == []


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (
            ChatGPTCredentialAmbiguousError("private detail"),
            "chatgpt_credential_ambiguous",
        ),
        (
            ChatGPTAuthorizationRequiredError("private detail"),
            "chatgpt_authorization_required",
        ),
        (
            ChatGPTAccountNotEligibleError("private detail"),
            "chatgpt_account_not_eligible",
        ),
        (
            ChatGPTPlanUsageLimitError("private detail"),
            "chatgpt_plan_usage_limit",
        ),
        (
            ChatGPTConfigurationError("private detail"),
            "chatgpt_configuration_error",
        ),
        (
            ChatGPTAmbiguousOutcomeError("private detail"),
            "chatgpt_response_ambiguous",
        ),
        (ChatGPTTransientError("private detail"), "chatgpt_transient"),
    ],
)
def test_typed_chatgpt_failures_have_stable_content_free_categories(
    error: Exception, category: str
) -> None:
    provider = FakeProvider()

    class FailingAssistant:
        def respond(self, _messages: object) -> AssistantResponse:
            raise error

    with pytest.raises(E2EProofError) as caught:
        run(provider, assistant=FailingAssistant())  # type: ignore[arg-type]

    assert caught.value.category == category
    assert "private" not in json.dumps(caught.value.as_dict())
    assert provider.created == []


def test_missing_response_identity_never_publishes() -> None:
    provider = FakeProvider()

    class MissingIdentityAssistant:
        def respond(self, _messages: object) -> AssistantResponse:
            return AssistantResponse(PRIVATE_ANSWER, None)

    with pytest.raises(E2EProofError) as caught:
        run(provider, assistant=MissingIdentityAssistant())  # type: ignore[arg-type]

    assert caught.value.category == "assistant_response_identity_missing"
    assert provider.created == []


def test_invalid_assistant_or_create_contract_fails_closed() -> None:
    provider = FakeProvider()

    class InvalidAssistant:
        def respond(self, _messages: object) -> object:
            return object()

    with pytest.raises(E2EProofError) as caught:
        run(provider, assistant=InvalidAssistant())  # type: ignore[arg-type]
    assert caught.value.category == "assistant_contract_invalid"
    assert provider.created == []

    def invalid_create(*, title: str, content: str) -> object:
        provider.created.append((title, content))
        return object()

    provider.create_text_note = invalid_create  # type: ignore[method-assign]
    with pytest.raises(E2EProofError) as caught:
        run(provider)
    assert caught.value.category == "publication_outcome_unknown"
    assert caught.value.mutation_attempted is True
    assert len(provider.created) == 1


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: E2EInputBinding("bad id with spaces", NoteKind.AUDIO, "0" * 64),
        lambda: E2EInputBinding(SOURCE_ID, NoteKind.AUDIO, "bad"),
        lambda: E2EProofLimits(max_output_bytes=0),
    ],
)
def test_configuration_validation(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()  # type: ignore[operator]


def test_invalid_generated_operation_id_and_title_prefix_fail_before_reads() -> None:
    provider = FakeProvider()
    with pytest.raises(ValueError, match="operation_id"):
        run_e2e_proof(
            provider=provider,
            transcriber=FakeTranscriber(),
            assistant=FakeAssistant(),
            binding=binding(PRIVATE_AUDIO),
            operation_id_factory=lambda: "bad",
        )
    with pytest.raises(ValueError, match="title"):
        run_e2e_proof(
            provider=provider,
            transcriber=FakeTranscriber(),
            assistant=FakeAssistant(),
            binding=binding(PRIVATE_AUDIO),
            output_title_prefix="bad\ntitle",
            operation_id_factory=lambda: OPERATION_ID,
        )
    assert provider.created == []
