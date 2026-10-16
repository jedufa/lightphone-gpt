from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO
import hashlib
import json
from pathlib import Path

import pytest

from lightphone_gpt import cli
import lightphone_gpt.light_input_audit as input_audit
from lightphone_gpt.adapters.light_garado import LightWritePolicy
from lightphone_gpt.models import NoteKind, RemoteNote


SOURCE_SHA = "c" * 40
PRIVATE_PERSONAL_TITLE = "Private personal journal title"
PRIVATE_AUDIO_TITLE = "Assistant synthetic audio fixture"
PRIVATE_TEXT_TITLE = "Assistant synthetic text fixture"
PRIVATE_AUDIO_ID = "opaque-audio-note-id"
PRIVATE_TEXT_ID = "opaque-text-note-id"


def private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    return path


def title_digest(title: str) -> str:
    return hashlib.sha256(title.encode("utf-8")).hexdigest()


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "lightphone_username": "private-owner@example.test",
        "lightphone_password": "private-light-password",
        "lightphone_device_id": "private-device-id",
        "LIGHTPHONE_GPT_OPERATOR_SOURCE_SHA": SOURCE_SHA,
        "LIGHTPHONE_GPT_STATE_DIR": str(private_directory(tmp_path / "state")),
        "LIGHTPHONE_GPT_INPUT_BINDINGS_JSON": json.dumps(
            [
                {
                    "name": "audio_fixture",
                    "kind": "audio",
                    "title_sha256": title_digest(PRIVATE_AUDIO_TITLE),
                },
                {
                    "name": "text_fixture",
                    "kind": "text",
                    "title_sha256": title_digest(PRIVATE_TEXT_TITLE),
                },
            ]
        ),
    }


def receipt_paths(tmp_path: Path) -> tuple[Path, Path]:
    receipts = private_directory(tmp_path / "receipts")
    evidence = private_directory(tmp_path / "evidence")
    return receipts / "binding.json", evidence / "binding-summary.json"


class FakeProvider:
    def __init__(
        self,
        calls: list[object],
        notes: list[RemoteNote],
        **kwargs: object,
    ) -> None:
        self.calls = calls
        self.notes = notes
        calls.append(("constructed", kwargs))

    def list_notes(self):
        self.calls.append(("listed",))
        return list(self.notes)

    def close(self) -> None:
        self.calls.append(("closed",))


def designated_notes() -> list[RemoteNote]:
    return [
        RemoteNote("unrelated-personal-id", NoteKind.TEXT, PRIVATE_PERSONAL_TITLE),
        RemoteNote(PRIVATE_AUDIO_ID, NoteKind.AUDIO, PRIVATE_AUDIO_TITLE),
        RemoteNote(PRIVATE_TEXT_ID, NoteKind.TEXT, PRIVATE_TEXT_TITLE),
    ]


def test_binding_uses_metadata_only_and_separates_private_ids_from_summary(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    private_path, summary_path = receipt_paths(tmp_path)
    calls: list[object] = []

    result = input_audit.run_input_binding(
        receipt_path=private_path,
        summary_path=summary_path,
        confirm_read=True,
        environ=env,
        provider_factory=lambda **kwargs: FakeProvider(
            calls, designated_notes(), **kwargs
        ),
        now=lambda: datetime(2026, 10, 15, 20, 0, tzinfo=UTC),
    )

    private = json.loads(private_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    public = json.dumps([result, summary], sort_keys=True)
    private_serialized = json.dumps(private, sort_keys=True)

    assert result["ok"] is True
    assert result["bindings_requested"] == 2
    assert result["bindings_resolved"] == 2
    assert result["resolved_kinds"] == {"audio": 1, "text": 1}
    assert result["all_unique"] is True
    assert private["status"] == "complete"
    assert private["result"]["bindings"] == [
        {
            "kind": "audio",
            "name": "audio_fixture",
            "note_id": PRIVATE_AUDIO_ID,
        },
        {
            "kind": "text",
            "name": "text_fixture",
            "note_id": PRIVATE_TEXT_ID,
        },
    ]
    assert summary["status"] == "complete"
    assert summary["result"] == {
        key: result[key]
        for key in (
            "operation",
            "source_sha",
            "write_policy",
            "notes_scanned",
            "bindings_requested",
            "bindings_resolved",
            "resolved_kinds",
            "all_unique",
            "timing_ms",
        )
    }
    assert calls[0][0] == "constructed"
    assert calls[0][1]["write_policy"] is LightWritePolicy.READ_ONLY
    assert calls[1:] == [("listed",), ("closed",)]
    assert private_path.stat().st_mode & 0o777 == 0o600
    assert summary_path.stat().st_mode & 0o777 == 0o600

    for private_value in (
        PRIVATE_PERSONAL_TITLE,
        PRIVATE_AUDIO_TITLE,
        PRIVATE_TEXT_TITLE,
        PRIVATE_AUDIO_ID,
        PRIVATE_TEXT_ID,
        title_digest(PRIVATE_AUDIO_TITLE),
        title_digest(PRIVATE_TEXT_TITLE),
        env["lightphone_username"],
        env["lightphone_password"],
        env["lightphone_device_id"],
    ):
        assert private_value not in public
    for private_title in (
        PRIVATE_PERSONAL_TITLE,
        PRIVATE_AUDIO_TITLE,
        PRIVATE_TEXT_TITLE,
    ):
        assert private_title not in private_serialized


@pytest.mark.parametrize(
    ("notes", "reason"),
    [
        ([], "binding_not_found"),
        (
            [
                RemoteNote("duplicate-a", NoteKind.AUDIO, PRIVATE_AUDIO_TITLE),
                RemoteNote("duplicate-b", NoteKind.AUDIO, PRIVATE_AUDIO_TITLE),
                RemoteNote(PRIVATE_TEXT_ID, NoteKind.TEXT, PRIVATE_TEXT_TITLE),
            ],
            "binding_not_unique",
        ),
        (
            [
                RemoteNote(PRIVATE_AUDIO_ID, NoteKind.TEXT, PRIVATE_AUDIO_TITLE),
                RemoteNote(PRIVATE_TEXT_ID, NoteKind.TEXT, PRIVATE_TEXT_TITLE),
            ],
            "binding_kind_mismatch",
        ),
        (
            [
                RemoteNote("same-opaque-id", NoteKind.AUDIO, PRIVATE_AUDIO_TITLE),
                RemoteNote("same-opaque-id", NoteKind.TEXT, PRIVATE_TEXT_TITLE),
            ],
            "duplicate_resolved_note_id",
        ),
    ],
)
def test_binding_fails_closed_without_persisting_partial_ids(
    tmp_path: Path,
    notes: list[RemoteNote],
    reason: str,
) -> None:
    env = environment(tmp_path)
    private_path, summary_path = receipt_paths(tmp_path)

    with pytest.raises(input_audit.LightInputAuditError) as raised:
        input_audit.run_input_binding(
            receipt_path=private_path,
            summary_path=summary_path,
            confirm_read=True,
            environ=env,
            provider_factory=lambda **kwargs: FakeProvider([], notes, **kwargs),
        )

    assert raised.value.category == "input_binding_failed"
    private = private_path.read_text(encoding="utf-8")
    summary = summary_path.read_text(encoding="utf-8")
    assert json.loads(private)["result"]["failure_reason"] == reason
    assert json.loads(summary)["result"]["failure_reason"] == reason
    for private_value in (
        PRIVATE_AUDIO_ID,
        PRIVATE_TEXT_ID,
        "duplicate-a",
        "duplicate-b",
        "same-opaque-id",
        PRIVATE_AUDIO_TITLE,
        PRIVATE_TEXT_TITLE,
    ):
        assert private_value not in private + summary


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: [],
        lambda value: [value[0], value[0]],
        lambda value: [{**value[0], "extra": "forbidden"}],
        lambda value: [{**value[0], "name": "Unsafe Name"}],
        lambda value: [{**value[0], "kind": "unknown"}],
        lambda value: [{**value[0], "title_sha256": "not-a-digest"}],
    ],
)
def test_malformed_binding_designations_stop_before_cloud(
    tmp_path: Path,
    mutate,
) -> None:
    env = environment(tmp_path)
    original = json.loads(env["LIGHTPHONE_GPT_INPUT_BINDINGS_JSON"])
    env["LIGHTPHONE_GPT_INPUT_BINDINGS_JSON"] = json.dumps(mutate(original))
    private_path, summary_path = receipt_paths(tmp_path)
    calls = 0

    def forbidden_provider(**_kwargs: object):
        nonlocal calls
        calls += 1
        raise AssertionError("cloud must not be accessed")

    with pytest.raises(input_audit.LightInputAuditError) as raised:
        input_audit.run_input_binding(
            receipt_path=private_path,
            summary_path=summary_path,
            confirm_read=True,
            environ=env,
            provider_factory=forbidden_provider,
        )

    assert raised.value.category == "configuration_error"
    assert calls == 0
    assert not private_path.exists()
    assert not summary_path.exists()


def test_binding_requires_confirmation_before_receipts_or_cloud(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    private_path, summary_path = receipt_paths(tmp_path)
    calls = 0

    def forbidden_provider(**_kwargs: object):
        nonlocal calls
        calls += 1
        raise AssertionError("cloud must not be accessed")

    with pytest.raises(input_audit.LightInputAuditError) as raised:
        input_audit.run_input_binding(
            receipt_path=private_path,
            summary_path=summary_path,
            confirm_read=False,
            environ=env,
            provider_factory=forbidden_provider,
        )

    assert raised.value.category == "confirmation_required"
    assert calls == 0
    assert not private_path.exists()
    assert not summary_path.exists()


def test_existing_private_binding_receipt_stops_before_cloud(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    private_path, summary_path = receipt_paths(tmp_path)
    private_path.write_text('{"status":"pending"}\n', encoding="utf-8")
    private_path.chmod(0o600)
    calls = 0

    def forbidden_provider(**_kwargs: object):
        nonlocal calls
        calls += 1
        raise AssertionError("cloud must not be accessed")

    with pytest.raises(input_audit.LightInputAuditError) as raised:
        input_audit.run_input_binding(
            receipt_path=private_path,
            summary_path=summary_path,
            confirm_read=True,
            environ=env,
            provider_factory=forbidden_provider,
        )

    assert raised.value.category == "receipt_already_exists"
    assert calls == 0
    assert not summary_path.exists()


def test_binding_cli_is_sanitized_and_does_not_build_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = environment(tmp_path)
    private_path, summary_path = receipt_paths(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("continuous runtime must remain isolated")

    monkeypatch.setattr(cli, "build_runtime", forbidden)
    monkeypatch.setattr(
        cli,
        "run_light_input_binding_command",
        lambda _args, _env: {
            "ok": True,
            "operation": "light_input_binding_summary",
            "bindings_resolved": 2,
        },
    )
    stdout = StringIO()
    stderr = StringIO()

    code = cli.main(
        [
            "light-input-bind",
            "--receipt",
            str(private_path),
            "--summary",
            str(summary_path),
            "--confirm-read",
        ],
        environ=env,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert json.loads(stdout.getvalue())["bindings_resolved"] == 2
    assert not stderr.getvalue()


def test_binding_parser_rejects_plaintext_title_without_echoing_it(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    private_path, summary_path = receipt_paths(tmp_path)
    stdout = StringIO()
    stderr = StringIO()

    code = cli.main(
        [
            "light-input-bind",
            "--title",
            PRIVATE_TEXT_TITLE,
            "--receipt",
            str(private_path),
            "--summary",
            str(summary_path),
            "--confirm-read",
        ],
        environ=env,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 2
    assert not stdout.getvalue()
    assert json.loads(stderr.getvalue()) == {
        "ok": False,
        "error": "configuration_error",
    }
    assert PRIVATE_TEXT_TITLE not in stderr.getvalue()
