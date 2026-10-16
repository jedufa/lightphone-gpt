from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from lightphone_gpt import cli
import lightphone_gpt.light_input_audit as input_audit
from lightphone_gpt.adapters.light_garado import LightWritePolicy
from lightphone_gpt.models import NoteKind, RemoteNote


PRIVATE_NOTE_ID = "9d2f744d-36b2-41f6-8c14-private-note"
PRIVATE_TITLE = "Synthetic private title"
PRIVATE_AUDIO = b"private-mpeg-audio-payload"
PRIVATE_TRANSCRIPT = "Assistant, keep this private question out of receipts"
SOURCE_SHA = "b" * 40


@dataclass
class FakeCloud:
    download_calls: int = 0
    close_calls: int = 0


class FakeProvider:
    def __init__(self, cloud: FakeCloud, **kwargs: object) -> None:
        self.cloud = cloud
        self.kwargs = kwargs

    def download_audio_with_metadata(self, note_id: str):
        self.cloud.download_calls += 1
        assert note_id == PRIVATE_NOTE_ID
        return (
            RemoteNote(
                note_id=note_id,
                kind=NoteKind.AUDIO,
                title=PRIVATE_TITLE,
                mime_type="audio/mpeg",
            ),
            PRIVATE_AUDIO,
        )

    def close(self) -> None:
        self.cloud.close_calls += 1


class FakeTranscriber:
    def __init__(self, calls: list[object], **kwargs: object) -> None:
        self.calls = calls
        calls.append(("constructed", kwargs))

    def transcribe(self, audio: bytes, *, mime_type: str | None = None) -> str:
        self.calls.append(("transcribe", audio, mime_type))
        return PRIVATE_TRANSCRIPT


def private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    return path


def model_bundle(root: Path, content: bytes = b"reviewed-model-bytes") -> tuple[Path, Path]:
    model = private_directory(root / "input-audit-model")
    model_file = model / "model.bin"
    model_file.write_bytes(content)
    model_file.chmod(0o600)
    manifest = root / "input-audit-model.manifest.json"
    input_audit.write_input_audit_model_manifest(model, manifest)
    return model, manifest


def environment(tmp_path: Path) -> dict[str, str]:
    state = private_directory(tmp_path / "state")
    audit_temp = private_directory(tmp_path / "audit-temp")
    model, manifest = model_bundle(tmp_path)
    return {
        "lightphone_username": "private-owner@example.test",
        "lightphone_password": "private-light-password",
        "lightphone_device_id": "private-device-id",
        "LIGHTPHONE_GPT_INPUT_AUDIT_NOTE_ID": PRIVATE_NOTE_ID,
        "LIGHTPHONE_GPT_OPERATOR_SOURCE_SHA": SOURCE_SHA,
        "LIGHTPHONE_GPT_STATE_DIR": str(state),
        "LIGHTPHONE_GPT_INPUT_AUDIT_TEMP_DIR": str(audit_temp),
        "LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_DIR": str(model),
        "LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_MANIFEST": str(manifest),
    }


def receipt_path(tmp_path: Path, name: str = "input-audit.json") -> Path:
    return private_directory(tmp_path / "receipts") / name


class MonotonicClock:
    def __init__(self) -> None:
        self.value = 1_000_000_000

    def __call__(self) -> int:
        current = self.value
        self.value += 10_000_000
        return current


def test_audit_downloads_one_exact_note_and_retains_only_metadata(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    path = receipt_path(tmp_path)
    cloud = FakeCloud()
    transcription_calls: list[object] = []
    provider_kwargs: dict[str, object] = {}
    times = iter(
        [
            datetime(2026, 10, 15, 12, 0, tzinfo=UTC),
            datetime(2026, 10, 15, 12, 0, 1, tzinfo=UTC),
        ]
    )

    def provider_factory(**kwargs: object) -> FakeProvider:
        provider_kwargs.update(kwargs)
        return FakeProvider(cloud, **kwargs)

    def transcriber_factory(**kwargs: object) -> FakeTranscriber:
        return FakeTranscriber(transcription_calls, **kwargs)

    result = input_audit.run_input_audit(
        receipt_path=path,
        confirm_read=True,
        environ=env,
        provider_factory=provider_factory,
        transcriber_factory=transcriber_factory,
        monotonic_ns=MonotonicClock(),
        now=lambda: next(times),
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps([result, receipt], sort_keys=True)

    assert result["ok"] is True
    assert result["accepted"] is True
    assert result["pipeline_completed"] is True
    assert result["audio_bytes"] == len(PRIVATE_AUDIO)
    assert result["mime_type"] == "audio/mpeg"
    assert result["transcript_characters"] == len(PRIVATE_TRANSCRIPT)
    assert result["prefix_recognized"] is True
    assert result["route_kind"] == "question"
    assert result["timing_ms"] == {
        "cloud_download": 10,
        "transcription": 10,
        "routing": 10,
        "total": 70,
    }
    assert receipt["status"] == "complete"
    assert receipt["source_sha"] == SOURCE_SHA
    assert receipt["result"]["route_kind"] == "question"
    assert path.stat().st_mode & 0o777 == 0o600
    assert cloud.download_calls == 1
    assert cloud.close_calls == 1
    assert provider_kwargs["write_policy"] is LightWritePolicy.READ_ONLY
    assert provider_kwargs["device_id"] == "private-device-id"
    assert transcription_calls[0][0] == "constructed"
    transcriber_kwargs = transcription_calls[0][1]
    assert transcriber_kwargs["model"] == env[
        "LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_DIR"
    ]
    assert transcriber_kwargs["model_cache"] is None
    assert transcriber_kwargs["device"] == "cpu"
    assert transcriber_kwargs["compute_type"] == "int8"
    assert transcriber_kwargs["local_files_only"] is True
    assert transcriber_kwargs["temp_root"] == Path(
        env["LIGHTPHONE_GPT_INPUT_AUDIT_TEMP_DIR"]
    )
    assert transcription_calls[1] == (
        "transcribe",
        PRIVATE_AUDIO,
        "audio/mpeg",
    )
    for private in (
        PRIVATE_NOTE_ID,
        PRIVATE_TITLE,
        PRIVATE_AUDIO.decode(),
        PRIVATE_TRANSCRIPT,
        env["lightphone_username"],
        env["lightphone_password"],
        env["lightphone_device_id"],
    ):
        assert private not in serialized
    assert "note_id_sha256" not in serialized
    assert "transcript_sha256" not in serialized


def test_text_audit_reads_bounded_utf8_and_never_constructs_transcriber(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    env["LIGHTPHONE_GPT_INPUT_AUDIT_KIND"] = "text"
    del env["LIGHTPHONE_GPT_INPUT_AUDIT_TEMP_DIR"]
    del env["LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_DIR"]
    del env["LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_MANIFEST"]
    path = receipt_path(tmp_path)
    private_text = "Assistant, typed private question \N{SNOWMAN}"
    private_title = "Private handset text title"
    calls: list[object] = []

    class TextProvider:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("constructed", kwargs))

        def read_text_with_metadata(self, note_id: str):
            calls.append(("read", note_id))
            return (
                RemoteNote(
                    note_id=note_id,
                    kind=NoteKind.TEXT,
                    title=private_title,
                    mime_type="text/plain; charset=utf-8",
                ),
                private_text.encode("utf-8"),
            )

        def close(self) -> None:
            calls.append(("closed",))

    def forbidden_transcriber(**_kwargs: object):
        raise AssertionError("text input must bypass speech transcription")

    result = input_audit.run_input_audit(
        receipt_path=path,
        confirm_read=True,
        environ=env,
        provider_factory=TextProvider,
        transcriber_factory=forbidden_transcriber,
        monotonic_ns=MonotonicClock(),
        now=lambda: datetime(2026, 10, 15, 12, 0, tzinfo=UTC),
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps([result, receipt], ensure_ascii=False)

    assert result["ok"] is True
    assert result["source_kind"] == "text"
    assert result["content_bytes"] == len(private_text.encode("utf-8"))
    assert result["text_characters"] == len(private_text)
    assert result["transcribed"] is False
    assert result["prefix_recognized"] is True
    assert result["route_kind"] == "question"
    assert result["timing_ms"] == {
        "cloud_download": 10,
        "text_decode": 10,
        "routing": 10,
        "total": 70,
    }
    assert calls[0][0] == "constructed"
    assert calls[0][1]["write_policy"] is LightWritePolicy.READ_ONLY
    assert calls[1] == ("read", PRIVATE_NOTE_ID)
    assert calls[2] == ("closed",)
    assert private_text not in serialized
    assert private_title not in serialized
    assert PRIVATE_NOTE_ID not in serialized
    assert "content_sha256" not in serialized


@pytest.mark.parametrize(
    ("payload", "failed_stage"),
    [
        (b"Assistant, " + b"x" * (64 * 1024), "source_validation"),
        (b"Assistant, \xff", "text_decode"),
    ],
)
def test_text_audit_rejects_oversized_or_invalid_utf8_without_persisting_it(
    tmp_path: Path,
    payload: bytes,
    failed_stage: str,
) -> None:
    env = environment(tmp_path)
    env["LIGHTPHONE_GPT_INPUT_AUDIT_KIND"] = "text"
    path = receipt_path(tmp_path)

    class InvalidTextProvider:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def read_text_with_metadata(self, note_id: str):
            return RemoteNote(note_id, NoteKind.TEXT, PRIVATE_TITLE), payload

        def close(self) -> None:
            pass

    with pytest.raises(input_audit.LightInputAuditError) as raised:
        input_audit.run_input_audit(
            receipt_path=path,
            confirm_read=True,
            environ=env,
            provider_factory=InvalidTextProvider,
            transcriber_factory=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("text input must bypass transcription")
            ),
        )

    receipt = path.read_text(encoding="utf-8")
    assert raised.value.category == "input_audit_failed"
    assert json.loads(receipt)["result"] == {
        "error": "input_audit_failed",
        "failed_stage": failed_stage,
    }
    assert "Assistant" not in receipt
    assert "xxxxxxxx" not in receipt
    assert PRIVATE_TITLE not in receipt


def test_model_preparation_dereferences_pin_and_writes_complete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = private_directory(tmp_path / "install-cache")
    repository_cache = cache / "models--Systran--faster-whisper-base"
    blobs = repository_cache / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "config").write_bytes(b"model-config")
    snapshot = (
        repository_cache
        / "snapshots"
        / input_audit.INPUT_AUDIT_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").symlink_to("../../blobs/config")
    (snapshot / "model.bin").write_bytes(b"model-weights")
    destination = tmp_path / "candidate-model"
    manifest = tmp_path / "candidate-model.manifest.json"
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    input_audit.prepare_input_audit_model(
        cache, destination, manifest
    )

    assert calls == [
        {
            "repo_id": input_audit.INPUT_AUDIT_MODEL_REPOSITORY,
            "revision": input_audit.INPUT_AUDIT_MODEL_REVISION,
            "cache_dir": str(cache),
            "local_files_only": False,
        }
    ]
    assert (destination / "config.json").read_bytes() == b"model-config"
    assert (destination / "model.bin").read_bytes() == b"model-weights"
    assert not any(path.is_symlink() for path in destination.rglob("*"))
    input_audit.verify_input_audit_model(destination, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert [entry["path"] for entry in payload["files"]] == [
        "config.json",
        "model.bin",
    ]


def test_model_preparation_rejects_special_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = private_directory(tmp_path / "install-cache")
    snapshot = cache / "snapshots" / input_audit.INPUT_AUDIT_MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    os.mkfifo(snapshot / "special")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **_kwargs: str(snapshot)),
    )
    destination = tmp_path / "candidate-model"
    manifest = tmp_path / "candidate-model.manifest.json"

    with pytest.raises(input_audit.LightInputAuditError) as raised:
        input_audit.prepare_input_audit_model(
            cache, destination, manifest
        )

    assert raised.value.category == "configuration_error"
    assert not destination.exists()
    assert not manifest.exists()


def test_candidate_model_tamper_does_not_affect_an_older_candidate(
    tmp_path: Path,
) -> None:
    old_root = private_directory(tmp_path / "old-version")
    new_root = private_directory(tmp_path / "new-version")
    old_model, old_manifest = model_bundle(old_root, b"old-model")
    new_model, new_manifest = model_bundle(new_root, b"new-model")

    (new_model / "model.bin").write_bytes(b"tampered-new-model")

    input_audit.verify_input_audit_model(old_model, old_manifest)
    with pytest.raises(input_audit.LightInputAuditError) as raised:
        input_audit.verify_input_audit_model(new_model, new_manifest)
    assert raised.value.category == "configuration_error"


def test_missing_confirmation_or_config_stops_before_cloud(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    path = receipt_path(tmp_path)
    calls = 0

    def forbidden_provider(**_kwargs: object):
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not be constructed")

    with pytest.raises(input_audit.LightInputAuditError) as unconfirmed:
        input_audit.run_input_audit(
            receipt_path=path,
            confirm_read=False,
            environ=env,
            provider_factory=forbidden_provider,
        )
    assert unconfirmed.value.category == "confirmation_required"
    assert not path.exists()


def test_tampered_candidate_model_stops_before_cloud(tmp_path: Path) -> None:
    env = environment(tmp_path)
    path = receipt_path(tmp_path)
    model_file = Path(env["LIGHTPHONE_GPT_INPUT_AUDIT_MODEL_DIR"]) / "model.bin"
    model_file.write_bytes(b"tampered-after-install")
    calls = 0

    def forbidden_provider(**_kwargs: object):
        nonlocal calls
        calls += 1
        raise AssertionError("cloud must not be accessed with a tampered model")

    with pytest.raises(input_audit.LightInputAuditError) as raised:
        input_audit.run_input_audit(
            receipt_path=path,
            confirm_read=True,
            environ=env,
            provider_factory=forbidden_provider,
        )

    assert raised.value.category == "configuration_error"
    assert calls == 0
    assert not path.exists()

    env["LIGHTPHONE_GPT_INPUT_AUDIT_NOTE_ID"] = "unsafe/id"
    with pytest.raises(input_audit.LightInputAuditError) as misconfigured:
        input_audit.run_input_audit(
            receipt_path=path,
            confirm_read=True,
            environ=env,
            provider_factory=forbidden_provider,
        )
    assert misconfigured.value.category == "configuration_error"
    assert calls == 0
    assert not path.exists()


def test_existing_receipt_prevents_a_second_download(tmp_path: Path) -> None:
    env = environment(tmp_path)
    path = receipt_path(tmp_path)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pending",
                "result": None,
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)

    with pytest.raises(input_audit.LightInputAuditError) as raised:
        input_audit.run_input_audit(
            receipt_path=path,
            confirm_read=True,
            environ=env,
            provider_factory=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("provider must not be constructed")
            ),
        )

    assert raised.value.category == "receipt_already_exists"


def test_failure_receipt_and_stderr_do_not_expose_private_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = environment(tmp_path)
    path = receipt_path(tmp_path)
    cloud = FakeCloud()

    class FailingProvider(FakeProvider):
        def download_audio_with_metadata(self, note_id: str):
            self.cloud.download_calls += 1
            raise RuntimeError(
                f"upstream exposed {note_id} {PRIVATE_TITLE} {PRIVATE_TRANSCRIPT}"
            )

    def command(args, environ):
        return input_audit.run_input_audit(
            receipt_path=args.receipt,
            confirm_read=args.confirm_read,
            environ=environ,
            provider_factory=lambda **kwargs: FailingProvider(cloud, **kwargs),
        )

    monkeypatch.setattr(cli, "run_light_input_audit_command", command)
    stdout = StringIO()
    stderr = StringIO()
    code = cli.main(
        [
            "light-input-audit",
            "--receipt",
            str(path),
            "--confirm-read",
        ],
        environ=env,
        stdout=stdout,
        stderr=stderr,
    )
    receipt = path.read_text(encoding="utf-8")
    public = stderr.getvalue() + receipt

    assert code == 1
    assert not stdout.getvalue()
    assert json.loads(stderr.getvalue())["error"] == "input_audit_failed"
    assert json.loads(receipt)["result"] == {
        "error": "input_audit_failed",
        "failed_stage": "cloud_download",
    }
    assert cloud.download_calls == 1
    assert cloud.close_calls == 1
    for private in (
        PRIVATE_NOTE_ID,
        PRIVATE_TITLE,
        PRIVATE_TRANSCRIPT,
        env["lightphone_username"],
        env["lightphone_password"],
    ):
        assert private not in public


def test_cli_command_does_not_build_continuous_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = environment(tmp_path)
    path = receipt_path(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("continuous runtime must remain isolated")

    monkeypatch.setattr(cli, "build_runtime", forbidden)
    monkeypatch.setattr(
        cli,
        "run_light_input_audit_command",
        lambda _args, _env: {"ok": True, "route_kind": "question"},
    )
    stdout = StringIO()
    stderr = StringIO()

    code = cli.main(
        [
            "light-input-audit",
            "--receipt",
            str(path),
            "--confirm-read",
        ],
        environ=env,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert json.loads(stdout.getvalue())["route_kind"] == "question"
    assert not stderr.getvalue()


def test_parser_rejects_note_id_argument_without_echoing_it(
    tmp_path: Path,
) -> None:
    stdout = StringIO()
    stderr = StringIO()

    code = cli.main(
        [
            "light-input-audit",
            "--note-id",
            PRIVATE_NOTE_ID,
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--confirm-read",
        ],
        environ=environment(tmp_path),
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 2
    assert not stdout.getvalue()
    assert json.loads(stderr.getvalue()) == {
        "ok": False,
        "error": "configuration_error",
    }
    assert PRIVATE_NOTE_ID not in stderr.getvalue()


def test_routing_can_report_ignored_without_retaining_transcript(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    path = receipt_path(tmp_path)
    cloud = FakeCloud()
    private_ordinary = "This ordinary recording mentions my assistant later"

    class OrdinaryTranscriber:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def transcribe(self, _audio: bytes, *, mime_type: str | None = None) -> str:
            assert mime_type == "audio/mpeg"
            return private_ordinary

    result = input_audit.run_input_audit(
        receipt_path=path,
        confirm_read=True,
        environ=env,
        provider_factory=lambda **kwargs: FakeProvider(cloud, **kwargs),
        transcriber_factory=OrdinaryTranscriber,
    )

    public = json.dumps(result) + path.read_text(encoding="utf-8")
    assert result["ok"] is False
    assert result["pipeline_completed"] is True
    assert result["accepted"] is False
    assert result["error"] == "prefix_not_recognized"
    assert result["prefix_recognized"] is False
    assert result["route_kind"] == "ignored"
    assert private_ordinary not in public


def test_audio_audit_accepts_observed_vocative_without_retaining_it(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    path = receipt_path(tmp_path)
    cloud = FakeCloud()
    private_vocative = "Do you, Assistant, remember the earlier answer?"

    class VocativeTranscriber:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def transcribe(self, _audio: bytes, *, mime_type: str | None = None) -> str:
            assert mime_type == "audio/mpeg"
            return private_vocative

    result = input_audit.run_input_audit(
        receipt_path=path,
        confirm_read=True,
        environ=env,
        provider_factory=lambda **kwargs: FakeProvider(cloud, **kwargs),
        transcriber_factory=VocativeTranscriber,
    )

    public = json.dumps(result) + path.read_text(encoding="utf-8")
    assert result["ok"] is True
    assert result["accepted"] is True
    assert result["route_kind"] == "question"
    assert private_vocative not in public


def test_text_audit_keeps_vocative_form_strictly_rejected(tmp_path: Path) -> None:
    env = environment(tmp_path)
    env["LIGHTPHONE_GPT_INPUT_AUDIT_KIND"] = "text"
    path = receipt_path(tmp_path)
    private_vocative = "Do you, Assistant, process this typed note?"

    class TextProvider:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def read_text_with_metadata(self, note_id: str):
            return (
                RemoteNote(note_id, NoteKind.TEXT, PRIVATE_TITLE),
                private_vocative.encode("utf-8"),
            )

        def close(self) -> None:
            return None

    result = input_audit.run_input_audit(
        receipt_path=path,
        confirm_read=True,
        environ=env,
        provider_factory=TextProvider,
        transcriber_factory=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("text input must bypass transcription")
        ),
    )

    public = json.dumps(result) + path.read_text(encoding="utf-8")
    assert result["ok"] is False
    assert result["accepted"] is False
    assert result["route_kind"] == "ignored"
    assert result["error"] == "prefix_not_recognized"
    assert private_vocative not in public


def test_stale_audit_audio_is_removed_but_unrelated_temp_data_is_preserved(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    temp_root = Path(env["LIGHTPHONE_GPT_INPUT_AUDIT_TEMP_DIR"])
    stale = temp_root / "lightphone-gpt-stt-interrupted"
    stale.mkdir()
    (stale / "input.mp3").write_bytes(PRIVATE_AUDIO)
    unrelated = temp_root / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("unrelated", encoding="utf-8")
    cloud = FakeCloud()

    input_audit.run_input_audit(
        receipt_path=receipt_path(tmp_path),
        confirm_read=True,
        environ=env,
        provider_factory=lambda **kwargs: FakeProvider(cloud, **kwargs),
        transcriber_factory=lambda **kwargs: FakeTranscriber([], **kwargs),
    )

    assert not stale.exists()
    assert (unrelated / "keep").read_text(encoding="utf-8") == "unrelated"


def test_transcriber_failure_removes_temporary_audio_and_records_safe_failure(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    path = receipt_path(tmp_path)
    cloud = FakeCloud()

    class FailingTranscriber:
        def __init__(self, **kwargs: object) -> None:
            self.temp_root = Path(kwargs["temp_root"])

        def transcribe(self, audio: bytes, *, mime_type: str | None = None) -> str:
            assert mime_type == "audio/mpeg"
            work = self.temp_root / "lightphone-gpt-stt-failed"
            work.mkdir()
            (work / "input.mp3").write_bytes(audio)
            raise RuntimeError(f"failed on {PRIVATE_TRANSCRIPT}")

    with pytest.raises(input_audit.LightInputAuditError) as raised:
        input_audit.run_input_audit(
            receipt_path=path,
            confirm_read=True,
            environ=env,
            provider_factory=lambda **kwargs: FakeProvider(cloud, **kwargs),
            transcriber_factory=FailingTranscriber,
        )

    receipt = path.read_text(encoding="utf-8")
    temp_root = Path(env["LIGHTPHONE_GPT_INPUT_AUDIT_TEMP_DIR"])
    assert raised.value.category == "input_audit_failed"
    assert json.loads(receipt)["result"] == {
        "error": "input_audit_failed",
        "failed_stage": "transcription",
    }
    assert not list(temp_root.glob("lightphone-gpt-stt-*"))
    assert PRIVATE_AUDIO.decode() not in receipt
    assert PRIVATE_TRANSCRIPT not in receipt


def test_cli_returns_failure_when_prefix_is_not_recognized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "run_light_input_audit_command",
        lambda _args, _env: {
            "ok": False,
            "pipeline_completed": True,
            "accepted": False,
            "error": "prefix_not_recognized",
        },
    )
    stdout = StringIO()
    stderr = StringIO()

    code = cli.main(
        [
            "light-input-audit",
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--confirm-read",
        ],
        environ=environment(tmp_path),
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert json.loads(stdout.getvalue())["accepted"] is False
    assert not stderr.getvalue()
