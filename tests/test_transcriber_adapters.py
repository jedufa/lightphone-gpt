from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from lightphone_gpt.adapters.errors import TranscriptionAdapterError
from lightphone_gpt.adapters.transcriber import (
    FakeTranscriber,
    FasterWhisperTranscriber,
    WhisperCppTranscriber,
)
from lightphone_gpt.contracts import PermanentDependencyError, TransientDependencyError


def test_fake_transcriber_routes_by_digest_without_retaining_audio() -> None:
    audio = b"private-audio-bytes"
    digest = hashlib.sha256(audio).hexdigest()
    adapter = FakeTranscriber(by_sha256={digest: " Assistant, hello. "})

    assert adapter.transcribe(audio, mime_type="audio/mp4") == "Assistant, hello."
    assert adapter.calls == ((digest, "audio/mp4"),)
    assert audio not in repr(adapter.calls).encode()


def test_whisper_cpp_runs_without_a_shell_and_reads_output_file(tmp_path: Path) -> None:
    executable = tmp_path / "fake whisper executable"
    executable.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys
output = pathlib.Path(sys.argv[sys.argv.index('-of') + 1] + '.txt')
source = pathlib.Path(sys.argv[sys.argv.index('-f') + 1])
assert source.suffix == '.m4a'
assert source.read_bytes() == b'fake-m4a'
output.write_text(' Assistant, simulated transcript. \\n', encoding='utf-8')
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model-placeholder")
    temp_root = tmp_path / "private-temp"
    temp_root.mkdir()

    adapter = WhisperCppTranscriber(
        executable=executable,
        model=model,
        temp_root=temp_root,
    )

    assert (
        adapter.transcribe(b"fake-m4a", mime_type="audio/mp4")
        == "Assistant, simulated transcript."
    )
    assert list(temp_root.iterdir()) == []


def test_whisper_cpp_sanitizes_process_output(tmp_path: Path) -> None:
    executable = tmp_path / "failing-whisper"
    executable.write_text(
        "#!/bin/sh\necho 'secret transcript material' >&2\nexit 9\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model-placeholder")
    adapter = WhisperCppTranscriber(executable=executable, model=model)

    with pytest.raises(TranscriptionAdapterError) as captured:
        adapter.transcribe(b"audio", mime_type="audio/wav")

    assert "secret transcript material" not in str(captured.value)


def test_faster_whisper_is_lazy_local_and_cleans_staged_audio(tmp_path: Path) -> None:
    factory_arguments: dict[str, object] = {}
    staged_paths: list[Path] = []

    class FakeModel:
        def transcribe(self, path, **kwargs):
            staged = Path(path)
            staged_paths.append(staged)
            assert staged.read_bytes() == b"fake-m4a"
            assert kwargs == {"beam_size": 3, "language": "en"}

            def segments():
                yield SimpleNamespace(text=" Assistant,")
                yield SimpleNamespace(text=" hello. ")

            return segments(), SimpleNamespace(language="en")

    def model_factory(name, **kwargs):
        factory_arguments.update(name=name, **kwargs)
        return FakeModel()

    cache = tmp_path / "model-cache"
    temp_root = tmp_path / "private-temp"
    temp_root.mkdir()
    adapter = FasterWhisperTranscriber(
        model="tiny.en",
        model_cache=cache,
        device="cpu",
        compute_type="int8",
        beam_size=3,
        language="en",
        local_files_only=True,
        temp_root=temp_root,
        model_factory=model_factory,
        duration_probe=lambda _path: 2.0,
    )

    assert adapter.transcribe(b"fake-m4a", mime_type="audio/mp4") == "Assistant, hello."
    assert factory_arguments == {
        "name": "tiny.en",
        "device": "cpu",
        "compute_type": "int8",
        "download_root": os.fspath(cache),
        "local_files_only": True,
    }
    assert staged_paths and not staged_paths[0].exists()
    assert list(temp_root.iterdir()) == []


def test_staged_audio_path_uses_isolated_directory_and_suffix_allowlist(
    tmp_path: Path,
) -> None:
    temp_root = tmp_path / "private-temp"
    temp_root.mkdir()
    observed: list[Path] = []

    class FakeModel:
        def transcribe(self, path, **_kwargs):
            staged = Path(path)
            observed.append(staged)
            assert staged.name == "input.audio"
            assert staged.parent.parent == temp_root
            return iter([SimpleNamespace(text="Assistant, safe")]), SimpleNamespace()

    adapter = FasterWhisperTranscriber(
        temp_root=temp_root,
        model_factory=lambda *_args, **_kwargs: FakeModel(),
        duration_probe=lambda _path: 1.0,
    )

    assert adapter.transcribe(b"audio", mime_type="../../escape") == "Assistant, safe"
    assert observed and not observed[0].exists()
    assert list(temp_root.iterdir()) == []


def test_faster_whisper_missing_model_is_a_sanitized_configuration_error() -> None:
    def failing_factory(*_args, **_kwargs):
        raise RuntimeError("/private/model/path and token")

    adapter = FasterWhisperTranscriber(
        model_factory=failing_factory,
        duration_probe=lambda _path: 1.0,
    )
    with pytest.raises(PermanentDependencyError) as captured:
        adapter.transcribe(b"audio")

    assert "/private/model/path" not in str(captured.value)


def test_transcription_operational_errors_are_retryable() -> None:
    assert isinstance(
        TranscriptionAdapterError("temporary"), TransientDependencyError
    )


@pytest.mark.parametrize(
    "duration_limit",
    [0, -1, math.nan, math.inf, -math.inf, True, "invalid"],
)
def test_faster_whisper_duration_limit_must_be_positive_and_finite(
    duration_limit: object,
) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        FasterWhisperTranscriber(
            max_audio_duration_seconds=duration_limit,  # type: ignore[arg-type]
        )


def test_faster_whisper_rejects_overlong_audio_before_loading_model(
    tmp_path: Path,
) -> None:
    temp_root = tmp_path / "private-temp"
    temp_root.mkdir()
    probe_paths: list[Path] = []
    model_loads = 0

    def duration_probe(path: Path) -> float:
        probe_paths.append(path)
        assert path.read_bytes() == b"private-audio-content"
        return 300.01

    def forbidden_model(*_args, **_kwargs):
        nonlocal model_loads
        model_loads += 1
        raise AssertionError("model must not load for rejected audio")

    adapter = FasterWhisperTranscriber(
        temp_root=temp_root,
        model_factory=forbidden_model,
        duration_probe=duration_probe,
    )

    with pytest.raises(PermanentDependencyError, match="duration limit") as captured:
        adapter.transcribe(b"private-audio-content", mime_type="audio/mp4")

    assert "private-audio-content" not in str(captured.value)
    assert model_loads == 0
    assert probe_paths and not probe_paths[0].exists()
    assert list(temp_root.iterdir()) == []


@pytest.mark.parametrize(
    "observed_duration",
    [None, 0, -1, math.nan, math.inf, -math.inf, True, "invalid"],
)
def test_faster_whisper_rejects_unknown_or_invalid_duration_before_model(
    tmp_path: Path, observed_duration: object
) -> None:
    model_loads = 0

    def forbidden_model(*_args, **_kwargs):
        nonlocal model_loads
        model_loads += 1
        raise AssertionError("model must not load for invalid duration")

    adapter = FasterWhisperTranscriber(
        temp_root=tmp_path,
        model_factory=forbidden_model,
        duration_probe=lambda _path: observed_duration,  # type: ignore[return-value]
    )

    with pytest.raises(
        PermanentDependencyError, match="duration could not be determined"
    ):
        adapter.transcribe(b"private-audio")

    assert model_loads == 0
    assert list(tmp_path.iterdir()) == []


def test_faster_whisper_sanitizes_duration_probe_failure(tmp_path: Path) -> None:
    def failing_probe(_path: Path) -> float:
        raise RuntimeError("private audio filename and metadata")

    adapter = FasterWhisperTranscriber(
        temp_root=tmp_path,
        model_factory=lambda *_args, **_kwargs: pytest.fail(
            "model must not load when probing fails"
        ),
        duration_probe=failing_probe,
    )

    with pytest.raises(PermanentDependencyError) as captured:
        adapter.transcribe(b"private-audio")

    assert "private audio filename" not in str(captured.value)
    assert "private-audio" not in str(captured.value)
    assert list(tmp_path.iterdir()) == []


def test_faster_whisper_default_probe_uses_local_pyav_metadata_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []
    model_calls = 0

    class Container:
        duration = 250_000
        streams = SimpleNamespace(
            audio=(SimpleNamespace(duration=None, time_base=None),)
        )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def open_container(path, *, mode):
        opened.append(Path(path))
        assert mode == "r"
        assert Path(path).read_bytes() == b"synthetic-container"
        return Container()

    fake_av = SimpleNamespace(open=open_container, time_base=1_000_000)
    monkeypatch.setitem(sys.modules, "av", fake_av)

    class FakeModel:
        def transcribe(self, _path, **_kwargs):
            nonlocal model_calls
            model_calls += 1
            return iter([SimpleNamespace(text="Assistant, safe")]), SimpleNamespace()

    adapter = FasterWhisperTranscriber(
        max_audio_duration_seconds=0.25,
        temp_root=tmp_path,
        model_factory=lambda *_args, **_kwargs: FakeModel(),
    )

    assert adapter.transcribe(b"synthetic-container") == "Assistant, safe"
    assert model_calls == 1
    assert opened and not opened[0].exists()
    assert list(tmp_path.iterdir()) == []
