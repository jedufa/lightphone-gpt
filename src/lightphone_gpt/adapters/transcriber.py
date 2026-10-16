"""Local transcription adapters."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from threading import Lock
from typing import Any

from lightphone_gpt.contracts import PermanentDependencyError

from .errors import TranscriptionAdapterError


DEFAULT_MAX_AUDIO_DURATION_SECONDS = 300.0


def _probe_audio_duration_seconds(path: Path) -> float:
    """Read duration from local container metadata without decoding speech."""

    try:
        import av
    except ImportError:
        raise PermanentDependencyError(
            "local audio duration support is unavailable"
        ) from None

    try:
        with av.open(os.fspath(path), mode="r") as container:
            audio_streams = tuple(container.streams.audio)
            if not audio_streams:
                raise ValueError
            candidates: list[float] = []
            if container.duration is not None:
                candidates.append(float(container.duration) / float(av.time_base))
            for stream in audio_streams:
                if stream.duration is not None and stream.time_base is not None:
                    candidates.append(float(stream.duration * stream.time_base))
    except PermanentDependencyError:
        raise
    except Exception:
        raise PermanentDependencyError(
            "audio duration could not be determined"
        ) from None

    valid = [value for value in candidates if math.isfinite(value) and value > 0]
    if not valid:
        raise PermanentDependencyError(
            "audio duration could not be determined"
        )
    return max(valid)


class FakeTranscriber:
    """Deterministic transcriber that retains only audio digests, never audio."""

    def __init__(
        self,
        transcripts: Iterable[str] = (),
        *,
        by_sha256: Mapping[str, str] | None = None,
        default_transcript: str | None = None,
    ) -> None:
        self._transcripts = deque(transcripts)
        self._by_sha256 = dict(by_sha256 or {})
        self._default = default_transcript
        self._calls: list[tuple[str, str | None]] = []
        self._lock = Lock()

    def transcribe(self, audio: bytes, *, mime_type: str | None = None) -> str:
        digest = hashlib.sha256(audio).hexdigest()
        with self._lock:
            self._calls.append((digest, mime_type))
            transcript = self._by_sha256.get(digest)
            if transcript is None and self._transcripts:
                transcript = self._transcripts.popleft()
            if transcript is None:
                transcript = self._default
        if transcript is None:
            raise PermanentDependencyError(
                f"no fake transcript configured for audio sha256 {digest}"
            )
        normalized = transcript.strip()
        if not normalized:
            raise TranscriptionAdapterError("transcription returned no text")
        return normalized

    @property
    def calls(self) -> tuple[tuple[str, str | None], ...]:
        with self._lock:
            return tuple(self._calls)


class WhisperCppTranscriber:
    """Run a local whisper.cpp CLI with isolated temporary input/output files."""

    _SUFFIXES = {
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/m4a": ".m4a",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
    }

    def __init__(
        self,
        *,
        executable: str | os.PathLike[str],
        model: str | os.PathLike[str],
        timeout_seconds: float = 180.0,
        max_audio_bytes: int = 32 * 1024 * 1024,
        extra_args: Iterable[str] = (),
        temp_root: str | os.PathLike[str] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_audio_bytes < 1:
            raise ValueError("max_audio_bytes must be positive")
        self._executable = os.fspath(executable)
        self._model = Path(model)
        self._timeout_seconds = timeout_seconds
        self._max_audio_bytes = max_audio_bytes
        self._extra_args = tuple(str(value) for value in extra_args)
        self._temp_root = Path(temp_root) if temp_root is not None else None
        self._environment = (
            dict(environment)
            if environment is not None
            else {"PATH": os.defpath, "LC_ALL": "C.UTF-8"}
        )

    def transcribe(self, audio: bytes, *, mime_type: str | None = None) -> str:
        if not audio:
            raise PermanentDependencyError("audio must not be empty")
        if len(audio) > self._max_audio_bytes:
            raise PermanentDependencyError("audio exceeds configured size limit")

        executable = self._resolve_executable()
        if not self._model.is_file():
            raise PermanentDependencyError("whisper.cpp model file is unavailable")
        suffix = self._SUFFIXES.get((mime_type or "").lower(), ".audio")

        try:
            with tempfile.TemporaryDirectory(
                prefix="lightphone-gpt-stt-",
                dir=self._temp_root,
            ) as directory:
                work = Path(directory)
                audio_path = work / f"input{suffix}"
                output_base = work / "transcript"
                audio_path.write_bytes(audio)
                command = [
                    executable,
                    "-m",
                    os.fspath(self._model),
                    "-f",
                    os.fspath(audio_path),
                    "-otxt",
                    "-of",
                    os.fspath(output_base),
                    "-nt",
                    "-np",
                    *self._extra_args,
                ]
                completed = subprocess.run(
                    command,
                    shell=False,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self._timeout_seconds,
                    env=self._environment,
                )
                if completed.returncode != 0:
                    raise TranscriptionAdapterError(
                        "whisper.cpp transcription failed"
                    )
                output_path = Path(f"{output_base}.txt")
                if not output_path.is_file():
                    raise TranscriptionAdapterError(
                        "whisper.cpp produced no transcript file"
                    )
                transcript = output_path.read_text(encoding="utf-8").strip()
        except subprocess.TimeoutExpired:
            raise TranscriptionAdapterError("whisper.cpp transcription timed out") from None
        except OSError:
            raise TranscriptionAdapterError("whisper.cpp process failed") from None

        if not transcript:
            raise TranscriptionAdapterError("transcription returned no text")
        return transcript

    def _resolve_executable(self) -> str:
        candidate = Path(self._executable)
        if candidate.parent != Path(".") or candidate.is_absolute():
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                raise PermanentDependencyError(
                    "whisper.cpp executable is unavailable"
                )
            return os.fspath(candidate)
        resolved = shutil.which(self._executable, path=self._environment.get("PATH"))
        if resolved is None:
            raise PermanentDependencyError("whisper.cpp executable is unavailable")
        return resolved


class FasterWhisperTranscriber:
    """Run faster-whisper locally while keeping recordings in temporary files."""

    def __init__(
        self,
        *,
        model: str = "base",
        model_cache: str | os.PathLike[str] | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        language: str | None = None,
        local_files_only: bool = True,
        max_audio_bytes: int = 32 * 1024 * 1024,
        max_audio_duration_seconds: float = DEFAULT_MAX_AUDIO_DURATION_SECONDS,
        temp_root: str | os.PathLike[str] | None = None,
        model_factory: Callable[..., Any] | None = None,
        duration_probe: Callable[[Path], float | None] | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        if beam_size < 1:
            raise ValueError("beam_size must be positive")
        if max_audio_bytes < 1:
            raise ValueError("max_audio_bytes must be positive")
        try:
            duration_limit = float(max_audio_duration_seconds)
        except (TypeError, ValueError):
            duration_limit = math.nan
        if (
            isinstance(max_audio_duration_seconds, bool)
            or not math.isfinite(duration_limit)
            or duration_limit <= 0
        ):
            raise ValueError("max_audio_duration_seconds must be positive and finite")
        self._model_name = model
        self._model_cache = (
            os.fspath(model_cache) if model_cache is not None else None
        )
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._language = language
        self._local_files_only = local_files_only
        self._max_audio_bytes = max_audio_bytes
        self._max_audio_duration_seconds = duration_limit
        self._temp_root = Path(temp_root) if temp_root is not None else None
        self._model_factory = model_factory
        self._duration_probe = (
            duration_probe
            if duration_probe is not None
            else _probe_audio_duration_seconds
        )
        self._model: Any | None = None
        self._lock = Lock()

    def transcribe(self, audio: bytes, *, mime_type: str | None = None) -> str:
        if not audio:
            raise PermanentDependencyError("audio must not be empty")
        if len(audio) > self._max_audio_bytes:
            raise PermanentDependencyError("audio exceeds configured size limit")
        suffix = WhisperCppTranscriber._SUFFIXES.get(
            (mime_type or "").lower(), ".audio"
        )

        try:
            with tempfile.TemporaryDirectory(
                prefix="lightphone-gpt-stt-",
                dir=self._temp_root,
            ) as directory:
                audio_path = Path(directory) / f"input{suffix}"
                audio_path.write_bytes(audio)
                duration = self._checked_duration(audio_path)
                if duration > self._max_audio_duration_seconds:
                    raise PermanentDependencyError(
                        "audio exceeds configured duration limit"
                    )
                model = self._get_model()
                with self._lock:
                    segments, _info = model.transcribe(
                        os.fspath(audio_path),
                        beam_size=self._beam_size,
                        language=self._language,
                    )
                    parts = [
                        str(segment.text).strip()
                        for segment in segments
                        if str(segment.text).strip()
                    ]
        except (PermanentDependencyError, TranscriptionAdapterError):
            raise
        except Exception:
            raise TranscriptionAdapterError("local transcription failed") from None

        transcript = " ".join(parts).strip()
        if not transcript:
            raise TranscriptionAdapterError("transcription returned no text")
        return transcript

    def _checked_duration(self, path: Path) -> float:
        try:
            duration = self._duration_probe(path)
            if isinstance(duration, bool) or duration is None:
                raise ValueError
            normalized = float(duration)
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError
            return normalized
        except Exception:
            raise PermanentDependencyError(
                "audio duration could not be determined"
            ) from None

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            factory = self._model_factory
            if factory is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError:
                    raise PermanentDependencyError(
                        "faster-whisper is not installed"
                    ) from None
                factory = WhisperModel
            try:
                self._model = factory(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                    download_root=self._model_cache,
                    local_files_only=self._local_files_only,
                )
            except Exception:
                raise PermanentDependencyError(
                    "faster-whisper model is unavailable"
                ) from None
            return self._model
