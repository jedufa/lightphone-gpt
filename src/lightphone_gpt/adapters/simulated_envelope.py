"""Strict transcript envelopes for simulator-only audio notes.

The simulator deliberately does not treat arbitrary bytes as UTF-8 text.  A
distinct MIME type and magic header make it hard to accidentally route a real
recording through the test transcriber.
"""

from __future__ import annotations

import json
from typing import Final

from lightphone_gpt.contracts import PermanentDependencyError


SIMULATED_TRANSCRIPT_MIME: Final = (
    "application/vnd.lightphone-gpt.simulated-transcript+json;version=1"
)
_MAGIC: Final = b"LIGHTPHONE_GPT_SIMULATED_TRANSCRIPT_V1\n"
_MAX_ENVELOPE_BYTES: Final = 64 * 1024


def encode_simulated_transcript(transcript: str) -> bytes:
    """Encode a synthetic transcript for the durable Light simulator."""

    normalized = transcript.strip()
    if not normalized:
        raise ValueError("simulated transcript must not be blank")
    payload = json.dumps(
        {"transcript": normalized},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    envelope = _MAGIC + payload
    if len(envelope) > _MAX_ENVELOPE_BYTES:
        raise ValueError("simulated transcript envelope is too large")
    return envelope


class SimulatedEnvelopeTranscriber:
    """Decode only explicit simulator envelopes and retain no input bytes."""

    def transcribe(self, audio: bytes, *, mime_type: str | None = None) -> str:
        if mime_type != SIMULATED_TRANSCRIPT_MIME:
            raise PermanentDependencyError(
                "simulated transcriber requires its dedicated MIME type"
            )
        if not audio or len(audio) > _MAX_ENVELOPE_BYTES or not audio.startswith(_MAGIC):
            raise PermanentDependencyError("invalid simulated transcript envelope")
        try:
            decoded = json.loads(audio[len(_MAGIC) :].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise PermanentDependencyError(
                "invalid simulated transcript envelope"
            ) from None
        if not isinstance(decoded, dict) or set(decoded) != {"transcript"}:
            raise PermanentDependencyError("invalid simulated transcript envelope")
        transcript = decoded.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            raise PermanentDependencyError("invalid simulated transcript envelope")
        return transcript.strip()
