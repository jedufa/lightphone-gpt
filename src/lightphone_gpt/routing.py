"""Turn a local transcript into one of the deliberately small interactions."""

from __future__ import annotations

import re

from .models import RouteKind, RouteResult


_PREFIX = re.compile(r"^\s*assistant\b(?P<rest>.*)$", re.IGNORECASE | re.DOTALL)
_SEPARATOR = re.compile(r"^[\s,:;.!?\-\u2013\u2014]+")
_SPOKEN_FILLERS = ("uh", "um")
_SPOKEN_FILLER_PREFIX = re.compile(
    rf"^\s*(?:{'|'.join(_SPOKEN_FILLERS)})\b"
    r"[\s,:\-\u2013\u2014]+(?P<wake>assistant)\b(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_SPOKEN_VOCATIVE_PREFIX = re.compile(
    r"^\s*(?P<preamble>do\s+you)\b"
    r"[\s,:\-\u2013\u2014]+(?P<wake>assistant)\b(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_MAX_SPOKEN_PREFIX_CHARS = 24


def route_transcript(transcript: str) -> RouteResult:
    """Recognize the strict text prefix and two deterministic commands.

    The prefix is case-insensitive and may be followed by ordinary punctuation.
    Content before the prefix is rejected so an incidental mention of the word
    "assistant" cannot submit a text note.
    """

    match = _PREFIX.match(transcript)
    if match is None:
        return RouteResult(RouteKind.IGNORED)

    return _route_after_assistant(match.group("rest"))


def route_spoken_transcript(transcript: str) -> RouteResult:
    """Route speech while tolerating narrowly defined recognition artifacts.

    Local speech recognition can put a short filler before the wake word or
    render a leading question as a vocative (``Do you, Assistant, ...``).
    Only explicit sentence-initial forms are accepted. Arbitrary leading words
    and later mentions remain ignored so an ordinary recording is not submitted
    merely because it contains the word ``assistant``.
    """

    direct = _PREFIX.match(transcript)
    if direct is not None:
        return _route_after_assistant(direct.group("rest"))

    filler = _SPOKEN_FILLER_PREFIX.match(transcript)
    if filler is not None and filler.start("wake") <= _MAX_SPOKEN_PREFIX_CHARS:
        return _route_after_assistant(filler.group("rest"))

    vocative = _SPOKEN_VOCATIVE_PREFIX.match(transcript)
    if (
        vocative is not None
        and vocative.start("wake") <= _MAX_SPOKEN_PREFIX_CHARS
    ):
        body = _body_after_assistant(vocative.group("rest"))
        if body is not None:
            return RouteResult(RouteKind.QUESTION, f"do you {body}")

    return RouteResult(RouteKind.IGNORED)


def _route_after_assistant(rest: str) -> RouteResult:
    body = _body_after_assistant(rest)
    if body is None:
        return RouteResult(RouteKind.IGNORED)

    command = re.sub(r"[\s.!?]+$", "", body).strip().casefold()
    command = re.sub(r"\s+", " ", command)
    if command == "continue":
        return RouteResult(RouteKind.CONTINUE)
    if command == "new conversation":
        return RouteResult(RouteKind.NEW_CONVERSATION)
    if command == "reset":
        return RouteResult(RouteKind.RESET)
    return RouteResult(RouteKind.QUESTION, body)


def _body_after_assistant(rest: str) -> str | None:
    """Return content after a complete wake word, or ``None`` if malformed."""

    if rest and not (rest[0].isspace() or _SEPARATOR.match(rest)):
        return None
    body = _SEPARATOR.sub("", rest, count=1).strip()
    if not body:
        return None
    return body
