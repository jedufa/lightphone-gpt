"""Light Phone-sized response splitting and plain-text rendering."""

from __future__ import annotations

from datetime import datetime, timezone
import re


_COMPLETION_SUFFIX = re.compile(
    r"\n\nCompleted (?P<minute>[0-9]{4}-[0-9]{2}-[0-9]{2} "
    r"[0-9]{2}:[0-9]{2}) UTC\Z"
)


def _normalize(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value


def split_answer(
    answer: str,
    *,
    min_chars: int = 400,
    target_chars: int = 520,
    max_chars: int = 600,
) -> tuple[str, ...]:
    """Split once, persist the sections, and let `continue` reveal them."""

    if not 0 < min_chars <= target_chars <= max_chars:
        raise ValueError("expected 0 < min <= target <= max")
    remaining = _normalize(answer)
    if not remaining:
        raise ValueError("answer cannot be empty")
    sections: list[str] = []
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        candidates: list[int] = []
        for pattern in (r"\n\n", r"(?<=[.!?])\s+", r"\s+"):
            candidates = [
                match.start()
                for match in re.finditer(pattern, window)
                if min_chars <= match.start() <= max_chars
            ]
            if candidates:
                break
        if candidates:
            cut = min(candidates, key=lambda value: (abs(value - target_chars), -value))
        else:
            fallback = window.rfind(" ", 1, max_chars + 1)
            cut = fallback if fallback > 0 else max_chars
        section = remaining[:cut].strip()
        if not section:
            section = remaining[:max_chars]
            cut = max_chars
        sections.append(section)
        remaining = remaining[cut:].strip()
    if remaining:
        sections.append(remaining)
    return tuple(sections)


def format_response(
    transcript: str,
    section: str,
    *,
    completed_at: datetime,
) -> str:
    timestamp = completed_at
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return (
        f"YOU SAID\n{_normalize(transcript)}\n\n"
        f"ANSWER\n{_normalize(section)}\n\n"
        f"Completed {timestamp:%Y-%m-%d %H:%M} UTC"
    )


def publication_completed_at(content: str) -> datetime:
    """Recover the durable completion minute from bridge-rendered content."""

    match = _COMPLETION_SUFFIX.search(content)
    if match is None:
        raise ValueError("stored publication has no valid completion time")
    try:
        parsed = datetime.strptime(match.group("minute"), "%Y-%m-%d %H:%M")
    except ValueError:
        raise ValueError("stored publication has no valid completion time") from None
    return parsed.replace(tzinfo=timezone.utc)
