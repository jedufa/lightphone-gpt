"""Strict generation and recognition of bridge output-note titles."""

from __future__ import annotations

from datetime import datetime
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_LOCAL_TIMEZONE = "UTC"
_CLOCK_SUFFIX = re.compile(r" (?:[01][0-9]|2[0-3]):[0-5][0-9]\Z")


def local_timezone(name: str) -> ZoneInfo:
    """Resolve one explicit IANA timezone or reject it at configuration time."""

    if not name or name != name.strip():
        raise ValueError("LIGHTPHONE_GPT_LOCAL_TIMEZONE is invalid")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("LIGHTPHONE_GPT_LOCAL_TIMEZONE is invalid") from None


def format_output_title(prefix: str, completed_at: datetime, timezone_name: str) -> str:
    """Render the answer completion time in the configured local timezone."""

    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise ValueError("output completion time must include a timezone")
    local = completed_at.astimezone(local_timezone(timezone_name))
    return f"{prefix} {local:%H:%M}"


def is_output_title(title: str, prefix: str) -> bool:
    """Accept generated titles plus the exact legacy title during migration."""

    if title == prefix:
        return True
    if not title.startswith(prefix):
        return False
    return _CLOCK_SUFFIX.fullmatch(title[len(prefix) :]) is not None
