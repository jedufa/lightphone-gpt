from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lightphone_gpt.output_titles import format_output_title, is_output_title


def test_output_title_uses_configured_timezone_and_daylight_saving() -> None:
    assert (
        format_output_title(
            "AI READY",
            datetime(2026, 1, 15, 18, 35, tzinfo=UTC),
            "America/New_York",
        )
        == "AI READY 13:35"
    )
    assert (
        format_output_title(
            "AI READY",
            datetime(2026, 7, 15, 17, 35, tzinfo=UTC),
            "America/New_York",
        )
        == "AI READY 13:35"
    )


def test_output_title_requires_aware_time_and_valid_iana_zone() -> None:
    with pytest.raises(ValueError, match="completion time"):
        format_output_title(
            "AI READY", datetime(2026, 1, 15, 18, 35), "UTC"
        )
    with pytest.raises(ValueError, match="LOCAL_TIMEZONE"):
        format_output_title(
            "AI READY", datetime(2026, 1, 15, 18, 35, tzinfo=UTC), "Moon/Base"
        )


@pytest.mark.parametrize(
    "title",
    [
        "AI READY",
        "AI READY 00:00",
        "AI READY 09:07",
        "AI READY 19:35",
        "AI READY 23:59",
    ],
)
def test_output_title_policy_accepts_generated_and_legacy_titles(title: str) -> None:
    assert is_output_title(title, "AI READY")


@pytest.mark.parametrize(
    "title",
    [
        "AI READY ",
        "AI READY 9:07",
        "AI READY 24:00",
        "AI READY 19:60",
        "AI READY 19:35 extra",
        "AI READY TOMORROW",
        "MY AI READY 19:35",
    ],
)
def test_output_title_policy_rejects_noncanonical_variants(title: str) -> None:
    assert not is_output_title(title, "AI READY")
