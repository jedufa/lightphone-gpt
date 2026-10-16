from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lightphone_gpt.formatting import format_response, publication_completed_at


def test_publication_completion_minute_round_trips_from_stored_content() -> None:
    completed = datetime(2026, 10, 16, 17, 35, 49, tzinfo=UTC)
    content = format_response("Question", "Answer", completed_at=completed)

    assert publication_completed_at(content) == datetime(
        2026, 10, 16, 17, 35, tzinfo=UTC
    )


@pytest.mark.parametrize(
    "content",
    [
        "Completed 2026-10-16 17:35 UTC",
        "text\n\nCompleted 2026-13-16 17:35 UTC",
        "text\n\nCompleted 2026-10-16 24:00 UTC",
        "text\n\nCompleted 2026-10-16 17:35 UTC\n",
    ],
)
def test_publication_completion_parser_rejects_noncanonical_content(
    content: str,
) -> None:
    with pytest.raises(ValueError, match="completion time"):
        publication_completed_at(content)
