from __future__ import annotations

import json

import pytest

from lightphone_gpt.health import (
    HealthServer,
    HealthSnapshot,
    HealthState,
    health_response,
)


def test_health_server_reports_only_typed_aggregate_state() -> None:
    state = HealthState(
        HealthSnapshot(
            phase="running",
            ready=True,
            database_ok=True,
            cycles=3,
            last_cycle_ok=True,
            jobs_total=9,
            jobs_actionable=2,
            jobs_failed=1,
        )
    )
    status, payload = health_response(state, "/health")

    assert status == 200
    assert payload["jobs_total"] == 9
    assert payload["jobs_actionable"] == 2
    encoded = json.dumps(payload)
    assert "note" not in encoded.lower()
    assert "transcript" not in encoded.lower()


def test_unready_health_is_503_and_unknown_paths_are_404() -> None:
    health_status, _ = health_response(HealthState(), "/health")
    missing_status, missing_body = health_response(
        HealthState(), "/anything-else"
    )

    assert health_status == 503
    assert missing_status == 404
    assert missing_body == {"error": "not_found"}


@pytest.mark.parametrize(
    ("cycle_ok", "database_ok"),
    [(False, True), (True, False), (False, False)],
)
def test_failed_cycle_or_database_marks_health_unready(
    cycle_ok: bool, database_ok: bool
) -> None:
    state = HealthState()
    snapshot = state.record_cycle(
        ok=cycle_ok,
        database_ok=database_ok,
        jobs_total=1,
        jobs_actionable=1,
        jobs_failed=0,
    )
    status, _ = health_response(state, "/health")

    assert snapshot.ready is False
    assert snapshot.phase == "degraded"
    assert status == 503


def test_output_ambiguity_marks_health_unready() -> None:
    state = HealthState()
    snapshot = state.record_cycle(
        ok=True,
        database_ok=True,
        jobs_total=2,
        jobs_actionable=1,
        jobs_failed=1,
        output_blocked=True,
    )
    status, payload = health_response(state, "/health")

    assert snapshot.ready is False
    assert snapshot.phase == "degraded"
    assert payload["output_blocked"] is True
    assert status == 503


def test_unresolved_failure_degrades_but_completed_failure_is_historical() -> None:
    state = HealthState()
    unresolved = state.record_cycle(
        ok=True,
        database_ok=True,
        jobs_total=1,
        jobs_actionable=1,
        jobs_failed=1,
        jobs_unresolved_failed=1,
        jobs_retry_waiting=1,
    )
    assert unresolved.ready is False
    assert health_response(state, "/health")[0] == 503

    historical = state.record_cycle(
        ok=True,
        database_ok=True,
        jobs_total=1,
        jobs_actionable=0,
        jobs_failed=1,
        jobs_unresolved_failed=0,
        jobs_terminal_failed=1,
    )
    assert historical.ready is True
    assert historical.jobs_failed == 1
    assert health_response(state, "/health")[0] == 200


def test_unresolved_source_cleanup_is_content_free_and_degrades_health() -> None:
    state = HealthState()
    snapshot = state.record_cycle(
        ok=False,
        database_ok=True,
        jobs_total=1,
        jobs_actionable=0,
        jobs_failed=0,
        source_cleanups_unresolved=1,
        source_cleanup_blocked=True,
    )
    status, payload = health_response(state, "/health")

    assert snapshot.ready is False
    assert status == 503
    assert payload["source_cleanups_unresolved"] == 1
    assert payload["source_cleanup_blocked"] is True
    assert "source_note_id" not in json.dumps(payload)


def test_health_server_refuses_non_loopback_bind() -> None:
    with pytest.raises(ValueError, match="loopback"):
        HealthServer("0.0.0.0", 0, HealthState())
