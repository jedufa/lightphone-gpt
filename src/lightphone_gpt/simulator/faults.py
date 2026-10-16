"""Small deterministic fault scripts for provider contract tests.

Faults are deliberately explicit: a test selects an operation, call number and
whether the failure happens before or after the operation's durable mutation.
This makes ambiguous-write recovery reproducible without random timing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock

from lightphone_gpt.contracts import TransientDependencyError


class SimulatedProviderError(TransientDependencyError):
    """A synthetic transport or provider failure."""


class FaultPhase(StrEnum):
    BEFORE = "before"
    AFTER = "after"


@dataclass(frozen=True, slots=True)
class _Fault:
    operation: str
    call_number: int
    phase: FaultPhase
    message: str


class FaultScript:
    """Trigger configured failures on exact operation call numbers.

    Call counts include successful calls and are shared by the before/after
    phases of one operation. Each scripted fault is consumed once.
    """

    def __init__(self) -> None:
        self._faults: dict[tuple[str, int, FaultPhase], _Fault] = {}
        self._calls: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def fail(
        self,
        operation: str,
        *,
        call: int = 1,
        phase: FaultPhase | str = FaultPhase.BEFORE,
        message: str = "simulated provider failure",
    ) -> "FaultScript":
        if not operation or call < 1:
            raise ValueError("operation must be non-empty and call must be positive")
        normalized_phase = FaultPhase(phase)
        key = (operation, call, normalized_phase)
        with self._lock:
            if key in self._faults:
                raise ValueError(f"fault already configured for {key!r}")
            self._faults[key] = _Fault(operation, call, normalized_phase, message)
        return self

    def before(self, operation: str) -> None:
        with self._lock:
            self._calls[operation] += 1
            call_number = self._calls[operation]
        self._trigger(operation, call_number, FaultPhase.BEFORE)

    def after(self, operation: str) -> None:
        with self._lock:
            call_number = self._calls.get(operation, 0)
            if call_number == 0:
                raise RuntimeError(f"after called without before for {operation!r}")
        self._trigger(operation, call_number, FaultPhase.AFTER)

    def call_count(self, operation: str) -> int:
        with self._lock:
            return self._calls.get(operation, 0)

    def _trigger(
        self, operation: str, call_number: int, phase: FaultPhase
    ) -> None:
        key = (operation, call_number, phase)
        with self._lock:
            fault = self._faults.pop(key, None)
        if fault is not None:
            raise SimulatedProviderError(
                f"{fault.message} ({operation} call {call_number}, {phase.value})"
            )
