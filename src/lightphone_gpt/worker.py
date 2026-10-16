"""Sequential bridge engine with durable boundaries around every side effect."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Literal

from .contracts import (
    Assistant,
    LightProvider,
    PermanentDependencyError,
    Transcriber,
    TransientDependencyError,
)
from .chatgpt_errors import (
    ChatGPTAccountNotEligibleError,
    ChatGPTAmbiguousOutcomeError,
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTCredentialAmbiguousError,
    ChatGPTPlanUsageLimitError,
)
from .conversation import DEFAULT_SYSTEM_PROMPT, build_prompt
from .formatting import format_response, publication_completed_at, split_answer
from .models import (
    DeletionReceipt,
    EnqueueOutcome,
    JobRecord,
    JobState,
    MutationOutcome,
    MutationReceipt,
    NoteKind,
    RemoteNote,
    RouteKind,
    SourceCleanupRecord,
    SourceCleanupState,
)
from .routing import route_spoken_transcript, route_transcript
from .output_titles import (
    DEFAULT_LOCAL_TIMEZONE,
    format_output_title,
    is_output_title,
    local_timezone as resolve_local_timezone,
)
from .store import (
    MISSING_OUTPUT_UPDATE_ERROR,
    Store,
    sha256_bytes,
    sha256_text,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    output_mode: Literal["rolling", "per_turn"] = "rolling"
    output_title: str = "AI READY"
    local_timezone: str = DEFAULT_LOCAL_TIMEZONE
    max_audio_bytes: int = 25 * 1024 * 1024
    max_text_bytes: int = 64 * 1024
    max_transcription_attempts: int = 8
    max_model_attempts: int = 8
    max_publication_attempts: int = 5
    max_update_readback_attempts: int = 8
    confirm_output_deletion_after: int = 2
    delete_source_after_publish: bool = False
    max_source_delete_readbacks: int = 8
    confirm_source_deletion_after: int = 2
    max_pending_jobs: int = 100
    max_question_chars: int = 8_000
    max_model_calls_per_day: int = 100
    retry_delay: timedelta = timedelta(seconds=30)
    retry_max_delay: timedelta = timedelta(minutes=10)
    answer_min_chars: int = 400
    answer_target_chars: int = 520
    answer_max_chars: int = 600
    max_context_turns: int = 10
    max_context_chars: int = 24_000
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if self.output_mode not in {"rolling", "per_turn"}:
            raise ValueError("output_mode must be rolling or per_turn")
        if (
            not self.output_title
            or self.output_title != self.output_title.strip()
            or len(self.output_title) > 94
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in self.output_title
            )
        ):
            raise ValueError("output_title is invalid")
        resolve_local_timezone(self.local_timezone)
        for value in (
            self.max_audio_bytes,
            self.max_text_bytes,
            self.max_transcription_attempts,
            self.max_model_attempts,
            self.max_publication_attempts,
            self.max_update_readback_attempts,
            self.confirm_output_deletion_after,
            self.max_source_delete_readbacks,
            self.confirm_source_deletion_after,
            self.max_pending_jobs,
            self.max_question_chars,
            self.max_model_calls_per_day,
        ):
            if value <= 0:
                raise ValueError("worker limits must be positive")
        if self.max_context_turns < 0:
            raise ValueError("max_context_turns cannot be negative")
        if self.max_context_chars < self.max_question_chars:
            raise ValueError("max_context_chars must cover one maximum-size question")
        if not (
            0 < self.answer_min_chars
            <= self.answer_target_chars
            <= self.answer_max_chars
        ):
            raise ValueError("answer character limits must be positive and ordered")
        if self.retry_delay.total_seconds() < 0:
            raise ValueError("retry_delay cannot be negative")
        if self.retry_max_delay.total_seconds() <= 0:
            raise ValueError("retry_max_delay must be positive")
        if self.retry_max_delay < self.retry_delay:
            raise ValueError("retry_max_delay cannot be shorter than retry_delay")
        if self.confirm_source_deletion_after < 2:
            raise ValueError("source deletion confirmation requires at least two reads")


@dataclass(frozen=True, slots=True)
class PollReport:
    baselined: int = 0
    enqueued: int = 0
    duplicates: int = 0
    integrity_errors: int = 0
    rejected: int = 0
    ignored: int = 0
    output_retired: str | None = None
    output_gate: str | None = None
    queue_limited: bool = False
    source_cleanup: str | None = None
    source_cleanup_error: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkReport:
    worked: bool
    note_id: str | None = None
    state: JobState | None = None
    detail: str | None = None


class Worker:
    def __init__(
        self,
        *,
        store: Store,
        provider: LightProvider,
        transcriber: Transcriber,
        assistant: Assistant,
        config: WorkerConfig = WorkerConfig(),
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.store = store
        self.provider = provider
        self.transcriber = transcriber
        self.assistant = assistant
        self.config = config
        self.clock = clock
        self.store.configure_source_cleanup(
            enabled=config.delete_source_after_publish,
            now=self.clock(),
        )

    def poll(self) -> PollReport:
        """Reconcile output and discover audio/text in stable cloud order."""

        now = self.clock()
        retired: str | None = None
        try:
            self.store.gate_interrupted_publication(now=now)
            if self.config.output_mode == "rolling":
                retired = self._reconcile_output(now)
        except Exception as error:  # daemon boundary; no state was accepted
            return PollReport(
                output_retired=retired,
                output_gate=self.store.output_gate(),
                error=_safe_error(error),
            )

        output_gate = self.store.output_gate()
        # Discovery remains read-only under either publication gate. A newly
        # admitted Assistant request can then supersede the unconfirmable
        # delivery without repeating it. Under an uncertain create, exclude an
        # exact attempted output title conservatively; a title never grants
        # ownership and is never used to mutate or delete a note.

        try:
            notes = list(self.provider.list_notes())
        except Exception as error:  # daemon boundary; no state was accepted
            return PollReport(
                output_retired=retired,
                output_gate=output_gate,
                error=_safe_error(error),
            )

        cleanup_detail: str | None = None
        cleanup_error: str | None = None
        if self.config.delete_source_after_publish:
            try:
                cleanup_detail = self._reconcile_source_cleanup(notes, now)
            except Exception:
                # Source cleanup is independent of intake/publication. Preserve
                # the ambiguity and continue useful work while health reports
                # the unresolved cleanup.
                cleanup_error = "source cleanup read-back failed"

        if not self.store.is_enrolled():
            count = self.store.establish_baseline(notes, now=now)
            return PollReport(
                baselined=count,
                output_retired=retired,
                source_cleanup=cleanup_detail,
                source_cleanup_error=cleanup_error,
            )

        text_baseline_pass = not self.store.is_text_ingress_enrolled()
        baselined = 0
        if text_baseline_pass:
            baselined = self.store.establish_text_ingress_baseline(notes, now=now)

        enqueued = duplicates = integrity_errors = rejected = ignored = 0
        pending = self.store.pending_job_count()
        queue_limited = False
        assistant_gate = self.store.assistant_gate()
        output_recovery_gate_active = output_gate is not None
        inference_recovery_gate_active = assistant_gate == "inference_ambiguous"
        remote_recovery_gate_active = (
            output_recovery_gate_active or inference_recovery_gate_active
        )
        local_reset_blocker = (
            None
            if remote_recovery_gate_active
            else self.store.local_reset_discovery_blocker(now=now)
        )
        source_notes = sorted(
            (
                note
                for note in notes
                if note.kind in {NoteKind.AUDIO, NoteKind.TEXT}
                and not (text_baseline_pass and note.kind is NoteKind.TEXT)
            ),
            key=_note_order,
        )
        for note in source_notes:
            if self.store.is_known_output_note(note.note_id):
                duplicates += 1
                continue
            if (
                output_gate == "create_ambiguous"
                and note.kind is NoteKind.TEXT
                and self.store.is_possible_ambiguous_create_title(note.title)
            ):
                duplicates += 1
                continue
            if self.store.has_seen_request(note.note_id):
                if self.store.source_kind_changed(note.note_id, note.kind):
                    outcome = self.store.quarantine_source_kind_change(
                        note, now=now
                    )
                    if outcome is EnqueueOutcome.INTEGRITY_ERROR:
                        integrity_errors += 1
                    else:
                        duplicates += 1
                    continue
                if not self.store.should_verify_source(
                    note.note_id, note.kind, note.updated_at
                ):
                    duplicates += 1
                    continue
            if (
                not self.store.has_seen_request(note.note_id)
                and pending >= self.config.max_pending_jobs
            ):
                if output_recovery_gate_active:
                    capacity_available = (
                        not self.store.has_unrouted_output_reset_probe()
                    )
                elif inference_recovery_gate_active:
                    capacity_available = (
                        not self.store.has_unrouted_inference_reset_probe()
                    )
                elif local_reset_blocker is not None:
                    capacity_available = not self.store.has_unrouted_local_reset_probe(
                        after_queue_seq=local_reset_blocker.queue_seq
                    )
                else:
                    capacity_available = False
                if not capacity_available:
                    queue_limited = True
                    break
            try:
                content = (
                    self.provider.download_audio(note.note_id)
                    if note.kind is NoteKind.AUDIO
                    else self.provider.read_note_content(note.note_id)
                )
            except Exception as error:
                # Stop here so a later note cannot overtake an unavailable one.
                return PollReport(
                    enqueued=enqueued,
                    duplicates=duplicates,
                    integrity_errors=integrity_errors,
                    rejected=rejected,
                    ignored=ignored,
                    output_retired=retired,
                    output_gate=output_gate,
                    source_cleanup=cleanup_detail,
                    source_cleanup_error=cleanup_error,
                    error=_safe_error(error),
                )
            if (
                note.kind is NoteKind.AUDIO
                and len(content) > self.config.max_audio_bytes
            ):
                outcome = self.store.reject_oversized_audio(
                    note,
                    audio_sha256=sha256_bytes(content),
                    now=now,
                )
            elif (
                note.kind is NoteKind.TEXT
                and len(content) > self.config.max_text_bytes
            ):
                outcome = self.store.reject_oversized_text(
                    note,
                    now=now,
                )
            elif note.kind is NoteKind.AUDIO:
                outcome = self.store.enqueue_audio(note, content, now=now)
            else:
                try:
                    text = content.decode("utf-8-sig").strip()
                except UnicodeDecodeError:
                    outcome = self.store.reject_invalid_text(note, now=now)
                else:
                    if route_transcript(text).kind is RouteKind.IGNORED:
                        outcome = self.store.ignore_text_at_intake(
                            note,
                            content_sha256=sha256_bytes(content),
                            now=now,
                        )
                    else:
                        outcome = self.store.enqueue_text(note, content, now=now)
            if outcome == EnqueueOutcome.ENQUEUED:
                enqueued += 1
                pending += 1
            elif outcome == EnqueueOutcome.IGNORED:
                ignored += 1
            elif outcome == EnqueueOutcome.DUPLICATE:
                duplicates += 1
            elif outcome == EnqueueOutcome.INTEGRITY_ERROR:
                integrity_errors += 1
            elif outcome == EnqueueOutcome.REJECTED:
                rejected += 1
            else:  # pragma: no cover - enum exhaustiveness guard
                raise AssertionError(f"unexpected enqueue outcome: {outcome}")
        return PollReport(
            baselined=baselined,
            enqueued=enqueued,
            duplicates=duplicates,
            integrity_errors=integrity_errors,
            rejected=rejected,
            ignored=ignored,
            output_retired=retired,
            output_gate=self.store.output_gate(),
            queue_limited=queue_limited,
            source_cleanup=cleanup_detail,
            source_cleanup_error=cleanup_error,
        )

    def run_once(self) -> WorkReport:
        """Advance the oldest runnable job by one durable stage."""

        self.recover_time_based_gates()
        self.recover_interrupted_work()
        output_gate = self.store.output_gate()
        assistant_gate = self.store.assistant_gate()
        if output_gate is not None or assistant_gate is not None:
            if assistant_gate not in {None, "inference_ambiguous"}:
                return WorkReport(False, detail="assistant_gate")
            return self._run_gated_once()
        now = self.clock()
        local_blocker = self.store.local_reset_discovery_blocker(now=now)
        if local_blocker is not None:
            return self._run_local_reset_once(local_blocker, now)
        if self.config.delete_source_after_publish:
            cleanup = self.store.oldest_pending_source_cleanup(now=now)
            if cleanup is not None:
                return self._delete_source(cleanup, now)
        job = self.store.oldest_actionable(now=now)
        if job is None:
            return WorkReport(False)
        if job.state == JobState.FAILED:
            job = self.store.restore_retry(job.note_id, now=now)
        if (
            job.state == JobState.ANSWERED
            and self.config.output_mode == "rolling"
            and self.store.output_missing_pending()
        ):
            # A single missing observation is not enough to treat a deliberately
            # deleted output as gone. Keep the completed answer durable and wait
            # for the next poll to either clear the observation or retire and
            # advance the output generation.
            return WorkReport(
                False,
                note_id=job.note_id,
                state=job.state,
                detail="output_missing_pending",
            )
        if job.state == JobState.QUEUED:
            detail = self._transcribe(job, now)
        elif job.state == JobState.TRANSCRIBED and job.route_kind is RouteKind.RESET:
            applied = self.store.apply_assistant_reset(
                reset_note_id=job.note_id,
                answer="Assistant reset complete.",
                now=now,
            )
            detail = "reset" if applied else "reset_blocked"
        elif job.state == JobState.TRANSCRIBED:
            detail = self._answer(job, now)
        elif job.state == JobState.ANSWERED:
            detail = self._publish(job, now)
        else:  # guarded by Store.oldest_actionable
            raise AssertionError(f"unexpected runnable state: {job.state}")
        latest = self.store.get_job(job.note_id)
        return WorkReport(
            True,
            note_id=job.note_id,
            state=latest.state if latest else None,
            detail=detail,
        )

    def _run_gated_once(self) -> WorkReport:
        """Route only enough local input to recover from a safe ambiguity."""

        now = self.clock()
        reset = self.store.oldest_gated_reset_candidate()
        if reset is not None:
            applied = self.store.apply_assistant_reset(
                reset_note_id=reset.note_id,
                answer="Assistant reset complete.",
                now=now,
            )
            latest = self.store.get_job(reset.note_id)
            return WorkReport(
                applied,
                note_id=reset.note_id,
                state=latest.state if latest is not None else None,
                detail="reset" if applied else "reset_blocked",
            )

        interrupter = self.store.oldest_publication_interrupt_candidate()
        if interrupter is not None:
            if interrupter.route_kind is RouteKind.RESET:
                applied = self.store.apply_assistant_reset(
                    reset_note_id=interrupter.note_id,
                    answer="Assistant reset complete.",
                    now=now,
                )
                detail = "reset" if applied else "reset_blocked"
            else:
                applied = self.store.supersede_ambiguous_publication(
                    interrupter_note_id=interrupter.note_id,
                    now=now,
                )
                detail = "publication_superseded" if applied else "output_gate"
            latest = self.store.get_job(interrupter.note_id)
            return WorkReport(
                applied,
                note_id=interrupter.note_id,
                state=latest.state if latest is not None else None,
                detail=detail,
            )

        job = self.store.oldest_gated_routing_job()
        if job is None:
            detail = (
                "output_gate"
                if self.store.output_gate() is not None
                else "assistant_gate"
            )
            return WorkReport(False, detail=detail)
        detail = self._transcribe(job, now)
        latest = self.store.get_job(job.note_id)
        if latest is None:
            raise RuntimeError("gated routing job disappeared")
        if latest.route_kind is RouteKind.RESET:
            applied = self.store.apply_assistant_reset(
                reset_note_id=job.note_id,
                answer="Assistant reset complete.",
                now=now,
            )
            detail = "reset" if applied else "reset_blocked"
        elif (
            self.store.output_gate() is not None
            and latest.route_kind
            in {
                RouteKind.QUESTION,
                RouteKind.CONTINUE,
                RouteKind.NEW_CONVERSATION,
            }
        ):
            applied = self.store.supersede_ambiguous_publication(
                interrupter_note_id=job.note_id,
                now=now,
            )
            if applied:
                detail = "publication_superseded"
        latest = self.store.get_job(job.note_id)
        return WorkReport(
            True,
            note_id=job.note_id,
            state=latest.state if latest is not None else None,
            detail=detail,
        )

    def _run_local_reset_once(
        self, blocker: JobRecord, now: datetime
    ) -> WorkReport:
        """Discover an exact reset without overtaking a local queue blocker."""

        reset = self.store.oldest_local_reset_candidate(
            after_queue_seq=blocker.queue_seq
        )
        if reset is not None:
            applied = self.store.apply_assistant_reset(
                reset_note_id=reset.note_id,
                answer="Assistant reset complete.",
                now=now,
            )
            latest = self.store.get_job(reset.note_id)
            return WorkReport(
                applied,
                note_id=reset.note_id,
                state=latest.state if latest is not None else None,
                detail="reset" if applied else "reset_blocked",
            )

        job = self.store.oldest_local_reset_routing_job(
            after_queue_seq=blocker.queue_seq
        )
        if job is None:
            return WorkReport(
                False,
                note_id=blocker.note_id,
                state=blocker.state,
                detail=(
                    "job_recovery_required"
                    if blocker.manual_retry_allowed
                    else "retry_wait"
                ),
            )
        detail = self._transcribe(job, now, defer_route=True)
        latest = self.store.get_job(job.note_id)
        if latest is None:
            raise RuntimeError("local reset discovery job disappeared")
        if latest.route_kind is RouteKind.RESET:
            applied = self.store.apply_assistant_reset(
                reset_note_id=job.note_id,
                answer="Assistant reset complete.",
                now=now,
            )
            detail = "reset" if applied else "reset_blocked"
            latest = self.store.get_job(job.note_id)
        return WorkReport(
            True,
            note_id=job.note_id,
            state=latest.state if latest is not None else None,
            detail=detail,
        )

    def _delete_source(
        self, cleanup: SourceCleanupRecord, now: datetime
    ) -> WorkReport:
        """Cross one durable DELETE boundary for an already published source."""

        # Kept as a local import-free structural assertion so malformed store
        # data cannot reach the provider mutation.
        source_note_id = getattr(cleanup, "source_note_id", None)
        source_kind = getattr(cleanup, "source_kind", None)
        source_sha256 = getattr(cleanup, "source_sha256", None)
        precheck_failures = getattr(cleanup, "precheck_failures", None)
        if (
            not isinstance(source_note_id, str)
            or not isinstance(source_kind, NoteKind)
            or not isinstance(source_sha256, str)
            or not isinstance(precheck_failures, int)
        ):
            raise RuntimeError("source cleanup candidate is malformed")
        self.store.begin_source_cleanup_attempt(
            source_note_id,
            expected_kind=source_kind,
            expected_sha256=source_sha256,
            now=now,
        )
        try:
            receipt = self.provider.delete_source_note(
                source_note_id,
                expected_kind=source_kind,
                expected_content_sha256=source_sha256,
            )
        except Exception:
            # The provider did not prove where it failed. The durable attempt
            # remains ambiguous and no later cycle may issue another DELETE.
            self.store.mark_source_cleanup_unknown(
                source_note_id,
                error="source delete outcome is unknown",
                now=now,
            )
            return WorkReport(
                True,
                note_id=source_note_id,
                state=JobState.PUBLISHED,
                detail="source_cleanup_unknown",
            )
        if not isinstance(receipt, DeletionReceipt):
            self.store.mark_source_cleanup_unknown(
                source_note_id,
                error="source delete returned an invalid receipt",
                now=now,
            )
            return WorkReport(
                True,
                note_id=source_note_id,
                state=JobState.PUBLISHED,
                detail="source_cleanup_unknown",
            )
        if (
            not isinstance(receipt.outcome, MutationOutcome)
            or type(receipt.deletion_attempted) is not bool
            or (
                receipt.note is not None
                and (
                    not isinstance(receipt.note, RemoteNote)
                    or receipt.note.note_id != source_note_id
                )
            )
            or (
                receipt.outcome is MutationOutcome.VERIFIED
                and receipt.note is not None
            )
        ):
            self.store.mark_source_cleanup_unknown(
                source_note_id,
                error="source delete returned malformed receipt metadata",
                now=now,
            )
            return WorkReport(
                True,
                note_id=source_note_id,
                state=JobState.PUBLISHED,
                detail="source_cleanup_unknown",
            )
        if receipt.outcome is MutationOutcome.VERIFIED:
            self.store.record_verified_source_cleanup_absence(
                source_note_id,
                deletion_attempted=receipt.deletion_attempted,
                now=now,
            )
            return WorkReport(
                True,
                note_id=source_note_id,
                state=JobState.PUBLISHED,
                detail="source_cleanup_verification_pending",
            )
        if receipt.outcome is MutationOutcome.CONFLICT:
            if receipt.deletion_attempted:
                self.store.mark_source_cleanup_unknown(
                    source_note_id,
                    error="source delete returned an invalid conflict receipt",
                    now=now,
                )
                detail = "source_cleanup_unknown"
            else:
                self.store.mark_source_cleanup_conflict(
                    source_note_id,
                    error=receipt.detail or "source note changed after acceptance",
                    now=now,
                )
                detail = "source_cleanup_conflict"
            return WorkReport(
                True,
                note_id=source_note_id,
                state=JobState.PUBLISHED,
                detail=detail,
            )
        if receipt.outcome is MutationOutcome.UNKNOWN and not receipt.deletion_attempted:
            attempt = precheck_failures + 1
            self.store.source_cleanup_not_attempted(
                source_note_id,
                error=receipt.detail or "source delete precheck unavailable",
                retry_at=now + self._retry_delay(attempt),
                now=now,
            )
            return WorkReport(
                True,
                note_id=source_note_id,
                state=JobState.PUBLISHED,
                detail="source_cleanup_retry_scheduled",
            )
        self.store.mark_source_cleanup_unknown(
            source_note_id,
            error=receipt.detail or "source delete outcome is unknown",
            now=now,
        )
        return WorkReport(
            True,
            note_id=source_note_id,
            state=JobState.PUBLISHED,
            detail="source_cleanup_unknown",
        )

    def _reconcile_source_cleanup(
        self, notes: list[RemoteNote], now: datetime
    ) -> str | None:
        cleanup = self.store.next_source_cleanup_readback()
        if cleanup is None:
            return None
        matches = [note for note in notes if note.note_id == cleanup.source_note_id]
        if not matches:
            confirmed = self.store.record_source_cleanup_absent(
                cleanup.source_note_id,
                confirmation_reads=self.config.confirm_source_deletion_after,
                now=now,
            )
            return "confirmed" if confirmed else "missing"
        first = matches[0]
        if any(note.kind is not first.kind for note in matches[1:]):
            self.store.mark_source_cleanup_conflict(
                cleanup.source_note_id,
                error="source note metadata is inconsistent",
                now=now,
            )
            return "conflict"
        if cleanup.state is SourceCleanupState.CONFLICT:
            self.store.mark_source_cleanup_conflict(
                cleanup.source_note_id,
                error="source note remains present after a cleanup conflict",
                now=now,
            )
            return "conflict"
        if cleanup.readback_attempts >= self.config.max_source_delete_readbacks:
            # Keep checking the cheap list result so a later manual/cloud
            # deletion can still be confirmed, but stop downloading content.
            self.store.record_source_cleanup_still_present(
                cleanup.source_note_id, now=now
            )
            return "exhausted"
        content = (
            self.provider.download_audio(cleanup.source_note_id)
            if first.kind is NoteKind.AUDIO
            else self.provider.read_note_content(cleanup.source_note_id)
        )
        return self.store.record_source_cleanup_present(
            cleanup.source_note_id,
            observed_kind=first.kind,
            observed_sha256=sha256_bytes(content),
            maximum_readbacks=self.config.max_source_delete_readbacks,
            now=now,
        )

    def recover_time_based_gates(self) -> bool:
        """Resume a daily-limit turn once a new UTC allowance day begins."""

        status = self.store.assistant_gate_status()
        if status is None or status["gate"] != "daily_model_limit":
            return False
        gated_at = status["gated_at"]
        source_note_id = status["source_note_id"]
        if not isinstance(gated_at, datetime) or not isinstance(source_note_id, str):
            raise RuntimeError("assistant gate state is inconsistent")
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise RuntimeError("worker clock must be timezone-aware")
        gate_day = gated_at.astimezone(timezone.utc).date()
        current_day = now.astimezone(timezone.utc).date()
        if current_day <= gate_day:
            return False
        return self.store.clear_assistant_gate(
            expected_gate="daily_model_limit",
            expected_note_id=source_note_id,
            now=now,
        )

    def recover_interrupted_work(self) -> bool:
        """Install durable gates for interrupted model or publication calls."""

        now = self.clock()
        publication = self.store.gate_interrupted_publication(now=now)
        model = self.store.gate_interrupted_model(now=now)
        return publication or model

    def drain(self, *, max_steps: int = 100) -> list[WorkReport]:
        reports: list[WorkReport] = []
        for _ in range(max_steps):
            report = self.run_once()
            if not report.worked:
                break
            reports.append(report)
        return reports

    def _transcribe(
        self, job: JobRecord, now: datetime, *, defer_route: bool = False
    ) -> str:
        if job.route_kind is not None:
            if (
                job.state is not JobState.QUEUED
                or job.route_kind
                not in {
                    RouteKind.QUESTION,
                    RouteKind.CONTINUE,
                    RouteKind.NEW_CONVERSATION,
                }
                or job.transcript is None
                or job.audio is not None
            ):
                raise RuntimeError("deferred local route is inconsistent")
            return self._apply_route(
                job,
                kind=job.route_kind,
                text=(
                    job.transcript
                    if job.route_kind is RouteKind.QUESTION
                    else None
                ),
                now=now,
                defer_route=False,
            )
        if job.audio is None:
            self.store.mark_failure(
                job.note_id,
                stage="transcription",
                resume_state=JobState.QUEUED,
                error="staged audio is missing",
                retryable=False,
                now=now,
            )
            return "missing_audio"
        if job.transcription_attempts >= self.config.max_transcription_attempts:
            self.store.mark_failure(
                job.note_id,
                stage="transcription",
                resume_state=JobState.QUEUED,
                error="transcription outcome unknown after maximum attempts",
                retryable=False,
                manual_retry_allowed=True,
                preserve_audio=True,
                now=now,
            )
            return "attempts_exhausted"
        attempt = self.store.increment_attempt(
            job.note_id, "transcription", now=now
        )
        try:
            if job.source_kind is NoteKind.TEXT:
                transcript = job.audio.decode("utf-8-sig").strip()
            else:
                transcript = self.transcriber.transcribe(
                    job.audio, mime_type=job.mime_type
                ).strip()
            if not transcript:
                raise PermanentDependencyError("transcription was empty")
        except Exception as error:
            return self._stage_error(
                job,
                stage="transcription",
                resume_state=JobState.QUEUED,
                error=error,
                attempt=attempt,
                maximum=self.config.max_transcription_attempts,
                now=now,
            )

        route = (
            route_transcript(transcript)
            if job.source_kind is NoteKind.TEXT
            else route_spoken_transcript(transcript)
        )
        return self._apply_route(
            job,
            kind=route.kind,
            text=route.text,
            now=now,
            defer_route=defer_route,
        )

    def _apply_route(
        self,
        job: JobRecord,
        *,
        kind: RouteKind,
        text: str | None,
        now: datetime,
        defer_route: bool,
    ) -> str:
        if kind == RouteKind.IGNORED:
            # Neither the ordinary recording nor its transcript survives locally.
            self.store.mark_ignored(job.note_id, reason="prefix_missing", now=now)
            return "ignored"
        if kind == RouteKind.QUESTION:
            assert text is not None
            if len(text) > self.config.max_question_chars:
                self.store.mark_failure(
                    job.note_id,
                    stage="routing",
                    resume_state=JobState.QUEUED,
                    error="assistant request exceeded the configured character limit",
                    retryable=False,
                    now=now,
                )
                return "question_too_long"
        if defer_route and kind is not RouteKind.RESET:
            deferred_transcript = {
                RouteKind.QUESTION: text,
                RouteKind.CONTINUE: "Continue",
                RouteKind.NEW_CONVERSATION: "New conversation",
            }.get(kind)
            if deferred_transcript is None:
                raise RuntimeError("unsupported deferred local route")
            self.store.defer_local_reset_probe_route(
                job.note_id,
                route_kind=kind,
                transcript=deferred_transcript,
                now=now,
            )
            return "reset_probe_deferred"
        if kind == RouteKind.NEW_CONVERSATION:
            self.store.mark_new_conversation(
                job.note_id, now=now, answer="New conversation started."
            )
            return "new_conversation"
        if kind == RouteKind.RESET:
            self.store.mark_reset_routed(job.note_id, now=now)
            return "reset_routed"
        if kind == RouteKind.CONTINUE:
            self.store.mark_continue(
                job.note_id,
                now=now,
                exhausted_answer="There is no more of the previous answer.",
            )
            return "continue"
        assert kind == RouteKind.QUESTION and text is not None
        self.store.mark_question_transcribed(job.note_id, question=text, now=now)
        return "question"

    def _answer(self, job: JobRecord, now: datetime) -> str:
        if job.model_attempts >= self.config.max_model_attempts:
            # Reaching this branch while still transcribed means an attempt was
            # not closed by a typed success/failure transition. Its remote
            # outcome cannot be proven, so require the existing exact gate.
            self.store.pause_assistant(
                job.note_id,
                gate="inference_ambiguous",
                error="model outcome unknown after maximum attempts",
                now=now,
            )
            return "assistant_paused"
        messages = build_prompt(
            self.store,
            job,
            system_prompt=self.config.system_prompt,
            max_prior_turns=self.config.max_context_turns,
            max_context_chars=self.config.max_context_chars,
        )
        if not self.store.reserve_model_call(
            now=now,
            daily_limit=self.config.max_model_calls_per_day,
        ):
            self.store.pause_assistant(
                job.note_id,
                gate="daily_model_limit",
                error="daily model-call limit reached",
                now=now,
            )
            return "assistant_paused"
        attempt = self.store.begin_model_attempt(job.note_id, now=now)
        try:
            response = self.assistant.respond(messages)
            answer = response.text.strip()
            sections = split_answer(
                answer,
                min_chars=self.config.answer_min_chars,
                target_chars=self.config.answer_target_chars,
                max_chars=self.config.answer_max_chars,
            )
        except Exception as error:
            return self._stage_error(
                job,
                stage="model",
                resume_state=JobState.TRANSCRIBED,
                error=error,
                attempt=attempt,
                maximum=self.config.max_model_attempts,
                now=now,
            )
        self.store.mark_answered(
            job.note_id,
            answer=answer,
            sections=sections,
            response_id=response.response_id,
            now=now,
        )
        return "answered"

    def _publish(self, job: JobRecord, now: datetime) -> str:
        if job.publication_attempts >= self.config.max_publication_attempts:
            self.store.mark_failure(
                job.note_id,
                stage="publication",
                resume_state=JobState.ANSWERED,
                error="publication attempts exhausted",
                retryable=False,
                now=now,
            )
            return "attempts_exhausted"
        if not job.transcript or not job.answer_sections:
            self.store.mark_failure(
                job.note_id,
                stage="publication",
                resume_state=JobState.ANSWERED,
                error="stored answer is incomplete",
                retryable=False,
                now=now,
            )
            return "incomplete_answer"
        legacy_prepared_title = (
            job.publication_content is not None
            and job.publication_title_sha256
            == sha256_text(self.config.output_title)
        )
        completed_at = (
            publication_completed_at(job.publication_content)
            if job.publication_content is not None
            and not legacy_prepared_title
            else job.updated_at
        )
        desired_title = format_output_title(
            self.config.output_title,
            completed_at,
            self.config.local_timezone,
        )
        # Existing prepared turns used the exact unsuffixed title. Preserve
        # their durable mutation boundary during an in-place upgrade.
        if legacy_prepared_title:
            desired_title = self.config.output_title
        proposed_content = format_response(
            job.transcript, job.answer_sections[0], completed_at=completed_at
        )
        content, desired_hash, publication_mode = self.store.prepare_publication(
            job.note_id,
            content=proposed_content,
            title=desired_title,
            output_mode=self.config.output_mode,
            now=now,
        )

        if publication_mode == "per_turn":
            if job.publication_attempts > 0:
                self._mark_unknown_mutation(
                    job,
                    "create was interrupted; refusing a blind retry",
                    now,
                    gate="create_ambiguous",
                )
                return "create_unknown"
            attempt = self.store.increment_attempt(
                job.note_id, "publication", now=now
            )
            return self._create_output(
                job,
                content,
                desired_hash,
                title=desired_title,
                output_mode=publication_mode,
                attempt=attempt,
                now=now,
            )

        snapshot = self.store.output_snapshot()
        if snapshot.epoch != job.accepted_output_epoch:
            self.store.mark_failure(
                job.note_id,
                stage="publication",
                resume_state=JobState.ANSWERED,
                error="reply surface changed after request acceptance",
                retryable=False,
                now=now,
            )
            return "output_epoch_changed"
        if snapshot.note_id is None:
            if job.publication_attempts > 0:
                self._mark_unknown_mutation(
                    job,
                    "create was interrupted; refusing a blind retry",
                    now,
                    gate="create_ambiguous",
                )
                return "create_unknown"
            attempt = self.store.increment_attempt(
                job.note_id, "publication", now=now
            )
            return self._create_output(
                job,
                content,
                desired_hash,
                title=desired_title,
                output_mode=publication_mode,
                attempt=attempt,
                now=now,
            )
        if snapshot.content_sha256 is None:
            self.store.mark_failure(
                job.note_id,
                stage="publication",
                resume_state=JobState.ANSWERED,
                error="reply surface has no verified content hash",
                retryable=False,
                now=now,
            )
            return "missing_output_hash"

        if job.publication_attempts > 0:
            self._mark_unknown_mutation(
                job,
                "update was interrupted after its durable attempt boundary",
                now,
                gate="update_ambiguous",
            )
            return "update_unknown"

        try:
            current_output = self.provider.get_note(snapshot.note_id)
        except Exception as error:
            return self._stage_error(
                job,
                stage="publication",
                resume_state=JobState.ANSWERED,
                error=error,
                attempt=1,
                maximum=self.config.max_publication_attempts,
                now=now,
            )
        expected_title = self.config.output_title
        if current_output is not None:
            if (
                not isinstance(current_output, RemoteNote)
                or current_output.note_id != snapshot.note_id
                or current_output.kind is not NoteKind.TEXT
                or not is_output_title(
                    current_output.title, self.config.output_title
                )
            ):
                self.store.retire_output(
                    snapshot.note_id, reason="modified", now=now
                )
                self.store.mark_failure(
                    job.note_id,
                    stage="publication",
                    resume_state=JobState.ANSWERED,
                    error="reply note metadata changed outside the bridge",
                    retryable=False,
                    now=now,
                )
                return "conflict"
            expected_title = current_output.title

        if not self.store.bind_rolling_publication_target(
            job.note_id,
            output_note_id=snapshot.note_id,
            expected_output_epoch=snapshot.epoch,
            expected_content_sha256=snapshot.content_sha256,
            now=now,
        ):
            self.store.mark_failure(
                job.note_id,
                stage="publication",
                resume_state=JobState.ANSWERED,
                error="reply surface changed before update",
                retryable=False,
                now=now,
            )
            return "output_target_changed"

        self.store.increment_attempt(job.note_id, "publication", now=now)

        try:
            receipt = self.provider.update_text_note(
                snapshot.note_id,
                title=desired_title,
                content=content,
                expected_title=expected_title,
                expected_content_sha256=snapshot.content_sha256,
            )
        except Exception as error:
            self._mark_unknown_mutation(
                job,
                _safe_error(error),
                now,
                gate="update_ambiguous",
            )
            return "unknown"
        if receipt.outcome == MutationOutcome.CONFLICT:
            if (
                receipt.note is None
                and receipt.observed_content_sha256 is None
                and receipt.detail
                in {"missing", "note_missing"}
            ):
                # A delete can race the update even when no prior poll observed
                # it. Pause publication and let two normal poll observations
                # establish that the note is really gone. Once confirmed, the
                # store can safely move this exact answer to the next output
                # generation without repeating model work.
                self._mark_unknown_mutation(
                    job,
                    MISSING_OUTPUT_UPDATE_ERROR,
                    now,
                    gate="update_ambiguous",
                )
                return "missing"
            if (
                receipt.observed_content_sha256 == desired_hash
                and receipt.note is not None
                and receipt.note.note_id == snapshot.note_id
                and receipt.note.kind is NoteKind.TEXT
                and is_output_title(
                    receipt.note.title, self.config.output_title
                )
            ):
                if self.store.update_output_hash(
                    snapshot.note_id,
                    expected_content_sha256=snapshot.content_sha256,
                    content_sha256=desired_hash,
                    now=now,
                ):
                    self.store.mark_published(
                        job.note_id, output_note_id=snapshot.note_id, now=now
                    )
                    return "recovered"
                self._mark_unknown_mutation(
                    job,
                    "local output receipt conflicted",
                    now,
                    gate="update_ambiguous",
                )
                return "local_conflict"
            if receipt.observed_content_sha256 == desired_hash:
                self._mark_unknown_mutation(
                    job,
                    "conflict receipt for desired content was incomplete",
                    now,
                    gate="update_ambiguous",
                )
                return "invalid_receipt"
            self.store.retire_output(snapshot.note_id, reason="content_conflict", now=now)
            self.store.mark_failure(
                job.note_id,
                stage="publication",
                resume_state=JobState.ANSWERED,
                error=receipt.detail or "reply note changed outside the bridge",
                retryable=False,
                now=now,
            )
            return "conflict"
        if receipt.outcome == MutationOutcome.UNKNOWN:
            self._mark_unknown_mutation(
                job, receipt.detail, now, gate="update_ambiguous"
            )
            return "unknown"
        if (
            receipt.note is None
            or receipt.note.note_id != snapshot.note_id
            or receipt.note.kind is not NoteKind.TEXT
            or receipt.note.title != desired_title
            or receipt.observed_content_sha256 != desired_hash
        ):
            self._mark_unknown_mutation(
                job,
                "verified receipt was incomplete",
                now,
                gate="update_ambiguous",
            )
            return "invalid_receipt"
        if not self.store.update_output_hash(
            snapshot.note_id,
            expected_content_sha256=snapshot.content_sha256,
            content_sha256=desired_hash,
            now=now,
        ):
            self._mark_unknown_mutation(
                job,
                "local output receipt conflicted",
                now,
                gate="update_ambiguous",
            )
            return "local_conflict"
        self.store.mark_published(
            job.note_id, output_note_id=snapshot.note_id, now=now
        )
        return "published"

    def _create_output(
        self,
        job: JobRecord,
        content: str,
        desired_hash: str,
        *,
        title: str,
        output_mode: Literal["rolling", "per_turn"],
        attempt: int,
        now: datetime,
    ) -> str:
        try:
            receipt = self.provider.create_text_note(
                title=title, content=content
            )
        except Exception as error:
            # Without a returned note id, a crashed/ambiguous create cannot be
            # distinguished from a successful create. Never blind-retry it.
            self._mark_unknown_mutation(
                job,
                f"create outcome unknown: {_safe_error(error)}",
                now,
                gate="create_ambiguous",
            )
            return "create_unknown"
        if not isinstance(receipt, MutationReceipt):
            self._mark_unknown_mutation(
                job,
                "create returned an invalid receipt",
                now,
                gate="create_ambiguous",
            )
            return "invalid_receipt"

        try:
            outcome = receipt.outcome
            note = receipt.note
            observed_hash = receipt.observed_content_sha256
            detail = receipt.detail
        except Exception:
            self._mark_unknown_mutation(
                job,
                "create returned an unreadable receipt",
                now,
                gate="create_ambiguous",
            )
            return "invalid_receipt"

        if note is not None and not isinstance(note, RemoteNote):
            self._mark_unknown_mutation(
                job,
                "create returned malformed receipt metadata",
                now,
                gate="create_ambiguous",
            )
            return "invalid_receipt"

        note_id = None if note is None else _usable_note_id(note.note_id)
        # Any identity returned by our create attempt is bridge-owned even when
        # the rest of the receipt is malformed. Persist it before validation so
        # it can never be mistaken for a handset request during recovery.
        if note_id is not None:
            try:
                self.store.remember_output_note(
                    note_id,
                    source_note_id=job.note_id,
                    output_epoch=job.accepted_output_epoch,
                    publication_attempt=attempt,
                    content_sha256=_usable_sha256(observed_hash),
                    now=now,
                )
            except (RuntimeError, ValueError):
                self._mark_unknown_mutation(
                    job,
                    "returned output identity conflicts with durable state",
                    now,
                    gate="create_ambiguous",
                )
                return "invalid_receipt"
        if note is not None and note_id is None:
            self._mark_unknown_mutation(
                job,
                "create returned an invalid note identity",
                now,
                gate="create_ambiguous",
            )
            return "invalid_receipt"
        if not isinstance(outcome, MutationOutcome):
            self._mark_unknown_mutation(
                job,
                "create returned an invalid mutation outcome",
                now,
                gate="create_ambiguous",
            )
            return "invalid_receipt"
        if note is not None and (
            not isinstance(note.kind, NoteKind)
            or not isinstance(note.title, str)
        ):
            self._mark_unknown_mutation(
                job,
                "create returned invalid note metadata",
                now,
                gate="create_ambiguous",
            )
            return "invalid_receipt"
        if observed_hash is not None and not isinstance(observed_hash, str):
            self._mark_unknown_mutation(
                job,
                "create returned an invalid content hash",
                now,
                gate="create_ambiguous",
            )
            return "invalid_receipt"
        if outcome != MutationOutcome.VERIFIED:
            self._mark_unknown_mutation(
                job,
                detail if isinstance(detail, str) else "create was not verified",
                now,
                gate="create_ambiguous",
            )
            return outcome.value
        if (
            note is None
            or note_id is None
            or note.kind is not NoteKind.TEXT
            or note.title != title
            or observed_hash != desired_hash
        ):
            self._mark_unknown_mutation(
                job,
                "verified create receipt was incomplete",
                now,
                gate="create_ambiguous",
            )
            return "invalid_receipt"

        if output_mode == "rolling":
            registered = self.store.register_created_output(
                expected_epoch=job.accepted_output_epoch,
                note_id=note_id,
                content_sha256=desired_hash,
                now=now,
            )
            if not registered:
                self._mark_unknown_mutation(
                    job,
                    "local output generation changed during create",
                    now,
                    gate="create_ambiguous",
                )
                return "local_conflict"
        self.store.mark_published(
            job.note_id, output_note_id=note_id, now=now
        )
        return "published"

    def _reconcile_output(self, now: datetime) -> str | None:
        readback_exhausted = self.store.update_readback_exhausted(
            maximum=self.config.max_update_readback_attempts
        )
        snapshot = self.store.output_snapshot()
        if snapshot.note_id is None:
            return None
        note = self.provider.get_note(snapshot.note_id)
        if note is None:
            missing = self.store.record_output_missing(snapshot.note_id, now=now)
            resolution = self.store.resolve_missing_publication(
                output_note_id=snapshot.note_id,
                maximum_readbacks=self.config.max_update_readback_attempts,
                deletion_confirmation_reads=(
                    self.config.confirm_output_deletion_after
                ),
                now=now,
            )
            if resolution in {"pending", "unknown"}:
                return resolution
            if resolution == "deleted_confirmed":
                retired = self.store.retire_output(
                    snapshot.note_id, reason="deleted_confirmed", now=now
                )
                return "deleted" if retired else "unknown"
            if missing < self.config.confirm_output_deletion_after:
                return "missing"
            retired = self.store.retire_output(
                snapshot.note_id, reason="deleted_confirmed", now=now
            )
            return "deleted" if retired else "unknown"
        self.store.clear_output_missing(snapshot.note_id)
        if note.kind != NoteKind.TEXT or not is_output_title(
            note.title, self.config.output_title
        ):
            self.store.retire_output(snapshot.note_id, reason="modified", now=now)
            return "modified"
        content_hash = sha256_bytes(self.provider.read_note_content(snapshot.note_id))
        if content_hash == snapshot.content_sha256:
            if readback_exhausted:
                # A fresh exact read after the bounded non-desired window still
                # shows the previous content. That window can contain exact-old
                # and missing observations; this final sample must be exact-old.
                # Quarantine this rolling surface and abandon only the
                # unconfirmable delivery. Re-creating the stored answer could
                # duplicate it if the delayed update later appears, while
                # leaving the gate active would make one lost response a
                # permanent outage.
                abandoned = self.store.abandon_stale_ambiguous_update(
                    output_note_id=snapshot.note_id,
                    observed_content_sha256=content_hash,
                    maximum_readbacks=self.config.max_update_readback_attempts,
                    now=now,
                )
                return "abandoned" if abandoned else "unknown"
            resolution = self.store.resolve_unapplied_publication(
                output_note_id=snapshot.note_id,
                observed_content_sha256=content_hash,
                maximum_readbacks=self.config.max_update_readback_attempts,
                now=now,
            )
            return resolution
        if content_hash != snapshot.content_sha256:
            recovered = self.store.reconcile_applied_publication(
                output_note_id=snapshot.note_id,
                observed_content_sha256=content_hash,
                now=now,
            )
            if recovered is not None:
                return "recovered"
            self.store.retire_output(snapshot.note_id, reason="modified", now=now)
            return "modified"
        return None

    def _stage_error(
        self,
        job: JobRecord,
        *,
        stage: str,
        resume_state: JobState,
        error: Exception,
        attempt: int,
        maximum: int,
        now: datetime,
    ) -> str:
        assistant_gate: str | None = None
        if isinstance(error, ChatGPTCredentialAmbiguousError):
            assistant_gate = "credential_ambiguous"
        elif isinstance(error, ChatGPTAmbiguousOutcomeError):
            assistant_gate = "inference_ambiguous"
        elif isinstance(error, ChatGPTAuthorizationRequiredError):
            assistant_gate = "authorization_required"
        elif isinstance(error, ChatGPTPlanUsageLimitError):
            assistant_gate = "plan_limit"
        elif isinstance(error, ChatGPTAccountNotEligibleError):
            assistant_gate = "account_not_eligible"
        elif isinstance(error, ChatGPTConfigurationError):
            assistant_gate = "configuration_error"
        if stage == "model" and assistant_gate is not None:
            self.store.pause_assistant(
                job.note_id,
                gate=assistant_gate,
                error=_safe_error(error),
                now=now,
            )
            return "assistant_paused"
        transient = isinstance(error, TransientDependencyError)
        retryable = transient and attempt < maximum
        manually_recoverable = transient and not retryable
        self.store.mark_failure(
            job.note_id,
            stage=stage,
            resume_state=resume_state,
            error=_safe_error(error),
            retryable=retryable,
            now=now,
            next_retry_at=(
                now + self._retry_delay(attempt) if retryable else None
            ),
            manual_retry_allowed=manually_recoverable,
            preserve_audio=(stage == "transcription" and manually_recoverable),
        )
        return "retry_scheduled" if retryable else "failed"

    def _retry_delay(self, attempt: int) -> timedelta:
        """Return a bounded exponential delay for a completed safe failure."""

        exponent = min(max(attempt - 1, 0), 30)
        seconds = self.config.retry_delay.total_seconds() * (2**exponent)
        return timedelta(
            seconds=min(seconds, self.config.retry_max_delay.total_seconds())
        )

    def _mark_unknown_mutation(
        self,
        job: JobRecord,
        detail: str | None,
        now: datetime,
        *,
        gate: Literal["update_ambiguous", "create_ambiguous"],
    ) -> None:
        self.store.mark_publication_ambiguous(
            job.note_id,
            gate=gate,
            error=detail or "remote mutation outcome is unknown",
            now=now,
        )


def _safe_error(error: Exception) -> str:
    message = str(error).replace("\n", " ").strip()
    if message:
        return f"{type(error).__name__}: {message}"[:500]
    return type(error).__name__


def _usable_note_id(value: object) -> str | None:
    """Accept only bounded, nonempty opaque identities from a provider."""

    if not isinstance(value, str) or not value or len(value) > 4096:
        return None
    return value


def _usable_sha256(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    if any(character not in "0123456789abcdef" for character in value):
        return None
    return value


def _note_order(note: RemoteNote) -> tuple[float, str]:
    """Order by absolute provider update time, then permanent note identity."""

    updated_at = note.updated_at
    if updated_at is None:
        timestamp = float("-inf")
    else:
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        timestamp = updated_at.timestamp()
    return timestamp, note.note_id
