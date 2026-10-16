"""Administrative CLI for the Pre-0 simulator and one-shot probes."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import sys
from typing import Iterator, TextIO

from .adapters.light_garado import GaradoLightProvider, LightWritePolicy
from .adapters.simulated_envelope import (
    SIMULATED_TRANSCRIPT_MIME,
    encode_simulated_transcript,
)
from .config import ServiceConfig
from .chatgpt_auth import run_command as run_chatgpt_auth_command
from .chatgpt_errors import (
    ChatGPTAuthorizationRequiredError,
    ChatGPTConfigurationError,
    ChatGPTTransientError,
)
from .e2e_command import E2ECommandError, run_e2e_command
from .light_test_note import (
    LightTestNoteError,
    run_command as run_light_test_note_command,
)
from .light_input_audit import (
    LightInputAuditError,
    run_binding_command as run_light_input_binding_command,
    run_command as run_light_input_audit_command,
)
from .models import (
    AppliedCreateVerification,
    CreateRecoveryCandidate,
    NoteKind,
    RemoteNote,
    UpdateRecoveryCandidate,
)
from .retention import apply_retention
from .runtime_lock import RuntimeLock
from .service import (
    Pre0GuardError,
    build_runtime,
    run_cycle,
    runtime_status,
    serve,
)
from .store import Store


_ASSISTANT_RECOVERY_GATES = (
    "authorization_required",
    "plan_limit",
    "credential_ambiguous",
    "inference_ambiguous",
    "account_not_eligible",
    "configuration_error",
    "daily_model_limit",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lightphone-gpt")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="initialize state and establish a baseline")
    commands.add_parser("status", help="show content-free durable state counts")
    commands.add_parser("cycle", help="poll once and drain runnable work")
    commands.add_parser("drain", help="drain already queued work without polling")
    commands.add_parser("retention", help="apply the configured retention policy")
    commands.add_parser("serve", help="run the loopback-only Pre-0 service")

    chatgpt_auth = commands.add_parser(
        "chatgpt-auth", help="manage local Sign in with ChatGPT authorization"
    )
    chatgpt_auth_commands = chatgpt_auth.add_subparsers(
        dest="chatgpt_auth_command", required=True
    )
    auth_login = chatgpt_auth_commands.add_parser(
        "login", help="authorize interactively in the local browser"
    )
    auth_login.add_argument("--credential-file", type=Path)
    auth_login.add_argument("--timeout-seconds", type=float, default=600.0)
    auth_status = chatgpt_auth_commands.add_parser(
        "status", help="show content-free local authorization status"
    )
    auth_status.add_argument("--credential-file", type=Path)
    auth_models = chatgpt_auth_commands.add_parser(
        "models", help="list models visible to the authorized ChatGPT account"
    )
    auth_models.add_argument("--credential-file", type=Path)
    auth_models.add_argument("--timeout-seconds", type=float, default=15.0)

    simulator = commands.add_parser("simulator", help="manage simulated notes")
    simulator_commands = simulator.add_subparsers(
        dest="simulator_command", required=True
    )
    seed = simulator_commands.add_parser("seed", help="seed one simulated audio note")
    input_group = seed.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--transcript")
    input_group.add_argument("--audio-file", type=Path)
    seed.add_argument("--mime-type", default="audio/mp4")
    seed_text = simulator_commands.add_parser(
        "seed-text", help="seed one simulated text note"
    )
    seed_text.add_argument("--content", required=True)
    seed_text.add_argument("--title", default="Text note")
    simulator_commands.add_parser("list", help="list note metadata without content")
    delete = simulator_commands.add_parser("delete", help="delete a simulated note")
    delete.add_argument("note_id")

    commands.add_parser(
        "light-probe",
        help="explicit one-shot, read-only aggregate Light Cloud probe",
    )

    light_e2e = commands.add_parser(
        "light-e2e",
        help="run one protected real Light-to-ChatGPT-to-Light proof",
    )
    light_e2e.add_argument(
        "--confirm-run",
        action="store_true",
        help="confirm the protected one-shot proof",
    )

    light_input_audit = commands.add_parser(
        "light-input-audit",
        help="read and route one exact Light audio or text note",
    )
    light_input_audit.add_argument("--receipt", required=True, type=Path)
    light_input_audit.add_argument("--confirm-read", action="store_true")

    light_input_bind = commands.add_parser(
        "light-input-bind",
        help="resolve protected test-note title digests using metadata only",
    )
    light_input_bind.add_argument("--receipt", required=True, type=Path)
    light_input_bind.add_argument("--summary", required=True, type=Path)
    light_input_bind.add_argument("--confirm-read", action="store_true")

    light_test_note = commands.add_parser(
        "light-test-note",
        help="operate one receipt-owned synthetic Phase 0 Light note",
    )
    light_test_note_commands = light_test_note.add_subparsers(
        dest="light_test_note_command", required=True
    )

    def add_receipt(command: argparse.ArgumentParser) -> None:
        command.add_argument("--receipt", required=True, type=Path)

    test_create = light_test_note_commands.add_parser(
        "create", help="create one generated synthetic test note"
    )
    add_receipt(test_create)
    test_create.add_argument("--confirm-create", action="store_true")

    test_inspect = light_test_note_commands.add_parser(
        "inspect", help="read and compare cloud state without mutation"
    )
    add_receipt(test_inspect)

    test_update = light_test_note_commands.add_parser(
        "update", help="advance the generated test note by one revision"
    )
    add_receipt(test_update)
    test_update.add_argument("--expected-note-id", required=True)
    test_update.add_argument("--expected-content-sha256", required=True)
    test_update.add_argument("--confirm-update", action="store_true")

    test_delete = light_test_note_commands.add_parser(
        "delete", help="delete one exact receipt-owned synthetic test note"
    )
    add_receipt(test_delete)
    test_delete.add_argument("--expected-note-id", required=True)
    test_delete.add_argument("--expected-content-sha256", required=True)
    test_delete.add_argument("--confirm-delete", action="store_true")

    test_resolve = light_test_note_commands.add_parser(
        "resolve", help="resolve an uncertain intent using read-only cloud checks"
    )
    add_receipt(test_resolve)
    test_resolve.add_argument(
        "--decision", required=True, choices=("applied", "not-applied")
    )
    test_resolve.add_argument("--confirm-run-id", required=True)
    test_resolve.add_argument("--observed-note-id")
    test_resolve.add_argument("--observed-content-sha256")

    recovery = commands.add_parser(
        "publication-recovery",
        help="inspect or resolve a locally recorded uncertain publication",
    )
    recovery_commands = recovery.add_subparsers(
        dest="recovery_command", required=True
    )
    recovery_commands.add_parser(
        "inspect",
        help="show exact content-free identifiers for the active output gate",
    )
    resolve_not_applied = recovery_commands.add_parser(
        "resolve-create-not-applied",
        help="record a manually reviewed not-applied create outcome",
    )
    resolve_applied = recovery_commands.add_parser(
        "resolve-create-applied",
        help="verify one exact Light note read-only, then record an applied create",
    )

    def add_recovery_identity(command: argparse.ArgumentParser) -> None:
        command.add_argument("--source-note-id", required=True)
        command.add_argument("--expected-publication-sha256", required=True)
        command.add_argument("--expected-output-epoch", required=True, type=int)
        command.add_argument(
            "--confirm",
            action="store_true",
            help="confirm the exact gated publication under review",
        )

    add_recovery_identity(resolve_not_applied)
    add_recovery_identity(resolve_applied)
    resolve_applied.add_argument("--output-note-id", required=True)

    resolve_update = recovery_commands.add_parser(
        "resolve-update-applied",
        help="verify the exact rolling note read-only, then complete its update",
    )
    resume_update = recovery_commands.add_parser(
        "resume-update-readback",
        help="grant another bounded read-only confirmation window",
    )

    def add_update_recovery_identity(command: argparse.ArgumentParser) -> None:
        add_recovery_identity(command)
        command.add_argument("--expected-output-note-id", required=True)
        command.add_argument(
            "--expected-previous-content-sha256",
            required=True,
        )
        command.add_argument(
            "--expected-readback-attempts",
            required=True,
            type=int,
        )

    add_update_recovery_identity(resolve_update)
    add_update_recovery_identity(resume_update)

    assistant_recovery = commands.add_parser(
        "assistant-recovery",
        help="inspect or resume a durably paused assistant turn",
    )
    assistant_recovery_commands = assistant_recovery.add_subparsers(
        dest="assistant_recovery_command", required=True
    )
    assistant_recovery_commands.add_parser(
        "inspect",
        help="show content-free identifiers for an active assistant gate",
    )
    assistant_resume = assistant_recovery_commands.add_parser(
        "resume",
        help="resume the exact assistant turn after administrative review",
    )
    assistant_resume.add_argument(
        "--expected-gate",
        required=True,
        choices=_ASSISTANT_RECOVERY_GATES,
    )
    assistant_resume.add_argument("--source-note-id", required=True)
    assistant_resume.add_argument(
        "--confirm",
        action="store_true",
        help="confirm the exact gated assistant turn under review",
    )

    job_recovery = commands.add_parser(
        "job-recovery",
        help="inspect or retry one exact exhausted processing stage",
    )
    job_recovery_commands = job_recovery.add_subparsers(
        dest="job_recovery_command", required=True
    )
    job_recovery_commands.add_parser(
        "list",
        help="list bounded content-free safe-retry candidates",
    )
    job_inspect = job_recovery_commands.add_parser(
        "inspect",
        help="show content-free recovery state for one exact source note",
    )
    job_inspect.add_argument("--source-note-id", required=True)
    job_retry = job_recovery_commands.add_parser(
        "retry",
        help="retry one exact, safely repeatable exhausted stage",
    )
    job_retry.add_argument("--source-note-id", required=True)
    job_retry.add_argument(
        "--expected-stage",
        required=True,
        choices=("transcription", "model"),
    )
    job_retry.add_argument("--expected-attempts", required=True, type=int)
    job_retry.add_argument(
        "--confirm",
        action="store_true",
        help="confirm retry of the exact exhausted processing stage",
    )

    ignored_audio = commands.add_parser(
        "ignored-audio-recovery",
        help="inspect or restage one exact ignored audio request",
    )
    ignored_audio_commands = ignored_audio.add_subparsers(
        dest="ignored_audio_recovery_command", required=True
    )
    ignored_audio_inspect = ignored_audio_commands.add_parser(
        "inspect",
        help="show the content-free identity for one exact ignored audio request",
    )
    ignored_audio_inspect.add_argument("--source-note-id", required=True)
    ignored_audio_restage = ignored_audio_commands.add_parser(
        "restage",
        help="read and atomically restage one exact prefix-missing audio request",
    )
    ignored_audio_restage.add_argument("--source-note-id", required=True)
    ignored_audio_restage.add_argument(
        "--expected-audio-sha256", required=True
    )
    ignored_audio_restage.add_argument(
        "--expected-state", required=True, choices=("ignored",)
    )
    ignored_audio_restage.add_argument(
        "--expected-source-kind", required=True, choices=("audio",)
    )
    ignored_audio_restage.add_argument(
        "--expected-ignored-reason", required=True, choices=("prefix_missing",)
    )
    ignored_audio_restage.add_argument(
        "--expected-transcription-attempts", required=True, type=int
    )
    ignored_audio_restage.add_argument(
        "--expected-model-attempts", required=True, type=int
    )
    ignored_audio_restage.add_argument(
        "--expected-publication-attempts", required=True, type=int
    )
    ignored_audio_restage.add_argument(
        "--confirm",
        action="store_true",
        help="confirm recovery of the exact ignored audio request",
    )
    return parser


def _emit(stream: TextIO, payload: Mapping[str, object]) -> None:
    stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    stream.write("\n")


def _parse_light_input_audit_args(
    argv: Sequence[str],
) -> argparse.Namespace | None:
    """Parse the privacy-sensitive audit without reflecting rejected argv."""

    if tuple(argv) in {
        ("light-input-audit", "-h"),
        ("light-input-audit", "--help"),
    }:
        return None
    receipt: Path | None = None
    confirm_read = False
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--receipt" and receipt is None and index + 1 < len(argv):
            receipt = Path(argv[index + 1])
            index += 2
            continue
        if token == "--confirm-read" and not confirm_read:
            confirm_read = True
            index += 1
            continue
        raise LightInputAuditError("configuration_error")
    if receipt is None:
        raise LightInputAuditError("configuration_error")
    return argparse.Namespace(
        command="light-input-audit",
        receipt=receipt,
        confirm_read=confirm_read,
    )


def _parse_light_input_binding_args(
    argv: Sequence[str],
) -> argparse.Namespace | None:
    """Parse metadata binding without reflecting rejected arguments."""

    if tuple(argv) in {
        ("light-input-bind", "-h"),
        ("light-input-bind", "--help"),
    }:
        return None
    receipt: Path | None = None
    summary: Path | None = None
    confirm_read = False
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--receipt" and receipt is None and index + 1 < len(argv):
            receipt = Path(argv[index + 1])
            index += 2
            continue
        if token == "--summary" and summary is None and index + 1 < len(argv):
            summary = Path(argv[index + 1])
            index += 2
            continue
        if token == "--confirm-read" and not confirm_read:
            confirm_read = True
            index += 1
            continue
        raise LightInputAuditError("configuration_error")
    if receipt is None or summary is None:
        raise LightInputAuditError("configuration_error")
    return argparse.Namespace(
        command="light-input-bind",
        receipt=receipt,
        summary=summary,
        confirm_read=confirm_read,
    )


def _summarize_work(reports: Sequence[object]) -> dict[str, object]:
    states: Counter[str] = Counter()
    details: Counter[str] = Counter()
    for report in reports:
        state = getattr(report, "state", None)
        detail = getattr(report, "detail", None)
        if state is not None:
            states[str(state)] += 1
        if detail is not None:
            details[str(detail)] += 1
    return {
        "steps": len(reports),
        "result_states": dict(sorted(states.items())),
        "result_details": dict(sorted(details.items())),
    }


def _config(environ: Mapping[str, str]) -> ServiceConfig:
    return ServiceConfig.from_env(environ)


class PublicationRecoveryError(RuntimeError):
    """A content-free, stable failure category for recovery automation."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class AssistantRecoveryError(RuntimeError):
    """A content-free, stable failure category for assistant recovery."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class JobRecoveryError(RuntimeError):
    """A content-free, stable failure category for exact job recovery."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class IgnoredAudioRecoveryError(RuntimeError):
    """A content-free, stable failure category for ignored-audio recovery."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


@contextmanager
def _operator_store(config: ServiceConfig) -> Iterator[Store]:
    """Open only the existing bridge database under the runtime lock."""

    if config.state_dir.is_symlink() or not config.state_dir.is_dir():
        raise ValueError("the configured state directory does not exist or is unsafe")
    database = config.bridge_database
    if database.is_symlink() or not database.is_file():
        raise ValueError("the bridge database does not exist or is unsafe")
    with RuntimeLock(config.state_dir / "runtime.lock"):
        try:
            with Store(database) as store:
                yield store
        finally:
            if database.is_file() and not database.is_symlink():
                database.chmod(0o600)


def _publication_recovery(
    args: argparse.Namespace,
    config: ServiceConfig,
    environ: Mapping[str, str],
) -> dict[str, object]:
    with _operator_store(config) as store:
        if args.recovery_command == "inspect":
            gate = store.output_gate()
            if gate is None:
                return {"ok": True, "active": False}
            if gate == "update_ambiguous":
                update = store.ambiguous_update_candidate()
                if update is None:
                    raise PublicationRecoveryError("recovery_gate_missing")
                return {
                    "ok": True,
                    "active": True,
                    "gate": "update_ambiguous",
                    "source_note_id": update.source_note_id,
                    "publication_sha256": update.publication_sha256,
                    "publication_title_sha256": (
                        update.publication_title_sha256
                    ),
                    "publication_attempts": update.publication_attempts,
                    "output_epoch": update.output_epoch,
                    "stored_output_mode": "rolling",
                    "output_note_id": update.output_note_id,
                    "previous_content_sha256": (
                        update.previous_content_sha256
                    ),
                    "readback_attempts": update.readback_attempts,
                    "readback_limit": config.max_update_readback_attempts,
                    "readback_exhausted": (
                        update.readback_attempts
                        >= config.max_update_readback_attempts
                    ),
                    "gated_at": update.gated_at.isoformat(),
                }
            candidate = store.ambiguous_create_candidate()
            if candidate is None:
                raise PublicationRecoveryError("recovery_gate_missing")
            return {
                "ok": True,
                "active": True,
                "gate": "create_ambiguous",
                "source_note_id": candidate.source_note_id,
                "publication_sha256": candidate.publication_sha256,
                "publication_title_sha256": candidate.publication_title_sha256,
                "publication_attempts": candidate.publication_attempts,
                "output_epoch": candidate.output_epoch,
                "stored_output_mode": candidate.output_mode,
                "provisional_output_note_id": (
                    candidate.provisional_output_note_id
                ),
                "gated_at": candidate.gated_at.isoformat(),
            }
        if not args.confirm:
            raise ValueError("publication recovery requires --confirm")

        if args.recovery_command in {
            "resolve-update-applied",
            "resume-update-readback",
        }:
            update = store.ambiguous_update_candidate()
            _match_update_recovery_candidate(update, args)
            assert update is not None
            _validate_operator_identifier(args.expected_output_note_id)
            if args.recovery_command == "resolve-update-applied":
                verification = _verify_applied_publication(
                    update,
                    output_note_id=args.expected_output_note_id,
                    environ=environ,
                )
                result = store.resolve_ambiguous_update_applied(
                    source_note_id=args.source_note_id,
                    expected_publication_sha256=(
                        args.expected_publication_sha256
                    ),
                    expected_output_epoch=args.expected_output_epoch,
                    expected_output_note_id=args.expected_output_note_id,
                    expected_previous_content_sha256=(
                        args.expected_previous_content_sha256
                    ),
                    expected_readback_attempts=args.expected_readback_attempts,
                    verification=verification,
                    now=datetime.now(UTC),
                )
                verification_method: str | None = verification.method
            else:
                if (
                    update.readback_attempts
                    < config.max_update_readback_attempts
                ):
                    raise PublicationRecoveryError(
                        "update_readback_not_exhausted"
                    )
                result = store.resume_ambiguous_update_readback(
                    source_note_id=args.source_note_id,
                    expected_publication_sha256=(
                        args.expected_publication_sha256
                    ),
                    expected_output_epoch=args.expected_output_epoch,
                    expected_output_note_id=args.expected_output_note_id,
                    expected_previous_content_sha256=(
                        args.expected_previous_content_sha256
                    ),
                    expected_readback_attempts=args.expected_readback_attempts,
                    now=datetime.now(UTC),
                )
                verification_method = None
            store.checkpoint()
            return {
                "ok": True,
                "audit_id": result.audit_id,
                "action": result.action,
                "source_note_id": result.source_note_id,
                "publication_sha256": result.publication_sha256,
                "output_note_id": result.output_note_id,
                "output_epoch": result.output_epoch,
                "prior_readback_attempts": result.prior_readback_attempts,
                "decided_at": result.decided_at.isoformat(),
                "verification_method": verification_method,
            }

        candidate = store.ambiguous_create_candidate()
        _match_recovery_candidate(candidate, args)
        assert candidate is not None

        if args.recovery_command == "resolve-create-not-applied":
            decision = "not_applied"
            verification = None
        elif args.recovery_command == "resolve-create-applied":
            _validate_operator_identifier(args.output_note_id)
            if (
                candidate.provisional_output_note_id is not None
                and not hmac.compare_digest(
                    candidate.provisional_output_note_id,
                    args.output_note_id,
                )
            ):
                raise PublicationRecoveryError("recovery_output_identity_mismatch")
            verification = _verify_applied_publication(
                candidate,
                output_note_id=args.output_note_id,
                environ=environ,
            )
            decision = "applied"
        else:
            raise AssertionError("unknown publication recovery command")

        result = store.resolve_ambiguous_create(
            source_note_id=args.source_note_id,
            expected_publication_sha256=args.expected_publication_sha256,
            expected_output_epoch=args.expected_output_epoch,
            decision=decision,
            verification=verification,
            now=datetime.now(UTC),
        )
        store.checkpoint()
        return {
            "ok": True,
            "audit_id": result.audit_id,
            "decision": result.decision,
            "source_note_id": result.source_note_id,
            "publication_sha256": result.publication_sha256,
            "output_note_id": result.output_note_id,
            "output_epoch": result.output_epoch,
            "decided_at": result.decided_at.isoformat(),
            "verification_method": (
                verification.method if verification is not None else None
            ),
        }


def _assistant_recovery(
    args: argparse.Namespace,
    config: ServiceConfig,
) -> dict[str, object]:
    """Inspect or clear one exact durable assistant gate without adapters."""

    with _operator_store(config) as store:
        status = store.assistant_gate_status()
        if args.assistant_recovery_command == "inspect":
            if status is None:
                return {"ok": True, "active": False}
            gated_at = status["gated_at"]
            if not isinstance(gated_at, datetime):
                raise AssistantRecoveryError("assistant_recovery_state_invalid")
            return {
                "ok": True,
                "active": True,
                "gate": status["gate"],
                "source_note_id": status["source_note_id"],
                "gated_at": gated_at.isoformat(),
            }

        if args.assistant_recovery_command != "resume":
            raise AssertionError("unknown assistant recovery command")
        if not args.confirm:
            raise AssistantRecoveryError(
                "assistant_recovery_confirmation_required"
            )
        if status is None:
            raise AssistantRecoveryError("assistant_recovery_gate_missing")
        if (
            not hmac.compare_digest(str(status["gate"]), args.expected_gate)
            or not hmac.compare_digest(
                str(status["source_note_id"]), args.source_note_id
            )
        ):
            raise AssistantRecoveryError(
                "assistant_recovery_candidate_mismatch"
            )
        try:
            resumed = store.clear_assistant_gate(
                expected_gate=args.expected_gate,
                expected_note_id=args.source_note_id,
                now=datetime.now(UTC),
            )
        except (KeyError, RuntimeError, ValueError):
            raise AssistantRecoveryError(
                "assistant_recovery_state_invalid"
            ) from None
        if not resumed:
            raise AssistantRecoveryError("assistant_recovery_gate_missing")
        store.checkpoint()
        return {
            "ok": True,
            "resumed": True,
            "gate": args.expected_gate,
            "source_note_id": args.source_note_id,
        }


def _job_recovery(
    args: argparse.Namespace,
    config: ServiceConfig,
) -> dict[str, object]:
    """Inspect or retry an exhausted safe stage without constructing adapters."""

    with _operator_store(config) as store:
        if args.job_recovery_command == "list":
            candidates, total = store.list_job_recovery_candidates()
            return {
                "ok": True,
                "candidates": [
                    {
                        "source_note_id": candidate.source_note_id,
                        "failure_stage": candidate.failure_stage,
                        "attempts": candidate.attempts,
                        "failed_at": (
                            candidate.failed_at.isoformat()
                            if candidate.failed_at is not None
                            else None
                        ),
                    }
                    for candidate in candidates
                ],
                "total": total,
                "truncated": total > len(candidates),
            }
        try:
            candidate = store.job_recovery_candidate(args.source_note_id)
        except ValueError:
            raise JobRecoveryError("job_recovery_identity_invalid") from None
        if args.job_recovery_command == "inspect":
            if candidate is None:
                return {"ok": True, "active": False}
            return {
                "ok": True,
                "active": True,
                "source_note_id": candidate.source_note_id,
                "failure_stage": candidate.failure_stage,
                "attempts": candidate.attempts,
                "retry_allowed": candidate.retry_allowed,
                "failed_at": (
                    candidate.failed_at.isoformat()
                    if candidate.failed_at is not None
                    else None
                ),
            }

        if args.job_recovery_command != "retry":
            raise AssertionError("unknown job recovery command")
        if not args.confirm:
            raise JobRecoveryError("job_recovery_confirmation_required")
        if candidate is None:
            raise JobRecoveryError("job_recovery_candidate_missing")
        if (
            candidate.failure_stage != args.expected_stage
            or candidate.attempts != args.expected_attempts
            or not candidate.retry_allowed
        ):
            raise JobRecoveryError("job_recovery_candidate_mismatch")
        try:
            result = store.retry_terminal_job(
                source_note_id=args.source_note_id,
                expected_stage=args.expected_stage,
                expected_attempts=args.expected_attempts,
                now=datetime.now(UTC),
            )
        except KeyError:
            raise JobRecoveryError("job_recovery_candidate_missing") from None
        except ValueError:
            raise JobRecoveryError("job_recovery_candidate_mismatch") from None
        except RuntimeError:
            raise JobRecoveryError("job_recovery_state_invalid") from None
        store.checkpoint()
        return {
            "ok": True,
            "retried": True,
            "audit_id": result.audit_id,
            "source_note_id": result.source_note_id,
            "failure_stage": result.failure_stage,
            "prior_attempts": result.prior_attempts,
            "resumed_state": result.resumed_state.value,
            "retried_at": result.retried_at.isoformat(),
        }


def _ignored_audio_recovery(
    args: argparse.Namespace,
    config: ServiceConfig,
    environ: Mapping[str, str],
) -> dict[str, object]:
    """Inspect or safely restage one exact prefix-missing audio request."""

    with _operator_store(config) as store:
        try:
            candidate = store.ignored_audio_recovery_candidate(
                args.source_note_id
            )
        except ValueError:
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_identity_invalid"
            ) from None

        if args.ignored_audio_recovery_command == "inspect":
            if candidate is None:
                return {"ok": True, "active": False}
            return {
                "ok": True,
                "active": True,
                "source_note_id": candidate.source_note_id,
                "retained_sha256": candidate.retained_sha256,
                "queue_seq": candidate.queue_seq,
                "state": "ignored",
                "source_kind": "audio",
                "ignored_reason": "prefix_missing",
                "stt_attempts": candidate.transcription_attempts,
                "model_attempts": candidate.model_attempts,
                "publication_attempts": candidate.publication_attempts,
                "ignored_at": candidate.ignored_at.isoformat(),
            }

        if args.ignored_audio_recovery_command != "restage":
            raise AssertionError("unknown ignored audio recovery command")
        if not args.confirm:
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_confirmation_required"
            )
        if (
            len(args.expected_audio_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in args.expected_audio_sha256
            )
            or args.expected_transcription_attempts < 1
            or args.expected_model_attempts != 0
            or args.expected_publication_attempts != 0
        ):
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_expectation_invalid"
            )

        candidate_matches = bool(
            candidate is not None
            and hmac.compare_digest(
                candidate.retained_sha256, args.expected_audio_sha256
            )
            and candidate.transcription_attempts
            == args.expected_transcription_attempts
            and candidate.model_attempts == args.expected_model_attempts
            and candidate.publication_attempts
            == args.expected_publication_attempts
        )
        if not candidate_matches:
            try:
                already_restaged = store.ignored_audio_recovery_already_restaged(
                    args.source_note_id,
                    expected_sha256=args.expected_audio_sha256,
                    expected_transcription_attempts=(
                        args.expected_transcription_attempts
                    ),
                    expected_model_attempts=args.expected_model_attempts,
                    expected_publication_attempts=(
                        args.expected_publication_attempts
                    ),
                )
            except ValueError:
                already_restaged = False
            if not already_restaged:
                raise IgnoredAudioRecoveryError(
                    "ignored_audio_recovery_candidate_mismatch"
                )

        username = environ.get("lightphone_username", "").strip()
        password = environ.get("lightphone_password", "")
        if not username or not password:
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_credentials_missing"
            )
        try:
            provider = GaradoLightProvider(
                username=username,
                password=password,
                device_id=environ.get("lightphone_device_id") or None,
                write_policy=LightWritePolicy.READ_ONLY,
            )
        except Exception:
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_provider_setup_failed"
            ) from None

        try:
            try:
                note, audio = provider.download_audio_with_metadata(
                    args.source_note_id
                )
            except Exception:
                raise IgnoredAudioRecoveryError(
                    "ignored_audio_recovery_remote_read_failed"
                ) from None
        finally:
            provider.close()

        if not isinstance(note, RemoteNote):
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_remote_metadata_invalid"
            )
        if note.note_id != args.source_note_id:
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_remote_identity_mismatch"
            )
        if note.kind is not NoteKind.AUDIO:
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_remote_kind_mismatch"
            )
        if not audio or len(audio) > config.worker_config().max_audio_bytes:
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_remote_payload_invalid"
            )
        observed_sha256 = hashlib.sha256(audio).hexdigest()
        if not hmac.compare_digest(
            observed_sha256, args.expected_audio_sha256
        ):
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_remote_hash_mismatch"
            )

        try:
            result = store.restage_ignored_audio(
                note,
                audio,
                expected_sha256=args.expected_audio_sha256,
                expected_state=args.expected_state,
                expected_source_kind=args.expected_source_kind,
                expected_ignored_reason=args.expected_ignored_reason,
                expected_transcription_attempts=(
                    args.expected_transcription_attempts
                ),
                expected_model_attempts=args.expected_model_attempts,
                expected_publication_attempts=(
                    args.expected_publication_attempts
                ),
                now=datetime.now(UTC),
            )
        except KeyError:
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_candidate_missing"
            ) from None
        except ValueError:
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_candidate_mismatch"
            ) from None
        except RuntimeError:
            raise IgnoredAudioRecoveryError(
                "ignored_audio_recovery_state_invalid"
            ) from None
        store.checkpoint()
        return {
            "ok": True,
            "action": (
                "already_restaged" if result.already_restaged else "restaged"
            ),
            "source_note_id": result.source_note_id,
            "retained_sha256": result.retained_sha256,
            "queue_seq": result.queue_seq,
            "audit_id": result.audit_id,
            "prior_stt_attempts": (
                result.prior_transcription_attempts
            ),
            "accepted_output_epoch": result.accepted_output_epoch,
            "accepted_output_note_id": result.accepted_output_note_id,
            "stt_attempts": 0,
            "model_attempts": args.expected_model_attempts,
            "publication_attempts": args.expected_publication_attempts,
            "restaged_at": result.restaged_at.isoformat(),
        }


def _match_recovery_candidate(
    candidate: CreateRecoveryCandidate | None,
    args: argparse.Namespace,
) -> None:
    if candidate is None:
        raise PublicationRecoveryError("recovery_gate_missing")
    if (
        candidate.source_note_id != args.source_note_id
        or candidate.output_epoch != args.expected_output_epoch
        or not hmac.compare_digest(
            candidate.publication_sha256,
            args.expected_publication_sha256,
        )
    ):
        raise PublicationRecoveryError("recovery_candidate_mismatch")


def _match_update_recovery_candidate(
    candidate: UpdateRecoveryCandidate | None,
    args: argparse.Namespace,
) -> None:
    if candidate is None:
        raise PublicationRecoveryError("recovery_gate_missing")
    if (
        candidate.source_note_id != args.source_note_id
        or candidate.output_epoch != args.expected_output_epoch
        or candidate.output_note_id != args.expected_output_note_id
        or candidate.readback_attempts != args.expected_readback_attempts
        or not hmac.compare_digest(
            candidate.publication_sha256,
            args.expected_publication_sha256,
        )
        or not hmac.compare_digest(
            candidate.previous_content_sha256,
            args.expected_previous_content_sha256,
        )
    ):
        raise PublicationRecoveryError("recovery_candidate_mismatch")


def _validate_operator_identifier(value: str) -> None:
    if (
        not value
        or value != value.strip()
        or len(value) > 1024
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise PublicationRecoveryError("recovery_output_identity_invalid")


def _verify_applied_publication(
    candidate: CreateRecoveryCandidate | UpdateRecoveryCandidate,
    *,
    output_note_id: str,
    environ: Mapping[str, str],
) -> AppliedCreateVerification:
    username = environ.get("lightphone_username", "").strip()
    password = environ.get("lightphone_password", "")
    if not username or not password:
        raise PublicationRecoveryError("recovery_credentials_missing")

    try:
        provider = GaradoLightProvider(
            username=username,
            password=password,
            device_id=environ.get("lightphone_device_id") or None,
            write_policy=LightWritePolicy.READ_ONLY,
        )
    except Exception:
        raise PublicationRecoveryError("recovery_provider_setup_failed") from None
    started_at = datetime.now(UTC)
    try:
        first_note, first_title_sha, first_content_sha = _read_recovery_sample(
            provider,
            output_note_id=output_note_id,
            expected_title_sha256=candidate.publication_title_sha256,
            expected_content_sha256=candidate.publication_sha256,
        )
        second_note, second_title_sha, second_content_sha = _read_recovery_sample(
            provider,
            output_note_id=output_note_id,
            expected_title_sha256=candidate.publication_title_sha256,
            expected_content_sha256=candidate.publication_sha256,
        )
    finally:
        provider.close()
    completed_at = datetime.now(UTC)

    if (
        first_note.updated_at is None
        or second_note.updated_at is None
        or first_note.updated_at != second_note.updated_at
        or first_note.note_id != second_note.note_id
        or first_note.kind is not second_note.kind
        or not hmac.compare_digest(first_title_sha, second_title_sha)
        or not hmac.compare_digest(first_content_sha, second_content_sha)
    ):
        raise PublicationRecoveryError("recovery_remote_state_unstable")

    return AppliedCreateVerification(
        method="light_read_only_v1",
        output_note_id=output_note_id,
        observed_content_sha256=second_content_sha,
        observed_title_sha256=second_title_sha,
        observed_kind=second_note.kind,
        remote_updated_at=second_note.updated_at,
        started_at=started_at,
        completed_at=completed_at,
    )


def _read_recovery_sample(
    provider: GaradoLightProvider,
    *,
    output_note_id: str,
    expected_title_sha256: str,
    expected_content_sha256: str,
) -> tuple[RemoteNote, str, str]:
    try:
        note = provider.get_note(output_note_id)
    except Exception:
        raise PublicationRecoveryError("recovery_remote_read_failed") from None
    if note is None:
        raise PublicationRecoveryError("recovery_remote_note_missing")
    if not isinstance(note, RemoteNote):
        raise PublicationRecoveryError("recovery_remote_metadata_incomplete")
    if note.note_id != output_note_id:
        raise PublicationRecoveryError("recovery_remote_identity_mismatch")
    if note.kind is not NoteKind.TEXT:
        raise PublicationRecoveryError("recovery_remote_kind_mismatch")
    if not isinstance(note.title, str) or not isinstance(note.updated_at, datetime):
        raise PublicationRecoveryError("recovery_remote_metadata_incomplete")

    title_sha256 = hashlib.sha256(note.title.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(title_sha256, expected_title_sha256):
        raise PublicationRecoveryError("recovery_remote_title_mismatch")
    try:
        content = provider.read_note_content(output_note_id)
        if not isinstance(content, bytes):
            raise TypeError
        content_sha256 = hashlib.sha256(content).hexdigest()
    except Exception:
        raise PublicationRecoveryError("recovery_remote_read_failed") from None
    if not hmac.compare_digest(content_sha256, expected_content_sha256):
        raise PublicationRecoveryError("recovery_remote_content_mismatch")
    return note, title_sha256, content_sha256


def _light_probe(environ: Mapping[str, str]) -> dict[str, object]:
    username = environ.get("lightphone_username", "").strip()
    password = environ.get("lightphone_password", "")
    if not username or not password:
        raise ValueError("Light credentials are required")
    # READ_ONLY makes accidental mutation fail before any cloud write call.
    provider = GaradoLightProvider(
        username=username,
        password=password,
        device_id=environ.get("lightphone_device_id") or None,
        write_policy=LightWritePolicy.READ_ONLY,
    )
    try:
        notes = provider.list_notes()
    finally:
        provider.close()
    kinds = Counter(note.kind.value for note in notes)
    return {
        "ok": True,
        "write_policy": "read_only",
        "notes_total": len(notes),
        "audio_notes": kinds.get(NoteKind.AUDIO.value, 0),
        "text_notes": kinds.get(NoteKind.TEXT.value, 0),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    env = os.environ if environ is None else environ
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    raw_argv = list(argv) if argv is not None else sys.argv[1:]

    if raw_argv and raw_argv[0] in {"light-input-audit", "light-input-bind"}:
        try:
            if raw_argv[0] == "light-input-audit":
                args = _parse_light_input_audit_args(raw_argv)
            else:
                args = _parse_light_input_binding_args(raw_argv)
        except LightInputAuditError as error:
            _emit(err, {"ok": False, "error": error.category})
            return 2
        if args is None:
            if raw_argv[0] == "light-input-audit":
                out.write(
                    "usage: lightphone-gpt light-input-audit "
                    "--receipt RECEIPT [--confirm-read]\n"
                )
            else:
                out.write(
                    "usage: lightphone-gpt light-input-bind "
                    "--receipt RECEIPT --summary SUMMARY [--confirm-read]\n"
                )
            return 0
    else:
        args = _parser().parse_args(raw_argv)

    try:
        if args.command == "chatgpt-auth":
            _emit(out, run_chatgpt_auth_command(args, env))
            return 0
        if args.command == "light-probe":
            _emit(out, _light_probe(env))
            return 0
        if args.command == "light-e2e":
            if not args.confirm_run:
                raise E2ECommandError("confirmation_required")
            _emit(out, run_e2e_command(environ=env))
            return 0
        if args.command == "light-input-audit":
            result = run_light_input_audit_command(args, env)
            _emit(out, result)
            return 0 if result.get("ok") else 1
        if args.command == "light-input-bind":
            result = run_light_input_binding_command(args, env)
            _emit(out, result)
            return 0 if result.get("ok") else 1
        if args.command == "light-test-note":
            if args.light_test_note_command == "resolve":
                args.decision = args.decision.replace("-", "_")
            result = run_light_test_note_command(args, env)
            _emit(out, result)
            return 0 if result.get("ok") else 1
        config = _config(env)
        if args.command == "publication-recovery":
            _emit(out, _publication_recovery(args, config, env))
            return 0
        if args.command == "assistant-recovery":
            _emit(out, _assistant_recovery(args, config))
            return 0
        if args.command == "job-recovery":
            _emit(out, _job_recovery(args, config))
            return 0
        if args.command == "ignored-audio-recovery":
            _emit(out, _ignored_audio_recovery(args, config, env))
            return 0
        if args.command == "serve":
            serve(config)
            return 0

        with build_runtime(config) as runtime:
            if args.command == "init":
                report = runtime.worker.poll()
                _emit(
                    out,
                    {
                        "ok": report.error is None and report.output_gate is None,
                        "initialized": True,
                        "baselined": report.baselined,
                        **runtime_status(runtime),
                    },
                )
            elif args.command == "status":
                _emit(out, {"ok": True, **runtime_status(runtime)})
            elif args.command == "cycle":
                report = run_cycle(runtime)
                _emit(
                    out,
                    {
                        "ok": report.ok,
                        "poll": {
                            "baselined": report.poll.baselined,
                            "enqueued": report.poll.enqueued,
                            "duplicates": report.poll.duplicates,
                            "integrity_errors": report.poll.integrity_errors,
                            "rejected": report.poll.rejected,
                            "ignored": report.poll.ignored,
                            "output_retired": report.poll.output_retired,
                            "output_blocked": report.poll.output_gate is not None,
                            "error": report.poll.error is not None,
                        },
                        "work": _summarize_work(report.work),
                        **runtime_status(runtime),
                    },
                )
                return 0 if report.ok else 1
            elif args.command == "drain":
                reports = runtime.worker.drain(max_steps=config.drain_max_steps)
                _emit(
                    out,
                    {
                        "ok": True,
                        "work": _summarize_work(reports),
                        **runtime_status(runtime),
                    },
                )
            elif args.command == "retention":
                report = apply_retention(
                    runtime.store,
                    now=datetime.now(UTC),
                    policy=config.retention_policy(),
                )
                _emit(
                    out,
                    {
                        "ok": True,
                        "conversations_redacted": report.conversations_redacted,
                        "jobs_deleted": report.jobs_deleted,
                    },
                )
            elif args.command == "simulator":
                if args.simulator_command == "seed":
                    if args.transcript is not None:
                        content = encode_simulated_transcript(args.transcript)
                        mime_type = SIMULATED_TRANSCRIPT_MIME
                    else:
                        audio_path = args.audio_file
                        if not audio_path.is_file():
                            raise ValueError("audio file does not exist")
                        if audio_path.stat().st_size > 64 * 1024 * 1024:
                            raise ValueError("audio file is too large")
                        content = audio_path.read_bytes()
                        mime_type = args.mime_type
                    note = runtime.provider.seed_audio_note(
                        content, mime_type=mime_type
                    )
                    _emit(out, {"ok": True, "note_id": note.note_id})
                elif args.simulator_command == "seed-text":
                    note = runtime.provider.seed_text_note(
                        args.content, title=args.title
                    )
                    _emit(out, {"ok": True, "note_id": note.note_id})
                elif args.simulator_command == "list":
                    notes = runtime.provider.list_notes()
                    _emit(
                        out,
                        {
                            "ok": True,
                            "notes": [
                                {
                                    "note_id": note.note_id,
                                    "kind": note.kind.value,
                                    "mime_type": note.mime_type,
                                    "updated_at": (
                                        note.updated_at.isoformat()
                                        if note.updated_at
                                        else None
                                    ),
                                }
                                for note in notes
                            ],
                        },
                    )
                elif args.simulator_command == "delete":
                    existed = runtime.provider.get_note(args.note_id) is not None
                    runtime.provider.delete_note(args.note_id)
                    _emit(out, {"ok": True, "existed": existed})
                else:
                    raise AssertionError("unknown simulator command")
            else:
                raise AssertionError("unknown command")
        return 0
    except ChatGPTAuthorizationRequiredError:
        _emit(err, {"ok": False, "error": "authorization_required"})
        return 1
    except ChatGPTTransientError:
        _emit(err, {"ok": False, "error": "temporarily_unavailable"})
        return 1
    except ChatGPTConfigurationError:
        _emit(err, {"ok": False, "error": "configuration_error"})
        return 2
    except PublicationRecoveryError as error:
        _emit(err, {"ok": False, "error": error.category})
        return 1
    except AssistantRecoveryError as error:
        _emit(err, {"ok": False, "error": error.category})
        return 1
    except JobRecoveryError as error:
        _emit(err, {"ok": False, "error": error.category})
        return (
            2
            if error.category == "job_recovery_confirmation_required"
            else 1
        )
    except IgnoredAudioRecoveryError as error:
        _emit(err, {"ok": False, "error": error.category})
        return (
            2
            if error.category
            == "ignored_audio_recovery_confirmation_required"
            else 1
        )
    except E2ECommandError as error:
        _emit(err, error.as_dict())
        return 2 if error.category in {
            "configuration_error",
            "confirmation_required",
        } else 1
    except LightTestNoteError as error:
        _emit(
            err,
            {
                "ok": False,
                "error": "configuration_error",
                "detail": str(error),
            },
        )
        return 2
    except LightInputAuditError as error:
        payload: dict[str, object] = {"ok": False, "error": error.category}
        if error.operation_id is not None:
            payload["operation_id"] = error.operation_id
        _emit(err, payload)
        return 2 if error.category in {
            "configuration_error",
            "confirmation_required",
            "receipt_already_exists",
        } else 1
    except (ValueError, Pre0GuardError) as error:
        if args.command == "light-test-note":
            # A dependency can raise ValueError with request, response, or
            # credential material. Only our fixed LightTestNoteError messages
            # are safe to echo for this command.
            _emit(err, {"ok": False, "error": "operation_failed"})
            return 1
        _emit(
            err,
            {
                "ok": False,
                "error": "configuration_error",
                "detail": str(error),
            },
        )
        return 2
    except Exception:
        # Adapter errors intentionally contain no credential or note content,
        # but avoid echoing third-party exception details at this final boundary.
        category = (
            "light_probe_failed"
            if args.command == "light-probe"
            else "operation_failed"
        )
        _emit(err, {"ok": False, "error": category})
        return 1


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
