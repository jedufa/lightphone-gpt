from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from io import StringIO
import json
from pathlib import Path

import pytest

from lightphone_gpt import cli
import lightphone_gpt.light_test_note as light_test_note
from lightphone_gpt.adapters.light_garado import LightWritePolicy
from lightphone_gpt.models import (
    DeletionReceipt,
    MutationOutcome,
    MutationReceipt,
    NoteKind,
    RemoteNote,
)


ENV = {
    "lightphone_username": "owner@example.test",
    "lightphone_password": "credential-must-stay-private",
    "lightphone_device_id": "device-1",
}


def invoke(argv: list[str], env: dict[str, str] | None = None):
    stdout = StringIO()
    stderr = StringIO()
    code = cli.main(argv, environ=env or ENV, stdout=stdout, stderr=stderr)
    output = json.loads(stdout.getvalue()) if stdout.getvalue() else None
    error = json.loads(stderr.getvalue()) if stderr.getvalue() else None
    return code, output, error


def receipt_location(tmp_path: Path) -> Path:
    directory = tmp_path / "phase0"
    directory.mkdir(mode=0o700)
    return directory / "test-note.json"


@dataclass
class FakeCloud:
    notes: dict[str, tuple[str, bytes]] = field(default_factory=dict)
    kinds: dict[str, NoteKind] = field(default_factory=dict)
    constructed: list[dict[str, object]] = field(default_factory=list)
    create_calls: int = 0
    update_calls: int = 0
    delete_calls: int = 0
    get_note_calls: int = 0
    create_outcome: MutationOutcome = MutationOutcome.VERIFIED
    update_outcome: MutationOutcome = MutationOutcome.VERIFIED
    delete_failure: str | None = None
    close_calls: int = 0


class FakeProvider:
    def __init__(self, cloud: FakeCloud, **kwargs: object) -> None:
        self.cloud = cloud
        self.kwargs = kwargs
        cloud.constructed.append(kwargs)

    def list_notes(self):
        return [
            RemoteNote(note_id, self.cloud.kinds.get(note_id, NoteKind.TEXT), title)
            for note_id, (title, _content) in sorted(self.cloud.notes.items())
        ]

    def get_note(self, note_id: str):
        self.cloud.get_note_calls += 1
        value = self.cloud.notes.get(note_id)
        if value is None:
            return None
        return RemoteNote(
            note_id,
            self.cloud.kinds.get(note_id, NoteKind.TEXT),
            value[0],
        )

    def read_note_content(self, note_id: str) -> bytes:
        return self.cloud.notes[note_id][1]

    def create_text_note(self, *, title: str, content: str) -> MutationReceipt:
        self.cloud.create_calls += 1
        note_id = f"test-{self.cloud.create_calls}"
        raw = content.encode()
        self.cloud.notes[note_id] = (title, raw)
        self.cloud.kinds[note_id] = NoteKind.TEXT
        note = RemoteNote(note_id, NoteKind.TEXT, title)
        return MutationReceipt(
            self.cloud.create_outcome,
            note,
            hashlib.sha256(raw).hexdigest(),
            "synthetic outcome",
        )

    def delete_test_note(
        self,
        note_id: str,
        *,
        expected_title: str,
        expected_content_sha256: str,
    ) -> DeletionReceipt:
        self.cloud.delete_calls += 1
        if self.cloud.delete_failure == "before":
            raise RuntimeError("synthetic failure before apply")
        title, content = self.cloud.notes[note_id]
        assert title == expected_title
        assert hashlib.sha256(content).hexdigest() == expected_content_sha256
        del self.cloud.notes[note_id]
        self.cloud.kinds.pop(note_id, None)
        if self.cloud.delete_failure == "after":
            raise RuntimeError("synthetic failure after apply")
        return DeletionReceipt(
            MutationOutcome.VERIFIED,
            deletion_attempted=True,
            detail="synthetic delete verified",
        )

    def update_text_note(
        self,
        note_id: str,
        *,
        title: str,
        content: str,
        expected_title: str,
        expected_content_sha256: str,
    ) -> MutationReceipt:
        self.cloud.update_calls += 1
        current_title, current_content = self.cloud.notes[note_id]
        assert current_title == expected_title
        assert hashlib.sha256(current_content).hexdigest() == expected_content_sha256
        raw = content.encode()
        self.cloud.notes[note_id] = (title, raw)
        return MutationReceipt(
            self.cloud.update_outcome,
            RemoteNote(note_id, NoteKind.TEXT, title),
            hashlib.sha256(raw).hexdigest(),
            "synthetic outcome",
        )

    def close(self) -> None:
        self.cloud.close_calls += 1


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch) -> FakeCloud:
    state = FakeCloud()

    def factory(**kwargs: object) -> FakeProvider:
        return FakeProvider(state, **kwargs)

    monkeypatch.setattr(light_test_note, "GaradoLightProvider", factory)
    return state


def create_verified(path: Path) -> dict[str, object]:
    code, output, error = invoke(
        [
            "light-test-note",
            "create",
            "--receipt",
            str(path),
            "--confirm-create",
        ]
    )
    assert code == 0 and error is None
    return output


def delete_args(path: Path, created: dict[str, object]) -> list[str]:
    return [
        "light-test-note",
        "delete",
        "--receipt",
        str(path),
        "--expected-note-id",
        str(created["note_id"]),
        "--expected-content-sha256",
        str(created["content_sha256"]),
        "--confirm-delete",
    ]


def test_missing_confirmation_never_constructs_provider_or_receipt(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)

    code, output, error = invoke(
        ["light-test-note", "create", "--receipt", str(path)]
    )

    assert code == 2 and output is None
    assert error["detail"] == "create requires --confirm-create"
    assert cloud.constructed == []
    assert not path.exists()


def test_missing_update_confirmation_stops_before_receipt_and_provider(
    tmp_path: Path,
    cloud: FakeCloud,
) -> None:
    path = receipt_location(tmp_path)
    args = [
        "light-test-note",
        "update",
        "--receipt",
        str(path),
        "--expected-note-id",
        "test-1",
        "--expected-content-sha256",
        "0" * 64,
    ]

    code, output, error = invoke(args)

    assert code == 2 and output is None
    assert "confirm-update" in error["detail"]
    assert cloud.constructed == []


def test_delete_requires_explicit_confirmation_before_cloud_access(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    constructed = len(cloud.constructed)

    code, output, error = invoke(
        [
            "light-test-note",
            "delete",
            "--receipt",
            str(path),
            "--expected-note-id",
            created["note_id"],
            "--expected-content-sha256",
            created["content_sha256"],
        ]
    )

    assert code == 2 and output is None
    assert error["detail"] == "delete requires --confirm-delete"
    assert len(cloud.constructed) == constructed
    assert cloud.delete_calls == 0


def test_delete_is_exact_durable_private_and_idempotent(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    title, body = cloud.notes[str(created["note_id"])]
    cloud.notes["same-title"] = (title, body)
    cloud.kinds["same-title"] = NoteKind.TEXT
    reads_before = cloud.get_note_calls

    code, output, error = invoke(delete_args(path, created))

    assert code == 0 and error is None
    assert output == {
        "absence_observations": 2,
        "delete_retried": False,
        "exact_identity_authority": True,
        "ok": True,
        "source_note_deleted": True,
        "status": "deleted",
        "title_deletion_authority": False,
    }
    assert cloud.delete_calls == 1
    assert cloud.get_note_calls - reads_before == 3
    assert str(created["note_id"]) not in cloud.notes
    assert cloud.notes["same-title"] == (title, body)
    serialized = json.dumps(output)
    assert str(created["note_id"]) not in serialized
    assert str(created["content_sha256"]) not in serialized
    assert str(created["run_id"]) not in serialized
    persisted = json.loads(path.read_text())
    assert persisted["status"] == "deleted"
    assert persisted["active_intent"] is None
    assert persisted["last_intent"]["operation"] == "delete"

    constructed = len(cloud.constructed)
    reads_before = cloud.get_note_calls
    code, repeated, error = invoke(delete_args(path, created))
    assert code == 0 and error is None and repeated == output
    assert cloud.delete_calls == 1
    assert cloud.get_note_calls == reads_before
    assert len(cloud.constructed) == constructed

    code, inspected, error = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)]
    )
    assert code == 0 and error is None and inspected == output
    assert cloud.get_note_calls == reads_before
    assert len(cloud.constructed) == constructed


@pytest.mark.parametrize("change", ["title", "content", "kind"])
def test_delete_changed_note_records_conflict_without_mutation(
    tmp_path: Path, cloud: FakeCloud, change: str
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    note_id = str(created["note_id"])
    title, body = cloud.notes[note_id]
    if change == "title":
        cloud.notes[note_id] = ("user changed title", body)
    elif change == "content":
        cloud.notes[note_id] = (title, b"user changed content")
    else:
        cloud.kinds[note_id] = NoteKind.AUDIO

    code, output, error = invoke(delete_args(path, created))

    assert code == 1 and error is None
    assert output["status"] == "conflict"
    assert output["source_note_deleted"] is False
    assert cloud.delete_calls == 0
    assert note_id in cloud.notes
    persisted = json.loads(path.read_text())
    assert persisted["status"] == "conflict"
    assert persisted["active_intent"]["operation"] == "delete"


def test_delete_already_absent_uses_two_reads_and_never_title_matches(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    note_id = str(created["note_id"])
    title, body = cloud.notes.pop(note_id)
    cloud.kinds.pop(note_id)
    cloud.notes["duplicate"] = (title, body)
    cloud.kinds["duplicate"] = NoteKind.TEXT
    reads_before = cloud.get_note_calls

    code, output, error = invoke(delete_args(path, created))

    assert code == 0 and error is None
    assert output["status"] == "deleted"
    assert output["absence_observations"] == 2
    assert cloud.get_note_calls - reads_before == 3
    assert cloud.delete_calls == 0
    assert "duplicate" in cloud.notes


def test_delete_failure_before_apply_is_never_retried(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    note_id = str(created["note_id"])
    cloud.delete_failure = "before"

    code, output, error = invoke(delete_args(path, created))

    assert code == 1 and output is None
    assert error == {"error": "operation_failed", "ok": False}
    assert cloud.delete_calls == 1
    assert note_id in cloud.notes
    assert json.loads(path.read_text())["status"] == "unknown"

    cloud.delete_failure = None
    code, output, error = invoke(delete_args(path, created))
    assert code == 1 and error is None
    assert output["status"] == "unknown"
    assert cloud.delete_calls == 1
    assert note_id in cloud.notes

    code, output, error = invoke(delete_args(path, created))
    assert code == 1 and error is None
    assert output["status"] == "unknown"
    assert cloud.delete_calls == 1


def test_delete_failure_after_apply_reconciles_by_reads_without_retry(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    cloud.delete_failure = "after"

    code, output, error = invoke(delete_args(path, created))

    assert code == 1 and output is None
    assert error == {"error": "operation_failed", "ok": False}
    assert cloud.delete_calls == 1
    assert json.loads(path.read_text())["status"] == "unknown"
    reads_before = cloud.get_note_calls

    cloud.delete_failure = None
    code, output, error = invoke(delete_args(path, created))
    assert code == 0 and error is None
    assert output["status"] == "deleted"
    assert output["absence_observations"] == 2
    assert cloud.get_note_calls - reads_before == 2
    assert cloud.delete_calls == 1


def test_crash_after_durable_intent_recovers_read_only(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    stored = json.loads(path.read_text())
    intent = light_test_note._new_intent(
        stored,
        "delete",
        desired_revision=stored["revision"],
    )
    with light_test_note.ReceiptStore(path).locked() as store:
        light_test_note._set_active(store, stored, intent, "pending")

    code, output, error = invoke(delete_args(path, created))

    assert code == 1 and error is None
    assert output["status"] == "unknown"
    assert cloud.delete_calls == 0
    assert cloud.constructed[-1]["write_policy"] is LightWritePolicy.READ_ONLY
    assert json.loads(path.read_text())["status"] == "unknown"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("note_id", "different-note"), ("content_sha256", "f" * 64)],
)
def test_delete_requires_exact_receipt_expectations_before_provider(
    tmp_path: Path,
    cloud: FakeCloud,
    field: str,
    replacement: str,
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    args = delete_args(path, created)
    option = "--expected-note-id" if field == "note_id" else "--expected-content-sha256"
    args[args.index(option) + 1] = replacement
    constructed = len(cloud.constructed)

    code, output, error = invoke(args)

    assert code == 2 and output is None
    assert error["error"] == "configuration_error"
    assert cloud.delete_calls == 0
    assert len(cloud.constructed) == constructed


def test_delete_requires_exact_account_binding_before_provider(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    constructed = len(cloud.constructed)
    changed_env = dict(ENV, lightphone_device_id="different-device")

    code, output, error = invoke(delete_args(path, created), changed_env)

    assert code == 2 and output is None
    assert "configured Light account and device" in error["detail"]
    assert cloud.delete_calls == 0
    assert len(cloud.constructed) == constructed


def test_verified_create_is_generated_durable_and_content_free(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)

    output = create_verified(path)
    stored = json.loads(path.read_text())

    assert output["status"] == "ready"
    assert output["note_id"] == "test-1"
    assert len(output["run_id"]) == 32
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert stored["note_id"] == "test-1"
    assert stored["status"] == "ready"
    assert stored["last_intent"]["outcome"] == "verified"
    title, body = cloud.notes["test-1"]
    assert title == f"LIGHTPHONE GPT TEST {output['run_id']} R0"
    assert body.decode() == (
        "Lightphone GPT Phase 0 sync test\n"
        f"Run: {output['run_id']}\n"
        "Revision: 0\n"
    )
    serialized = json.dumps(output)
    assert body.decode() not in serialized
    assert ENV["lightphone_username"] not in serialized
    assert ENV["lightphone_password"] not in serialized
    assert cloud.constructed[0]["write_policy"] is LightWritePolicy.TEST_NOTES
    assert cloud.constructed[0]["test_title_prefix"] == "LIGHTPHONE GPT TEST"
    assert cloud.close_calls == 1


def test_operator_command_never_builds_the_continuous_runtime(
    tmp_path: Path,
    cloud: FakeCloud,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = receipt_location(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the continuous runtime must stay isolated")

    monkeypatch.setattr(cli, "build_runtime", forbidden)

    output = create_verified(path)

    assert output["status"] == "ready"
    assert cloud.create_calls == 1


def test_inspect_succeeds_only_for_exact_ready_remote_state(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    create_verified(path)

    code, inspected, error = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)]
    )

    assert code == 0 and error is None
    assert inspected["ok"] is True
    assert inspected["remote_state"] == "expected"
    assert inspected["expected_title_matches"] is True
    assert inspected["expected_content_matches"] is True
    assert "observed_content_sha256" not in inspected


def test_unknown_create_is_not_retried_and_can_be_resolved_read_only(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    cloud.create_outcome = MutationOutcome.UNKNOWN

    code, created, error = invoke(
        [
            "light-test-note",
            "create",
            "--receipt",
            str(path),
            "--confirm-create",
        ]
    )

    assert code == 1 and error is None and created["status"] == "unknown"
    assert cloud.create_calls == 1
    constructed_before = len(cloud.constructed)
    code, output, error = invoke(
        [
            "light-test-note",
            "create",
            "--receipt",
            str(path),
            "--confirm-create",
        ]
    )
    assert code == 2 and output is None
    assert error["detail"] == "the create receipt already exists"
    assert cloud.create_calls == 1
    assert len(cloud.constructed) == constructed_before

    code, inspected, error = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)]
    )
    assert code == 1 and error is None
    assert inspected["candidate_count"] == 1
    assert inspected["candidates"][0]["note_id"] == "test-1"
    assert inspected["candidates"][0]["content_matches"] is True
    assert cloud.constructed[-1]["write_policy"] is LightWritePolicy.READ_ONLY

    code, resolved, error = invoke(
        [
            "light-test-note",
            "resolve",
            "--receipt",
            str(path),
            "--decision",
            "applied",
            "--confirm-run-id",
            created["run_id"],
            "--observed-note-id",
            "test-1",
            "--observed-content-sha256",
            created["content_sha256"],
        ]
    )
    assert code == 0 and error is None
    assert resolved["status"] == "ready"
    assert resolved["note_id"] == "test-1"
    assert cloud.create_calls == 1
    assert cloud.constructed[-1]["write_policy"] is LightWritePolicy.READ_ONLY


def test_duplicate_create_candidates_remain_unresolved(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    cloud.create_outcome = MutationOutcome.UNKNOWN
    code, created, _ = invoke(
        [
            "light-test-note",
            "create",
            "--receipt",
            str(path),
            "--confirm-create",
        ]
    )
    assert code == 1
    title, body = cloud.notes["test-1"]
    cloud.notes["duplicate"] = (title, body)

    code, inspected, _ = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)]
    )
    assert code == 1
    assert [item["note_id"] for item in inspected["candidates"]] == [
        "duplicate",
        "test-1",
    ]

    code, resolved, error = invoke(
        [
            "light-test-note",
            "resolve",
            "--receipt",
            str(path),
            "--decision",
            "applied",
            "--confirm-run-id",
            created["run_id"],
            "--observed-note-id",
            "duplicate",
            "--observed-content-sha256",
            created["content_sha256"],
        ]
    )
    assert code == 2 and resolved is None
    assert error["detail"] == "the create result is not uniquely verified"
    persisted = json.loads(path.read_text())
    assert persisted["status"] == "unknown"
    assert persisted["note_id"] is None
    assert cloud.create_calls == 1


def test_uncertain_create_cannot_be_cleared_by_absence_and_late_note_is_found(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    cloud.create_outcome = MutationOutcome.UNKNOWN
    code, created, error = invoke(
        [
            "light-test-note",
            "create",
            "--receipt",
            str(path),
            "--confirm-create",
        ]
    )
    assert code == 1 and error is None
    delayed_note = cloud.notes.pop("test-1")
    cloud.kinds.pop("test-1")

    code, inspected, error = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)]
    )
    assert code == 1 and error is None
    assert inspected["candidate_count"] == 0
    constructed_before_resolution = len(cloud.constructed)

    code, resolved, error = invoke(
        [
            "light-test-note",
            "resolve",
            "--receipt",
            str(path),
            "--decision",
            "not-applied",
            "--confirm-run-id",
            created["run_id"],
        ]
    )
    assert code == 2 and resolved is None
    assert error["detail"] == (
        "an uncertain create cannot be resolved as not applied"
    )
    assert len(cloud.constructed) == constructed_before_resolution
    assert json.loads(path.read_text())["status"] == "unknown"

    cloud.notes["test-1"] = delayed_note
    cloud.kinds["test-1"] = NoteKind.TEXT
    code, inspected, error = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)]
    )
    assert code == 1 and error is None
    assert inspected["candidate_count"] == 1
    assert inspected["candidates"][0]["note_id"] == "test-1"
    assert inspected["candidates"][0]["content_matches"] is True


def test_partial_create_remains_unresolved_without_exposing_remote_hash(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    cloud.create_outcome = MutationOutcome.UNKNOWN
    code, created, error = invoke(
        [
            "light-test-note",
            "create",
            "--receipt",
            str(path),
            "--confirm-create",
        ]
    )
    assert code == 1 and error is None
    title, _body = cloud.notes["test-1"]
    private_edit = b"short private value"
    private_hash = hashlib.sha256(private_edit).hexdigest()
    cloud.notes["test-1"] = (title, private_edit)

    code, inspected, error = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)]
    )
    assert code == 1 and error is None
    assert inspected["candidates"] == [
        {"content_matches": False, "note_id": "test-1", "text_note": True}
    ]
    assert private_hash not in json.dumps(inspected)

    for decision in ("applied", "not-applied"):
        argv = [
            "light-test-note",
            "resolve",
            "--receipt",
            str(path),
            "--decision",
            decision,
            "--confirm-run-id",
            created["run_id"],
        ]
        if decision == "applied":
            argv.extend(
                [
                    "--observed-note-id",
                    "test-1",
                    "--observed-content-sha256",
                    created["content_sha256"],
                ]
            )
        code, output, error = invoke(argv)
        assert code == 2 and output is None
        assert error["error"] == "configuration_error"
        assert json.loads(path.read_text())["status"] == "unknown"


def test_partial_update_matches_neither_safe_resolution_state(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    stored = json.loads(path.read_text())
    intent = light_test_note._new_intent(stored, "update", desired_revision=1)
    with light_test_note.ReceiptStore(path).locked() as store:
        light_test_note._set_active(store, stored, intent, "unknown")

    # Model the real provider's two-call update after the content PUT succeeds
    # but before the title PATCH succeeds.
    cloud.notes[created["note_id"]] = (
        intent["previous_title"],
        light_test_note._body(created["run_id"], 1).encode(),
    )
    code, inspected, error = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)]
    )
    assert code == 1 and error is None
    assert inspected["remote_state"] == "changed"
    assert inspected["desired_state_matches"] is False
    assert "observed_content_sha256" not in inspected

    for decision in ("applied", "not-applied"):
        argv = [
            "light-test-note",
            "resolve",
            "--receipt",
            str(path),
            "--decision",
            decision,
            "--confirm-run-id",
            created["run_id"],
        ]
        if decision == "applied":
            argv.extend(
                [
                    "--observed-note-id",
                    created["note_id"],
                    "--observed-content-sha256",
                    intent["desired_content_sha256"],
                ]
            )
        code, output, error = invoke(argv)
        assert code == 2 and output is None
        assert error["error"] == "configuration_error"
        assert json.loads(path.read_text())["status"] == "unknown"


def test_update_uses_exact_receipt_identity_and_unknown_never_retries(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    cloud.update_outcome = MutationOutcome.UNKNOWN
    args = [
        "light-test-note",
        "update",
        "--receipt",
        str(path),
        "--expected-note-id",
        created["note_id"],
        "--expected-content-sha256",
        created["content_sha256"],
        "--confirm-update",
    ]

    code, updated, error = invoke(args)

    assert code == 1 and error is None and updated["status"] == "unknown"
    assert cloud.update_calls == 1
    assert updated["revision"] == 0
    code, output, error = invoke(args)
    assert code == 2 and output is None
    assert error["detail"] == "the receipt is not ready for another cloud mutation"
    assert cloud.update_calls == 1

    code, resolved, error = invoke(
        [
            "light-test-note",
            "resolve",
            "--receipt",
            str(path),
            "--decision",
            "applied",
            "--confirm-run-id",
            created["run_id"],
            "--observed-note-id",
            created["note_id"],
            "--observed-content-sha256",
            updated["operation"] and hashlib.sha256(
                cloud.notes[created["note_id"]][1]
            ).hexdigest(),
        ]
    )
    assert code == 0 and error is None
    assert resolved["status"] == "ready"
    assert resolved["revision"] == 1
    assert cloud.update_calls == 1
    assert cloud.constructed[-1]["write_policy"] is LightWritePolicy.READ_ONLY


def test_wrong_identity_or_external_edit_performs_zero_mutations(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)

    code, output, error = invoke(
        [
            "light-test-note",
            "update",
            "--receipt",
            str(path),
            "--expected-note-id",
            "different-id",
            "--expected-content-sha256",
            created["content_sha256"],
            "--confirm-update",
        ]
    )
    assert code == 2 and output is None
    assert cloud.update_calls == 0

    title, _ = cloud.notes[created["note_id"]]
    cloud.notes[created["note_id"]] = (title, b"external edit")
    code, output, error = invoke(
        [
            "light-test-note",
            "update",
            "--receipt",
            str(path),
            "--expected-note-id",
            created["note_id"],
            "--expected-content-sha256",
            created["content_sha256"],
            "--confirm-update",
        ]
    )
    assert code == 1 and error is None
    assert output["status"] == "conflict"
    assert cloud.update_calls == 0
    assert cloud.notes[created["note_id"]][1] == b"external edit"


@pytest.mark.parametrize(
    ("remote_change", "remote_state"),
    [("absent", "absent"), ("changed", "changed"), ("wrong_kind", "wrong_kind")],
)
def test_inspect_fails_when_ready_receipt_no_longer_matches_remote(
    tmp_path: Path,
    cloud: FakeCloud,
    remote_change: str,
    remote_state: str,
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    note_id = created["note_id"]
    if remote_change == "absent":
        del cloud.notes[note_id]
        cloud.kinds.pop(note_id)
    elif remote_change == "changed":
        title, _body = cloud.notes[note_id]
        cloud.notes[note_id] = (title, b"user supplied private text")
    else:
        cloud.kinds[note_id] = NoteKind.AUDIO

    code, inspected, error = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)]
    )

    assert code == 1 and error is None
    assert inspected["ok"] is False
    assert inspected["remote_state"] == remote_state
    assert "observed_content_sha256" not in inspected
    assert hashlib.sha256(b"user supplied private text").hexdigest() not in json.dumps(
        inspected
    )


def test_not_applied_update_resolution_requires_exact_previous_state(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    stored = json.loads(path.read_text())
    intent = light_test_note._new_intent(stored, "update", desired_revision=1)
    with light_test_note.ReceiptStore(path).locked() as store:
        light_test_note._set_active(store, stored, intent, "unknown")

    code, resolved, error = invoke(
        [
            "light-test-note",
            "resolve",
            "--receipt",
            str(path),
            "--decision",
            "not-applied",
            "--confirm-run-id",
            created["run_id"],
        ]
    )
    assert code == 0 and error is None
    assert resolved["status"] == "ready"
    assert resolved["revision"] == 0
    assert cloud.update_calls == 0


def test_pending_receipt_is_durably_promoted_to_unknown_before_inspection(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    stored = json.loads(path.read_text())
    intent = light_test_note._new_intent(stored, "update", desired_revision=1)
    with light_test_note.ReceiptStore(path).locked() as store:
        light_test_note._set_active(store, stored, intent, "pending")

    code, inspected, error = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)]
    )

    assert code == 1 and error is None
    assert inspected["status"] == "unknown"
    persisted = json.loads(path.read_text())
    assert persisted["status"] == "unknown"
    assert persisted["active_intent"]["outcome"] == "unknown"
    assert persisted["note_id"] == created["note_id"]
    assert cloud.update_calls == 0


@pytest.mark.parametrize("unsafe", ["relative", "wrong_parent_mode", "symlink"])
def test_unsafe_receipt_locations_are_rejected_before_provider_construction(
    tmp_path: Path,
    cloud: FakeCloud,
    monkeypatch: pytest.MonkeyPatch,
    unsafe: str,
) -> None:
    if unsafe == "relative":
        monkeypatch.chdir(tmp_path)
        path = Path("receipt.json")
    elif unsafe == "wrong_parent_mode":
        directory = tmp_path / "open"
        directory.mkdir(mode=0o755)
        path = directory / "receipt.json"
    else:
        directory = tmp_path / "safe"
        directory.mkdir(mode=0o700)
        target = directory / "target.json"
        target.write_text("unused")
        target.chmod(0o600)
        path = directory / "receipt.json"
        path.symlink_to(target)

    code, output, error = invoke(
        [
            "light-test-note",
            "create",
            "--receipt",
            str(path),
            "--confirm-create",
        ]
    )

    assert code == 2 and output is None and error["error"] == "configuration_error"
    assert cloud.constructed == []


def test_receipt_permissions_binding_and_lock_are_enforced(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    created = create_verified(path)
    constructed = len(cloud.constructed)

    path.chmod(0o640)
    code, output, error = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)]
    )
    assert code == 2 and output is None
    assert "0600" in error["detail"]
    assert len(cloud.constructed) == constructed
    path.chmod(0o600)

    changed_env = dict(ENV, lightphone_device_id="different-device")
    code, output, error = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)], changed_env
    )
    assert code == 2 and output is None
    assert "configured Light account and device" in error["detail"]
    assert len(cloud.constructed) == constructed

    store = light_test_note.ReceiptStore(path)
    with store.locked():
        code, output, error = invoke(
            ["light-test-note", "inspect", "--receipt", str(path)]
        )
    assert code == 2 and output is None
    assert "another test-note operator" in error["detail"]
    assert len(cloud.constructed) == constructed
    assert created["note_id"] in cloud.notes


def test_receipt_rejects_unrecognized_fields_before_cloud_access(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    create_verified(path)
    stored = json.loads(path.read_text())
    stored["lightphone_password"] = "must-never-be-accepted"
    path.write_text(json.dumps(stored))
    path.chmod(0o600)
    constructed = len(cloud.constructed)

    code, output, error = invoke(
        ["light-test-note", "inspect", "--receipt", str(path)]
    )

    assert code == 2 and output is None
    assert error["detail"] == "the test-note receipt is invalid"
    assert len(cloud.constructed) == constructed
    assert "must-never-be-accepted" not in json.dumps(error)


def test_missing_credentials_and_remote_errors_never_echo_sensitive_values(
    tmp_path: Path, cloud: FakeCloud, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = receipt_location(tmp_path)
    code, output, error = invoke(
        [
            "light-test-note",
            "create",
            "--receipt",
            str(path),
            "--confirm-create",
        ],
        {"lightphone_username": "private-user"},
    )
    assert code == 2 and output is None
    assert "private-user" not in json.dumps(error)
    assert cloud.constructed == []

    class FailingProvider(FakeProvider):
        def list_notes(self):
            raise ValueError("upstream included credential-must-stay-private")

    monkeypatch.setattr(
        light_test_note,
        "GaradoLightProvider",
        lambda **kwargs: FailingProvider(cloud, **kwargs),
    )
    code, output, error = invoke(
        [
            "light-test-note",
            "create",
            "--receipt",
            str(path),
            "--confirm-create",
        ]
    )
    assert code == 1 and output is None
    serialized = json.dumps(error)
    assert error == {"error": "operation_failed", "ok": False}
    assert ENV["lightphone_password"] not in serialized

    class FailingCreateProvider(FakeProvider):
        def create_text_note(self, *, title: str, content: str) -> MutationReceipt:
            super().create_text_note(title=title, content=content)
            raise ValueError("write response included credential-must-stay-private")

    monkeypatch.setattr(
        light_test_note,
        "GaradoLightProvider",
        lambda **kwargs: FailingCreateProvider(cloud, **kwargs),
    )
    code, output, error = invoke(
        [
            "light-test-note",
            "create",
            "--receipt",
            str(path),
            "--confirm-create",
        ]
    )
    assert code == 1 and output is None
    assert error == {"error": "operation_failed", "ok": False}
    assert "credential-must-stay-private" not in json.dumps(error)
    assert json.loads(path.read_text())["status"] == "unknown"


def test_explicit_device_id_is_required_before_provider_construction(
    tmp_path: Path, cloud: FakeCloud
) -> None:
    path = receipt_location(tmp_path)
    env = {key: value for key, value in ENV.items() if key != "lightphone_device_id"}

    code, output, error = invoke(
        [
            "light-test-note",
            "create",
            "--receipt",
            str(path),
            "--confirm-create",
        ],
        env,
    )

    assert code == 2 and output is None
    assert error["detail"] == (
        "an explicit Light device ID is required for test-note operations"
    )
    assert cloud.constructed == []
    assert not path.exists()

