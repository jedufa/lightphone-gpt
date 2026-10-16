"""Durable, single-worker SQLite state for the bridge.

The database is the hand-off point between every side effect.  Audio is staged
before transcription, the transcript is committed before a model call, and the
answer is committed before a Light mutation.  Repeating a stage can therefore
repeat computation, but it cannot publish two different stored answers.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator, Literal, Sequence

from .models import (
    AppliedCreateVerification,
    CreateRecoveryCandidate,
    CreateRecoveryResult,
    EnqueueOutcome,
    IgnoredAudioRecoveryCandidate,
    IgnoredAudioRecoveryResult,
    JobRecord,
    JobRecoveryCandidate,
    JobRecoveryResult,
    JobState,
    NoteKind,
    OutputSnapshot,
    RemoteNote,
    RouteKind,
    SourceCleanupRecord,
    SourceCleanupState,
    UpdateRecoveryCandidate,
    UpdateRecoveryResult,
)


_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    closed_at TEXT,
    redacted_at TEXT
);

CREATE TABLE IF NOT EXISTS bridge_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    enrolled_at TEXT,
    active_conversation_id INTEGER,
    active_output_note_id TEXT,
    output_epoch INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO bridge_state(singleton) VALUES (1);

CREATE TABLE IF NOT EXISTS output_surfaces (
    note_id TEXT PRIMARY KEY,
    epoch INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    retired_at TEXT,
    retired_reason TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    note_id TEXT PRIMARY KEY,
    queue_seq INTEGER NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (
        state IN ('ignored','queued','transcribed','answered','published','failed')
    ),
    remote_updated_at TEXT,
    mime_type TEXT,
    audio_sha256 TEXT,
    audio_blob BLOB,
    route_kind TEXT CHECK (
        route_kind IS NULL OR
        route_kind IN ('ignored','question','continue','new_conversation')
    ),
    transcript TEXT,
    answer TEXT,
    answer_sections TEXT NOT NULL DEFAULT '[]',
    next_section_index INTEGER NOT NULL DEFAULT 0,
    publication_content TEXT,
    publication_sha256 TEXT,
    conversation_id INTEGER REFERENCES conversations(id),
    source_job_note_id TEXT,
    response_id TEXT,
    output_note_id TEXT,
    accepted_output_epoch INTEGER NOT NULL,
    accepted_output_note_id TEXT,
    failure_stage TEXT,
    resume_state TEXT,
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0,1)),
    next_retry_at TEXT,
    last_error TEXT,
    ignored_reason TEXT,
    transcription_attempts INTEGER NOT NULL DEFAULT 0,
    model_attempts INTEGER NOT NULL DEFAULT 0,
    publication_attempts INTEGER NOT NULL DEFAULT 0,
    discovered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs(queue_seq);
CREATE INDEX IF NOT EXISTS jobs_conversation_idx
    ON jobs(conversation_id, queue_seq);
CREATE INDEX IF NOT EXISTS jobs_completed_idx ON jobs(completed_at);
"""


_SCHEMA_V2 = """
CREATE TABLE request_ledger (
    note_id TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    remote_updated_at TEXT,
    audio_sha256 TEXT,
    disposition TEXT NOT NULL
);

INSERT INTO request_ledger(
    note_id, first_seen_at, last_seen_at, remote_updated_at,
    audio_sha256, disposition
)
SELECT note_id, discovered_at, updated_at, remote_updated_at,
       audio_sha256,
       CASE
           WHEN ignored_reason = 'predates_enrollment' THEN 'baseline'
           WHEN failure_stage = 'intake' THEN 'rejected'
           WHEN failure_stage = 'integrity' THEN 'integrity_error'
           ELSE 'accepted'
       END
  FROM jobs;

ALTER TABLE jobs ADD COLUMN delivered_sections INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN source_section_index INTEGER;

UPDATE jobs
   SET delivered_sections = CASE
       WHEN route_kind = 'question' AND state = 'published' THEN
           MIN(
               json_array_length(answer_sections),
               1 + (
                   SELECT COUNT(*) FROM jobs AS continuation
                    WHERE continuation.source_job_note_id = jobs.note_id
                      AND continuation.route_kind = 'continue'
                      AND continuation.state = 'published'
               )
           )
       WHEN state = 'published' AND answer IS NOT NULL THEN 1
       ELSE 0
   END;

UPDATE jobs
   SET next_section_index = delivered_sections
 WHERE route_kind = 'question';
"""


_SCHEMA_V3 = """
ALTER TABLE bridge_state ADD COLUMN output_gate TEXT CHECK (
    output_gate IS NULL OR output_gate IN ('update_ambiguous','create_ambiguous')
);
ALTER TABLE bridge_state ADD COLUMN output_gate_job_note_id TEXT;
ALTER TABLE bridge_state ADD COLUMN output_gate_at TEXT;
"""


_SCHEMA_V4 = """
ALTER TABLE bridge_state ADD COLUMN text_ingress_enrolled_at TEXT;

ALTER TABLE jobs ADD COLUMN source_kind TEXT CHECK (
    source_kind IS NULL OR source_kind IN ('audio','text')
);
UPDATE jobs SET source_kind = 'audio' WHERE source_kind IS NULL;

ALTER TABLE request_ledger ADD COLUMN source_kind TEXT CHECK (
    source_kind IS NULL OR source_kind IN ('audio','text')
);
UPDATE request_ledger SET source_kind = 'audio' WHERE source_kind IS NULL;

INSERT OR IGNORE INTO request_ledger (
    note_id, first_seen_at, last_seen_at, remote_updated_at,
    audio_sha256, disposition, source_kind
)
SELECT output.note_id, output.created_at, output.created_at, NULL,
       output.content_sha256, 'bridge_output', 'text'
  FROM output_surfaces AS output;

INSERT OR IGNORE INTO request_ledger (
    note_id, first_seen_at, last_seen_at, remote_updated_at,
    audio_sha256, disposition, source_kind
)
SELECT job.output_note_id, job.updated_at, job.updated_at, NULL,
       job.publication_sha256, 'bridge_output', 'text'
  FROM jobs AS job
 WHERE job.output_note_id IS NOT NULL;

UPDATE request_ledger
   SET disposition = 'bridge_output', source_kind = 'text'
 WHERE note_id IN (
       SELECT note_id FROM output_surfaces
       UNION
       SELECT output_note_id FROM jobs WHERE output_note_id IS NOT NULL
 );
"""


_SCHEMA_V5 = """
ALTER TABLE jobs ADD COLUMN publication_output_mode TEXT CHECK (
    publication_output_mode IS NULL OR
    publication_output_mode IN ('rolling','per_turn')
);
ALTER TABLE jobs ADD COLUMN publication_title_sha256 TEXT CHECK (
    publication_title_sha256 IS NULL OR length(publication_title_sha256) = 64
);

CREATE TABLE provisional_output_identities (
    note_id TEXT PRIMARY KEY,
    source_note_id TEXT NOT NULL,
    output_epoch INTEGER NOT NULL,
    publication_attempt INTEGER NOT NULL CHECK (publication_attempt > 0),
    content_sha256 TEXT CHECK (
        content_sha256 IS NULL OR length(content_sha256) = 64
    ),
    recorded_at TEXT NOT NULL,
    UNIQUE(source_note_id, output_epoch, publication_attempt)
);

CREATE TABLE publication_recovery_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gated_at TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    gate TEXT NOT NULL CHECK (gate = 'create_ambiguous'),
    source_note_id TEXT NOT NULL,
    publication_sha256 TEXT NOT NULL CHECK (length(publication_sha256) = 64),
    publication_title_sha256 TEXT NOT NULL CHECK (
        length(publication_title_sha256) = 64
    ),
    publication_attempts INTEGER NOT NULL CHECK (publication_attempts > 0),
    output_epoch INTEGER NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('applied','not_applied')),
    output_mode TEXT NOT NULL CHECK (output_mode IN ('rolling','per_turn')),
    output_note_id TEXT,
    observed_content_sha256 TEXT,
    observed_title_sha256 TEXT,
    observed_kind TEXT CHECK (
        observed_kind IS NULL OR observed_kind = 'text'
    ),
    remote_updated_at TEXT,
    verification_method TEXT CHECK (
        verification_method IS NULL OR verification_method = 'light_read_only_v1'
    ),
    verification_started_at TEXT,
    verification_completed_at TEXT,
    UNIQUE(gate, source_note_id),
    CHECK (
        (decision = 'applied' AND output_note_id IS NOT NULL
            AND observed_content_sha256 IS NOT NULL
            AND observed_title_sha256 IS NOT NULL
            AND observed_kind = 'text'
            AND remote_updated_at IS NOT NULL
            AND verification_method = 'light_read_only_v1'
            AND verification_started_at IS NOT NULL
            AND verification_completed_at IS NOT NULL)
        OR
        (decision = 'not_applied' AND output_note_id IS NULL
            AND observed_content_sha256 IS NULL
            AND observed_title_sha256 IS NULL
            AND observed_kind IS NULL
            AND remote_updated_at IS NULL
            AND verification_method IS NULL
            AND verification_started_at IS NULL
            AND verification_completed_at IS NULL)
    )
);
"""


_SCHEMA_V6 = """
ALTER TABLE output_surfaces
    ADD COLUMN missing_observations INTEGER NOT NULL DEFAULT 0
    CHECK (missing_observations >= 0);
ALTER TABLE output_surfaces ADD COLUMN missing_since TEXT;
"""


_SCHEMA_V7 = """
ALTER TABLE bridge_state ADD COLUMN assistant_gate TEXT CHECK (
    assistant_gate IS NULL OR assistant_gate IN (
        'authorization_required',
        'plan_limit',
        'credential_ambiguous',
        'inference_ambiguous',
        'account_not_eligible',
        'configuration_error',
        'daily_model_limit'
    )
);
ALTER TABLE bridge_state ADD COLUMN assistant_gate_job_note_id TEXT;
ALTER TABLE bridge_state ADD COLUMN assistant_gate_at TEXT;

CREATE TABLE model_usage_days (
    utc_day TEXT PRIMARY KEY,
    attempts INTEGER NOT NULL CHECK (attempts >= 0),
    updated_at TEXT NOT NULL
);
"""


_SCHEMA_V8 = """
ALTER TABLE jobs ADD COLUMN manual_retry_allowed INTEGER NOT NULL DEFAULT 0
    CHECK (manual_retry_allowed IN (0,1));
ALTER TABLE jobs ADD COLUMN model_attempt_active INTEGER NOT NULL DEFAULT 0
    CHECK (model_attempt_active IN (0,1));

-- A pre-v8 transcribed job with a recorded model attempt cannot be proven not
-- to have reached the remote model. Preserve it behind the ambiguity gate on
-- the first worker pass instead of silently repeating the request.
UPDATE jobs
   SET model_attempt_active = 1
 WHERE state = 'transcribed' AND model_attempts > 0;

CREATE TABLE job_recovery_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    retried_at TEXT NOT NULL,
    source_note_id TEXT NOT NULL,
    failure_stage TEXT NOT NULL CHECK (
        failure_stage IN ('transcription','model')
    ),
    prior_attempts INTEGER NOT NULL CHECK (prior_attempts > 0),
    resumed_state TEXT NOT NULL CHECK (
        resumed_state IN ('queued','transcribed')
    ),
    FOREIGN KEY(source_note_id) REFERENCES jobs(note_id) ON DELETE CASCADE
);
CREATE INDEX job_recovery_source_idx
    ON job_recovery_audit(source_note_id, id);
"""


_SCHEMA_V9 = """
ALTER TABLE bridge_state
    ADD COLUMN output_gate_readback_attempts INTEGER NOT NULL DEFAULT 0
    CHECK (output_gate_readback_attempts >= 0);

CREATE TABLE update_recovery_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gated_at TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('applied','resume_readback')),
    source_note_id TEXT NOT NULL,
    publication_sha256 TEXT NOT NULL CHECK (length(publication_sha256) = 64),
    publication_title_sha256 TEXT NOT NULL CHECK (
        length(publication_title_sha256) = 64
    ),
    publication_attempts INTEGER NOT NULL CHECK (publication_attempts > 0),
    output_epoch INTEGER NOT NULL CHECK (output_epoch >= 0),
    output_note_id TEXT NOT NULL,
    previous_content_sha256 TEXT NOT NULL CHECK (
        length(previous_content_sha256) = 64
    ),
    prior_readback_attempts INTEGER NOT NULL CHECK (
        prior_readback_attempts >= 0
    ),
    observed_content_sha256 TEXT,
    observed_title_sha256 TEXT,
    observed_kind TEXT CHECK (
        observed_kind IS NULL OR observed_kind = 'text'
    ),
    remote_updated_at TEXT,
    verification_method TEXT CHECK (
        verification_method IS NULL OR verification_method = 'light_read_only_v1'
    ),
    verification_started_at TEXT,
    verification_completed_at TEXT,
    CHECK (
        (action = 'applied'
            AND observed_content_sha256 IS NOT NULL
            AND observed_title_sha256 IS NOT NULL
            AND observed_kind = 'text'
            AND remote_updated_at IS NOT NULL
            AND verification_method = 'light_read_only_v1'
            AND verification_started_at IS NOT NULL
            AND verification_completed_at IS NOT NULL)
        OR
        (action = 'resume_readback'
            AND observed_content_sha256 IS NULL
            AND observed_title_sha256 IS NULL
            AND observed_kind IS NULL
            AND remote_updated_at IS NULL
            AND verification_method IS NULL
            AND verification_started_at IS NULL
            AND verification_completed_at IS NULL)
    )
);
CREATE INDEX update_recovery_source_idx
    ON update_recovery_audit(source_note_id, id);

-- v8 could misclassify one exact stale read as proof that an ambiguous rolling
-- update was not applied, clear its gate, and leave the turn terminal. Restore
-- only an unambiguous latest candidate whose output generation is unchanged.
-- This reinstates read-only recovery and never resets its mutation attempt.
WITH legacy_candidate AS (
    SELECT job.note_id, COALESCE(job.completed_at, job.updated_at) AS gated_at
      FROM jobs AS job
      JOIN bridge_state AS bridge ON bridge.singleton = 1
      JOIN output_surfaces AS surface
        ON surface.note_id = bridge.active_output_note_id
       AND surface.retired_at IS NULL
       AND surface.epoch = bridge.output_epoch
     WHERE bridge.output_gate IS NULL
       AND job.state = 'failed'
       AND job.failure_stage = 'publication'
       AND job.resume_state = 'answered'
       AND job.last_error =
           'remote read-back confirmed update was not applied'
       AND job.publication_attempts > 0
       AND job.publication_output_mode = 'rolling'
       AND job.publication_content IS NOT NULL
       AND job.publication_sha256 IS NOT NULL
       AND job.publication_title_sha256 IS NOT NULL
       AND job.output_note_id IS NULL
       AND job.accepted_output_epoch = bridge.output_epoch
       AND (
           job.accepted_output_note_id IS NULL
           OR job.accepted_output_note_id = bridge.active_output_note_id
       )
       AND surface.content_sha256 != job.publication_sha256
       AND NOT EXISTS (
           SELECT 1 FROM jobs AS later
            WHERE later.queue_seq > job.queue_seq
              AND later.state != 'ignored'
       )
)
UPDATE bridge_state
   SET output_gate = 'update_ambiguous',
       output_gate_job_note_id = (SELECT note_id FROM legacy_candidate),
       output_gate_at = (SELECT gated_at FROM legacy_candidate),
       output_gate_readback_attempts = 0
 WHERE singleton = 1
   AND output_gate IS NULL
   AND (SELECT COUNT(*) FROM legacy_candidate) = 1;
"""


_SCHEMA_V10 = """
ALTER TABLE bridge_state
    ADD COLUMN source_cleanup_enabled INTEGER NOT NULL DEFAULT 0
    CHECK (source_cleanup_enabled IN (0,1));
ALTER TABLE bridge_state ADD COLUMN source_cleanup_policy_started_at TEXT;

CREATE TABLE source_cleanups (
    source_note_id TEXT PRIMARY KEY,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('audio','text')),
    source_sha256 TEXT NOT NULL CHECK (length(source_sha256) = 64),
    state TEXT NOT NULL CHECK (
        state IN ('pending','attempted','conflict','confirmed')
    ),
    delete_attempts INTEGER NOT NULL DEFAULT 0
        CHECK (delete_attempts IN (0,1)),
    precheck_failures INTEGER NOT NULL DEFAULT 0
        CHECK (precheck_failures >= 0),
    readback_attempts INTEGER NOT NULL DEFAULT 0
        CHECK (readback_attempts >= 0),
    missing_observations INTEGER NOT NULL DEFAULT 0
        CHECK (missing_observations >= 0),
    next_retry_at TEXT,
    last_error TEXT,
    scheduled_at TEXT NOT NULL,
    attempted_at TEXT,
    confirmed_at TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_note_id) REFERENCES jobs(note_id) ON DELETE CASCADE,
    CHECK (
        (state = 'pending' AND delete_attempts = 0
            AND attempted_at IS NULL AND confirmed_at IS NULL)
        OR
        (state IN ('attempted','conflict') AND delete_attempts IN (0,1)
            AND confirmed_at IS NULL)
        OR
        (state = 'confirmed' AND confirmed_at IS NOT NULL)
    )
);
CREATE INDEX source_cleanups_state_idx
    ON source_cleanups(state, updated_at, source_note_id);
"""


_SCHEMA_V11 = """
ALTER TABLE bridge_state
    ADD COLUMN notifier_sequence INTEGER NOT NULL DEFAULT 0
    CHECK (notifier_sequence >= 0);
ALTER TABLE jobs ADD COLUMN notifier_event_sequence INTEGER
    CHECK (notifier_event_sequence IS NULL OR notifier_event_sequence > 0);

-- Preserve the legacy public cursor as the migration baseline. Future ready
-- events use this database-owned counter, so retention and queue-sequence
-- reuse cannot make the cursor regress or suppress a later publication.
UPDATE bridge_state
   SET notifier_sequence = (
       SELECT COALESCE(MAX(queue_seq), 0)
         FROM jobs
        WHERE state = 'published'
   )
 WHERE singleton = 1;

UPDATE jobs
   SET notifier_event_sequence = queue_seq
 WHERE state = 'published';

CREATE UNIQUE INDEX jobs_notifier_event_sequence_idx
    ON jobs(notifier_event_sequence)
 WHERE notifier_event_sequence IS NOT NULL;
"""


_SCHEMA_V12 = """
-- SQLite cannot widen a CHECK constraint in place. Rebuild only the jobs
-- table so ``reset`` is a first-class, durable route that survives a crash
-- between local speech routing and recovery of an ambiguous remote call.
CREATE TABLE jobs_v12 (
    note_id TEXT PRIMARY KEY,
    queue_seq INTEGER NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (
        state IN ('ignored','queued','transcribed','answered','published','failed')
    ),
    remote_updated_at TEXT,
    mime_type TEXT,
    audio_sha256 TEXT,
    audio_blob BLOB,
    route_kind TEXT CHECK (
        route_kind IS NULL OR route_kind IN (
            'ignored','question','continue','new_conversation','reset'
        )
    ),
    transcript TEXT,
    answer TEXT,
    answer_sections TEXT NOT NULL DEFAULT '[]',
    next_section_index INTEGER NOT NULL DEFAULT 0,
    publication_content TEXT,
    publication_sha256 TEXT,
    conversation_id INTEGER REFERENCES conversations(id),
    source_job_note_id TEXT,
    response_id TEXT,
    output_note_id TEXT,
    accepted_output_epoch INTEGER NOT NULL,
    accepted_output_note_id TEXT,
    failure_stage TEXT,
    resume_state TEXT,
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0,1)),
    next_retry_at TEXT,
    last_error TEXT,
    ignored_reason TEXT,
    transcription_attempts INTEGER NOT NULL DEFAULT 0,
    model_attempts INTEGER NOT NULL DEFAULT 0,
    publication_attempts INTEGER NOT NULL DEFAULT 0,
    discovered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    delivered_sections INTEGER NOT NULL DEFAULT 0,
    source_section_index INTEGER,
    source_kind TEXT CHECK (
        source_kind IS NULL OR source_kind IN ('audio','text')
    ),
    publication_output_mode TEXT CHECK (
        publication_output_mode IS NULL OR
        publication_output_mode IN ('rolling','per_turn')
    ),
    publication_title_sha256 TEXT CHECK (
        publication_title_sha256 IS NULL OR length(publication_title_sha256) = 64
    ),
    manual_retry_allowed INTEGER NOT NULL DEFAULT 0
        CHECK (manual_retry_allowed IN (0,1)),
    model_attempt_active INTEGER NOT NULL DEFAULT 0
        CHECK (model_attempt_active IN (0,1)),
    notifier_event_sequence INTEGER
        CHECK (notifier_event_sequence IS NULL OR notifier_event_sequence > 0),
    admitted_output_gate TEXT CHECK (
        admitted_output_gate IS NULL OR
        admitted_output_gate IN ('create_ambiguous','update_ambiguous')
    ),
    admitted_output_gate_job_note_id TEXT,
    admitted_assistant_gate TEXT CHECK (
        admitted_assistant_gate IS NULL OR
        admitted_assistant_gate = 'inference_ambiguous'
    ),
    admitted_assistant_gate_job_note_id TEXT,
    CHECK (
        (admitted_output_gate IS NULL
            AND admitted_output_gate_job_note_id IS NULL)
        OR
        (admitted_output_gate IS NOT NULL
            AND admitted_output_gate_job_note_id IS NOT NULL)
    ),
    CHECK (
        (admitted_assistant_gate IS NULL
            AND admitted_assistant_gate_job_note_id IS NULL)
        OR
        (admitted_assistant_gate IS NOT NULL
            AND admitted_assistant_gate_job_note_id IS NOT NULL)
    )
);

INSERT INTO jobs_v12 (
    note_id, queue_seq, state, remote_updated_at, mime_type, audio_sha256,
    audio_blob, route_kind, transcript, answer, answer_sections,
    next_section_index, publication_content, publication_sha256,
    conversation_id, source_job_note_id, response_id, output_note_id,
    accepted_output_epoch, accepted_output_note_id, failure_stage,
    resume_state, retryable, next_retry_at, last_error, ignored_reason,
    transcription_attempts, model_attempts, publication_attempts,
    discovered_at, updated_at, completed_at, delivered_sections,
    source_section_index, source_kind, publication_output_mode,
    publication_title_sha256, manual_retry_allowed, model_attempt_active,
    notifier_event_sequence
)
SELECT
    note_id, queue_seq, state, remote_updated_at, mime_type, audio_sha256,
    audio_blob, route_kind, transcript, answer, answer_sections,
    next_section_index, publication_content, publication_sha256,
    conversation_id, source_job_note_id, response_id, output_note_id,
    accepted_output_epoch, accepted_output_note_id, failure_stage,
    resume_state, retryable, next_retry_at, last_error, ignored_reason,
    transcription_attempts, model_attempts, publication_attempts,
    discovered_at, updated_at, completed_at, delivered_sections,
    source_section_index, source_kind, publication_output_mode,
    publication_title_sha256, manual_retry_allowed, model_attempt_active,
    notifier_event_sequence
FROM jobs;

DROP TABLE jobs;
ALTER TABLE jobs_v12 RENAME TO jobs;
CREATE INDEX jobs_queue_idx ON jobs(queue_seq);
CREATE INDEX jobs_conversation_idx ON jobs(conversation_id, queue_seq);
CREATE INDEX jobs_completed_idx ON jobs(completed_at);
CREATE UNIQUE INDEX jobs_notifier_event_sequence_idx
    ON jobs(notifier_event_sequence)
 WHERE notifier_event_sequence IS NOT NULL;
"""


_LATEST_SCHEMA_VERSION = 12

MISSING_OUTPUT_UPDATE_ERROR = "reply note was missing during update"
MISSING_OUTPUT_READBACK_EXHAUSTED_ERROR = (
    "reply note remained absent through bounded update read-back"
)
STALE_UPDATE_DELIVERY_ABANDONED_ERROR = (
    "uncertain update delivery abandoned after bounded non-desired read-back"
)
BEST_EFFORT_PUBLICATION_SUPERSEDED_ERROR = (
    "uncertain publication superseded by a newer assistant request"
)
BEST_EFFORT_RESET_SUPERSEDED_ERROR = "pending request superseded by assistant reset"
BEST_EFFORT_INFERENCE_RESET_ERROR = (
    "uncertain model turn abandoned by assistant reset"
)
_RESTAGED_IGNORED_AUDIO_REASON = "prefix_missing_recovery_queued"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _encode_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _as_utc(value).isoformat(timespec="microseconds")


def _decode_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _require_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _require_operator_identifier(value: str, *, field: str) -> None:
    if (
        not value
        or value != value.strip()
        or len(value) > 1024
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{field} is not a valid exact identifier")


class Store:
    """Owns the bridge database and all state transitions."""

    def __init__(self, path: str | Path) -> None:
        path_string = str(path)
        if path_string != ":memory:":
            Path(path_string).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._connection = sqlite3.connect(
            path_string,
            timeout=5.0,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA secure_delete = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        try:
            self._migrate()
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        try:
            self.checkpoint()
        finally:
            self._connection.close()

    def checkpoint(self) -> None:
        """Flush and truncate WAL frames after sensitive row transitions."""

        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """Exposed for health checks and narrowly scoped diagnostics."""

        return self._connection

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _migrate(self) -> None:
        current = int(
            self._connection.execute("PRAGMA user_version").fetchone()[0]
        )
        if current > _LATEST_SCHEMA_VERSION:
            raise RuntimeError(
                "database schema is newer than this lightphone-gpt build"
            )
        migrations = {
            1: _SCHEMA_V1,
            2: _SCHEMA_V2,
            3: _SCHEMA_V3,
            4: _SCHEMA_V4,
            5: _SCHEMA_V5,
            6: _SCHEMA_V6,
            7: _SCHEMA_V7,
            8: _SCHEMA_V8,
            9: _SCHEMA_V9,
            10: _SCHEMA_V10,
            11: _SCHEMA_V11,
            12: _SCHEMA_V12,
        }
        for version in range(current + 1, _LATEST_SCHEMA_VERSION + 1):
            script = migrations[version]
            # Rebuilding a parent table while foreign keys are active would
            # cascade-delete its recovery/cleanup children. Disable FK actions
            # only around v12's transactional copy, then verify every binding
            # before exposing the migrated store.
            rebuild_jobs = version == 12
            if rebuild_jobs:
                self._connection.execute("PRAGMA foreign_keys = OFF")
            try:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    f"{script}\n"
                    f"PRAGMA user_version = {version};\n"
                    "COMMIT;"
                )
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise
            finally:
                if rebuild_jobs:
                    self._connection.execute("PRAGMA foreign_keys = ON")
            if rebuild_jobs:
                violation = self._connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchone()
                if violation is not None:
                    raise RuntimeError("database migration broke a foreign key")

    def is_enrolled(self) -> bool:
        row = self._connection.execute(
            "SELECT enrolled_at FROM bridge_state WHERE singleton = 1"
        ).fetchone()
        return bool(row["enrolled_at"])

    def latest_published_sequence(self) -> int:
        """Return the durable, strictly monotonic ready-event sequence."""

        row = self._connection.execute(
            "SELECT notifier_sequence AS value FROM bridge_state "
            "WHERE singleton = 1"
        ).fetchone()
        return int(row["value"])

    def enrolled_at(self) -> datetime | None:
        row = self._connection.execute(
            "SELECT enrolled_at FROM bridge_state WHERE singleton = 1"
        ).fetchone()
        return _decode_time(row["enrolled_at"])

    def is_text_ingress_enrolled(self) -> bool:
        row = self._connection.execute(
            "SELECT text_ingress_enrolled_at FROM bridge_state "
            "WHERE singleton = 1"
        ).fetchone()
        return bool(row["text_ingress_enrolled_at"])

    def configure_source_cleanup(self, *, enabled: bool, now: datetime) -> None:
        """Persist the explicit source-deletion policy for publication commits.

        Enabling the policy does not backfill old publications. A cleanup row
        is created atomically only when a later publication completes.
        """

        timestamp = _encode_time(now)
        with self._write() as connection:
            connection.execute(
                """
                UPDATE bridge_state
                   SET source_cleanup_enabled = ?,
                       source_cleanup_policy_started_at = CASE
                           WHEN ? = 1 THEN COALESCE(
                               source_cleanup_policy_started_at, ?
                           )
                           ELSE source_cleanup_policy_started_at
                       END
                 WHERE singleton = 1
                """,
                (int(enabled), int(enabled), timestamp),
            )

    def source_cleanup_enabled(self) -> bool:
        row = self._connection.execute(
            "SELECT source_cleanup_enabled FROM bridge_state WHERE singleton = 1"
        ).fetchone()
        return bool(row["source_cleanup_enabled"])

    def oldest_pending_source_cleanup(
        self, *, now: datetime
    ) -> SourceCleanupRecord | None:
        row = self._connection.execute(
            """
            SELECT cleanup.*
              FROM source_cleanups AS cleanup
              JOIN jobs AS job ON job.note_id = cleanup.source_note_id
             WHERE cleanup.state = 'pending'
               AND (
                   cleanup.next_retry_at IS NULL
                   OR cleanup.next_retry_at <= ?
               )
             ORDER BY job.queue_seq
             LIMIT 1
            """,
            (_encode_time(now),),
        ).fetchone()
        return self._source_cleanup_from_row(row) if row is not None else None

    def next_source_cleanup_readback(self) -> SourceCleanupRecord | None:
        """Return one unresolved post-DELETE item in fair read-only order."""

        row = self._connection.execute(
            """
            SELECT cleanup.*
              FROM source_cleanups AS cleanup
             WHERE cleanup.state IN ('attempted','conflict')
             ORDER BY cleanup.updated_at, cleanup.source_note_id
             LIMIT 1
            """
        ).fetchone()
        return self._source_cleanup_from_row(row) if row is not None else None

    def begin_source_cleanup_attempt(
        self,
        source_note_id: str,
        *,
        expected_kind: NoteKind,
        expected_sha256: str,
        now: datetime,
    ) -> SourceCleanupRecord:
        _require_sha256(expected_sha256, field="source cleanup hash")
        timestamp = _encode_time(now)
        with self._write() as connection:
            result = connection.execute(
                """
                UPDATE source_cleanups
                   SET state = 'attempted', delete_attempts = 1,
                       attempted_at = ?, next_retry_at = NULL,
                       last_error = NULL, updated_at = ?
                 WHERE source_note_id = ? AND source_kind = ?
                   AND source_sha256 = ? AND state = 'pending'
                   AND delete_attempts = 0
                """,
                (
                    timestamp,
                    timestamp,
                    source_note_id,
                    expected_kind.value,
                    expected_sha256,
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError("source cleanup attempt boundary is inconsistent")
        candidate = self.source_cleanup(source_note_id)
        assert candidate is not None
        return candidate

    def source_cleanup_not_attempted(
        self,
        source_note_id: str,
        *,
        error: str,
        retry_at: datetime,
        now: datetime,
    ) -> None:
        """Restore a safe retry only when the provider proves DELETE was not called."""

        timestamp = _encode_time(now)
        with self._write() as connection:
            result = connection.execute(
                """
                UPDATE source_cleanups
                   SET state = 'pending', delete_attempts = 0,
                       precheck_failures = precheck_failures + 1,
                       attempted_at = NULL, next_retry_at = ?,
                       last_error = ?, missing_observations = 0,
                       updated_at = ?
                 WHERE source_note_id = ? AND state = 'attempted'
                   AND delete_attempts = 1
                """,
                (
                    _encode_time(retry_at),
                    error[:500],
                    timestamp,
                    source_note_id,
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError("source cleanup safe-retry state is inconsistent")

    def mark_source_cleanup_unknown(
        self, source_note_id: str, *, error: str, now: datetime
    ) -> None:
        with self._write() as connection:
            result = connection.execute(
                """
                UPDATE source_cleanups
                   SET last_error = ?, updated_at = ?
                 WHERE source_note_id = ? AND state = 'attempted'
                   AND delete_attempts = 1
                """,
                (error[:500], _encode_time(now), source_note_id),
            )
            if result.rowcount != 1:
                raise RuntimeError("source cleanup ambiguity state is inconsistent")

    def mark_source_cleanup_conflict(
        self, source_note_id: str, *, error: str, now: datetime
    ) -> None:
        with self._write() as connection:
            result = connection.execute(
                """
                UPDATE source_cleanups
                   SET state = 'conflict', next_retry_at = NULL,
                       last_error = ?, missing_observations = 0,
                       updated_at = ?
                 WHERE source_note_id = ?
                   AND state IN ('pending','attempted','conflict')
                """,
                (error[:500], _encode_time(now), source_note_id),
            )
            if result.rowcount != 1:
                raise RuntimeError("source cleanup conflict state is inconsistent")

    def record_verified_source_cleanup_absence(
        self,
        source_note_id: str,
        *,
        deletion_attempted: bool,
        now: datetime,
    ) -> None:
        """Persist the provider's first exact absence without declaring success."""

        timestamp = _encode_time(now)
        with self._write() as connection:
            result = connection.execute(
                """
                UPDATE source_cleanups
                   SET state = 'attempted', delete_attempts = ?,
                       attempted_at = CASE WHEN ? THEN attempted_at ELSE NULL END,
                       missing_observations = 1, next_retry_at = NULL,
                       last_error = NULL, updated_at = ?
                 WHERE source_note_id = ? AND state = 'attempted'
                """,
                (
                    int(deletion_attempted),
                    int(deletion_attempted),
                    timestamp,
                    source_note_id,
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError("source cleanup absence state is inconsistent")

    def record_source_cleanup_absent(
        self,
        source_note_id: str,
        *,
        confirmation_reads: int,
        now: datetime,
    ) -> bool:
        if confirmation_reads < 2:
            raise ValueError("source deletion confirmation requires at least two reads")
        timestamp = _encode_time(now)
        with self._write() as connection:
            row = connection.execute(
                "SELECT state, missing_observations FROM source_cleanups "
                "WHERE source_note_id = ?",
                (source_note_id,),
            ).fetchone()
            if row is None or row["state"] not in {"attempted", "conflict"}:
                return False
            observations = int(row["missing_observations"]) + 1
            confirmed = observations >= confirmation_reads
            connection.execute(
                """
                UPDATE source_cleanups
                   SET state = CASE WHEN ? THEN 'confirmed' ELSE state END,
                       missing_observations = ?,
                       confirmed_at = CASE WHEN ? THEN ? ELSE confirmed_at END,
                       last_error = CASE WHEN ? THEN NULL ELSE last_error END,
                       updated_at = ?
                 WHERE source_note_id = ?
                """,
                (
                    int(confirmed),
                    observations,
                    int(confirmed),
                    timestamp,
                    int(confirmed),
                    timestamp,
                    source_note_id,
                ),
            )
            return confirmed

    def record_source_cleanup_present(
        self,
        source_note_id: str,
        *,
        observed_kind: NoteKind,
        observed_sha256: str | None,
        maximum_readbacks: int,
        now: datetime,
    ) -> str:
        if maximum_readbacks < 1:
            raise ValueError("source deletion read-back limit must be positive")
        if observed_sha256 is not None:
            _require_sha256(observed_sha256, field="observed source hash")
        timestamp = _encode_time(now)
        with self._write() as connection:
            row = connection.execute(
                "SELECT * FROM source_cleanups WHERE source_note_id = ?",
                (source_note_id,),
            ).fetchone()
            if row is None or row["state"] not in {"attempted", "conflict"}:
                return "resolved"
            exact = (
                row["source_kind"] == observed_kind.value
                and observed_sha256 is not None
                and hmac.compare_digest(str(row["source_sha256"]), observed_sha256)
            )
            if not exact:
                connection.execute(
                    """
                    UPDATE source_cleanups
                       SET state = 'conflict', missing_observations = 0,
                           last_error = 'source note changed after acceptance',
                           updated_at = ?
                     WHERE source_note_id = ?
                    """,
                    (timestamp, source_note_id),
                )
                return "conflict"
            attempts = int(row["readback_attempts"])
            if row["state"] == "attempted" and attempts < maximum_readbacks:
                attempts += 1
            connection.execute(
                """
                UPDATE source_cleanups
                   SET readback_attempts = ?, missing_observations = 0,
                       last_error = CASE
                           WHEN state = 'attempted'
                                AND ? >= ?
                           THEN 'source deletion outcome remains unknown'
                           ELSE last_error
                       END,
                       updated_at = ?
                 WHERE source_note_id = ?
                """,
                (attempts, attempts, maximum_readbacks, timestamp, source_note_id),
            )
            return "exhausted" if attempts >= maximum_readbacks else "present"

    def record_source_cleanup_still_present(
        self, source_note_id: str, *, now: datetime
    ) -> None:
        """Record metadata presence after bounded content reads are exhausted."""

        with self._write() as connection:
            result = connection.execute(
                """
                UPDATE source_cleanups
                   SET missing_observations = 0, updated_at = ?
                 WHERE source_note_id = ? AND state = 'attempted'
                """,
                (_encode_time(now), source_note_id),
            )
            if result.rowcount != 1:
                raise RuntimeError("source cleanup read-back state is inconsistent")

    def source_cleanup(self, source_note_id: str) -> SourceCleanupRecord | None:
        row = self._connection.execute(
            "SELECT * FROM source_cleanups WHERE source_note_id = ?",
            (source_note_id,),
        ).fetchone()
        return self._source_cleanup_from_row(row) if row is not None else None

    def source_cleanup_status(self) -> dict[str, int | bool]:
        rows = self._connection.execute(
            "SELECT state, COUNT(*) AS count FROM source_cleanups GROUP BY state"
        ).fetchall()
        counts = {str(row["state"]): int(row["count"]) for row in rows}
        unresolved = sum(
            counts.get(state, 0) for state in ("pending", "attempted", "conflict")
        )
        return {
            "enabled": self.source_cleanup_enabled(),
            "pending": counts.get("pending", 0),
            "uncertain": counts.get("attempted", 0),
            "conflicts": counts.get("conflict", 0),
            "confirmed": counts.get("confirmed", 0),
            "unresolved": unresolved,
            "blocked": bool(counts.get("attempted", 0) or counts.get("conflict", 0)),
        }

    def establish_baseline(
        self, notes: Sequence[RemoteNote], *, now: datetime
    ) -> int:
        """Ignore every note identity visible at enrollment without downloading."""

        timestamp = _encode_time(now)
        inserted = 0
        source_notes = sorted(
            notes,
            key=lambda item: (
                _encode_time(item.updated_at) or "",
                item.note_id,
            ),
        )
        with self._write() as connection:
            state = connection.execute(
                "SELECT enrolled_at, output_epoch, active_output_note_id "
                "FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            if state["enrolled_at"]:
                return 0
            next_sequence = connection.execute(
                "SELECT COALESCE(MAX(queue_seq), 0) + 1 AS value FROM jobs"
            ).fetchone()["value"]
            for note in source_notes:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO request_ledger (
                        note_id, first_seen_at, last_seen_at,
                        remote_updated_at, audio_sha256, disposition, source_kind
                    ) VALUES (?, ?, ?, ?, NULL, 'baseline', ?)
                    """,
                    (
                        note.note_id,
                        timestamp,
                        timestamp,
                        _encode_time(note.updated_at),
                        note.kind.value,
                    ),
                )
                result = connection.execute(
                    """
                    INSERT OR IGNORE INTO jobs (
                        note_id, queue_seq, state, remote_updated_at, mime_type,
                        source_kind, route_kind, accepted_output_epoch,
                        accepted_output_note_id, ignored_reason,
                        discovered_at, updated_at, completed_at
                    ) VALUES (?, ?, 'ignored', ?, ?, ?, 'ignored', ?, ?,
                              'predates_enrollment', ?, ?, ?)
                    """,
                    (
                        note.note_id,
                        next_sequence,
                        _encode_time(note.updated_at),
                        note.mime_type,
                        note.kind.value,
                        state["output_epoch"],
                        state["active_output_note_id"],
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
                if result.rowcount:
                    inserted += 1
                    next_sequence += 1
            connection.execute(
                "UPDATE bridge_state SET enrolled_at = ?, "
                "text_ingress_enrolled_at = ? WHERE singleton = 1",
                (timestamp, timestamp),
            )
        return inserted

    def establish_text_ingress_baseline(
        self, notes: Sequence[RemoteNote], *, now: datetime
    ) -> int:
        """Baseline text identities when upgrading an already-enrolled store.

        Version 3 only discovered audio. This one-time marker prevents existing
        cloud text from becoming requests merely because version 4 adds text
        input. No note content is needed for this transition.
        """

        timestamp = _encode_time(now)
        inserted = 0
        text_notes = sorted(
            (note for note in notes if note.kind is NoteKind.TEXT),
            key=lambda item: (_encode_time(item.updated_at) or "", item.note_id),
        )
        with self._write() as connection:
            state = connection.execute(
                "SELECT text_ingress_enrolled_at, output_epoch, "
                "active_output_note_id FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            if state["text_ingress_enrolled_at"]:
                return 0
            next_sequence = connection.execute(
                "SELECT COALESCE(MAX(queue_seq), 0) + 1 AS value FROM jobs"
            ).fetchone()["value"]
            for note in text_notes:
                ledger = connection.execute(
                    "SELECT audio_sha256, disposition, source_kind "
                    "FROM request_ledger WHERE note_id = ?",
                    (note.note_id,),
                ).fetchone()
                if ledger is not None:
                    if ledger["disposition"] == "bridge_output":
                        continue
                    if ledger["source_kind"] != NoteKind.TEXT.value:
                        existing = connection.execute(
                            "SELECT 1 FROM jobs WHERE note_id = ?",
                            (note.note_id,),
                        ).fetchone()
                        if existing is None:
                            self._insert_terminal_job(
                                connection,
                                note,
                                now=now,
                                audio_sha256=ledger["audio_sha256"],
                                failure_stage="integrity",
                                error="source kind changed for existing note id",
                            )
                            inserted += 1
                            next_sequence += 1
                        else:
                            connection.execute(
                                """
                                UPDATE jobs
                                   SET state = 'failed',
                                       failure_stage = 'integrity',
                                       resume_state = NULL, retryable = 0,
                                       next_retry_at = NULL,
                                       last_error =
                                           'source kind changed for existing note id',
                                       audio_blob = NULL, updated_at = ?,
                                       completed_at = ?
                                 WHERE note_id = ?
                                """,
                                (timestamp, timestamp, note.note_id),
                            )
                        connection.execute(
                            """
                            UPDATE request_ledger
                               SET last_seen_at = ?, remote_updated_at = ?,
                                   disposition = 'integrity_error'
                             WHERE note_id = ?
                            """,
                            (
                                timestamp,
                                _encode_time(note.updated_at),
                                note.note_id,
                            ),
                        )
                    continue

                connection.execute(
                    """
                    INSERT INTO request_ledger (
                        note_id, first_seen_at, last_seen_at,
                        remote_updated_at, audio_sha256, disposition, source_kind
                    ) VALUES (?, ?, ?, ?, NULL, 'baseline', 'text')
                    """,
                    (
                        note.note_id,
                        timestamp,
                        timestamp,
                        _encode_time(note.updated_at),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO jobs (
                        note_id, queue_seq, state, remote_updated_at, mime_type,
                        source_kind, route_kind, accepted_output_epoch,
                        accepted_output_note_id, ignored_reason,
                        discovered_at, updated_at, completed_at
                    ) VALUES (?, ?, 'ignored', ?, ?, 'text', 'ignored', ?, ?,
                              'predates_text_ingress', ?, ?, ?)
                    """,
                    (
                        note.note_id,
                        next_sequence,
                        _encode_time(note.updated_at),
                        note.mime_type,
                        state["output_epoch"],
                        state["active_output_note_id"],
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
                inserted += 1
                next_sequence += 1
            connection.execute(
                "UPDATE bridge_state SET text_ingress_enrolled_at = ? "
                "WHERE singleton = 1",
                (timestamp,),
            )
        return inserted

    def enqueue_audio(
        self,
        note: RemoteNote,
        audio: bytes,
        *,
        now: datetime,
    ) -> EnqueueOutcome:
        if note.kind is not NoteKind.AUDIO:
            raise ValueError("only audio notes can be queued as audio")
        return self._enqueue_source(note, audio, now=now)

    def enqueue_text(
        self,
        note: RemoteNote,
        content: bytes,
        *,
        now: datetime,
    ) -> EnqueueOutcome:
        """Stage text durably for local routing without speech transcription."""

        if note.kind is not NoteKind.TEXT:
            raise ValueError("only text notes can be queued as text")
        return self._enqueue_source(note, content, now=now)

    def ignore_text_at_intake(
        self,
        note: RemoteNote,
        *,
        content_sha256: str,
        now: datetime,
    ) -> EnqueueOutcome:
        """Ledger an unaddressed text note without persisting its contents."""

        if note.kind is not NoteKind.TEXT:
            raise ValueError("only text notes can be ignored as text")
        timestamp = _encode_time(now)
        remote_updated = _encode_time(note.updated_at)
        with self._write() as connection:
            existing = connection.execute(
                "SELECT audio_sha256, source_kind FROM jobs WHERE note_id = ?",
                (note.note_id,),
            ).fetchone()
            ledger = connection.execute(
                "SELECT audio_sha256, source_kind FROM request_ledger "
                "WHERE note_id = ?",
                (note.note_id,),
            ).fetchone()
            prior = existing if existing is not None else ledger
            if prior is not None:
                prior_digest = prior["audio_sha256"]
                if prior["source_kind"] != note.kind.value or (
                    prior_digest is not None and prior_digest != content_sha256
                ):
                    if existing is None:
                        self._insert_terminal_job(
                            connection,
                            note,
                            now=now,
                            audio_sha256=prior_digest,
                            failure_stage="integrity",
                            error="source content changed for existing note id",
                        )
                    else:
                        connection.execute(
                            """
                            UPDATE jobs
                               SET state = 'failed', failure_stage = 'integrity',
                                   resume_state = NULL, retryable = 0,
                                   next_retry_at = NULL,
                                   last_error =
                                       'source content changed for existing note id',
                                   audio_blob = NULL, updated_at = ?, completed_at = ?
                             WHERE note_id = ?
                            """,
                            (timestamp, timestamp, note.note_id),
                        )
                    connection.execute(
                        """
                        UPDATE request_ledger
                           SET last_seen_at = ?, remote_updated_at = ?,
                               disposition = 'integrity_error'
                         WHERE note_id = ?
                        """,
                        (timestamp, remote_updated, note.note_id),
                    )
                    return EnqueueOutcome.INTEGRITY_ERROR
                if existing is not None:
                    connection.execute(
                        "UPDATE jobs SET remote_updated_at = ?, updated_at = ? "
                        "WHERE note_id = ?",
                        (remote_updated, timestamp, note.note_id),
                    )
                connection.execute(
                    """
                    UPDATE request_ledger
                       SET last_seen_at = ?, remote_updated_at = ?
                     WHERE note_id = ?
                    """,
                    (timestamp, remote_updated, note.note_id),
                )
                return EnqueueOutcome.DUPLICATE

            bridge = connection.execute(
                "SELECT output_epoch, active_output_note_id, output_gate, "
                "output_gate_job_note_id, assistant_gate, "
                "assistant_gate_job_note_id "
                "FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            sequence = connection.execute(
                "SELECT COALESCE(MAX(queue_seq), 0) + 1 AS value FROM jobs"
            ).fetchone()["value"]
            connection.execute(
                """
                INSERT INTO jobs (
                    note_id, queue_seq, state, remote_updated_at, mime_type,
                    source_kind, audio_sha256, audio_blob, route_kind,
                    accepted_output_epoch, accepted_output_note_id,
                    ignored_reason, discovered_at, updated_at, completed_at
                ) VALUES (?, ?, 'ignored', ?, ?, 'text', NULL, NULL, 'ignored',
                          ?, ?, 'prefix_missing', ?, ?, ?)
                """,
                (
                    note.note_id,
                    sequence,
                    remote_updated,
                    note.mime_type,
                    bridge["output_epoch"],
                    bridge["active_output_note_id"],
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO request_ledger (
                    note_id, first_seen_at, last_seen_at,
                    remote_updated_at, audio_sha256, disposition, source_kind
                ) VALUES (?, ?, ?, ?, NULL, 'ignored', 'text')
                """,
                (
                    note.note_id,
                    timestamp,
                    timestamp,
                    remote_updated,
                ),
            )
        return EnqueueOutcome.IGNORED

    def _enqueue_source(
        self,
        note: RemoteNote,
        content: bytes,
        *,
        now: datetime,
    ) -> EnqueueOutcome:
        # audio_sha256/audio_blob are legacy column names. source_kind makes
        # their meaning explicit for text input without a destructive migration.
        digest = sha256_bytes(content)
        timestamp = _encode_time(now)
        remote_updated = _encode_time(note.updated_at)
        with self._write() as connection:
            existing = connection.execute(
                "SELECT audio_sha256, remote_updated_at, source_kind "
                "FROM jobs WHERE note_id = ?",
                (note.note_id,),
            ).fetchone()
            ledger = connection.execute(
                "SELECT audio_sha256, remote_updated_at, source_kind "
                "FROM request_ledger "
                "WHERE note_id = ?",
                (note.note_id,),
            ).fetchone()
            if existing is not None:
                prior_digest = existing["audio_sha256"]
                if existing["source_kind"] != note.kind.value or (
                    prior_digest is not None and prior_digest != digest
                ):
                    connection.execute(
                        """
                        UPDATE jobs
                           SET state = 'failed', failure_stage = 'integrity',
                               resume_state = NULL, retryable = 0,
                               next_retry_at = NULL,
                               last_error = 'source content changed for existing note id',
                               audio_blob = NULL, updated_at = ?, completed_at = ?
                         WHERE note_id = ?
                        """,
                        (timestamp, timestamp, note.note_id),
                    )
                    connection.execute(
                        """
                        UPDATE request_ledger
                           SET last_seen_at = ?, remote_updated_at = ?,
                               disposition = 'integrity_error'
                         WHERE note_id = ?
                        """,
                        (timestamp, remote_updated, note.note_id),
                    )
                    return EnqueueOutcome.INTEGRITY_ERROR
                connection.execute(
                    "UPDATE jobs SET remote_updated_at = ?, updated_at = ? "
                    "WHERE note_id = ?",
                    (remote_updated, timestamp, note.note_id),
                )
                connection.execute(
                    """
                    INSERT INTO request_ledger (
                        note_id, first_seen_at, last_seen_at,
                        remote_updated_at, audio_sha256, disposition, source_kind
                    ) VALUES (?, ?, ?, ?, ?, 'accepted', ?)
                    ON CONFLICT(note_id) DO UPDATE SET
                        last_seen_at = excluded.last_seen_at,
                        remote_updated_at = excluded.remote_updated_at,
                        source_kind = excluded.source_kind
                    """,
                    (
                        note.note_id,
                        timestamp,
                        timestamp,
                        remote_updated,
                        prior_digest or digest,
                        note.kind.value,
                    ),
                )
                return EnqueueOutcome.DUPLICATE

            if ledger is not None:
                prior_digest = ledger["audio_sha256"]
                if ledger["source_kind"] != note.kind.value or (
                    prior_digest is not None and prior_digest != digest
                ):
                    self._insert_terminal_job(
                        connection,
                        note,
                        now=now,
                        audio_sha256=prior_digest,
                        failure_stage="integrity",
                        error="source content changed for existing note id",
                    )
                    connection.execute(
                        """
                        UPDATE request_ledger
                           SET last_seen_at = ?, remote_updated_at = ?,
                               disposition = 'integrity_error'
                         WHERE note_id = ?
                        """,
                        (timestamp, remote_updated, note.note_id),
                    )
                    return EnqueueOutcome.INTEGRITY_ERROR
                connection.execute(
                    """
                    UPDATE request_ledger
                       SET last_seen_at = ?, remote_updated_at = ?
                     WHERE note_id = ?
                    """,
                    (timestamp, remote_updated, note.note_id),
                )
                return EnqueueOutcome.DUPLICATE

            bridge = connection.execute(
                "SELECT output_epoch, active_output_note_id, output_gate, "
                "output_gate_job_note_id, assistant_gate, "
                "assistant_gate_job_note_id "
                "FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            sequence = connection.execute(
                "SELECT COALESCE(MAX(queue_seq), 0) + 1 AS value FROM jobs"
            ).fetchone()["value"]
            connection.execute(
                """
                INSERT INTO jobs (
                    note_id, queue_seq, state, remote_updated_at, mime_type,
                    source_kind, audio_sha256, audio_blob, accepted_output_epoch,
                    accepted_output_note_id, discovered_at, updated_at,
                    admitted_output_gate, admitted_output_gate_job_note_id,
                    admitted_assistant_gate,
                    admitted_assistant_gate_job_note_id
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note.note_id,
                    sequence,
                    remote_updated,
                    note.mime_type,
                    note.kind.value,
                    digest,
                    sqlite3.Binary(content),
                    bridge["output_epoch"],
                    bridge["active_output_note_id"],
                    timestamp,
                    timestamp,
                    bridge["output_gate"],
                    bridge["output_gate_job_note_id"],
                    (
                        bridge["assistant_gate"]
                        if bridge["assistant_gate"] == "inference_ambiguous"
                        else None
                    ),
                    (
                        bridge["assistant_gate_job_note_id"]
                        if bridge["assistant_gate"] == "inference_ambiguous"
                        else None
                    ),
                ),
            )
            connection.execute(
                """
                INSERT INTO request_ledger (
                    note_id, first_seen_at, last_seen_at,
                    remote_updated_at, audio_sha256, disposition, source_kind
                ) VALUES (?, ?, ?, ?, ?, 'accepted', ?)
                """,
                (
                    note.note_id,
                    timestamp,
                    timestamp,
                    remote_updated,
                    digest,
                    note.kind.value,
                ),
            )
        return EnqueueOutcome.ENQUEUED

    def should_verify_source(
        self,
        note_id: str,
        source_kind: NoteKind,
        remote_updated_at: datetime | None,
    ) -> bool:
        row = self._connection.execute(
            "SELECT remote_updated_at, audio_sha256, source_kind, disposition "
            "FROM request_ledger "
            "WHERE note_id = ?",
            (note_id,),
        ).fetchone()
        if row is None:
            return False
        if row["disposition"] in {"integrity_error", "bridge_output"}:
            return False
        if row["source_kind"] != source_kind.value:
            return True
        if row["audio_sha256"] is None:
            return False
        return row["remote_updated_at"] != _encode_time(remote_updated_at)

    def source_kind_changed(self, note_id: str, source_kind: NoteKind) -> bool:
        row = self._connection.execute(
            "SELECT source_kind, disposition FROM request_ledger "
            "WHERE note_id = ?",
            (note_id,),
        ).fetchone()
        return bool(
            row is not None
            and row["disposition"] not in {"integrity_error", "bridge_output"}
            and row["source_kind"] != source_kind.value
        )

    def quarantine_source_kind_change(
        self, note: RemoteNote, *, now: datetime
    ) -> EnqueueOutcome:
        """Quarantine an identity kind change using metadata only."""

        timestamp = _encode_time(now)
        remote_updated = _encode_time(note.updated_at)
        with self._write() as connection:
            ledger = connection.execute(
                "SELECT audio_sha256, source_kind, disposition "
                "FROM request_ledger WHERE note_id = ?",
                (note.note_id,),
            ).fetchone()
            if (
                ledger is None
                or ledger["disposition"] in {"integrity_error", "bridge_output"}
                or ledger["source_kind"] == note.kind.value
            ):
                return EnqueueOutcome.DUPLICATE
            existing = connection.execute(
                "SELECT 1 FROM jobs WHERE note_id = ?", (note.note_id,)
            ).fetchone()
            if existing is None:
                self._insert_terminal_job(
                    connection,
                    note,
                    now=now,
                    audio_sha256=ledger["audio_sha256"],
                    failure_stage="integrity",
                    error="source kind changed for existing note id",
                )
            else:
                connection.execute(
                    """
                    UPDATE jobs
                       SET state = 'failed', failure_stage = 'integrity',
                           resume_state = NULL, retryable = 0,
                           next_retry_at = NULL,
                           last_error = 'source kind changed for existing note id',
                           audio_blob = NULL, updated_at = ?, completed_at = ?
                     WHERE note_id = ?
                    """,
                    (timestamp, timestamp, note.note_id),
                )
            connection.execute(
                """
                UPDATE request_ledger
                   SET last_seen_at = ?, remote_updated_at = ?,
                       disposition = 'integrity_error'
                 WHERE note_id = ?
                """,
                (timestamp, remote_updated, note.note_id),
            )
        return EnqueueOutcome.INTEGRITY_ERROR

    def should_verify_audio(
        self, note_id: str, remote_updated_at: datetime | None
    ) -> bool:
        """Compatibility wrapper for audio-only callers."""

        return self.should_verify_source(
            note_id, NoteKind.AUDIO, remote_updated_at
        )

    def has_seen_request(self, note_id: str) -> bool:
        return (
            self._connection.execute(
                "SELECT 1 FROM request_ledger WHERE note_id = ?", (note_id,)
            ).fetchone()
            is not None
        )

    def is_known_output_note(self, note_id: str) -> bool:
        """Return whether the permanent identity belongs to bridge output."""

        row = self._connection.execute(
            "SELECT disposition FROM request_ledger WHERE note_id = ?",
            (note_id,),
        ).fetchone()
        if row is not None and row["disposition"] == "bridge_output":
            return True
        return (
            self._connection.execute(
                """
                SELECT 1 FROM output_surfaces WHERE note_id = ?
                UNION ALL
                SELECT 1 FROM jobs WHERE output_note_id = ?
                LIMIT 1
                """,
                (note_id, note_id),
            ).fetchone()
            is not None
        )

    def known_output_note_ids(self) -> tuple[str, ...]:
        """Return durable output identities for a write-policy allowlist.

        The values are opaque provider identifiers.  The union deliberately
        includes retired surfaces and per-turn outputs: an old bridge output
        remains bridge-owned after it is no longer active and must never be
        admitted as user input.
        """

        rows = self._connection.execute(
            """
            SELECT note_id FROM request_ledger
             WHERE disposition = 'bridge_output'
            UNION
            SELECT note_id FROM output_surfaces
            UNION
            SELECT output_note_id FROM jobs
             WHERE output_note_id IS NOT NULL
            ORDER BY 1
            """
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def pending_job_count(self) -> int:
        """Count admitted work that has not reached a terminal disposition."""

        row = self._connection.execute(
            """
            SELECT COUNT(*) FROM jobs
             WHERE state IN ('queued','transcribed','answered')
                OR (state = 'failed' AND (
                    completed_at IS NULL OR manual_retry_allowed = 1
                ))
            """
        ).fetchone()
        return int(row[0])

    def manual_recovery_pending(self) -> bool:
        """Return whether an exhausted safe stage blocks sequential work."""

        row = self._connection.execute(
            """
            SELECT 1 FROM jobs
             WHERE state = 'failed' AND manual_retry_allowed = 1
             LIMIT 1
            """
        ).fetchone()
        return row is not None

    def local_reset_discovery_blocker(self, *, now: datetime) -> JobRecord | None:
        """Return the exact no-gate job currently preventing queue progress.

        Manual recovery is a global ordering gate in the worker. Otherwise an
        oldest retryable job blocks only while its durable retry time remains
        in the future. These are the two local stalls through which a later
        exact reset may be discovered without invoking the model or Light.
        """

        row = self._connection.execute(
            """
            SELECT * FROM jobs
             WHERE state = 'failed' AND manual_retry_allowed = 1
             ORDER BY queue_seq
             LIMIT 1
            """
        ).fetchone()
        if row is not None:
            return self._job_from_row(row)
        row = self._connection.execute(
            """
            SELECT * FROM jobs
             WHERE state IN ('queued','transcribed','answered')
                OR (state = 'failed' AND retryable = 1)
             ORDER BY queue_seq
             LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        job = self._job_from_row(row)
        if (
            job.state == JobState.FAILED
            and job.retryable
            and job.next_retry_at is not None
            and _as_utc(job.next_retry_at) > _as_utc(now)
        ):
            return job
        return None

    def remember_output_note(
        self,
        note_id: str,
        *,
        source_note_id: str,
        output_epoch: int,
        publication_attempt: int,
        content_sha256: str | None,
        now: datetime,
    ) -> None:
        """Bind a returned create identity to one exact publication attempt."""

        _require_operator_identifier(note_id, field="output note id")
        _require_operator_identifier(source_note_id, field="source note id")
        if note_id == source_note_id:
            raise ValueError("source and output note identifiers must differ")
        if output_epoch < 0 or publication_attempt < 1:
            raise ValueError("publication identity boundary is invalid")
        if content_sha256 is not None:
            _require_sha256(content_sha256, field="observed content hash")
        timestamp = _encode_time(now)
        assert timestamp is not None
        with self._write() as connection:
            source = connection.execute(
                """
                SELECT state, accepted_output_epoch, publication_attempts,
                       publication_sha256, publication_output_mode
                  FROM jobs WHERE note_id = ?
                """,
                (source_note_id,),
            ).fetchone()
            if (
                source is None
                or source["state"] != JobState.ANSWERED
                or int(source["accepted_output_epoch"]) != output_epoch
                or int(source["publication_attempts"]) != publication_attempt
                or source["publication_sha256"] is None
                or source["publication_output_mode"] not in {"rolling", "per_turn"}
            ):
                raise ValueError(
                    "provisional output does not match an active publication attempt"
                )

            binding = connection.execute(
                """
                SELECT source_note_id, output_epoch, publication_attempt,
                       content_sha256
                  FROM provisional_output_identities WHERE note_id = ?
                """,
                (note_id,),
            ).fetchone()
            attempt_binding = connection.execute(
                """
                SELECT note_id FROM provisional_output_identities
                 WHERE source_note_id = ? AND output_epoch = ?
                   AND publication_attempt = ?
                """,
                (source_note_id, output_epoch, publication_attempt),
            ).fetchone()
            ledger = connection.execute(
                "SELECT disposition, source_kind, audio_sha256 "
                "FROM request_ledger WHERE note_id = ?",
                (note_id,),
            ).fetchone()
            exact_binding = bool(
                binding is not None
                and binding["source_note_id"] == source_note_id
                and int(binding["output_epoch"]) == output_epoch
                and int(binding["publication_attempt"]) == publication_attempt
            )
            if binding is not None and not exact_binding:
                raise ValueError("output note identity belongs to another publication")
            if (
                attempt_binding is not None
                and attempt_binding["note_id"] != note_id
            ):
                raise ValueError("publication attempt already has another output identity")
            if ledger is not None:
                if (
                    ledger["disposition"] != "bridge_output"
                    or ledger["source_kind"] != NoteKind.TEXT
                    or not exact_binding
                ):
                    raise ValueError("output note identity conflicts with durable state")
                existing_hash = ledger["audio_sha256"]
                if (
                    existing_hash is not None
                    and content_sha256 is not None
                    and not hmac.compare_digest(str(existing_hash), content_sha256)
                ):
                    raise ValueError("output note content conflicts with durable state")
            elif binding is not None:
                raise RuntimeError("provisional output binding has no request ledger entry")

            if binding is None:
                connection.execute(
                    """
                    INSERT INTO provisional_output_identities(
                        note_id, source_note_id, output_epoch,
                        publication_attempt, content_sha256, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        note_id,
                        source_note_id,
                        output_epoch,
                        publication_attempt,
                        content_sha256,
                        timestamp,
                    ),
                )
            elif binding["content_sha256"] is None and content_sha256 is not None:
                connection.execute(
                    "UPDATE provisional_output_identities SET content_sha256 = ? "
                    "WHERE note_id = ?",
                    (content_sha256, note_id),
                )
            elif (
                binding["content_sha256"] is not None
                and content_sha256 is not None
                and not hmac.compare_digest(
                    str(binding["content_sha256"]), content_sha256
                )
            ):
                raise ValueError("provisional output content changed")

            if ledger is None:
                connection.execute(
                    """
                    INSERT INTO request_ledger (
                        note_id, first_seen_at, last_seen_at, remote_updated_at,
                        audio_sha256, disposition, source_kind
                    ) VALUES (?, ?, ?, NULL, ?, 'bridge_output', 'text')
                    """,
                    (note_id, timestamp, timestamp, content_sha256),
                )
            else:
                connection.execute(
                    """
                    UPDATE request_ledger
                       SET last_seen_at = ?,
                           audio_sha256 = COALESCE(?, audio_sha256)
                     WHERE note_id = ? AND disposition = 'bridge_output'
                       AND source_kind = 'text'
                    """,
                    (timestamp, content_sha256, note_id),
                )

    def reject_oversized_audio(
        self,
        note: RemoteNote,
        *,
        audio_sha256: str,
        now: datetime,
    ) -> EnqueueOutcome:
        """Record a terminal intake result without storing the audio bytes."""

        if note.kind is not NoteKind.AUDIO:
            raise ValueError("only audio notes can be rejected at intake")
        return self._reject_oversized_source(
            note,
            content_sha256=audio_sha256,
            error="recording exceeds configured size limit",
            now=now,
        )

    def reject_oversized_text(
        self,
        note: RemoteNote,
        *,
        now: datetime,
    ) -> EnqueueOutcome:
        """Reject oversized text without retaining content or its digest."""

        if note.kind is not NoteKind.TEXT:
            raise ValueError("only text notes can be rejected at intake")
        return self._reject_oversized_source(
            note,
            content_sha256=None,
            error="text note exceeds configured size limit",
            now=now,
        )

    def reject_invalid_text(
        self,
        note: RemoteNote,
        *,
        now: datetime,
    ) -> EnqueueOutcome:
        """Reject undecodable text without retaining content or its digest."""

        if note.kind is not NoteKind.TEXT:
            raise ValueError("only text notes can be rejected as text")
        return self._reject_oversized_source(
            note,
            content_sha256=None,
            error="text note is not valid UTF-8",
            now=now,
        )

    def _reject_oversized_source(
        self,
        note: RemoteNote,
        *,
        content_sha256: str | None,
        error: str,
        now: datetime,
    ) -> EnqueueOutcome:
        timestamp = _encode_time(now)
        remote_updated = _encode_time(note.updated_at)
        with self._write() as connection:
            ledger = connection.execute(
                "SELECT audio_sha256, source_kind FROM request_ledger "
                "WHERE note_id = ?",
                (note.note_id,),
            ).fetchone()
            if ledger is not None:
                prior_digest = ledger["audio_sha256"]
                if (
                    ledger["source_kind"] == note.kind.value
                    and (prior_digest is None or prior_digest == content_sha256)
                ):
                    connection.execute(
                        """
                        UPDATE request_ledger
                           SET last_seen_at = ?, remote_updated_at = ?
                         WHERE note_id = ?
                        """,
                        (timestamp, remote_updated, note.note_id),
                    )
                    return EnqueueOutcome.DUPLICATE
                existing = connection.execute(
                    "SELECT 1 FROM jobs WHERE note_id = ?", (note.note_id,)
                ).fetchone()
                if existing is None:
                    self._insert_terminal_job(
                        connection,
                        note,
                        now=now,
                        audio_sha256=prior_digest,
                        failure_stage="integrity",
                        error="source content changed for existing note id",
                    )
                else:
                    connection.execute(
                        """
                        UPDATE jobs
                           SET state = 'failed', failure_stage = 'integrity',
                               resume_state = NULL, retryable = 0,
                               next_retry_at = NULL,
                               last_error = 'source content changed for existing note id',
                               audio_blob = NULL, updated_at = ?, completed_at = ?
                         WHERE note_id = ?
                        """,
                        (timestamp, timestamp, note.note_id),
                    )
                connection.execute(
                    """
                    UPDATE request_ledger
                       SET last_seen_at = ?, remote_updated_at = ?,
                           disposition = 'integrity_error'
                     WHERE note_id = ?
                    """,
                    (timestamp, remote_updated, note.note_id),
                )
                return EnqueueOutcome.INTEGRITY_ERROR

            self._insert_terminal_job(
                connection,
                note,
                now=now,
                audio_sha256=content_sha256,
                failure_stage="intake",
                error=error,
            )
            connection.execute(
                """
                INSERT INTO request_ledger (
                    note_id, first_seen_at, last_seen_at,
                    remote_updated_at, audio_sha256, disposition, source_kind
                ) VALUES (?, ?, ?, ?, ?, 'rejected', ?)
                """,
                (
                    note.note_id,
                    timestamp,
                    timestamp,
                    remote_updated,
                    content_sha256,
                    note.kind.value,
                ),
            )
        return EnqueueOutcome.REJECTED

    def get_job(self, note_id: str) -> JobRecord | None:
        row = self._connection.execute(
            "SELECT * FROM jobs WHERE note_id = ?", (note_id,)
        ).fetchone()
        return self._job_from_row(row) if row else None

    def list_jobs(self) -> list[JobRecord]:
        return [
            self._job_from_row(row)
            for row in self._connection.execute(
                "SELECT * FROM jobs ORDER BY queue_seq"
            ).fetchall()
        ]

    def oldest_actionable(self, *, now: datetime) -> JobRecord | None:
        row = self._connection.execute(
            """
            SELECT * FROM jobs
             WHERE state IN ('queued','transcribed','answered')
                OR (state = 'failed' AND retryable = 1)
             ORDER BY queue_seq
             LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        job = self._job_from_row(row)
        if (
            job.state == JobState.FAILED
            and job.next_retry_at is not None
            and _as_utc(job.next_retry_at) > _as_utc(now)
        ):
            return None
        return job

    def oldest_gated_routing_job(self) -> JobRecord | None:
        """Return the next staged source that may be routed behind a safe gate.

        Routing and local transcription do not call the assistant or mutate
        Light Cloud. They are therefore the only processing stages allowed
        while a publication outcome is unknown, or while an interrupted model
        call waits for an explicit ``Assistant reset``.
        """

        row = self._connection.execute(
            """
            WITH gate_jobs AS (
                SELECT output_gate_job_note_id AS note_id,
                       output_gate AS gate, 'output' AS gate_class
                  FROM bridge_state
                 WHERE singleton = 1 AND output_gate IS NOT NULL
                UNION ALL
                SELECT assistant_gate_job_note_id AS note_id,
                       assistant_gate AS gate, 'assistant' AS gate_class
                  FROM bridge_state
                 WHERE singleton = 1
                   AND assistant_gate = 'inference_ambiguous'
            ), gate_position AS (
                SELECT MIN(job.queue_seq) AS queue_seq
                  FROM jobs AS job JOIN gate_jobs ON gate_jobs.note_id = job.note_id
            )
            SELECT candidate.*
              FROM jobs AS candidate, gate_position, bridge_state AS bridge
             WHERE gate_position.queue_seq IS NOT NULL
               AND candidate.queue_seq > gate_position.queue_seq
               AND candidate.state = 'queued'
               AND bridge.singleton = 1
               AND (
                   (bridge.output_gate IS NOT NULL
                       AND candidate.admitted_output_gate = bridge.output_gate
                       AND candidate.admitted_output_gate_job_note_id =
                           bridge.output_gate_job_note_id)
                   OR
                   (bridge.assistant_gate = 'inference_ambiguous'
                       AND candidate.admitted_assistant_gate = bridge.assistant_gate
                       AND candidate.admitted_assistant_gate_job_note_id =
                           bridge.assistant_gate_job_note_id)
               )
             ORDER BY candidate.queue_seq
             LIMIT 1
            """
        ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def oldest_gated_reset_candidate(self) -> JobRecord | None:
        """Return a durably routed reset newer than the active safe gate."""

        row = self._connection.execute(
            """
            WITH gate_jobs AS (
                SELECT output_gate_job_note_id AS note_id,
                       output_gate AS gate, 'output' AS gate_class
                  FROM bridge_state
                 WHERE singleton = 1 AND output_gate IS NOT NULL
                UNION ALL
                SELECT assistant_gate_job_note_id AS note_id,
                       assistant_gate AS gate, 'assistant' AS gate_class
                  FROM bridge_state
                 WHERE singleton = 1
                   AND assistant_gate = 'inference_ambiguous'
            ), gate_position AS (
                SELECT MIN(job.queue_seq) AS queue_seq
                  FROM jobs AS job JOIN gate_jobs ON gate_jobs.note_id = job.note_id
            )
            SELECT candidate.*
              FROM jobs AS candidate, gate_position, bridge_state AS bridge
             WHERE gate_position.queue_seq IS NOT NULL
               AND candidate.queue_seq > gate_position.queue_seq
               AND candidate.state = 'transcribed'
               AND candidate.route_kind = 'reset'
               AND candidate.model_attempts = 0
               AND candidate.publication_attempts = 0
               AND bridge.singleton = 1
               AND (
                   (bridge.output_gate IS NOT NULL
                       AND candidate.admitted_output_gate = bridge.output_gate
                       AND candidate.admitted_output_gate_job_note_id =
                           bridge.output_gate_job_note_id)
                   OR
                   (bridge.assistant_gate = 'inference_ambiguous'
                       AND candidate.admitted_assistant_gate = bridge.assistant_gate
                       AND candidate.admitted_assistant_gate_job_note_id =
                           bridge.assistant_gate_job_note_id)
               )
             ORDER BY candidate.queue_seq
             LIMIT 1
            """
        ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def oldest_publication_interrupt_candidate(self) -> JobRecord | None:
        """Return the first recognized request newer than an output ambiguity."""

        row = self._connection.execute(
            """
            SELECT candidate.*
              FROM bridge_state AS bridge
              JOIN jobs AS gated
                ON gated.note_id = bridge.output_gate_job_note_id
              JOIN jobs AS candidate
                ON candidate.queue_seq > gated.queue_seq
             WHERE bridge.singleton = 1
               AND bridge.output_gate IN ('create_ambiguous','update_ambiguous')
               AND candidate.state IN ('transcribed','answered')
               AND candidate.route_kind IN (
                   'question','continue','new_conversation','reset'
               )
               AND candidate.output_note_id IS NULL
               AND candidate.model_attempts = 0
               AND candidate.publication_attempts = 0
               AND candidate.admitted_output_gate = bridge.output_gate
               AND candidate.admitted_output_gate_job_note_id =
                   bridge.output_gate_job_note_id
             ORDER BY candidate.queue_seq
             LIMIT 1
            """
        ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def oldest_local_reset_candidate(
        self, *, after_queue_seq: int
    ) -> JobRecord | None:
        """Return an exact reset already routed behind a local queue stall."""

        row = self._connection.execute(
            """
            SELECT * FROM jobs
             WHERE queue_seq > ? AND state = 'transcribed'
               AND route_kind = 'reset' AND model_attempts = 0
               AND publication_attempts = 0 AND output_note_id IS NULL
             ORDER BY queue_seq
             LIMIT 1
            """,
            (after_queue_seq,),
        ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def oldest_local_reset_routing_job(
        self, *, after_queue_seq: int
    ) -> JobRecord | None:
        """Return the next unprobed queued input behind a local queue stall."""

        row = self._connection.execute(
            """
            SELECT * FROM jobs
             WHERE queue_seq > ? AND state = 'queued' AND route_kind IS NULL
               AND model_attempts = 0 AND publication_attempts = 0
               AND output_note_id IS NULL
             ORDER BY queue_seq
             LIMIT 1
            """,
            (after_queue_seq,),
        ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def has_unrouted_local_reset_probe(self, *, after_queue_seq: int) -> bool:
        """Return whether the exact local blocker already owns a probe slot.

        A routed, deferred request no longer consumes the one intake probe slot:
        it carries no source audio and cannot call the model or mutate Light while
        the blocker remains.  An unrouted queued request does consume that slot,
        which keeps repeated polls from admitting an unbounded number of source
        payloads before the worker has inspected the previous candidate.
        """

        row = self._connection.execute(
            """
            SELECT 1 FROM jobs
             WHERE queue_seq > ? AND state = 'queued' AND route_kind IS NULL
               AND model_attempts = 0 AND publication_attempts = 0
               AND output_note_id IS NULL
             LIMIT 1
            """,
            (after_queue_seq,),
        ).fetchone()
        return row is not None

    def has_unrouted_inference_reset_probe(self) -> bool:
        """Return whether the active inference ambiguity owns a probe slot."""

        row = self._connection.execute(
            """
            SELECT 1
              FROM bridge_state AS bridge
              JOIN jobs AS gated
                ON gated.note_id = bridge.assistant_gate_job_note_id
              JOIN jobs AS candidate
                ON candidate.queue_seq > gated.queue_seq
             WHERE bridge.singleton = 1
               AND bridge.assistant_gate = 'inference_ambiguous'
               AND candidate.state = 'queued'
               AND candidate.route_kind IS NULL
               AND candidate.model_attempts = 0
               AND candidate.publication_attempts = 0
               AND candidate.output_note_id IS NULL
               AND candidate.admitted_assistant_gate = bridge.assistant_gate
               AND candidate.admitted_assistant_gate_job_note_id =
                   bridge.assistant_gate_job_note_id
             LIMIT 1
            """
        ).fetchone()
        return row is not None

    def has_unrouted_output_reset_probe(self) -> bool:
        """Return whether the active publication ambiguity owns a probe slot."""

        row = self._connection.execute(
            """
            SELECT 1
              FROM bridge_state AS bridge
              JOIN jobs AS gated
                ON gated.note_id = bridge.output_gate_job_note_id
              JOIN jobs AS candidate
                ON candidate.queue_seq > gated.queue_seq
             WHERE bridge.singleton = 1
               AND bridge.output_gate IN ('create_ambiguous','update_ambiguous')
               AND candidate.state = 'queued'
               AND candidate.route_kind IS NULL
               AND candidate.model_attempts = 0
               AND candidate.publication_attempts = 0
               AND candidate.output_note_id IS NULL
               AND candidate.admitted_output_gate = bridge.output_gate
               AND candidate.admitted_output_gate_job_note_id =
                   bridge.output_gate_job_note_id
             LIMIT 1
            """
        ).fetchone()
        return row is not None

    def restore_retry(self, note_id: str, *, now: datetime) -> JobRecord:
        timestamp = _encode_time(now)
        with self._write() as connection:
            row = connection.execute(
                "SELECT state, retryable, resume_state FROM jobs WHERE note_id = ?",
                (note_id,),
            ).fetchone()
            if row is None:
                raise KeyError(note_id)
            if row["state"] != JobState.FAILED or not row["retryable"]:
                raise ValueError("job is not retryable")
            connection.execute(
                """
                UPDATE jobs SET state = resume_state, retryable = 0,
                    next_retry_at = NULL, manual_retry_allowed = 0,
                    updated_at = ?
                WHERE note_id = ?
                """,
                (timestamp, note_id),
            )
        job = self.get_job(note_id)
        assert job is not None
        return job

    def increment_attempt(self, note_id: str, stage: str, *, now: datetime) -> int:
        columns = {
            "transcription": "transcription_attempts",
            "model": "model_attempts",
            "publication": "publication_attempts",
        }
        try:
            column = columns[stage]
        except KeyError as error:
            raise ValueError(f"unknown stage: {stage}") from error
        with self._write() as connection:
            connection.execute(
                f"UPDATE jobs SET {column} = {column} + 1, updated_at = ? "
                "WHERE note_id = ?",
                (_encode_time(now), note_id),
            )
            row = connection.execute(
                f"SELECT {column} AS value FROM jobs WHERE note_id = ?",
                (note_id,),
            ).fetchone()
            if row is None:
                raise KeyError(note_id)
            return int(row["value"])

    def begin_model_attempt(self, note_id: str, *, now: datetime) -> int:
        """Persist an in-flight marker before crossing the model boundary.

        A process death after this commit leaves enough durable evidence to
        pause behind ``inference_ambiguous`` rather than repeat a request whose
        response may have completed remotely.
        """

        with self._write() as connection:
            result = connection.execute(
                """
                UPDATE jobs
                   SET model_attempts = model_attempts + 1,
                       model_attempt_active = 1, updated_at = ?
                 WHERE note_id = ? AND state = 'transcribed'
                   AND model_attempt_active = 0
                """,
                (_encode_time(now), note_id),
            )
            if result.rowcount != 1:
                raise RuntimeError("model attempt state is inconsistent")
            row = connection.execute(
                "SELECT model_attempts FROM jobs WHERE note_id = ?",
                (note_id,),
            ).fetchone()
            if row is None:  # pragma: no cover - guarded by the update
                raise KeyError(note_id)
            return int(row["model_attempts"])

    def mark_failure(
        self,
        note_id: str,
        *,
        stage: str,
        resume_state: JobState,
        error: str,
        retryable: bool,
        now: datetime,
        next_retry_at: datetime | None = None,
        manual_retry_allowed: bool = False,
        preserve_audio: bool = False,
    ) -> None:
        if manual_retry_allowed and (retryable or stage not in {"transcription", "model"}):
            raise ValueError("manual retry is only valid for terminal processing failures")
        if preserve_audio and stage != "transcription":
            raise ValueError("only transcription recovery may preserve staged audio")
        timestamp = _encode_time(now)
        completed = None if retryable else timestamp
        with self._write() as connection:
            connection.execute(
                """
                UPDATE jobs
                   SET state = 'failed', failure_stage = ?, resume_state = ?,
                       retryable = ?, manual_retry_allowed = ?,
                       next_retry_at = ?, last_error = ?,
                       audio_blob = CASE WHEN ? THEN audio_blob ELSE NULL END,
                       model_attempt_active = 0,
                       updated_at = ?, completed_at = ?
                 WHERE note_id = ?
                """,
                (
                    stage,
                    resume_state.value,
                    int(retryable),
                    int(manual_retry_allowed),
                    _encode_time(next_retry_at),
                    error[:500],
                    int(retryable or preserve_audio),
                    timestamp,
                    completed,
                    note_id,
                ),
            )

    def job_recovery_candidate(
        self, source_note_id: str
    ) -> JobRecoveryCandidate | None:
        """Inspect one exact failed job without returning user content."""

        _require_operator_identifier(source_note_id, field="source note id")
        row = self._connection.execute(
            """
            SELECT state, failure_stage, transcription_attempts, model_attempts,
                   manual_retry_allowed, completed_at
              FROM jobs WHERE note_id = ?
            """,
            (source_note_id,),
        ).fetchone()
        if row is None or row["state"] != JobState.FAILED:
            return None
        stage = str(row["failure_stage"] or "unknown")
        attempts = (
            int(row["transcription_attempts"])
            if stage == "transcription"
            else int(row["model_attempts"])
            if stage == "model"
            else 0
        )
        return JobRecoveryCandidate(
            source_note_id=source_note_id,
            failure_stage=stage,
            attempts=attempts,
            retry_allowed=bool(row["manual_retry_allowed"]),
            failed_at=_decode_time(row["completed_at"]),
        )

    def list_job_recovery_candidates(
        self, *, limit: int = 100
    ) -> tuple[list[JobRecoveryCandidate], int]:
        """List a bounded set of safe manual-retry candidates, oldest first."""

        if not 1 <= limit <= 100:
            raise ValueError("job recovery list limit is outside the allowed range")
        total = int(
            self._connection.execute(
                """
                SELECT COUNT(*) FROM jobs
                 WHERE state = 'failed' AND manual_retry_allowed = 1
                """
            ).fetchone()[0]
        )
        rows = self._connection.execute(
            """
            SELECT note_id, failure_stage, transcription_attempts,
                   model_attempts, completed_at
              FROM jobs
             WHERE state = 'failed' AND manual_retry_allowed = 1
             ORDER BY queue_seq
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        candidates = [
            JobRecoveryCandidate(
                source_note_id=str(row["note_id"]),
                failure_stage=str(row["failure_stage"]),
                attempts=(
                    int(row["transcription_attempts"])
                    if row["failure_stage"] == "transcription"
                    else int(row["model_attempts"])
                ),
                retry_allowed=True,
                failed_at=_decode_time(row["completed_at"]),
            )
            for row in rows
        ]
        return candidates, total

    def retry_terminal_job(
        self,
        *,
        source_note_id: str,
        expected_stage: Literal["transcription", "model"],
        expected_attempts: int,
        now: datetime,
    ) -> JobRecoveryResult:
        """Retry one exact exhausted safe stage while preserving its inputs.

        Publication failures and ambiguous model attempts are intentionally
        ineligible. The guards also prevent repeating a stored answer or any
        job that crossed a Light mutation boundary.
        """

        _require_operator_identifier(source_note_id, field="source note id")
        if expected_stage not in {"transcription", "model"}:
            raise ValueError("unsupported recovery stage")
        if expected_attempts < 1:
            raise ValueError("expected attempts must be positive")
        timestamp = _encode_time(now)
        assert timestamp is not None
        with self._write() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE note_id = ?",
                (source_note_id,),
            ).fetchone()
            if row is None:
                raise KeyError(source_note_id)
            attempts = (
                int(row["transcription_attempts"])
                if expected_stage == "transcription"
                else int(row["model_attempts"])
            )
            if (
                row["state"] != JobState.FAILED
                or row["failure_stage"] != expected_stage
                or attempts != expected_attempts
                or bool(row["retryable"])
                or not bool(row["manual_retry_allowed"])
                or bool(row["model_attempt_active"])
                or row["output_note_id"] is not None
                or int(row["publication_attempts"]) != 0
                or row["publication_content"] is not None
                or row["publication_sha256"] is not None
                or row["answer"] is not None
                or row["response_id"] is not None
            ):
                raise ValueError("job recovery identifiers or safety state do not match")

            if expected_stage == "transcription":
                resumed_state = JobState.QUEUED
                if (
                    row["resume_state"] != resumed_state
                    or row["audio_blob"] is None
                    or row["transcript"] is not None
                ):
                    raise ValueError("transcription recovery state is incomplete")
                attempt_column = "transcription_attempts"
            else:
                resumed_state = JobState.TRANSCRIBED
                if (
                    row["resume_state"] != resumed_state
                    or not row["transcript"]
                    or row["answer_sections"] != "[]"
                ):
                    raise ValueError("model recovery state is incomplete")
                attempt_column = "model_attempts"

            restored = connection.execute(
                f"""
                UPDATE jobs SET state = ?, failure_stage = NULL,
                    resume_state = NULL, retryable = 0,
                    manual_retry_allowed = 0, next_retry_at = NULL,
                    last_error = NULL, model_attempt_active = 0,
                    {attempt_column} = 0, updated_at = ?, completed_at = NULL
                 WHERE note_id = ? AND state = 'failed'
                   AND failure_stage = ? AND manual_retry_allowed = 1
                """,
                (
                    resumed_state.value,
                    timestamp,
                    source_note_id,
                    expected_stage,
                ),
            )
            if restored.rowcount != 1:
                raise RuntimeError("job recovery state changed")
            cursor = connection.execute(
                """
                INSERT INTO job_recovery_audit (
                    retried_at, source_note_id, failure_stage,
                    prior_attempts, resumed_state
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    source_note_id,
                    expected_stage,
                    expected_attempts,
                    resumed_state.value,
                ),
            )
            audit_id = int(cursor.lastrowid)
        return JobRecoveryResult(
            audit_id=audit_id,
            source_note_id=source_note_id,
            failure_stage=expected_stage,
            prior_attempts=expected_attempts,
            resumed_state=resumed_state,
            retried_at=_decode_time(timestamp),
        )

    def mark_ignored(self, note_id: str, *, reason: str, now: datetime) -> None:
        timestamp = _encode_time(now)
        with self._write() as connection:
            connection.execute(
                """
                UPDATE jobs SET state = 'ignored', route_kind = 'ignored',
                    transcript = NULL, answer = NULL, answer_sections = '[]',
                    audio_blob = NULL, ignored_reason = ?, failure_stage = NULL,
                    resume_state = NULL, retryable = 0, next_retry_at = NULL,
                    manual_retry_allowed = 0, model_attempt_active = 0,
                    last_error = NULL, updated_at = ?, completed_at = ?
                WHERE note_id = ?
                """,
                (reason, timestamp, timestamp, note_id),
            )

    def ignored_audio_recovery_candidate(
        self, source_note_id: str
    ) -> IgnoredAudioRecoveryCandidate | None:
        """Return one exact ignored-audio candidate without retained content."""

        _require_operator_identifier(source_note_id, field="source note id")
        row = self._connection.execute(
            """
            SELECT job.note_id, job.queue_seq, job.state, job.source_kind,
                   job.audio_sha256, job.audio_blob, job.route_kind,
                   job.transcript, job.answer, job.answer_sections,
                   job.next_section_index, job.delivered_sections,
                   job.publication_content, job.publication_sha256,
                   job.publication_title_sha256,
                   job.publication_output_mode, job.conversation_id,
                   job.source_job_note_id, job.source_section_index,
                   job.response_id, job.output_note_id, job.failure_stage,
                   job.resume_state, job.retryable,
                   job.manual_retry_allowed, job.model_attempt_active,
                   job.next_retry_at, job.last_error, job.ignored_reason,
                   job.transcription_attempts, job.model_attempts,
                   job.publication_attempts, job.completed_at,
                   ledger.audio_sha256 AS ledger_sha256,
                   ledger.source_kind AS ledger_source_kind,
                   ledger.disposition AS ledger_disposition,
                   bridge.output_gate_job_note_id,
                   bridge.assistant_gate_job_note_id
              FROM jobs AS job
              LEFT JOIN request_ledger AS ledger
                ON ledger.note_id = job.note_id
              JOIN bridge_state AS bridge ON bridge.singleton = 1
             WHERE job.note_id = ?
            """,
            (source_note_id,),
        ).fetchone()
        if row is None or not self._is_exact_ignored_audio_candidate(row):
            return None
        ignored_at = _decode_time(row["completed_at"])
        if ignored_at is None:
            return None
        return IgnoredAudioRecoveryCandidate(
            source_note_id=str(row["note_id"]),
            retained_sha256=str(row["audio_sha256"]),
            queue_seq=int(row["queue_seq"]),
            transcription_attempts=int(row["transcription_attempts"]),
            model_attempts=int(row["model_attempts"]),
            publication_attempts=int(row["publication_attempts"]),
            ignored_at=ignored_at,
        )

    def ignored_audio_recovery_already_restaged(
        self,
        source_note_id: str,
        *,
        expected_sha256: str,
        expected_transcription_attempts: int,
        expected_model_attempts: int,
        expected_publication_attempts: int,
    ) -> bool:
        """Recognize only the exact no-op state left by a completed restage."""

        _require_operator_identifier(source_note_id, field="source note id")
        _require_sha256(expected_sha256, field="expected source hash")
        row = self._connection.execute(
            """
            SELECT job.*, ledger.audio_sha256 AS ledger_sha256,
                   ledger.source_kind AS ledger_source_kind,
                   ledger.disposition AS ledger_disposition,
                   bridge.output_epoch AS current_output_epoch,
                   bridge.active_output_note_id AS current_output_note_id,
                   bridge.output_gate_job_note_id,
                   bridge.assistant_gate_job_note_id,
                   audit.id AS recovery_audit_id,
                   audit.retried_at AS recovery_retried_at,
                   audit.failure_stage AS recovery_failure_stage,
                   audit.prior_attempts AS recovery_prior_attempts,
                   audit.resumed_state AS recovery_resumed_state
              FROM jobs AS job
              LEFT JOIN request_ledger AS ledger
                ON ledger.note_id = job.note_id
              JOIN bridge_state AS bridge ON bridge.singleton = 1
              LEFT JOIN job_recovery_audit AS audit
                ON audit.id = (
                    SELECT MAX(candidate.id)
                      FROM job_recovery_audit AS candidate
                     WHERE candidate.source_note_id = job.note_id
                )
             WHERE job.note_id = ?
            """,
            (source_note_id,),
        ).fetchone()
        return bool(
            row is not None
            and self._is_exact_restaged_audio(
                row,
                expected_sha256=expected_sha256,
                expected_transcription_attempts=expected_transcription_attempts,
                expected_model_attempts=expected_model_attempts,
                expected_publication_attempts=expected_publication_attempts,
            )
        )

    def restage_ignored_audio(
        self,
        note: RemoteNote,
        audio: bytes,
        *,
        expected_sha256: str,
        expected_state: str,
        expected_source_kind: str,
        expected_ignored_reason: str,
        expected_transcription_attempts: int,
        expected_model_attempts: int,
        expected_publication_attempts: int,
        now: datetime,
    ) -> IgnoredAudioRecoveryResult:
        """Atomically restage only one exact prefix-missing audio request.

        The remote payload is supplied by a read-only caller. Every durable
        identity and processing boundary is rechecked inside the transaction;
        an exact already-restaged row is a safe no-op for command recovery.
        """

        _require_operator_identifier(note.note_id, field="source note id")
        _require_sha256(expected_sha256, field="expected source hash")
        if note.kind is not NoteKind.AUDIO:
            raise ValueError("remote source kind does not match audio recovery")
        observed_sha256 = sha256_bytes(audio)
        if not hmac.compare_digest(observed_sha256, expected_sha256):
            raise ValueError("remote source hash does not match recovery identity")
        if (
            expected_state != JobState.IGNORED.value
            or expected_source_kind != NoteKind.AUDIO.value
            or expected_ignored_reason != "prefix_missing"
            or expected_transcription_attempts < 1
            or expected_model_attempts != 0
            or expected_publication_attempts != 0
        ):
            raise ValueError("ignored audio recovery expectations are invalid")

        timestamp = _encode_time(now)
        assert timestamp is not None
        remote_updated = _encode_time(note.updated_at)
        with self._write() as connection:
            row = connection.execute(
                """
                SELECT job.*, ledger.audio_sha256 AS ledger_sha256,
                       ledger.source_kind AS ledger_source_kind,
                       ledger.disposition AS ledger_disposition,
                       bridge.output_epoch AS current_output_epoch,
                       bridge.active_output_note_id AS current_output_note_id,
                       bridge.output_gate_job_note_id,
                       bridge.assistant_gate_job_note_id,
                       audit.id AS recovery_audit_id,
                       audit.retried_at AS recovery_retried_at,
                       audit.failure_stage AS recovery_failure_stage,
                       audit.prior_attempts AS recovery_prior_attempts,
                       audit.resumed_state AS recovery_resumed_state
                  FROM jobs AS job
                  LEFT JOIN request_ledger AS ledger
                    ON ledger.note_id = job.note_id
                  JOIN bridge_state AS bridge ON bridge.singleton = 1
                  LEFT JOIN job_recovery_audit AS audit
                    ON audit.id = (
                        SELECT MAX(candidate.id)
                          FROM job_recovery_audit AS candidate
                         WHERE candidate.source_note_id = job.note_id
                    )
                 WHERE job.note_id = ?
                """,
                (note.note_id,),
            ).fetchone()
            if row is None:
                raise KeyError(note.note_id)
            already_restaged = row["state"] == JobState.QUEUED.value
            audit_id: int
            restaged_at: datetime
            if already_restaged:
                if not self._is_exact_restaged_audio(
                    row,
                    expected_sha256=expected_sha256,
                    expected_transcription_attempts=expected_transcription_attempts,
                    expected_model_attempts=expected_model_attempts,
                    expected_publication_attempts=expected_publication_attempts,
                ):
                    raise RuntimeError("restaged audio recovery state is inconsistent")
                audit_id = int(row["recovery_audit_id"])
                decoded_retried_at = _decode_time(row["recovery_retried_at"])
                if decoded_retried_at is None:
                    raise RuntimeError("restaged audio recovery audit is incomplete")
                restaged_at = decoded_retried_at
            else:
                if not self._ignored_audio_identity_matches(
                    row,
                    expected_sha256=expected_sha256,
                    expected_transcription_attempts=(
                        expected_transcription_attempts
                    ),
                    expected_model_attempts=expected_model_attempts,
                    expected_publication_attempts=(
                        expected_publication_attempts
                    ),
                ):
                    raise ValueError(
                        "ignored audio recovery identity does not match"
                    )
                if not self._is_exact_ignored_audio_candidate(row):
                    raise RuntimeError("ignored audio recovery state is inconsistent")
                updated = connection.execute(
                    """
                    UPDATE jobs
                       SET state = 'queued', remote_updated_at = ?, mime_type = ?,
                           audio_blob = ?, route_kind = NULL,
                           accepted_output_epoch = ?,
                           accepted_output_note_id = ?, ignored_reason = ?,
                           failure_stage = NULL, resume_state = NULL,
                           retryable = 0, manual_retry_allowed = 0,
                           model_attempt_active = 0, next_retry_at = NULL,
                           last_error = NULL, transcription_attempts = 0,
                           updated_at = ?, completed_at = NULL
                     WHERE note_id = ? AND state = 'ignored'
                       AND source_kind = 'audio'
                       AND ignored_reason = 'prefix_missing'
                       AND transcription_attempts = ?
                       AND model_attempts = 0
                       AND publication_attempts = 0
                    """,
                    (
                        remote_updated,
                        note.mime_type,
                        sqlite3.Binary(audio),
                        int(row["current_output_epoch"]),
                        row["current_output_note_id"],
                        _RESTAGED_IGNORED_AUDIO_REASON,
                        timestamp,
                        note.note_id,
                        expected_transcription_attempts,
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("ignored audio recovery state changed")
                ledger = connection.execute(
                    """
                    UPDATE request_ledger
                       SET last_seen_at = ?, remote_updated_at = ?
                     WHERE note_id = ? AND source_kind = 'audio'
                       AND disposition = 'accepted' AND audio_sha256 = ?
                    """,
                    (timestamp, remote_updated, note.note_id, expected_sha256),
                )
                if ledger.rowcount != 1:
                    raise RuntimeError("ignored audio recovery ledger changed")
                audit = connection.execute(
                    """
                    INSERT INTO job_recovery_audit (
                        retried_at, source_note_id, failure_stage,
                        prior_attempts, resumed_state
                    ) VALUES (?, ?, 'transcription', ?, 'queued')
                    """,
                    (
                        timestamp,
                        note.note_id,
                        expected_transcription_attempts,
                    ),
                )
                audit_id = int(audit.lastrowid)
                restaged_at = _decode_time(timestamp)
                assert restaged_at is not None

            return IgnoredAudioRecoveryResult(
                source_note_id=note.note_id,
                retained_sha256=expected_sha256,
                queue_seq=int(row["queue_seq"]),
                audit_id=audit_id,
                prior_transcription_attempts=expected_transcription_attempts,
                accepted_output_epoch=int(row["current_output_epoch"]),
                accepted_output_note_id=row["current_output_note_id"],
                already_restaged=already_restaged,
                restaged_at=restaged_at,
            )

    @staticmethod
    def _ignored_audio_identity_matches(
        row: sqlite3.Row,
        *,
        expected_sha256: str,
        expected_transcription_attempts: int,
        expected_model_attempts: int,
        expected_publication_attempts: int,
    ) -> bool:
        return bool(
            row["source_kind"] == NoteKind.AUDIO.value
            and row["ledger_source_kind"] == NoteKind.AUDIO.value
            and row["ledger_disposition"] == "accepted"
            and isinstance(row["audio_sha256"], str)
            and isinstance(row["ledger_sha256"], str)
            and hmac.compare_digest(str(row["audio_sha256"]), expected_sha256)
            and hmac.compare_digest(str(row["ledger_sha256"]), expected_sha256)
            and int(row["transcription_attempts"])
            == expected_transcription_attempts
            and int(row["model_attempts"]) == expected_model_attempts
            and int(row["publication_attempts"])
            == expected_publication_attempts
            and row["output_gate_job_note_id"] != row["note_id"]
            and row["assistant_gate_job_note_id"] != row["note_id"]
        )

    @classmethod
    def _is_exact_ignored_audio_candidate(cls, row: sqlite3.Row) -> bool:
        retained = row["audio_sha256"]
        return bool(
            isinstance(retained, str)
            and cls._ignored_audio_identity_matches(
                row,
                expected_sha256=retained,
                expected_transcription_attempts=int(
                    row["transcription_attempts"]
                ),
                expected_model_attempts=0,
                expected_publication_attempts=0,
            )
            and int(row["transcription_attempts"]) > 0
            and row["state"] == JobState.IGNORED.value
            and row["audio_blob"] is None
            and row["route_kind"] == RouteKind.IGNORED.value
            and row["ignored_reason"] == "prefix_missing"
            and row["transcript"] is None
            and row["answer"] is None
            and row["answer_sections"] == "[]"
            and int(row["next_section_index"]) == 0
            and int(row["delivered_sections"]) == 0
            and row["publication_content"] is None
            and row["publication_sha256"] is None
            and row["publication_title_sha256"] is None
            and row["publication_output_mode"] is None
            and row["conversation_id"] is None
            and row["source_job_note_id"] is None
            and row["source_section_index"] is None
            and row["response_id"] is None
            and row["output_note_id"] is None
            and row["failure_stage"] is None
            and row["resume_state"] is None
            and not bool(row["retryable"])
            and not bool(row["manual_retry_allowed"])
            and not bool(row["model_attempt_active"])
            and row["next_retry_at"] is None
            and row["last_error"] is None
            and row["completed_at"] is not None
        )

    @classmethod
    def _is_exact_restaged_audio(
        cls,
        row: sqlite3.Row,
        *,
        expected_sha256: str,
        expected_transcription_attempts: int,
        expected_model_attempts: int,
        expected_publication_attempts: int,
    ) -> bool:
        staged = row["audio_blob"]
        return bool(
            cls._ignored_audio_identity_matches(
                row,
                expected_sha256=expected_sha256,
                expected_transcription_attempts=0,
                expected_model_attempts=expected_model_attempts,
                expected_publication_attempts=expected_publication_attempts,
            )
            and row["state"] == JobState.QUEUED.value
            and staged is not None
            and hmac.compare_digest(
                sha256_bytes(bytes(staged)), expected_sha256
            )
            and row["route_kind"] is None
            and row["ignored_reason"] == _RESTAGED_IGNORED_AUDIO_REASON
            and row["transcript"] is None
            and row["answer"] is None
            and row["answer_sections"] == "[]"
            and int(row["next_section_index"]) == 0
            and int(row["delivered_sections"]) == 0
            and row["publication_content"] is None
            and row["publication_sha256"] is None
            and row["publication_title_sha256"] is None
            and row["publication_output_mode"] is None
            and row["conversation_id"] is None
            and row["source_job_note_id"] is None
            and row["source_section_index"] is None
            and row["response_id"] is None
            and row["output_note_id"] is None
            and row["failure_stage"] is None
            and row["resume_state"] is None
            and not bool(row["retryable"])
            and not bool(row["manual_retry_allowed"])
            and not bool(row["model_attempt_active"])
            and row["next_retry_at"] is None
            and row["last_error"] is None
            and row["completed_at"] is None
            and row["recovery_audit_id"] is not None
            and row["recovery_failure_stage"] == "transcription"
            and int(row["recovery_prior_attempts"])
            == expected_transcription_attempts
            and row["recovery_resumed_state"] == JobState.QUEUED.value
            and int(row["accepted_output_epoch"])
            == int(row["current_output_epoch"])
            and row["accepted_output_note_id"]
            == row["current_output_note_id"]
        )

    def mark_question_transcribed(
        self, note_id: str, *, question: str, now: datetime
    ) -> int:
        with self._write() as connection:
            conversation_id = self._ensure_active_conversation(connection, now)
            connection.execute(
                """
                UPDATE jobs SET state = 'transcribed', route_kind = 'question',
                    transcript = ?, audio_blob = NULL,
                    conversation_id = ?, failure_stage = NULL,
                    resume_state = NULL, retryable = 0, next_retry_at = NULL,
                    manual_retry_allowed = 0, model_attempt_active = 0,
                    last_error = NULL, ignored_reason = NULL,
                    updated_at = ?, completed_at = NULL
                WHERE note_id = ?
                """,
                (question, conversation_id, _encode_time(now), note_id),
            )
        return conversation_id

    def mark_reset_routed(self, note_id: str, *, now: datetime) -> None:
        """Persist reset recognition without changing any recovery gate."""

        with self._write() as connection:
            result = connection.execute(
                """
                UPDATE jobs SET state = 'transcribed', route_kind = 'reset',
                    transcript = 'Reset', audio_blob = NULL,
                    conversation_id = NULL, failure_stage = NULL,
                    resume_state = NULL, retryable = 0, next_retry_at = NULL,
                    manual_retry_allowed = 0, model_attempt_active = 0,
                    last_error = NULL, ignored_reason = NULL,
                    updated_at = ?, completed_at = NULL
                 WHERE note_id = ? AND state = 'queued'
                   AND model_attempts = 0 AND publication_attempts = 0
                """,
                (_encode_time(now), note_id),
            )
            if result.rowcount != 1:
                raise RuntimeError("reset routing state changed")

    def defer_local_reset_probe_route(
        self,
        note_id: str,
        *,
        route_kind: RouteKind,
        transcript: str,
        now: datetime,
    ) -> None:
        """Persist local routing while leaving conversational effects deferred.

        The job remains queued. This lets reset discovery continue after a
        restart without retranscribing earlier inputs, while ordinary queue
        processing later applies the route in its original order if no reset
        supersedes it.
        """

        if route_kind not in {
            RouteKind.QUESTION,
            RouteKind.CONTINUE,
            RouteKind.NEW_CONVERSATION,
        }:
            raise ValueError("only ordinary assistant routes may be deferred")
        if not transcript:
            raise ValueError("a deferred route needs a transcript")
        with self._write() as connection:
            result = connection.execute(
                """
                UPDATE jobs SET route_kind = ?, transcript = ?,
                    audio_blob = NULL, failure_stage = NULL,
                    resume_state = NULL, retryable = 0, next_retry_at = NULL,
                    manual_retry_allowed = 0, model_attempt_active = 0,
                    last_error = NULL, ignored_reason = NULL,
                    updated_at = ?, completed_at = NULL
                 WHERE note_id = ? AND state = 'queued'
                   AND route_kind IS NULL AND model_attempts = 0
                   AND publication_attempts = 0 AND output_note_id IS NULL
                """,
                (
                    route_kind.value,
                    transcript,
                    _encode_time(now),
                    note_id,
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError("local reset probe state changed")

    def mark_new_conversation(
        self, note_id: str, *, now: datetime, answer: str
    ) -> int:
        timestamp = _encode_time(now)
        with self._write() as connection:
            current = connection.execute(
                "SELECT active_conversation_id FROM bridge_state WHERE singleton = 1"
            ).fetchone()["active_conversation_id"]
            if current is not None:
                connection.execute(
                    "UPDATE conversations SET closed_at = COALESCE(closed_at, ?) "
                    "WHERE id = ?",
                    (timestamp, current),
                )
            cursor = connection.execute(
                "INSERT INTO conversations(started_at) VALUES (?)", (timestamp,)
            )
            conversation_id = int(cursor.lastrowid)
            connection.execute(
                "UPDATE bridge_state SET active_conversation_id = ? "
                "WHERE singleton = 1",
                (conversation_id,),
            )
            connection.execute(
                """
                UPDATE jobs SET state = 'answered',
                    route_kind = 'new_conversation', transcript = 'New conversation',
                    answer = ?, answer_sections = ?, next_section_index = 0,
                    delivered_sections = 0,
                    audio_blob = NULL, conversation_id = ?,
                    manual_retry_allowed = 0, model_attempt_active = 0,
                    ignored_reason = NULL, updated_at = ?
                WHERE note_id = ?
                """,
                (answer, json.dumps([answer]), conversation_id, timestamp, note_id),
            )
        return conversation_id

    def mark_continue(
        self,
        note_id: str,
        *,
        now: datetime,
        exhausted_answer: str,
    ) -> str:
        """Store the next section without consuming it before publication."""

        timestamp = _encode_time(now)
        with self._write() as connection:
            conversation_id = self._ensure_active_conversation(connection, now)
            candidate = connection.execute(
                """
                SELECT note_id, answer_sections, next_section_index
                  FROM jobs
                 WHERE conversation_id = ? AND route_kind = 'question'
                   AND delivered_sections > 0
                   AND answer IS NOT NULL AND queue_seq < (
                       SELECT queue_seq FROM jobs WHERE note_id = ?
                   )
                 ORDER BY queue_seq DESC
                 LIMIT 1
                """,
                (conversation_id, note_id),
            ).fetchone()
            answer = exhausted_answer
            source_note_id: str | None = None
            source_section_index: int | None = None
            if candidate is not None:
                sections = json.loads(candidate["answer_sections"])
                index = int(candidate["next_section_index"])
                if index < len(sections):
                    answer = str(sections[index])
                    source_note_id = candidate["note_id"]
                    source_section_index = index
            connection.execute(
                """
                UPDATE jobs SET state = 'answered', route_kind = 'continue',
                    transcript = 'Continue', answer = ?, answer_sections = ?,
                    next_section_index = 0, delivered_sections = 0,
                    audio_blob = NULL, conversation_id = ?,
                    source_job_note_id = ?, source_section_index = ?,
                    manual_retry_allowed = 0, model_attempt_active = 0,
                    ignored_reason = NULL, updated_at = ?
                WHERE note_id = ?
                """,
                (
                    answer,
                    json.dumps([answer]),
                    conversation_id,
                    source_note_id,
                    source_section_index,
                    timestamp,
                    note_id,
                ),
            )
        return answer

    def conversation_turns(
        self, conversation_id: int, *, before_queue_seq: int
    ) -> list[tuple[str, str]]:
        rows = self._connection.execute(
            """
            SELECT transcript, answer_sections, delivered_sections FROM jobs
             WHERE conversation_id = ? AND route_kind = 'question'
               AND transcript IS NOT NULL AND delivered_sections > 0
               AND queue_seq < ?
             ORDER BY queue_seq
            """,
            (conversation_id, before_queue_seq),
        ).fetchall()
        turns: list[tuple[str, str]] = []
        for row in rows:
            sections = [str(value) for value in json.loads(row["answer_sections"])]
            delivered = min(int(row["delivered_sections"]), len(sections))
            if delivered:
                turns.append((row["transcript"], "\n\n".join(sections[:delivered])))
        return turns

    def mark_answered(
        self,
        note_id: str,
        *,
        answer: str,
        sections: Sequence[str],
        response_id: str | None,
        now: datetime,
    ) -> None:
        if not sections or sections[0] == "":
            raise ValueError("an answer needs at least one non-empty section")
        with self._write() as connection:
            connection.execute(
                """
                UPDATE jobs SET state = 'answered', answer = ?,
                    answer_sections = ?, next_section_index = 0,
                    delivered_sections = 0,
                    response_id = ?, failure_stage = NULL, resume_state = NULL,
                    retryable = 0, next_retry_at = NULL, last_error = NULL,
                    manual_retry_allowed = 0, model_attempt_active = 0,
                    updated_at = ?
                WHERE note_id = ?
                """,
                (
                    answer,
                    json.dumps(list(sections)),
                    response_id,
                    _encode_time(now),
                    note_id,
                ),
            )

    def output_snapshot(self) -> OutputSnapshot:
        row = self._connection.execute(
            """
            SELECT b.output_epoch, b.active_output_note_id, o.content_sha256
              FROM bridge_state b
              LEFT JOIN output_surfaces o
                ON o.note_id = b.active_output_note_id
             WHERE b.singleton = 1
            """
        ).fetchone()
        return OutputSnapshot(
            epoch=int(row["output_epoch"]),
            note_id=row["active_output_note_id"],
            content_sha256=row["content_sha256"],
        )

    def output_missing_pending(self) -> bool:
        """Return whether the active rolling output has one unconfirmed miss."""

        row = self._connection.execute(
            """
            SELECT o.missing_observations
              FROM bridge_state AS b
              JOIN output_surfaces AS o
                ON o.note_id = b.active_output_note_id
             WHERE b.singleton = 1 AND o.retired_at IS NULL
            """
        ).fetchone()
        return bool(row is not None and int(row["missing_observations"]) > 0)

    def output_gate(self) -> str | None:
        """Return the content-free durable publication pause, if any."""

        row = self._connection.execute(
            "SELECT output_gate FROM bridge_state WHERE singleton = 1"
        ).fetchone()
        return str(row["output_gate"]) if row["output_gate"] is not None else None

    def is_possible_ambiguous_create_title(self, title: str) -> bool:
        """Conservatively identify the exact title of an uncertain create.

        This is an exclusion check only. It does not claim ownership of the
        remote note and must never authorize an update or deletion.
        """

        row = self._connection.execute(
            """
            SELECT job.publication_title_sha256
              FROM bridge_state AS bridge
              JOIN jobs AS job ON job.note_id = bridge.output_gate_job_note_id
             WHERE bridge.singleton = 1
               AND bridge.output_gate = 'create_ambiguous'
               AND job.state = 'failed'
               AND job.failure_stage = 'publication'
               AND job.resume_state = 'answered'
               AND job.publication_attempts > 0
               AND job.publication_title_sha256 IS NOT NULL
            """
        ).fetchone()
        return bool(
            row is not None
            and hmac.compare_digest(
                str(row["publication_title_sha256"]), sha256_text(title)
            )
        )

    def update_readback_exhausted(self, *, maximum: int) -> bool:
        """Return whether automatic reads for an uncertain update are spent."""

        if maximum < 1:
            raise ValueError("update read-back limit must be positive")
        row = self._connection.execute(
            "SELECT output_gate, output_gate_readback_attempts "
            "FROM bridge_state WHERE singleton = 1"
        ).fetchone()
        return bool(
            row["output_gate"] == "update_ambiguous"
            and int(row["output_gate_readback_attempts"]) >= maximum
        )

    def assistant_gate(self) -> str | None:
        """Return the durable assistant pause, if one requires review."""

        row = self._connection.execute(
            "SELECT assistant_gate FROM bridge_state WHERE singleton = 1"
        ).fetchone()
        return (
            str(row["assistant_gate"])
            if row["assistant_gate"] is not None
            else None
        )

    def assistant_gate_status(self) -> dict[str, object] | None:
        """Return content-free identifiers for administrative recovery."""

        row = self._connection.execute(
            """
            SELECT assistant_gate, assistant_gate_job_note_id, assistant_gate_at
              FROM bridge_state WHERE singleton = 1
            """
        ).fetchone()
        if row["assistant_gate"] is None:
            return None
        if row["assistant_gate_job_note_id"] is None or row["assistant_gate_at"] is None:
            raise RuntimeError("assistant gate state is inconsistent")
        return {
            "gate": str(row["assistant_gate"]),
            "source_note_id": str(row["assistant_gate_job_note_id"]),
            "gated_at": _decode_time(str(row["assistant_gate_at"])),
        }

    def gate_interrupted_model(self, *, now: datetime) -> bool:
        """Fail closed when a process ended during a remote model attempt."""

        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                "SELECT assistant_gate FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            if state["assistant_gate"] is not None:
                return False
            row = connection.execute(
                """
                SELECT note_id FROM jobs
                 WHERE state = 'transcribed' AND model_attempt_active = 1
                 ORDER BY queue_seq
                 LIMIT 1
                """
            ).fetchone()
            if row is None:
                return False
            note_id = str(row["note_id"])
            result = connection.execute(
                """
                UPDATE jobs SET state = 'failed', failure_stage = 'model',
                    resume_state = 'transcribed', retryable = 0,
                    manual_retry_allowed = 0, next_retry_at = NULL,
                    model_attempt_active = 0,
                    last_error = 'model attempt was interrupted with an unknown outcome',
                    updated_at = ?, completed_at = NULL
                 WHERE note_id = ? AND state = 'transcribed'
                   AND model_attempt_active = 1
                """,
                (timestamp, note_id),
            )
            if result.rowcount != 1:
                raise RuntimeError("interrupted model attempt changed during recovery")
            connection.execute(
                """
                UPDATE bridge_state
                   SET assistant_gate = 'inference_ambiguous',
                       assistant_gate_job_note_id = ?, assistant_gate_at = ?
                 WHERE singleton = 1 AND assistant_gate IS NULL
                """,
                (note_id, timestamp),
            )
            return True

    def pause_assistant(
        self,
        note_id: str,
        *,
        gate: str,
        error: str,
        now: datetime,
    ) -> None:
        """Pause every model turn while preserving the oldest turn for resume."""

        allowed = {
            "authorization_required",
            "plan_limit",
            "credential_ambiguous",
            "inference_ambiguous",
            "account_not_eligible",
            "configuration_error",
            "daily_model_limit",
        }
        if gate not in allowed:
            raise ValueError("unsupported assistant gate")
        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                "SELECT assistant_gate, assistant_gate_job_note_id "
                "FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            if state["assistant_gate"] is not None and (
                state["assistant_gate"] != gate
                or state["assistant_gate_job_note_id"] != note_id
            ):
                raise RuntimeError("a different assistant gate is already active")
            result = connection.execute(
                """
                UPDATE jobs SET state = 'failed', failure_stage = 'model',
                    resume_state = 'transcribed', retryable = 0,
                    next_retry_at = NULL, last_error = ?, audio_blob = NULL,
                    manual_retry_allowed = 0, model_attempt_active = 0,
                    updated_at = ?, completed_at = NULL
                 WHERE note_id = ? AND state = 'transcribed'
                """,
                (error[:500], timestamp, note_id),
            )
            if result.rowcount != 1:
                raise KeyError(note_id)
            connection.execute(
                """
                UPDATE bridge_state
                   SET assistant_gate = ?, assistant_gate_job_note_id = ?,
                       assistant_gate_at = ?
                 WHERE singleton = 1
                """,
                (gate, note_id, timestamp),
            )

    def clear_assistant_gate(
        self,
        *,
        expected_gate: str,
        expected_note_id: str,
        now: datetime,
    ) -> bool:
        """Resume the preserved turn after an explicit administrative review."""

        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                """
                SELECT assistant_gate, assistant_gate_job_note_id
                  FROM bridge_state WHERE singleton = 1
                """
            ).fetchone()
            if state["assistant_gate"] is None:
                return False
            if (
                state["assistant_gate"] != expected_gate
                or state["assistant_gate_job_note_id"] != expected_note_id
            ):
                raise ValueError("assistant recovery identifiers do not match")
            restored = connection.execute(
                """
                UPDATE jobs SET state = 'transcribed', failure_stage = NULL,
                    resume_state = NULL, retryable = 0, next_retry_at = NULL,
                    last_error = NULL, model_attempts = 0,
                    manual_retry_allowed = 0, model_attempt_active = 0,
                    updated_at = ?,
                    completed_at = NULL
                 WHERE note_id = ? AND state = 'failed'
                   AND failure_stage = 'model'
                   AND resume_state = 'transcribed'
                """,
                (timestamp, expected_note_id),
            )
            if restored.rowcount != 1:
                raise RuntimeError("assistant gate job state is inconsistent")
            cleared = connection.execute(
                """
                UPDATE bridge_state SET assistant_gate = NULL,
                    assistant_gate_job_note_id = NULL,
                    assistant_gate_at = NULL
                 WHERE singleton = 1 AND assistant_gate = ?
                   AND assistant_gate_job_note_id = ?
                """,
                (expected_gate, expected_note_id),
            )
            if cleared.rowcount != 1:
                raise RuntimeError("assistant gate changed during recovery")
        return True

    def reserve_model_call(self, *, now: datetime, daily_limit: int) -> bool:
        """Reserve one content-free UTC-day model-call allowance durably."""

        if daily_limit < 1:
            raise ValueError("daily model-call limit must be positive")
        current = _as_utc(now)
        day = current.date().isoformat()
        timestamp = _encode_time(current)
        with self._write() as connection:
            existing = connection.execute(
                "SELECT attempts FROM model_usage_days WHERE utc_day = ?",
                (day,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO model_usage_days(utc_day, attempts, updated_at) "
                    "VALUES (?, 1, ?)",
                    (day, timestamp),
                )
                return True
            if int(existing["attempts"]) >= daily_limit:
                return False
            connection.execute(
                "UPDATE model_usage_days SET attempts = attempts + 1, "
                "updated_at = ? WHERE utc_day = ?",
                (timestamp, day),
            )
            return True

    def model_calls_for_day(self, *, now: datetime) -> int:
        day = _as_utc(now).date().isoformat()
        row = self._connection.execute(
            "SELECT attempts FROM model_usage_days WHERE utc_day = ?", (day,)
        ).fetchone()
        return int(row["attempts"]) if row is not None else 0

    def ambiguous_create_candidate(self) -> CreateRecoveryCandidate | None:
        """Return exact, content-free identifiers for operator review.

        This is deliberately a local database read. Remote inspection and the
        decision about whether a create took effect remain separate operator
        actions.
        """

        state = self._connection.execute(
            "SELECT output_gate, output_gate_job_note_id, output_gate_at, "
            "output_epoch FROM bridge_state WHERE singleton = 1"
        ).fetchone()
        if state["output_gate"] is None:
            return None
        if state["output_gate"] != "create_ambiguous":
            raise ValueError("the active output gate is not an uncertain create")
        row = self._connection.execute(
            """
            SELECT job.note_id, job.state, job.failure_stage, job.resume_state,
                   job.publication_sha256, job.publication_content,
                   job.publication_title_sha256,
                   job.publication_attempts, job.accepted_output_epoch,
                   job.publication_output_mode,
                   provisional.note_id AS provisional_output_note_id,
                   provisional.content_sha256 AS provisional_content_sha256
              FROM jobs AS job
              LEFT JOIN provisional_output_identities AS provisional
                ON provisional.source_note_id = job.note_id
               AND provisional.output_epoch = job.accepted_output_epoch
               AND provisional.publication_attempt = job.publication_attempts
             WHERE job.note_id = ?
            """,
            (state["output_gate_job_note_id"],),
        ).fetchone()
        if (
            row is None
            or row["state"] != JobState.FAILED
            or row["failure_stage"] != "publication"
            or row["resume_state"] != JobState.ANSWERED
            or int(row["publication_attempts"]) < 1
            or row["publication_sha256"] is None
            or row["publication_content"] is None
            or row["publication_title_sha256"] is None
            or row["publication_output_mode"] not in {"rolling", "per_turn"}
            or int(row["accepted_output_epoch"]) != int(state["output_epoch"])
            or state["output_gate_at"] is None
        ):
            raise RuntimeError("uncertain-create recovery state is inconsistent")
        publication_sha256 = str(row["publication_sha256"])
        publication_title_sha256 = str(row["publication_title_sha256"])
        _require_sha256(publication_sha256, field="stored publication hash")
        _require_sha256(
            publication_title_sha256,
            field="stored publication title hash",
        )
        if not hmac.compare_digest(
            publication_sha256, sha256_text(str(row["publication_content"]))
        ):
            raise RuntimeError("stored publication content failed its integrity check")
        return CreateRecoveryCandidate(
            source_note_id=str(row["note_id"]),
            publication_sha256=publication_sha256,
            publication_title_sha256=publication_title_sha256,
            publication_attempts=int(row["publication_attempts"]),
            output_epoch=int(state["output_epoch"]),
            output_mode=row["publication_output_mode"],
            provisional_output_note_id=row["provisional_output_note_id"],
            provisional_content_sha256=row["provisional_content_sha256"],
            gated_at=_decode_time(str(state["output_gate_at"])),
        )

    def ambiguous_update_candidate(self) -> UpdateRecoveryCandidate | None:
        """Return exact, content-free identifiers for update recovery."""

        state = self._connection.execute(
            """
            SELECT output_gate, output_gate_job_note_id, output_gate_at,
                   output_gate_readback_attempts, output_epoch,
                   active_output_note_id
              FROM bridge_state WHERE singleton = 1
            """
        ).fetchone()
        if state["output_gate"] is None:
            return None
        if state["output_gate"] != "update_ambiguous":
            raise ValueError("the active output gate is not an uncertain update")
        row = self._connection.execute(
            """
            SELECT job.note_id, job.state, job.failure_stage, job.resume_state,
                   job.publication_content, job.publication_sha256,
                   job.publication_title_sha256, job.publication_attempts,
                   job.accepted_output_epoch, job.accepted_output_note_id,
                   job.publication_output_mode, job.output_note_id,
                   surface.epoch AS surface_epoch,
                   surface.content_sha256 AS previous_content_sha256
              FROM jobs AS job
              JOIN output_surfaces AS surface
                ON surface.note_id = ? AND surface.retired_at IS NULL
             WHERE job.note_id = ?
            """,
            (
                state["active_output_note_id"],
                state["output_gate_job_note_id"],
            ),
        ).fetchone()
        if (
            row is None
            or row["state"] != JobState.FAILED
            or row["failure_stage"] != "publication"
            or row["resume_state"] != JobState.ANSWERED
            or int(row["publication_attempts"]) < 1
            or row["publication_content"] is None
            or row["publication_sha256"] is None
            or row["publication_title_sha256"] is None
            or row["publication_output_mode"] != "rolling"
            or int(row["accepted_output_epoch"]) != int(state["output_epoch"])
            or int(row["surface_epoch"]) != int(state["output_epoch"])
            or row["accepted_output_note_id"]
            not in {None, state["active_output_note_id"]}
            or row["output_note_id"] is not None
            or row["previous_content_sha256"] is None
            or state["active_output_note_id"] is None
            or state["output_gate_at"] is None
        ):
            raise RuntimeError("uncertain-update recovery state is inconsistent")
        publication_sha256 = str(row["publication_sha256"])
        publication_title_sha256 = str(row["publication_title_sha256"])
        previous_content_sha256 = str(row["previous_content_sha256"])
        _require_sha256(publication_sha256, field="stored publication hash")
        _require_sha256(
            publication_title_sha256,
            field="stored publication title hash",
        )
        _require_sha256(previous_content_sha256, field="stored output hash")
        if not hmac.compare_digest(
            publication_sha256, sha256_text(str(row["publication_content"]))
        ):
            raise RuntimeError("stored publication content failed its integrity check")
        gated_at = _decode_time(str(state["output_gate_at"]))
        assert gated_at is not None
        return UpdateRecoveryCandidate(
            source_note_id=str(row["note_id"]),
            publication_sha256=publication_sha256,
            publication_title_sha256=publication_title_sha256,
            publication_attempts=int(row["publication_attempts"]),
            output_epoch=int(state["output_epoch"]),
            output_note_id=str(state["active_output_note_id"]),
            previous_content_sha256=previous_content_sha256,
            readback_attempts=int(state["output_gate_readback_attempts"]),
            gated_at=gated_at,
        )

    def gate_interrupted_publication(self, *, now: datetime) -> bool:
        """Pause an interrupted create or update before any mutation can repeat.

        A process can stop after persisting its publication attempt but before
        its receipt reaches SQLite. An interrupted create must be gated before
        cloud discovery can ingest a bridge-owned note. An interrupted rolling
        update must be gated before the worker can issue the same conditional
        update again. Read-only reconciliation can clear either applied update
        later without crossing another mutation boundary.
        """

        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                "SELECT output_gate, output_epoch, active_output_note_id "
                "FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            if state["output_gate"] is not None:
                return False
            row = connection.execute(
                """
                SELECT note_id, publication_output_mode,
                       accepted_output_note_id
                  FROM jobs
                 WHERE state = 'answered'
                   AND output_note_id IS NULL
                   AND publication_attempts > 0
                   AND publication_content IS NOT NULL
                   AND publication_sha256 IS NOT NULL
                   AND publication_title_sha256 IS NOT NULL
                   AND publication_output_mode IN ('rolling','per_turn')
                   AND accepted_output_epoch = ?
                 ORDER BY queue_seq
                 LIMIT 1
                """,
                (state["output_epoch"],),
            ).fetchone()
            if row is None:
                return False
            interrupted_update = (
                row["publication_output_mode"] == "rolling"
                and state["active_output_note_id"] is not None
            )
            if (
                row["publication_output_mode"] == "rolling"
                and row["accepted_output_note_id"] is not None
                and row["accepted_output_note_id"]
                != state["active_output_note_id"]
            ):
                raise RuntimeError("interrupted update output identity is inconsistent")
            if interrupted_update and row["accepted_output_note_id"] is None:
                # Pre-v9 workers did not bind requests admitted before the
                # first rolling output was created. The active identity in the
                # unchanged output epoch is the only target they could have
                # crossed after persisting this attempt boundary.
                connection.execute(
                    "UPDATE jobs SET accepted_output_note_id = ?, updated_at = ? "
                    "WHERE note_id = ? AND accepted_output_note_id IS NULL",
                    (
                        state["active_output_note_id"],
                        timestamp,
                        row["note_id"],
                    ),
                )
            gate = "update_ambiguous" if interrupted_update else "create_ambiguous"
            error = (
                "update was interrupted after its durable attempt boundary"
                if interrupted_update
                else "create was interrupted before local identity persistence"
            )
            result = connection.execute(
                """
                UPDATE jobs SET state = 'failed',
                    failure_stage = 'publication', resume_state = 'answered',
                    retryable = 0, next_retry_at = NULL,
                    last_error = ?,
                    updated_at = ?, completed_at = ?
                 WHERE note_id = ? AND state = 'answered'
                """,
                (error, timestamp, timestamp, row["note_id"]),
            )
            if result.rowcount != 1:
                raise RuntimeError("interrupted publication changed during recovery")
            connection.execute(
                """
                UPDATE bridge_state
                   SET output_gate = ?, output_gate_job_note_id = ?,
                       output_gate_at = ?, output_gate_readback_attempts = 0
                 WHERE singleton = 1 AND output_gate IS NULL
                """,
                (gate, row["note_id"], timestamp),
            )
            return True

    def resolve_ambiguous_create(
        self,
        *,
        source_note_id: str,
        expected_publication_sha256: str,
        expected_output_epoch: int,
        decision: Literal["applied", "not_applied"],
        verification: AppliedCreateVerification | None,
        now: datetime,
    ) -> CreateRecoveryResult:
        """Record one reviewed create outcome and atomically release its gate.

        The caller supplies the gated source identity, local payload digest and
        output generation from :meth:`ambiguous_create_candidate`. An applied
        decision additionally supplies a structured receipt from the CLI's
        read-only, two-sample Light verification. No remote operation or answer
        generation occurs here.
        """

        _require_operator_identifier(source_note_id, field="source note id")
        _require_sha256(
            expected_publication_sha256,
            field="expected publication hash",
        )
        if expected_output_epoch < 0:
            raise ValueError("expected output epoch must not be negative")
        if decision not in {"applied", "not_applied"}:
            raise ValueError("unsupported create recovery decision")
        if decision == "applied":
            if verification is None:
                raise ValueError("an applied recovery requires verified cloud evidence")
            if verification.method != "light_read_only_v1":
                raise ValueError("unsupported create recovery verification method")
            output_note_id = verification.output_note_id
            observed_content_sha256 = verification.observed_content_sha256
            _require_operator_identifier(output_note_id, field="output note id")
            _require_sha256(
                observed_content_sha256,
                field="observed content hash",
            )
            if output_note_id == source_note_id:
                raise ValueError("source and output note identifiers must differ")
            if not hmac.compare_digest(
                observed_content_sha256, expected_publication_sha256
            ):
                raise ValueError("observed content does not match stored publication")
            _require_sha256(
                verification.observed_title_sha256,
                field="observed title hash",
            )
            if verification.observed_kind is not NoteKind.TEXT:
                raise ValueError("verified recovery output is not a text note")
            if _as_utc(verification.completed_at) < _as_utc(
                verification.started_at
            ):
                raise ValueError("verification timestamps are inconsistent")
        elif verification is not None:
            raise ValueError("a not-applied recovery must not include cloud evidence")
        else:
            output_note_id = None
            observed_content_sha256 = None

        timestamp = _encode_time(now)
        assert timestamp is not None
        with self._write() as connection:
            state = connection.execute(
                """
                SELECT output_gate, output_gate_job_note_id, output_gate_at,
                       output_epoch, active_output_note_id
                  FROM bridge_state WHERE singleton = 1
                """
            ).fetchone()
            job = connection.execute(
                """
                SELECT state, failure_stage, resume_state, publication_sha256,
                       publication_content, publication_title_sha256,
                       publication_attempts,
                       publication_output_mode,
                       accepted_output_epoch, accepted_output_note_id,
                       output_note_id
                  FROM jobs WHERE note_id = ?
                """,
                (source_note_id,),
            ).fetchone()
            if (
                state["output_gate"] != "create_ambiguous"
                or state["output_gate_job_note_id"] != source_note_id
                or state["output_gate_at"] is None
                or int(state["output_epoch"]) != expected_output_epoch
                or job is None
                or job["state"] != JobState.FAILED
                or job["failure_stage"] != "publication"
                or job["resume_state"] != JobState.ANSWERED
                or int(job["publication_attempts"]) < 1
                or int(job["accepted_output_epoch"]) != expected_output_epoch
                or job["publication_sha256"] is None
                or job["publication_content"] is None
                or job["publication_title_sha256"] is None
                or job["publication_output_mode"] not in {"rolling", "per_turn"}
                or job["output_note_id"] is not None
                or not hmac.compare_digest(
                    str(job["publication_sha256"]),
                    expected_publication_sha256,
                )
                or not hmac.compare_digest(
                    sha256_text(str(job["publication_content"])),
                    expected_publication_sha256,
                )
            ):
                raise ValueError("recovery identifiers do not match the active gate")
            output_mode = str(job["publication_output_mode"])
            publication_title_sha256 = str(job["publication_title_sha256"])
            _require_sha256(
                publication_title_sha256,
                field="stored publication title hash",
            )

            if decision == "applied":
                assert output_note_id is not None
                assert observed_content_sha256 is not None
                assert verification is not None
                if not hmac.compare_digest(
                    verification.observed_title_sha256,
                    publication_title_sha256,
                ):
                    raise ValueError("observed title does not match stored publication")
                existing_surface = connection.execute(
                    "SELECT 1 FROM output_surfaces WHERE note_id = ?",
                    (output_note_id,),
                ).fetchone()
                existing_job = connection.execute(
                    "SELECT 1 FROM jobs WHERE note_id = ? OR output_note_id = ?",
                    (output_note_id, output_note_id),
                ).fetchone()
                ledger = connection.execute(
                    "SELECT disposition, audio_sha256, source_kind "
                    "FROM request_ledger WHERE note_id = ?",
                    (output_note_id,),
                ).fetchone()
                binding = connection.execute(
                    """
                    SELECT source_note_id, output_epoch, publication_attempt,
                           content_sha256
                      FROM provisional_output_identities WHERE note_id = ?
                    """,
                    (output_note_id,),
                ).fetchone()
                attempt_binding = connection.execute(
                    """
                    SELECT note_id FROM provisional_output_identities
                     WHERE source_note_id = ? AND output_epoch = ?
                       AND publication_attempt = ?
                    """,
                    (
                        source_note_id,
                        expected_output_epoch,
                        int(job["publication_attempts"]),
                    ),
                ).fetchone()
                if existing_surface is not None or existing_job is not None:
                    raise ValueError("output note identity is already in use")
                if (
                    attempt_binding is not None
                    and attempt_binding["note_id"] != output_note_id
                ):
                    raise ValueError(
                        "verified output identity differs from the publication receipt"
                    )
                if ledger is None:
                    if binding is not None:
                        raise RuntimeError(
                            "provisional output binding has no request ledger entry"
                        )
                else:
                    exact_binding = bool(
                        binding is not None
                        and binding["source_note_id"] == source_note_id
                        and int(binding["output_epoch"]) == expected_output_epoch
                        and int(binding["publication_attempt"])
                        == int(job["publication_attempts"])
                    )
                    if (
                        ledger["disposition"] != "bridge_output"
                        or ledger["source_kind"] != NoteKind.TEXT
                        or not exact_binding
                        or (
                            ledger["audio_sha256"] is not None
                            and not hmac.compare_digest(
                                str(ledger["audio_sha256"]),
                                observed_content_sha256,
                            )
                        )
                        or (
                            binding["content_sha256"] is not None
                            and not hmac.compare_digest(
                                str(binding["content_sha256"]),
                                observed_content_sha256,
                            )
                        )
                    ):
                        raise ValueError(
                            "output note identity conflicts with gated publication"
                        )
                if output_mode == "rolling":
                    if (
                        state["active_output_note_id"] is not None
                        or job["accepted_output_note_id"] is not None
                    ):
                        raise ValueError("rolling output generation is no longer empty")
                    connection.execute(
                        """
                        INSERT INTO output_surfaces(
                            note_id, epoch, content_sha256, created_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            output_note_id,
                            expected_output_epoch,
                            expected_publication_sha256,
                            timestamp,
                        ),
                    )
                    result = connection.execute(
                        """
                        UPDATE bridge_state SET active_output_note_id = ?
                         WHERE singleton = 1 AND output_gate = 'create_ambiguous'
                           AND output_gate_job_note_id = ?
                           AND output_epoch = ? AND active_output_note_id IS NULL
                        """,
                        (output_note_id, source_note_id, expected_output_epoch),
                    )
                    if result.rowcount != 1:
                        raise RuntimeError("output generation changed during recovery")
                self._complete_publication(
                    connection,
                    source_note_id,
                    output_note_id=output_note_id,
                    timestamp=timestamp,
                )
            else:
                connection.execute(
                    """
                    UPDATE jobs
                       SET last_error = 'operator confirmed uncertain create was not applied',
                           retryable = 0, next_retry_at = NULL,
                           updated_at = ?, completed_at = ?
                     WHERE note_id = ?
                    """,
                    (timestamp, timestamp, source_note_id),
                )

            cursor = connection.execute(
                """
                INSERT INTO publication_recovery_audit(
                    gated_at, decided_at, gate, source_note_id,
                    publication_sha256, publication_title_sha256,
                    publication_attempts, output_epoch,
                    decision, output_mode, output_note_id,
                    observed_content_sha256, observed_title_sha256,
                    observed_kind, remote_updated_at, verification_method,
                    verification_started_at, verification_completed_at
                ) VALUES (
                    ?, ?, 'create_ambiguous', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    state["output_gate_at"],
                    timestamp,
                    source_note_id,
                    expected_publication_sha256,
                    publication_title_sha256,
                    int(job["publication_attempts"]),
                    expected_output_epoch,
                    decision,
                    output_mode,
                    output_note_id,
                    observed_content_sha256,
                    (
                        verification.observed_title_sha256
                        if verification is not None
                        else None
                    ),
                    (
                        verification.observed_kind.value
                        if verification is not None
                        else None
                    ),
                    (
                        _encode_time(verification.remote_updated_at)
                        if verification is not None
                        else None
                    ),
                    verification.method if verification is not None else None,
                    (
                        _encode_time(verification.started_at)
                        if verification is not None
                        else None
                    ),
                    (
                        _encode_time(verification.completed_at)
                        if verification is not None
                        else None
                    ),
                ),
            )
            cleared = connection.execute(
                """
                UPDATE bridge_state SET output_gate = NULL,
                    output_gate_job_note_id = NULL, output_gate_at = NULL,
                    output_gate_readback_attempts = 0
                 WHERE singleton = 1 AND output_gate = 'create_ambiguous'
                   AND output_gate_job_note_id = ? AND output_epoch = ?
                """,
                (source_note_id, expected_output_epoch),
            )
            if cleared.rowcount != 1:
                raise RuntimeError("output gate changed during recovery")
            audit_id = int(cursor.lastrowid)

        return CreateRecoveryResult(
            audit_id=audit_id,
            decision=decision,
            source_note_id=source_note_id,
            publication_sha256=expected_publication_sha256,
            output_note_id=output_note_id,
            output_epoch=expected_output_epoch,
            output_mode=output_mode,
            decided_at=_as_utc(now),
        )

    def resolve_ambiguous_update_applied(
        self,
        *,
        source_note_id: str,
        expected_publication_sha256: str,
        expected_output_epoch: int,
        expected_output_note_id: str,
        expected_previous_content_sha256: str,
        expected_readback_attempts: int,
        verification: AppliedCreateVerification,
        now: datetime,
    ) -> UpdateRecoveryResult:
        """Accept exact read-only evidence that one uncertain update applied."""

        _require_operator_identifier(source_note_id, field="source note id")
        _require_operator_identifier(expected_output_note_id, field="output note id")
        _require_sha256(
            expected_publication_sha256,
            field="expected publication hash",
        )
        _require_sha256(
            expected_previous_content_sha256,
            field="expected previous output hash",
        )
        if expected_output_epoch < 0 or expected_readback_attempts < 0:
            raise ValueError("update recovery counters must not be negative")
        candidate = self.ambiguous_update_candidate()
        if (
            candidate is None
            or candidate.source_note_id != source_note_id
            or candidate.output_epoch != expected_output_epoch
            or candidate.output_note_id != expected_output_note_id
            or candidate.readback_attempts != expected_readback_attempts
            or not hmac.compare_digest(
                candidate.publication_sha256,
                expected_publication_sha256,
            )
            or not hmac.compare_digest(
                candidate.previous_content_sha256,
                expected_previous_content_sha256,
            )
        ):
            raise ValueError("update recovery identifiers do not match the active gate")
        if (
            verification.method != "light_read_only_v1"
            or verification.output_note_id != expected_output_note_id
            or verification.observed_kind is not NoteKind.TEXT
            or not hmac.compare_digest(
                verification.observed_content_sha256,
                expected_publication_sha256,
            )
            or not hmac.compare_digest(
                verification.observed_title_sha256,
                candidate.publication_title_sha256,
            )
            or _as_utc(verification.completed_at)
            < _as_utc(verification.started_at)
        ):
            raise ValueError("update verification does not match the active gate")

        timestamp = _encode_time(now)
        assert timestamp is not None
        with self._write() as connection:
            updated = connection.execute(
                """
                UPDATE output_surfaces SET content_sha256 = ?,
                    missing_observations = 0, missing_since = NULL
                 WHERE note_id = ? AND content_sha256 = ? AND retired_at IS NULL
                """,
                (
                    expected_publication_sha256,
                    expected_output_note_id,
                    expected_previous_content_sha256,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("update recovery output surface changed")
            connection.execute(
                "UPDATE jobs SET accepted_output_note_id = COALESCE("
                "accepted_output_note_id, ?) WHERE note_id = ?",
                (expected_output_note_id, source_note_id),
            )
            self._complete_publication(
                connection,
                source_note_id,
                output_note_id=expected_output_note_id,
                timestamp=timestamp,
            )
            cursor = connection.execute(
                """
                INSERT INTO update_recovery_audit(
                    gated_at, decided_at, action, source_note_id,
                    publication_sha256, publication_title_sha256,
                    publication_attempts, output_epoch, output_note_id,
                    previous_content_sha256, prior_readback_attempts,
                    observed_content_sha256, observed_title_sha256,
                    observed_kind, remote_updated_at, verification_method,
                    verification_started_at, verification_completed_at
                ) VALUES (?, ?, 'applied', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _encode_time(candidate.gated_at),
                    timestamp,
                    source_note_id,
                    candidate.publication_sha256,
                    candidate.publication_title_sha256,
                    candidate.publication_attempts,
                    candidate.output_epoch,
                    candidate.output_note_id,
                    candidate.previous_content_sha256,
                    candidate.readback_attempts,
                    verification.observed_content_sha256,
                    verification.observed_title_sha256,
                    verification.observed_kind.value,
                    _encode_time(verification.remote_updated_at),
                    verification.method,
                    _encode_time(verification.started_at),
                    _encode_time(verification.completed_at),
                ),
            )
            cleared = connection.execute(
                """
                UPDATE bridge_state SET output_gate = NULL,
                    output_gate_job_note_id = NULL, output_gate_at = NULL,
                    output_gate_readback_attempts = 0
                 WHERE singleton = 1 AND output_gate = 'update_ambiguous'
                   AND output_gate_job_note_id = ? AND output_epoch = ?
                   AND active_output_note_id = ?
                   AND output_gate_readback_attempts = ?
                """,
                (
                    source_note_id,
                    expected_output_epoch,
                    expected_output_note_id,
                    expected_readback_attempts,
                ),
            )
            if cleared.rowcount != 1:
                raise RuntimeError("update gate changed during recovery")
            audit_id = int(cursor.lastrowid)
        return UpdateRecoveryResult(
            audit_id=audit_id,
            action="applied",
            source_note_id=source_note_id,
            publication_sha256=expected_publication_sha256,
            output_note_id=expected_output_note_id,
            output_epoch=expected_output_epoch,
            prior_readback_attempts=expected_readback_attempts,
            decided_at=_as_utc(now),
        )

    def resume_ambiguous_update_readback(
        self,
        *,
        source_note_id: str,
        expected_publication_sha256: str,
        expected_output_epoch: int,
        expected_output_note_id: str,
        expected_previous_content_sha256: str,
        expected_readback_attempts: int,
        now: datetime,
    ) -> UpdateRecoveryResult:
        """Grant another bounded read-only window without repeating the update."""

        _require_operator_identifier(source_note_id, field="source note id")
        _require_operator_identifier(expected_output_note_id, field="output note id")
        _require_sha256(
            expected_publication_sha256,
            field="expected publication hash",
        )
        _require_sha256(
            expected_previous_content_sha256,
            field="expected previous output hash",
        )
        if expected_output_epoch < 0 or expected_readback_attempts < 1:
            raise ValueError("update recovery requires prior read-back attempts")
        candidate = self.ambiguous_update_candidate()
        if (
            candidate is None
            or candidate.source_note_id != source_note_id
            or candidate.output_epoch != expected_output_epoch
            or candidate.output_note_id != expected_output_note_id
            or candidate.readback_attempts != expected_readback_attempts
            or not hmac.compare_digest(
                candidate.publication_sha256,
                expected_publication_sha256,
            )
            or not hmac.compare_digest(
                candidate.previous_content_sha256,
                expected_previous_content_sha256,
            )
        ):
            raise ValueError("update recovery identifiers do not match the active gate")

        timestamp = _encode_time(now)
        assert timestamp is not None
        with self._write() as connection:
            reset = connection.execute(
                """
                UPDATE bridge_state SET output_gate_readback_attempts = 0
                 WHERE singleton = 1 AND output_gate = 'update_ambiguous'
                   AND output_gate_job_note_id = ? AND output_epoch = ?
                   AND active_output_note_id = ?
                   AND output_gate_readback_attempts = ?
                """,
                (
                    source_note_id,
                    expected_output_epoch,
                    expected_output_note_id,
                    expected_readback_attempts,
                ),
            )
            if reset.rowcount != 1:
                raise RuntimeError("update gate changed during recovery")
            connection.execute(
                """
                UPDATE jobs
                   SET last_error = 'update confirmation read-back resumed by operator',
                       updated_at = ?
                 WHERE note_id = ? AND state = 'failed'
                   AND failure_stage = 'publication'
                """,
                (timestamp, source_note_id),
            )
            cursor = connection.execute(
                """
                INSERT INTO update_recovery_audit(
                    gated_at, decided_at, action, source_note_id,
                    publication_sha256, publication_title_sha256,
                    publication_attempts, output_epoch, output_note_id,
                    previous_content_sha256, prior_readback_attempts
                ) VALUES (?, ?, 'resume_readback', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _encode_time(candidate.gated_at),
                    timestamp,
                    source_note_id,
                    candidate.publication_sha256,
                    candidate.publication_title_sha256,
                    candidate.publication_attempts,
                    candidate.output_epoch,
                    candidate.output_note_id,
                    candidate.previous_content_sha256,
                    candidate.readback_attempts,
                ),
            )
            audit_id = int(cursor.lastrowid)
        return UpdateRecoveryResult(
            audit_id=audit_id,
            action="resume_readback",
            source_note_id=source_note_id,
            publication_sha256=expected_publication_sha256,
            output_note_id=expected_output_note_id,
            output_epoch=expected_output_epoch,
            prior_readback_attempts=expected_readback_attempts,
            decided_at=_as_utc(now),
        )

    def mark_publication_ambiguous(
        self,
        note_id: str,
        *,
        gate: str,
        error: str,
        now: datetime,
    ) -> None:
        """Atomically fail one turn and stop every later remote publication."""

        if gate not in {"update_ambiguous", "create_ambiguous"}:
            raise ValueError("unsupported output ambiguity gate")
        timestamp = _encode_time(now)
        with self._write() as connection:
            current = connection.execute(
                "SELECT output_gate, output_gate_job_note_id FROM bridge_state "
                "WHERE singleton = 1"
            ).fetchone()
            if current["output_gate"] is not None and (
                current["output_gate"] != gate
                or current["output_gate_job_note_id"] != note_id
            ):
                raise RuntimeError("a different output ambiguity is already active")
            result = connection.execute(
                """
                UPDATE jobs SET state = 'failed', failure_stage = 'publication',
                    resume_state = 'answered', retryable = 0,
                    next_retry_at = NULL, last_error = ?, updated_at = ?,
                    completed_at = ?
                 WHERE note_id = ? AND state = 'answered'
                """,
                (error[:500], timestamp, timestamp, note_id),
            )
            if result.rowcount != 1:
                raise KeyError(note_id)
            connection.execute(
                """
                UPDATE bridge_state
                   SET output_gate = ?, output_gate_job_note_id = ?,
                       output_gate_at = ?, output_gate_readback_attempts = 0
                 WHERE singleton = 1
                """,
                (gate, note_id, timestamp),
            )

    def prepare_publication(
        self,
        note_id: str,
        *,
        content: str,
        title: str,
        output_mode: Literal["rolling", "per_turn"],
        now: datetime,
    ) -> tuple[str, str, Literal["rolling", "per_turn"]]:
        """Persist the exact payload and mode before an ambiguous mutation."""

        if output_mode not in {"rolling", "per_turn"}:
            raise ValueError("unsupported publication output mode")
        digest = sha256_text(content)
        title_digest = sha256_text(title)
        with self._write() as connection:
            row = connection.execute(
                "SELECT publication_content, publication_sha256, "
                "publication_title_sha256, publication_output_mode "
                "FROM jobs WHERE note_id = ?",
                (note_id,),
            ).fetchone()
            if row is None:
                raise KeyError(note_id)
            if row["publication_content"] is None:
                if (
                    row["publication_sha256"] is not None
                    or row["publication_title_sha256"] is not None
                    or row["publication_output_mode"] is not None
                ):
                    raise RuntimeError("stored publication boundary is inconsistent")
                connection.execute(
                    """
                    UPDATE jobs SET publication_content = ?,
                        publication_sha256 = ?, publication_title_sha256 = ?,
                        publication_output_mode = ?,
                        updated_at = ?
                     WHERE note_id = ?
                    """,
                    (
                        content,
                        digest,
                        title_digest,
                        output_mode,
                        _encode_time(now),
                        note_id,
                    ),
                )
                return content, digest, output_mode
            stored_content = str(row["publication_content"])
            stored_digest = row["publication_sha256"]
            stored_title_digest = row["publication_title_sha256"]
            stored_mode = row["publication_output_mode"]
            if (
                stored_digest is None
                or stored_title_digest is None
                or stored_mode not in {"rolling", "per_turn"}
                or not hmac.compare_digest(
                    str(stored_digest), sha256_text(stored_content)
                )
            ):
                raise RuntimeError("stored publication boundary is incomplete")
            _require_sha256(
                str(stored_title_digest),
                field="stored publication title hash",
            )
            if not hmac.compare_digest(str(stored_title_digest), title_digest):
                raise RuntimeError(
                    "stored publication title differs from runtime configuration"
                )
            return stored_content, str(stored_digest), stored_mode

    def bind_rolling_publication_target(
        self,
        note_id: str,
        *,
        output_note_id: str,
        expected_output_epoch: int,
        expected_content_sha256: str,
        now: datetime,
    ) -> bool:
        """Persist the exact rolling target before an update attempt begins."""

        _require_sha256(expected_content_sha256, field="expected output hash")
        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                """
                SELECT bridge.output_epoch, bridge.active_output_note_id,
                       surface.content_sha256
                  FROM bridge_state AS bridge
                  LEFT JOIN output_surfaces AS surface
                    ON surface.note_id = bridge.active_output_note_id
                   AND surface.retired_at IS NULL
                 WHERE bridge.singleton = 1
                """
            ).fetchone()
            job = connection.execute(
                """
                SELECT state, publication_output_mode, publication_attempts,
                       accepted_output_epoch, accepted_output_note_id,
                       output_note_id
                  FROM jobs WHERE note_id = ?
                """,
                (note_id,),
            ).fetchone()
            if (
                job is None
                or job["state"] != JobState.ANSWERED
                or job["publication_output_mode"] != "rolling"
                or int(job["publication_attempts"]) != 0
                or int(job["accepted_output_epoch"]) != expected_output_epoch
                or job["accepted_output_note_id"] not in {None, output_note_id}
                or job["output_note_id"] is not None
                or int(state["output_epoch"]) != expected_output_epoch
                or state["active_output_note_id"] != output_note_id
                or state["content_sha256"] != expected_content_sha256
            ):
                return False
            if job["accepted_output_note_id"] is None:
                result = connection.execute(
                    """
                    UPDATE jobs SET accepted_output_note_id = ?, updated_at = ?
                     WHERE note_id = ? AND accepted_output_note_id IS NULL
                       AND publication_attempts = 0 AND state = 'answered'
                    """,
                    (output_note_id, timestamp, note_id),
                )
                if result.rowcount != 1:
                    raise RuntimeError("rolling publication target changed")
            return True

    def reconcile_applied_publication(
        self,
        *,
        output_note_id: str,
        observed_content_sha256: str,
        now: datetime,
    ) -> str | None:
        """Complete a locally interrupted update found by remote read-back."""

        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                "SELECT output_epoch, active_output_note_id, output_gate, "
                "output_gate_job_note_id FROM bridge_state "
                "WHERE singleton = 1"
            ).fetchone()
            if state["active_output_note_id"] != output_note_id:
                return None
            if state["output_gate"] == "create_ambiguous":
                return None
            if state["output_gate"] == "update_ambiguous":
                job = connection.execute(
                    """
                    SELECT note_id FROM jobs
                     WHERE note_id = ? AND state IN ('answered','failed')
                       AND publication_attempts > 0
                       AND publication_sha256 = ?
                       AND accepted_output_epoch = ?
                    """,
                    (
                        state["output_gate_job_note_id"],
                        observed_content_sha256,
                        state["output_epoch"],
                    ),
                ).fetchone()
            else:
                job = connection.execute(
                    """
                    SELECT note_id FROM jobs
                     WHERE state = 'answered'
                       AND publication_attempts > 0
                       AND publication_sha256 = ?
                       AND accepted_output_epoch = ?
                     ORDER BY queue_seq
                     LIMIT 1
                    """,
                    (observed_content_sha256, state["output_epoch"]),
                ).fetchone()
            if job is None:
                return None
            connection.execute(
                "UPDATE output_surfaces SET content_sha256 = ? "
                "WHERE note_id = ? AND retired_at IS NULL",
                (observed_content_sha256, output_note_id),
            )
            self._complete_publication(
                connection,
                str(job["note_id"]),
                output_note_id=output_note_id,
                timestamp=timestamp,
            )
            connection.execute(
                """
                UPDATE bridge_state SET output_gate = NULL,
                    output_gate_job_note_id = NULL, output_gate_at = NULL,
                    output_gate_readback_attempts = 0
                 WHERE singleton = 1 AND output_gate = 'update_ambiguous'
                   AND output_gate_job_note_id = ?
                """,
                (str(job["note_id"]),),
            )
            return str(job["note_id"])

    def resolve_unapplied_publication(
        self,
        *,
        output_note_id: str,
        observed_content_sha256: str,
        maximum_readbacks: int,
        now: datetime,
    ) -> str | None:
        """Record one old-value read without concluding an update was unapplied.

        A Light object read can lag a completed upload. The exact pre-write
        value therefore consumes one durable, bounded read-back attempt while
        retaining the global update gate. The only automatic gate releases in
        this path are a no-op publication or a pre-mutation missing-note result,
        neither of which can repeat an uncertain update.
        """

        if maximum_readbacks < 1:
            raise ValueError("update read-back limit must be positive")
        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                "SELECT output_epoch, active_output_note_id, output_gate, "
                "output_gate_job_note_id, output_gate_readback_attempts "
                "FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            if (
                state["output_gate"] != "update_ambiguous"
                or state["active_output_note_id"] != output_note_id
            ):
                return None
            surface = connection.execute(
                "SELECT content_sha256 FROM output_surfaces "
                "WHERE note_id = ? AND retired_at IS NULL",
                (output_note_id,),
            ).fetchone()
            if (
                surface is None
                or surface["content_sha256"] != observed_content_sha256
            ):
                return None
            job = connection.execute(
                """
                SELECT note_id, publication_sha256, last_error, resume_state,
                       accepted_output_note_id, publication_output_mode
                  FROM jobs
                 WHERE note_id = ? AND state = 'failed'
                   AND failure_stage = 'publication'
                   AND publication_attempts > 0
                   AND accepted_output_epoch = ?
                """,
                (state["output_gate_job_note_id"], state["output_epoch"]),
            ).fetchone()
            if (
                job is None
                or job["resume_state"] != JobState.ANSWERED
                or job["publication_output_mode"] != "rolling"
            ):
                return None
            accepted_output_note_id = job["accepted_output_note_id"]
            if accepted_output_note_id is None:
                # Upgrade recovery for an ambiguity recorded by a worker that
                # predated explicit target binding. The active output identity
                # and epoch are already validated above.
                connection.execute(
                    "UPDATE jobs SET accepted_output_note_id = ?, updated_at = ? "
                    "WHERE note_id = ? AND accepted_output_note_id IS NULL",
                    (output_note_id, timestamp, str(job["note_id"])),
                )
            elif accepted_output_note_id != output_note_id:
                return None
            if job["publication_sha256"] == observed_content_sha256:
                self._complete_publication(
                    connection,
                    str(job["note_id"]),
                    output_note_id=output_note_id,
                    timestamp=timestamp,
                )
                outcome = "recovered"
            elif job["last_error"] == MISSING_OUTPUT_UPDATE_ERROR:
                connection.execute(
                    """
                    UPDATE jobs
                       SET state = 'answered', failure_stage = NULL,
                           resume_state = NULL, retryable = 0,
                           next_retry_at = NULL, last_error = NULL,
                           publication_attempts = 0,
                           updated_at = ?, completed_at = NULL
                     WHERE note_id = ?
                    """,
                    (timestamp, str(job["note_id"])),
                )
                outcome = "retry"
            else:
                attempts = int(state["output_gate_readback_attempts"])
                if attempts >= maximum_readbacks:
                    return "unknown"
                recorded = connection.execute(
                    """
                    UPDATE bridge_state
                       SET output_gate_readback_attempts =
                           output_gate_readback_attempts + 1
                     WHERE singleton = 1
                       AND output_gate = 'update_ambiguous'
                       AND output_gate_job_note_id = ?
                       AND output_gate_readback_attempts = ?
                    """,
                    (str(job["note_id"]), attempts),
                )
                if recorded.rowcount != 1:
                    raise RuntimeError("update read-back gate changed during recovery")
                attempts += 1
                if attempts >= maximum_readbacks:
                    connection.execute(
                        """
                        UPDATE jobs
                           SET last_error =
                               'update outcome remains unknown after bounded read-back',
                               updated_at = ?
                         WHERE note_id = ?
                        """,
                        (timestamp, str(job["note_id"])),
                    )
                    return "unknown"
                return "pending"
            cleared = connection.execute(
                """
                UPDATE bridge_state SET output_gate = NULL,
                    output_gate_job_note_id = NULL, output_gate_at = NULL,
                    output_gate_readback_attempts = 0
                 WHERE singleton = 1 AND output_gate = 'update_ambiguous'
                   AND output_gate_job_note_id = ?
                """,
                (str(job["note_id"]),),
            )
            if cleared.rowcount != 1:
                raise RuntimeError("update gate changed during recovery")
            return outcome

    def abandon_stale_ambiguous_update(
        self,
        *,
        output_note_id: str,
        observed_content_sha256: str,
        maximum_readbacks: int,
        now: datetime,
    ) -> bool:
        """Quarantine one exhausted rolling surface without repeating its write.

        Once the bounded non-desired read window is exhausted and one fresh
        read shows the exact pre-write value, continuing to gate every later
        turn makes one lost response a permanent service outage. The bounded
        counter may include old-value and missing observations; the final read
        must be the exact old value. Re-creating the stored answer is unsafe
        because the original update could become visible later and duplicate
        it. This transition therefore abandons only the unconfirmable delivery,
        keeps its answer and sole attempt record intact, retires the old surface
        locally, and carries untouched later work into a new output generation.

        The remote note is not mutated.  It remains a known bridge output, so
        it can never be ingested as a user request even if the delayed update
        eventually becomes visible.
        """

        if maximum_readbacks < 1:
            raise ValueError("update read-back limit must be positive")
        _require_sha256(observed_content_sha256, field="observed output hash")
        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                "SELECT output_epoch, active_output_note_id, output_gate, "
                "output_gate_job_note_id, output_gate_readback_attempts "
                "FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            if (
                state["output_gate"] != "update_ambiguous"
                or state["active_output_note_id"] != output_note_id
                or int(state["output_gate_readback_attempts"])
                < maximum_readbacks
            ):
                return False
            old_epoch = int(state["output_epoch"])
            surface = connection.execute(
                "SELECT content_sha256 FROM output_surfaces "
                "WHERE note_id = ? AND epoch = ? AND retired_at IS NULL",
                (output_note_id, old_epoch),
            ).fetchone()
            if (
                surface is None
                or not hmac.compare_digest(
                    str(surface["content_sha256"]), observed_content_sha256
                )
            ):
                return False
            job = connection.execute(
                """
                SELECT note_id, queue_seq, publication_sha256,
                       publication_attempts, publication_output_mode,
                       accepted_output_note_id
                  FROM jobs
                 WHERE note_id = ? AND state = 'failed'
                   AND failure_stage = 'publication'
                   AND resume_state = 'answered'
                   AND output_note_id IS NULL
                   AND publication_attempts = 1
                   AND accepted_output_epoch = ?
                """,
                (state["output_gate_job_note_id"], old_epoch),
            ).fetchone()
            if (
                job is None
                or int(job["publication_attempts"]) != 1
                or job["publication_output_mode"] != "rolling"
                or job["accepted_output_note_id"] != output_note_id
                or job["publication_sha256"] is None
                or hmac.compare_digest(
                    str(job["publication_sha256"]), observed_content_sha256
                )
            ):
                return False

            new_epoch = old_epoch + 1
            retired = connection.execute(
                """
                UPDATE output_surfaces
                   SET retired_at = ?, retired_reason = 'ambiguous_old_value'
                 WHERE note_id = ? AND epoch = ? AND retired_at IS NULL
                   AND content_sha256 = ?
                """,
                (timestamp, output_note_id, old_epoch, observed_content_sha256),
            )
            if retired.rowcount != 1:
                raise RuntimeError(
                    "stale ambiguous output changed during retirement"
                )
            failed = connection.execute(
                """
                UPDATE jobs SET last_error = ?, updated_at = ?
                 WHERE note_id = ? AND state = 'failed'
                   AND failure_stage = 'publication'
                   AND publication_attempts > 0
                """,
                (
                    STALE_UPDATE_DELIVERY_ABANDONED_ERROR,
                    timestamp,
                    str(job["note_id"]),
                ),
            )
            if failed.rowcount != 1:
                raise RuntimeError(
                    "stale ambiguous publication changed during retirement"
                )
            cleared = connection.execute(
                """
                UPDATE bridge_state
                   SET active_output_note_id = NULL,
                       output_epoch = ?, output_gate = NULL,
                       output_gate_job_note_id = NULL, output_gate_at = NULL,
                       output_gate_readback_attempts = 0
                 WHERE singleton = 1
                   AND output_epoch = ?
                   AND active_output_note_id = ?
                   AND output_gate = 'update_ambiguous'
                   AND output_gate_job_note_id = ?
                   AND output_gate_readback_attempts >= ?
                """,
                (
                    new_epoch,
                    old_epoch,
                    output_note_id,
                    str(job["note_id"]),
                    maximum_readbacks,
                ),
            )
            if cleared.rowcount != 1:
                raise RuntimeError("stale ambiguous update gate changed")

            # Later admitted work never crossed a Light mutation boundary. It
            # can safely target the fresh generation while the ambiguous turn
            # remains terminal and retains its exact answer for audit.
            connection.execute(
                """
                UPDATE jobs
                   SET accepted_output_epoch = ?,
                       accepted_output_note_id = NULL, updated_at = ?
                 WHERE queue_seq > ? AND accepted_output_epoch = ?
                   AND output_note_id IS NULL
                   AND publication_attempts = 0
                   AND (
                       state IN ('queued','transcribed','answered')
                       OR (state = 'failed' AND retryable = 1)
                   )
                """,
                (new_epoch, timestamp, int(job["queue_seq"]), old_epoch),
            )
        return True

    def supersede_ambiguous_publication(
        self,
        *,
        interrupter_note_id: str,
        now: datetime,
    ) -> bool:
        """Atomically abandon one uncertain delivery for a newer request.

        No remote operation occurs here. The old mutation attempt remains
        terminal and auditable, a known rolling surface is retired locally,
        and every untouched later request is rebound to a fresh generation.
        """

        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                """
                SELECT output_epoch, active_output_note_id, output_gate,
                       output_gate_job_note_id, assistant_gate
                  FROM bridge_state WHERE singleton = 1
                """
            ).fetchone()
            gate = state["output_gate"]
            if gate not in {"create_ambiguous", "update_ambiguous"}:
                return False
            if state["assistant_gate"] is not None:
                return False
            old_epoch = int(state["output_epoch"])
            gated = connection.execute(
                """
                SELECT note_id, queue_seq, state, failure_stage, resume_state,
                       publication_attempts, publication_sha256,
                       publication_title_sha256, publication_output_mode,
                       accepted_output_epoch, accepted_output_note_id,
                       output_note_id
                  FROM jobs WHERE note_id = ?
                """,
                (state["output_gate_job_note_id"],),
            ).fetchone()
            interrupter = connection.execute(
                """
                SELECT note_id, queue_seq, state, route_kind, model_attempts,
                       publication_attempts, accepted_output_epoch,
                       output_note_id, admitted_output_gate,
                       admitted_output_gate_job_note_id
                  FROM jobs WHERE note_id = ?
                """,
                (interrupter_note_id,),
            ).fetchone()
            if (
                gated is None
                or gated["state"] != JobState.FAILED
                or gated["failure_stage"] != "publication"
                or gated["resume_state"] != JobState.ANSWERED
                or int(gated["publication_attempts"]) != 1
                or gated["publication_sha256"] is None
                or gated["publication_title_sha256"] is None
                or int(gated["accepted_output_epoch"]) != old_epoch
                or gated["output_note_id"] is not None
                or interrupter is None
                or int(interrupter["queue_seq"]) <= int(gated["queue_seq"])
                or interrupter["state"] not in {
                    JobState.TRANSCRIBED,
                    JobState.ANSWERED,
                }
                or interrupter["route_kind"]
                not in {
                    RouteKind.QUESTION,
                    RouteKind.CONTINUE,
                    RouteKind.NEW_CONVERSATION,
                    RouteKind.RESET,
                }
                or int(interrupter["model_attempts"]) != 0
                or int(interrupter["publication_attempts"]) != 0
                or int(interrupter["accepted_output_epoch"]) != old_epoch
                or interrupter["output_note_id"] is not None
                or interrupter["admitted_output_gate"] != gate
                or interrupter["admitted_output_gate_job_note_id"]
                != gated["note_id"]
            ):
                raise RuntimeError("best-effort publication recovery is inconsistent")

            active_note_id = state["active_output_note_id"]
            if gate == "update_ambiguous":
                surface = connection.execute(
                    """
                    SELECT note_id FROM output_surfaces
                     WHERE note_id = ? AND epoch = ? AND retired_at IS NULL
                    """,
                    (active_note_id, old_epoch),
                ).fetchone()
                if (
                    active_note_id is None
                    or surface is None
                    or gated["publication_output_mode"] != "rolling"
                    or gated["accepted_output_note_id"] != active_note_id
                ):
                    raise RuntimeError("ambiguous update target changed")
                retired = connection.execute(
                    """
                    UPDATE output_surfaces
                       SET retired_at = ?,
                           retired_reason = 'superseded_by_new_request'
                     WHERE note_id = ? AND epoch = ? AND retired_at IS NULL
                    """,
                    (timestamp, active_note_id, old_epoch),
                )
                if retired.rowcount != 1:
                    raise RuntimeError("ambiguous update surface changed")
            elif active_note_id is not None:
                # A create with a locally registered surface is classified as
                # an update by gate_interrupted_publication. An active identity
                # under create_ambiguous therefore indicates corrupt state.
                raise RuntimeError("ambiguous create unexpectedly has a surface")

            failed = connection.execute(
                """
                UPDATE jobs SET last_error = ?, retryable = 0,
                    next_retry_at = NULL, manual_retry_allowed = 0,
                    model_attempt_active = 0, updated_at = ?, completed_at = ?
                 WHERE note_id = ? AND state = 'failed'
                   AND failure_stage = 'publication'
                   AND publication_attempts = 1
                """,
                (
                    BEST_EFFORT_PUBLICATION_SUPERSEDED_ERROR,
                    timestamp,
                    timestamp,
                    str(gated["note_id"]),
                ),
            )
            if failed.rowcount != 1:
                raise RuntimeError("ambiguous publication changed")

            # If routing had to pass older staged material to locate the first
            # recognized request, make that material terminal so the chosen
            # interrupter is deterministically next after restart.
            connection.execute(
                """
                UPDATE jobs SET state = 'failed', failure_stage = 'superseded',
                    resume_state = NULL, retryable = 0,
                    manual_retry_allowed = 0, model_attempt_active = 0,
                    next_retry_at = NULL, last_error = ?, audio_blob = NULL,
                    updated_at = ?, completed_at = ?
                 WHERE queue_seq > ? AND queue_seq < ?
                   AND output_note_id IS NULL AND publication_attempts = 0
                   AND (
                       state IN ('queued','transcribed','answered')
                       OR (state = 'failed' AND (
                           retryable = 1 OR manual_retry_allowed = 1
                           OR completed_at IS NULL
                       ))
                   )
                """,
                (
                    BEST_EFFORT_PUBLICATION_SUPERSEDED_ERROR,
                    timestamp,
                    timestamp,
                    int(gated["queue_seq"]),
                    int(interrupter["queue_seq"]),
                ),
            )

            new_epoch = old_epoch + 1
            cleared = connection.execute(
                """
                UPDATE bridge_state
                   SET active_output_note_id = NULL, output_epoch = ?,
                       output_gate = NULL, output_gate_job_note_id = NULL,
                       output_gate_at = NULL,
                       output_gate_readback_attempts = 0
                 WHERE singleton = 1 AND output_epoch = ?
                   AND output_gate = ? AND output_gate_job_note_id = ?
                """,
                (new_epoch, old_epoch, gate, str(gated["note_id"])),
            )
            if cleared.rowcount != 1:
                raise RuntimeError("publication gate changed during supersession")
            connection.execute(
                """
                UPDATE jobs
                   SET accepted_output_epoch = ?,
                       accepted_output_note_id = NULL, updated_at = ?
                 WHERE queue_seq >= ? AND accepted_output_epoch = ?
                   AND output_note_id IS NULL AND publication_attempts = 0
                   AND (
                       state IN ('queued','transcribed','answered')
                       OR (state = 'failed' AND retryable = 1)
                   )
                """,
                (
                    new_epoch,
                    timestamp,
                    int(interrupter["queue_seq"]),
                    old_epoch,
                ),
            )
        return True

    def apply_assistant_reset(
        self,
        *,
        reset_note_id: str,
        answer: str,
        now: datetime,
    ) -> bool:
        """Atomically reset safe ambiguity and start a fresh conversation.

        Only publication ambiguity and ``inference_ambiguous`` may be cleared.
        Authorization, account, configuration, plan, and daily-limit gates
        remain administrative boundaries and make this operation a no-op.
        """

        if not answer:
            raise ValueError("reset acknowledgement must not be empty")
        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                "SELECT * FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            assistant_gate = state["assistant_gate"]
            if assistant_gate not in {None, "inference_ambiguous"}:
                return False
            output_gate = state["output_gate"]
            if output_gate not in {None, "create_ambiguous", "update_ambiguous"}:
                return False
            reset = connection.execute(
                """
                SELECT note_id, queue_seq, state, route_kind, model_attempts,
                       publication_attempts, output_note_id,
                       admitted_output_gate, admitted_output_gate_job_note_id,
                       admitted_assistant_gate,
                       admitted_assistant_gate_job_note_id
                  FROM jobs WHERE note_id = ?
                """,
                (reset_note_id,),
            ).fetchone()
            if (
                reset is None
                or reset["state"] != JobState.TRANSCRIBED
                or reset["route_kind"] != RouteKind.RESET
                or int(reset["model_attempts"]) != 0
                or int(reset["publication_attempts"]) != 0
                or reset["output_note_id"] is not None
            ):
                raise RuntimeError("assistant reset request is inconsistent")
            if output_gate is not None and (
                reset["admitted_output_gate"] != output_gate
                or reset["admitted_output_gate_job_note_id"]
                != state["output_gate_job_note_id"]
            ):
                raise RuntimeError("reset was not admitted under this output gate")
            if assistant_gate == "inference_ambiguous" and (
                reset["admitted_assistant_gate"] != assistant_gate
                or reset["admitted_assistant_gate_job_note_id"]
                != state["assistant_gate_job_note_id"]
            ):
                raise RuntimeError("reset was not admitted under this model gate")
            reset_seq = int(reset["queue_seq"])
            old_epoch = int(state["output_epoch"])

            if output_gate is not None:
                gated = connection.execute(
                    """
                    SELECT note_id, queue_seq, state, failure_stage, resume_state,
                           publication_attempts, accepted_output_epoch,
                           accepted_output_note_id, publication_output_mode,
                           output_note_id
                      FROM jobs WHERE note_id = ?
                    """,
                    (state["output_gate_job_note_id"],),
                ).fetchone()
                if (
                    gated is None
                    or int(gated["queue_seq"]) >= reset_seq
                    or gated["state"] != JobState.FAILED
                    or gated["failure_stage"] != "publication"
                    or gated["resume_state"] != JobState.ANSWERED
                    or int(gated["publication_attempts"]) != 1
                    or int(gated["accepted_output_epoch"]) != old_epoch
                    or gated["output_note_id"] is not None
                ):
                    raise RuntimeError("reset publication gate is inconsistent")
                if output_gate == "update_ambiguous" and (
                    state["active_output_note_id"] is None
                    or gated["publication_output_mode"] != "rolling"
                    or gated["accepted_output_note_id"]
                    != state["active_output_note_id"]
                ):
                    raise RuntimeError("reset update target changed")
                connection.execute(
                    """
                    UPDATE jobs SET last_error = ?, retryable = 0,
                        next_retry_at = NULL, manual_retry_allowed = 0,
                        model_attempt_active = 0, updated_at = ?, completed_at = ?
                     WHERE note_id = ? AND state = 'failed'
                       AND failure_stage = 'publication'
                    """,
                    (
                        BEST_EFFORT_RESET_SUPERSEDED_ERROR,
                        timestamp,
                        timestamp,
                        str(gated["note_id"]),
                    ),
                )

            if assistant_gate == "inference_ambiguous":
                model_job = connection.execute(
                    """
                    SELECT note_id, queue_seq, state, failure_stage, resume_state
                      FROM jobs WHERE note_id = ?
                    """,
                    (state["assistant_gate_job_note_id"],),
                ).fetchone()
                if (
                    model_job is None
                    or int(model_job["queue_seq"]) >= reset_seq
                    or model_job["state"] != JobState.FAILED
                    or model_job["failure_stage"] != "model"
                    or model_job["resume_state"] != JobState.TRANSCRIBED
                ):
                    raise RuntimeError("reset model gate is inconsistent")
                connection.execute(
                    """
                    UPDATE jobs SET last_error = ?, retryable = 0,
                        next_retry_at = NULL, manual_retry_allowed = 0,
                        model_attempt_active = 0, updated_at = ?, completed_at = ?
                     WHERE note_id = ? AND state = 'failed'
                       AND failure_stage = 'model'
                    """,
                    (
                        BEST_EFFORT_INFERENCE_RESET_ERROR,
                        timestamp,
                        timestamp,
                        str(model_job["note_id"]),
                    ),
                )

            # Retiring the current known surface makes the acknowledgement a
            # fresh create. An unknown create remains unowned and untouched.
            active_output = state["active_output_note_id"]
            if active_output is not None:
                retired = connection.execute(
                    """
                    UPDATE output_surfaces
                       SET retired_at = ?, retired_reason = 'assistant_reset'
                     WHERE note_id = ? AND epoch = ? AND retired_at IS NULL
                    """,
                    (timestamp, active_output, old_epoch),
                )
                if retired.rowcount != 1:
                    raise RuntimeError("reset output surface changed")

            connection.execute(
                """
                UPDATE jobs SET state = 'failed', failure_stage = 'superseded',
                    resume_state = NULL, retryable = 0,
                    manual_retry_allowed = 0, model_attempt_active = 0,
                    next_retry_at = NULL, last_error = ?, audio_blob = NULL,
                    updated_at = ?, completed_at = ?
                 WHERE queue_seq < ? AND note_id NOT IN (
                       COALESCE(?, ''), COALESCE(?, '')
                   )
                   AND output_note_id IS NULL AND publication_attempts = 0
                   AND (
                       state IN ('queued','transcribed','answered')
                       OR (state = 'failed' AND (
                           retryable = 1 OR manual_retry_allowed = 1
                           OR completed_at IS NULL
                       ))
                   )
                """,
                (
                    BEST_EFFORT_RESET_SUPERSEDED_ERROR,
                    timestamp,
                    timestamp,
                    reset_seq,
                    state["output_gate_job_note_id"],
                    state["assistant_gate_job_note_id"],
                ),
            )

            current_conversation = state["active_conversation_id"]
            if current_conversation is not None:
                connection.execute(
                    "UPDATE conversations SET closed_at = COALESCE(closed_at, ?) "
                    "WHERE id = ?",
                    (timestamp, current_conversation),
                )
            cursor = connection.execute(
                "INSERT INTO conversations(started_at) VALUES (?)", (timestamp,)
            )
            conversation_id = int(cursor.lastrowid)
            new_epoch = old_epoch + 1
            changed = connection.execute(
                """
                UPDATE bridge_state
                   SET active_conversation_id = ?, active_output_note_id = NULL,
                       output_epoch = ?, output_gate = NULL,
                       output_gate_job_note_id = NULL, output_gate_at = NULL,
                       output_gate_readback_attempts = 0,
                       assistant_gate = NULL,
                       assistant_gate_job_note_id = NULL,
                       assistant_gate_at = NULL
                 WHERE singleton = 1 AND output_epoch = ?
                   AND output_gate IS ? AND assistant_gate IS ?
                """,
                (
                    conversation_id,
                    new_epoch,
                    old_epoch,
                    output_gate,
                    assistant_gate,
                ),
            )
            if changed.rowcount != 1:
                raise RuntimeError("recovery gates changed during reset")
            answered = connection.execute(
                """
                UPDATE jobs SET state = 'answered', route_kind = 'reset',
                    transcript = 'Reset', answer = ?, answer_sections = ?,
                    next_section_index = 0, delivered_sections = 0,
                    audio_blob = NULL, conversation_id = ?,
                    source_job_note_id = NULL, source_section_index = NULL,
                    response_id = NULL, accepted_output_epoch = ?,
                    accepted_output_note_id = NULL, failure_stage = NULL,
                    resume_state = NULL, retryable = 0, next_retry_at = NULL,
                    last_error = NULL, ignored_reason = NULL,
                    manual_retry_allowed = 0, model_attempt_active = 0,
                    updated_at = ?, completed_at = NULL
                 WHERE note_id = ? AND state = 'transcribed'
                   AND route_kind = 'reset' AND publication_attempts = 0
                """,
                (
                    answer,
                    json.dumps([answer]),
                    conversation_id,
                    new_epoch,
                    timestamp,
                    reset_note_id,
                ),
            )
            if answered.rowcount != 1:
                raise RuntimeError("reset request changed during recovery")
            connection.execute(
                """
                UPDATE jobs SET accepted_output_epoch = ?,
                    accepted_output_note_id = NULL, updated_at = ?
                 WHERE queue_seq > ? AND accepted_output_epoch = ?
                   AND output_note_id IS NULL AND publication_attempts = 0
                   AND state IN ('queued','transcribed','answered')
                """,
                (new_epoch, timestamp, reset_seq, old_epoch),
            )
        return True

    def resolve_missing_publication(
        self,
        *,
        output_note_id: str,
        maximum_readbacks: int,
        deletion_confirmation_reads: int,
        now: datetime,
    ) -> str | None:
        """Classify a missing read while an update outcome is uncertain.

        Absence after an attempted write is not proof that the write failed:
        Light metadata can temporarily lag the object upload. Only the exact
        durable error emitted before a provider crossed its mutation boundary
        permits normal deletion confirmation. Otherwise the exact output must
        remain absent for a complete consecutive read-back window before its
        identity can be retired. This missing-only window also works when an
        earlier old-content window had already exhausted its shared counter.
        """

        if maximum_readbacks < 1:
            raise ValueError("update read-back limit must be positive")
        if deletion_confirmation_reads < 1:
            raise ValueError("deletion confirmation count must be positive")
        missing_window = max(maximum_readbacks, deletion_confirmation_reads)
        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                "SELECT output_epoch, active_output_note_id, output_gate, "
                "output_gate_job_note_id, output_gate_readback_attempts "
                "FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            if state["output_gate"] != "update_ambiguous":
                return None
            if state["active_output_note_id"] != output_note_id:
                raise RuntimeError("uncertain update output identity changed")
            job = connection.execute(
                """
                SELECT note_id, last_error, resume_state,
                       accepted_output_note_id, publication_output_mode
                  FROM jobs
                 WHERE note_id = ? AND state = 'failed'
                   AND failure_stage = 'publication'
                   AND publication_attempts > 0
                   AND accepted_output_epoch = ?
                """,
                (state["output_gate_job_note_id"], state["output_epoch"]),
            ).fetchone()
            if (
                job is None
                or job["resume_state"] != JobState.ANSWERED
                or job["publication_output_mode"] != "rolling"
            ):
                raise RuntimeError("uncertain missing update state is inconsistent")
            accepted_output_note_id = job["accepted_output_note_id"]
            if accepted_output_note_id is None:
                connection.execute(
                    "UPDATE jobs SET accepted_output_note_id = ?, updated_at = ? "
                    "WHERE note_id = ? AND accepted_output_note_id IS NULL",
                    (output_note_id, timestamp, str(job["note_id"])),
                )
            elif accepted_output_note_id != output_note_id:
                raise RuntimeError("uncertain missing update target changed")
            if job["last_error"] == MISSING_OUTPUT_UPDATE_ERROR:
                return "pre_mutation_missing"

            surface = connection.execute(
                "SELECT missing_observations FROM output_surfaces "
                "WHERE note_id = ? AND retired_at IS NULL",
                (output_note_id,),
            ).fetchone()
            if surface is None:
                raise RuntimeError("uncertain update output surface is missing")
            missing_observations = int(surface["missing_observations"])
            if missing_observations >= missing_window:
                connection.execute(
                    """
                    UPDATE jobs SET last_error = ?, updated_at = ?
                     WHERE note_id = ?
                    """,
                    (
                        MISSING_OUTPUT_READBACK_EXHAUSTED_ERROR,
                        timestamp,
                        str(job["note_id"]),
                    ),
                )
                return "deleted_confirmed"

            attempts = int(state["output_gate_readback_attempts"])
            if attempts < maximum_readbacks:
                recorded = connection.execute(
                    """
                    UPDATE bridge_state
                       SET output_gate_readback_attempts =
                           output_gate_readback_attempts + 1
                     WHERE singleton = 1
                       AND output_gate = 'update_ambiguous'
                       AND output_gate_job_note_id = ?
                       AND output_gate_readback_attempts = ?
                    """,
                    (str(job["note_id"]), attempts),
                )
                if recorded.rowcount != 1:
                    raise RuntimeError("update read-back gate changed during recovery")
            return "pending"

    def register_created_output(
        self,
        *,
        expected_epoch: int,
        note_id: str,
        content_sha256: str,
        now: datetime,
    ) -> bool:
        with self._write() as connection:
            state = connection.execute(
                "SELECT output_epoch, active_output_note_id FROM bridge_state "
                "WHERE singleton = 1"
            ).fetchone()
            if (
                int(state["output_epoch"]) != expected_epoch
                or state["active_output_note_id"] is not None
            ):
                return False
            connection.execute(
                """
                INSERT INTO output_surfaces(
                    note_id, epoch, content_sha256, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (note_id, expected_epoch, content_sha256, _encode_time(now)),
            )
            connection.execute(
                "UPDATE bridge_state SET active_output_note_id = ? "
                "WHERE singleton = 1",
                (note_id,),
            )
        return True

    def update_output_hash(
        self,
        note_id: str,
        *,
        expected_content_sha256: str,
        content_sha256: str,
        now: datetime,
    ) -> bool:
        with self._write() as connection:
            result = connection.execute(
                """
                UPDATE output_surfaces SET content_sha256 = ?,
                    missing_observations = 0, missing_since = NULL
                 WHERE note_id = ? AND content_sha256 = ? AND retired_at IS NULL
                """,
                (content_sha256, note_id, expected_content_sha256),
            )
            return result.rowcount == 1

    def record_output_missing(self, note_id: str, *, now: datetime) -> int:
        """Durably count one missing observation of the active reply surface."""

        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                "SELECT active_output_note_id FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            if state["active_output_note_id"] != note_id:
                return 0
            result = connection.execute(
                """
                UPDATE output_surfaces
                   SET missing_observations = missing_observations + 1,
                       missing_since = COALESCE(missing_since, ?)
                 WHERE note_id = ? AND retired_at IS NULL
                """,
                (timestamp, note_id),
            )
            if result.rowcount != 1:
                return 0
            row = connection.execute(
                "SELECT missing_observations FROM output_surfaces WHERE note_id = ?",
                (note_id,),
            ).fetchone()
            return int(row["missing_observations"])

    def clear_output_missing(self, note_id: str) -> bool:
        """Clear transient absence evidence after the exact note reappears."""

        with self._write() as connection:
            result = connection.execute(
                """
                UPDATE output_surfaces
                   SET missing_observations = 0, missing_since = NULL
                 WHERE note_id = ? AND retired_at IS NULL
                   AND (missing_observations != 0 OR missing_since IS NOT NULL)
                """,
                (note_id,),
            )
            return result.rowcount == 1

    def retire_output(
        self, note_id: str, *, reason: str, now: datetime
    ) -> bool:
        timestamp = _encode_time(now)
        with self._write() as connection:
            state = connection.execute(
                "SELECT output_epoch, active_output_note_id, output_gate, "
                "output_gate_job_note_id FROM bridge_state WHERE singleton = 1"
            ).fetchone()
            if state["active_output_note_id"] != note_id:
                return False
            old_epoch = int(state["output_epoch"])
            new_epoch = old_epoch + 1
            if reason == "deleted_confirmed":
                attempted = connection.execute(
                    """
                    SELECT note_id, state, last_error
                      FROM jobs
                     WHERE accepted_output_epoch = ?
                       AND accepted_output_note_id = ?
                       AND output_note_id IS NULL
                       AND publication_attempts > 0
                       AND publication_output_mode = 'rolling'
                       AND state IN ('answered','failed')
                     ORDER BY queue_seq
                     LIMIT 1
                    """,
                    (old_epoch, note_id),
                ).fetchone()
                if attempted is not None and not (
                    state["output_gate"] == "update_ambiguous"
                    and state["output_gate_job_note_id"] == attempted["note_id"]
                    and attempted["state"] == JobState.FAILED
                    and attempted["last_error"] in {
                        MISSING_OUTPUT_UPDATE_ERROR,
                        MISSING_OUTPUT_READBACK_EXHAUSTED_ERROR,
                    }
                ):
                    # A missing cloud read cannot prove that an attempted
                    # update was unapplied. Keep its identity and gate intact.
                    return False
            connection.execute(
                """
                UPDATE output_surfaces SET retired_at = ?, retired_reason = ?
                 WHERE note_id = ? AND retired_at IS NULL
                """,
                (timestamp, reason, note_id),
            )
            connection.execute(
                """
                UPDATE bridge_state SET active_output_note_id = NULL,
                    output_epoch = output_epoch + 1,
                    output_gate = CASE
                        WHEN output_gate = 'update_ambiguous' THEN NULL
                        ELSE output_gate END,
                    output_gate_job_note_id = CASE
                        WHEN output_gate = 'update_ambiguous' THEN NULL
                        ELSE output_gate_job_note_id END,
                    output_gate_at = CASE
                        WHEN output_gate = 'update_ambiguous' THEN NULL
                        ELSE output_gate_at END,
                    output_gate_readback_attempts = CASE
                        WHEN output_gate = 'update_ambiguous' THEN 0
                        ELSE output_gate_readback_attempts END
                 WHERE singleton = 1
                """
            )
            if reason == "deleted_confirmed":
                # Requests accepted while deletion was still only suspected are
                # safe to carry forward: no create was attempted, and the old
                # identity is now durably absent. This is what lets the next
                # request create a new reply surface while an idle bridge leaves
                # the user's deletion alone.
                connection.execute(
                    """
                    UPDATE jobs
                       SET accepted_output_epoch = ?,
                           accepted_output_note_id = NULL,
                           updated_at = ?
                     WHERE accepted_output_epoch = ?
                       AND output_note_id IS NULL
                       AND publication_attempts = 0
                       AND (
                           state IN ('queued','transcribed','answered')
                           OR (state = 'failed' AND retryable = 1)
                       )
                    """,
                    (new_epoch, timestamp, old_epoch),
                )
                if (
                    state["output_gate"] == "update_ambiguous"
                    and state["output_gate_job_note_id"] is not None
                ):
                    # A missing update target cannot retain an ambiguous write
                    # once the configured read-back window proves the exact
                    # note remains absent. Keep the stored answer, discard only
                    # the obsolete update attempt, and publish it through the
                    # new generation.
                    restored = connection.execute(
                        """
                        UPDATE jobs
                           SET state = 'answered', failure_stage = NULL,
                               resume_state = NULL, retryable = 0,
                               next_retry_at = NULL, last_error = NULL,
                               completed_at = NULL,
                               publication_attempts = 0,
                               accepted_output_epoch = ?,
                               accepted_output_note_id = NULL,
                               updated_at = ?
                         WHERE note_id = ? AND accepted_output_epoch = ?
                           AND output_note_id IS NULL
                           AND state = 'failed'
                           AND failure_stage = 'publication'
                           AND resume_state = 'answered'
                        """,
                        (
                            new_epoch,
                            timestamp,
                            state["output_gate_job_note_id"],
                            old_epoch,
                        ),
                    )
                    if restored.rowcount != 1:
                        raise RuntimeError(
                            "confirmed deletion could not restore its gated answer"
                        )
        return True

    def mark_published(
        self, note_id: str, *, output_note_id: str, now: datetime
    ) -> None:
        timestamp = _encode_time(now)
        with self._write() as connection:
            self._complete_publication(
                connection,
                note_id,
                output_note_id=output_note_id,
                timestamp=timestamp,
            )
            connection.execute(
                """
                UPDATE bridge_state SET output_gate = NULL,
                    output_gate_job_note_id = NULL, output_gate_at = NULL,
                    output_gate_readback_attempts = 0
                 WHERE singleton = 1 AND output_gate_job_note_id = ?
                """,
                (note_id,),
            )

    def redact_closed_conversations(self, *, closed_before: datetime, now: datetime) -> int:
        with self._write() as connection:
            conversations = connection.execute(
                """
                SELECT id FROM conversations
                 WHERE closed_at IS NOT NULL AND closed_at < ?
                   AND redacted_at IS NULL
                   AND NOT EXISTS (
                       SELECT 1 FROM bridge_state AS bridge
                       JOIN jobs AS gated
                         ON gated.note_id = bridge.output_gate_job_note_id
                      WHERE bridge.singleton = 1
                        AND gated.conversation_id = conversations.id
                   )
                """,
                (_encode_time(closed_before),),
            ).fetchall()
            ids = [int(row["id"]) for row in conversations]
            for conversation_id in ids:
                connection.execute(
                    """
                    UPDATE jobs SET transcript = NULL, answer = NULL,
                        answer_sections = '[]', response_id = NULL,
                        publication_content = NULL, publication_sha256 = NULL,
                        publication_title_sha256 = NULL,
                        publication_output_mode = NULL,
                        audio_blob = NULL, updated_at = ?
                     WHERE conversation_id = ?
                    """,
                    (_encode_time(now), conversation_id),
                )
                connection.execute(
                    "UPDATE conversations SET redacted_at = ? WHERE id = ?",
                    (_encode_time(now), conversation_id),
                )
            return len(ids)

    def delete_expired_metadata(self, *, completed_before: datetime) -> int:
        """Delete terminal metadata, but retain every active conversation."""

        with self._write() as connection:
            result = connection.execute(
                """
                DELETE FROM jobs
                 WHERE completed_at IS NOT NULL AND completed_at < ?
                   AND manual_retry_allowed = 0
                   AND note_id != COALESCE(
                       (SELECT output_gate_job_note_id FROM bridge_state
                         WHERE singleton = 1),
                       ''
                   )
                   AND note_id NOT IN (
                       SELECT source_note_id FROM source_cleanups
                        WHERE state != 'confirmed'
                   )
                   AND (
                       conversation_id IS NULL OR conversation_id IN (
                           SELECT id FROM conversations WHERE closed_at IS NOT NULL
                       )
                   )
                """,
                (_encode_time(completed_before),),
            )
            return int(result.rowcount)

    def active_conversation_id(self) -> int | None:
        return self._connection.execute(
            "SELECT active_conversation_id FROM bridge_state WHERE singleton = 1"
        ).fetchone()["active_conversation_id"]

    def _insert_terminal_job(
        self,
        connection: sqlite3.Connection,
        note: RemoteNote,
        *,
        now: datetime,
        audio_sha256: str | None,
        failure_stage: str,
        error: str,
    ) -> None:
        state = connection.execute(
            "SELECT output_epoch, active_output_note_id "
            "FROM bridge_state WHERE singleton = 1"
        ).fetchone()
        sequence = connection.execute(
            "SELECT COALESCE(MAX(queue_seq), 0) + 1 AS value FROM jobs"
        ).fetchone()["value"]
        timestamp = _encode_time(now)
        connection.execute(
            """
            INSERT INTO jobs (
                note_id, queue_seq, state, remote_updated_at, mime_type,
                source_kind, audio_sha256, audio_blob, accepted_output_epoch,
                accepted_output_note_id, failure_stage, retryable,
                last_error, discovered_at, updated_at, completed_at
            ) VALUES (?, ?, 'failed', ?, ?, ?, ?, NULL, ?, ?, ?, 0, ?, ?, ?, ?)
            """,
            (
                note.note_id,
                sequence,
                _encode_time(note.updated_at),
                note.mime_type,
                note.kind.value,
                audio_sha256,
                state["output_epoch"],
                state["active_output_note_id"],
                failure_stage,
                error[:500],
                timestamp,
                timestamp,
                timestamp,
            ),
        )

    @staticmethod
    def _complete_publication(
        connection: sqlite3.Connection,
        note_id: str,
        *,
        output_note_id: str,
        timestamp: str,
    ) -> None:
        job = connection.execute(
            """
            SELECT route_kind, source_job_note_id, source_section_index,
                   publication_sha256, source_kind, audio_sha256, state,
                   notifier_event_sequence
              FROM jobs WHERE note_id = ?
            """,
            (note_id,),
        ).fetchone()
        if job is None:
            raise KeyError(note_id)
        cleanup_policy = connection.execute(
            "SELECT source_cleanup_enabled FROM bridge_state WHERE singleton = 1"
        ).fetchone()
        cleanup_enabled = bool(cleanup_policy["source_cleanup_enabled"])
        if cleanup_enabled:
            source_ledger = connection.execute(
                "SELECT disposition, source_kind, audio_sha256 "
                "FROM request_ledger WHERE note_id = ?",
                (note_id,),
            ).fetchone()
            if (
                note_id == output_note_id
                or source_ledger is None
                or source_ledger["disposition"] != "accepted"
                or source_ledger["source_kind"] != job["source_kind"]
                or source_ledger["audio_sha256"] is None
                or job["audio_sha256"] is None
                or not hmac.compare_digest(
                    str(source_ledger["audio_sha256"]),
                    str(job["audio_sha256"]),
                )
            ):
                raise RuntimeError("published source is not eligible for cleanup")
        ledger = connection.execute(
            "SELECT disposition, source_kind FROM request_ledger WHERE note_id = ?",
            (output_note_id,),
        ).fetchone()
        if ledger is not None and (
            ledger["disposition"] != "bridge_output"
            or ledger["source_kind"] != NoteKind.TEXT
        ):
            raise RuntimeError("output identity conflicts with request ledger")

        if job["route_kind"] == RouteKind.CONTINUE:
            source_note_id = job["source_job_note_id"]
            section_index = job["source_section_index"]
            if source_note_id is not None and section_index is not None:
                source = connection.execute(
                    "SELECT next_section_index FROM jobs WHERE note_id = ?",
                    (source_note_id,),
                ).fetchone()
                if source is None:
                    raise RuntimeError("continuation source is missing")
                current_index = int(source["next_section_index"])
                if current_index < int(section_index):
                    raise RuntimeError("continuation source cursor is inconsistent")
                if current_index == int(section_index):
                    connection.execute(
                        """
                        UPDATE jobs
                           SET next_section_index = ?,
                               delivered_sections = MAX(delivered_sections, ?),
                               updated_at = ?
                         WHERE note_id = ?
                        """,
                        (
                            int(section_index) + 1,
                            int(section_index) + 1,
                            timestamp,
                            source_note_id,
                        ),
                    )

        connection.execute(
            """
            UPDATE jobs SET state = 'published', output_note_id = ?,
                delivered_sections = CASE
                    WHEN answer IS NOT NULL THEN MAX(delivered_sections, 1)
                    ELSE delivered_sections
                END,
                next_section_index = CASE
                    WHEN route_kind = 'question' AND answer IS NOT NULL
                        THEN MAX(next_section_index, 1)
                    ELSE next_section_index
                END,
                failure_stage = NULL, resume_state = NULL, retryable = 0,
                next_retry_at = NULL, last_error = NULL,
                updated_at = ?, completed_at = ?
             WHERE note_id = ?
            """,
            (output_note_id, timestamp, timestamp, note_id),
        )
        if ledger is None:
            connection.execute(
                """
                INSERT INTO request_ledger (
                    note_id, first_seen_at, last_seen_at, remote_updated_at,
                    audio_sha256, disposition, source_kind
                ) VALUES (?, ?, ?, NULL, ?, 'bridge_output', 'text')
                """,
                (output_note_id, timestamp, timestamp, job["publication_sha256"]),
            )
        else:
            connection.execute(
                """
                UPDATE request_ledger
                   SET last_seen_at = ?, audio_sha256 = ?
                 WHERE note_id = ? AND disposition = 'bridge_output'
                   AND source_kind = 'text'
                """,
                (timestamp, job["publication_sha256"], output_note_id),
            )
        if cleanup_enabled:
            existing_cleanup = connection.execute(
                "SELECT source_kind, source_sha256 FROM source_cleanups "
                "WHERE source_note_id = ?",
                (note_id,),
            ).fetchone()
            if existing_cleanup is None:
                connection.execute(
                    """
                    INSERT INTO source_cleanups (
                        source_note_id, source_kind, source_sha256, state,
                        scheduled_at, updated_at
                    ) VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        note_id,
                        job["source_kind"],
                        job["audio_sha256"],
                        timestamp,
                        timestamp,
                    ),
                )
            elif (
                existing_cleanup["source_kind"] != job["source_kind"]
                or not hmac.compare_digest(
                    str(existing_cleanup["source_sha256"]),
                    str(job["audio_sha256"]),
                )
            ):
                raise RuntimeError("source cleanup identity conflicts with publication")
        if job["notifier_event_sequence"] is None:
            advanced = connection.execute(
                """
                UPDATE bridge_state
                   SET notifier_sequence = notifier_sequence + 1
                 WHERE singleton = 1
                   AND notifier_sequence < 9223372036854775807
                """
            )
            if advanced.rowcount != 1:
                raise RuntimeError("notifier sequence cannot advance")
            sequence = connection.execute(
                "SELECT notifier_sequence FROM bridge_state WHERE singleton = 1"
            ).fetchone()["notifier_sequence"]
            recorded = connection.execute(
                """
                UPDATE jobs SET notifier_event_sequence = ?
                 WHERE note_id = ? AND notifier_event_sequence IS NULL
                """,
                (sequence, note_id),
            )
            if recorded.rowcount != 1:
                raise RuntimeError("publication ready event changed concurrently")

    def _ensure_active_conversation(
        self, connection: sqlite3.Connection, now: datetime
    ) -> int:
        current = connection.execute(
            "SELECT active_conversation_id FROM bridge_state WHERE singleton = 1"
        ).fetchone()["active_conversation_id"]
        if current is not None:
            return int(current)
        cursor = connection.execute(
            "INSERT INTO conversations(started_at) VALUES (?)", (_encode_time(now),)
        )
        conversation_id = int(cursor.lastrowid)
        connection.execute(
            "UPDATE bridge_state SET active_conversation_id = ? WHERE singleton = 1",
            (conversation_id,),
        )
        return conversation_id

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobRecord:
        sections = tuple(str(value) for value in json.loads(row["answer_sections"]))
        return JobRecord(
            note_id=row["note_id"],
            queue_seq=int(row["queue_seq"]),
            state=JobState(row["state"]),
            remote_updated_at=_decode_time(row["remote_updated_at"]),
            mime_type=row["mime_type"],
            source_kind=NoteKind(row["source_kind"]),
            audio_sha256=row["audio_sha256"],
            audio=bytes(row["audio_blob"]) if row["audio_blob"] is not None else None,
            route_kind=RouteKind(row["route_kind"]) if row["route_kind"] else None,
            transcript=row["transcript"],
            answer=row["answer"],
            answer_sections=sections,
            next_section_index=int(row["next_section_index"]),
            delivered_sections=int(row["delivered_sections"]),
            source_job_note_id=row["source_job_note_id"],
            source_section_index=row["source_section_index"],
            publication_content=row["publication_content"],
            publication_sha256=row["publication_sha256"],
            publication_title_sha256=row["publication_title_sha256"],
            publication_output_mode=row["publication_output_mode"],
            conversation_id=row["conversation_id"],
            response_id=row["response_id"],
            output_note_id=row["output_note_id"],
            accepted_output_epoch=int(row["accepted_output_epoch"]),
            accepted_output_note_id=row["accepted_output_note_id"],
            failure_stage=row["failure_stage"],
            resume_state=JobState(row["resume_state"]) if row["resume_state"] else None,
            retryable=bool(row["retryable"]),
            manual_retry_allowed=bool(row["manual_retry_allowed"]),
            next_retry_at=_decode_time(row["next_retry_at"]),
            last_error=row["last_error"],
            transcription_attempts=int(row["transcription_attempts"]),
            model_attempts=int(row["model_attempts"]),
            publication_attempts=int(row["publication_attempts"]),
            discovered_at=_decode_time(row["discovered_at"]),
            updated_at=_decode_time(row["updated_at"]),
            completed_at=_decode_time(row["completed_at"]),
        )

    @staticmethod
    def _source_cleanup_from_row(row: sqlite3.Row) -> SourceCleanupRecord:
        return SourceCleanupRecord(
            source_note_id=str(row["source_note_id"]),
            source_kind=NoteKind(row["source_kind"]),
            source_sha256=str(row["source_sha256"]),
            state=SourceCleanupState(row["state"]),
            delete_attempts=int(row["delete_attempts"]),
            precheck_failures=int(row["precheck_failures"]),
            readback_attempts=int(row["readback_attempts"]),
            missing_observations=int(row["missing_observations"]),
            next_retry_at=_decode_time(row["next_retry_at"]),
            scheduled_at=_decode_time(row["scheduled_at"]),
            attempted_at=_decode_time(row["attempted_at"]),
            confirmed_at=_decode_time(row["confirmed_at"]),
            updated_at=_decode_time(row["updated_at"]),
        )
