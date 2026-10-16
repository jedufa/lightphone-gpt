# Architecture

## Components

The service is assembled from replaceable provider interfaces:

- A Light provider lists notes, reads content, and creates, updates, or deletes
  notes. The simulator and the unofficial Garado-backed provider implement the
  same contract.
- A transcriber turns bounded audio bytes into text. Simulation, whisper.cpp,
  and faster-whisper adapters are available.
- An assistant accepts a bounded local conversation and returns one completed
  answer. The mock and Sign in with ChatGPT adapters share this contract.
- `Store` owns durable SQLite workflow state. `Worker` applies state
  transitions and calls providers. `service.py` assembles and schedules them.

The engine does not branch on whether Light is simulated or real. This keeps
the simulator useful for regression testing when the unofficial API changes.

## Durable workflow

One worker processes conversational turns sequentially:

1. Poll note metadata and deduplicate by Light note ID.
2. Read and validate bounded content.
3. Transcribe audio locally when necessary and route the command.
4. Persist the accepted transcript.
5. Build bounded local conversation context and call the assistant.
6. Persist the complete answer and its sections.
7. Publish to the rolling output note and verify the observed result.
8. Advance conversation and notification state only after publication is
   durable.
9. Optionally delete the exact accepted source note through a separate guarded
   cleanup workflow.

Inference and publication are separate transaction boundaries. A publication
retry reuses the persisted answer. An ambiguous assistant result pauses the
turn because blindly retrying could produce conflicting answers.

## Identity and ownership

Light note IDs identify inputs and outputs. Titles and filenames are display
metadata and may repeat. The bridge mutates or deletes only identities already
bound in its durable state or in a narrowly scoped operator receipt. A title
such as `AI READY` alone never authorizes a write.

A fresh production database first records the IDs of existing notes as a
baseline. This prevents old personal notes from becoming requests after setup.

## Failure handling

Safe pre-result failures use persisted bounded backoff. Calls with an uncertain
remote outcome are not automatically repeated. An uncertain update is checked
with read-only observations; recovery either confirms the desired value,
retires a missing output surface, or leaves a durable gate for operator review.
Source deletion uses the same single-attempt principle and stable absence
observations.

Polling continues to admit inputs while an output publication or model result
is ambiguous. A later recognized request can atomically abandon the affected
delivery and continue on a fresh output generation without repeating its
external call. When the oldest turn is waiting for a safe retry or has exhausted
one, the worker performs bounded local routing of later inputs so an exact
`Assistant, reset` can clear that conversational stall. The reset command does
not clear authorization, account, configuration, or daily-limit gates.

SQLite changes that establish an attempt boundary happen before the external
mutation. On restart, the worker reconciles durable state rather than assuming
the call failed.

## Conversation and retention

Published turns enter the active conversation. `new conversation` closes that
context. `continue` reveals already stored sections and does not make a model
call. `reset` starts fresh context and acknowledges the reset without a model
call. Unpublished, superseded, or abandoned turns do not become model context.

Rolling answer titles include the local completion minute as
`AI READY HH:MM`. The configured IANA timezone controls that display value;
the answer body retains its UTC completion timestamp.

Retention separates audio, transcripts/answers, closed conversation content,
and longer-lived metadata. SQLite secure deletion and WAL checkpointing reduce
local residue but cannot erase filesystem snapshots, backups, Light Cloud, or
third-party retention.

## Trust boundaries

```text
Handset <-> Light Cloud <-> bridge host <-> OpenAI
                            |    |
                            |    +-- protected credentials
                            +------- SQLite, audio, models, logs
```

The bridge trusts note content only as bounded user input. It never exposes
tools to the model and never executes model or note text. Health and optional
notification listeners bind to numeric loopback by default. A reverse proxy or
service manager is a deployment concern and is not included in this public
candidate.
