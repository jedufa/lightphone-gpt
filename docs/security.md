# Security and privacy model

## Assets

Protect the Light password, ChatGPT OAuth tokens, host identifier, SQLite
database and WAL files, audio recordings, transcripts, answers, speech models,
logs, backups, and any provisioned notification credential. The database and
backups are user-content stores even when encrypted transport is used.

## Threats and controls

| Threat | Main controls | Residual risk |
|---|---|---|
| Old or unrelated notes become requests | Initial metadata baseline; strict `Assistant` routing; note-ID deduplication | Upstream metadata or adapter defects |
| A duplicate title is mistaken for an owned note | IDs, stored hashes, and explicit bindings establish ownership | Light offers fewer conditional-write primitives than an ideal API |
| An uncertain write is repeated | Durable attempt boundary and read-only reconciliation | A delayed remote mutation can still appear later |
| Note text causes code execution | Content is bounded data; no model tools; no shell evaluation | Dependency vulnerabilities remain possible |
| Credentials leak through output | Secret fields use redacted representations; errors and receipts are content-free | Process environment, host compromise, or misconfigured service logs |
| Temporary audio persists | Private temporary directories and bounded cleanup | Filesystem snapshots and crash artifacts may retain data |
| A malicious pull request reaches production | Public candidate has no deployment workflow and CI receives no production secrets | Repository administrators must preserve that separation |
| Dependency compromise | Exact direct pins, upstream attribution, simulator-first operation | Transitive dependencies and future updates require review |

## Logging and diagnostics

Do not enable debug traces around authentication or provider responses on a
real account. Administrative commands may emit opaque IDs and hashes needed
for recovery; treat their terminal and receipt directory as private. Never log
authorization URLs containing an ID-token hint, bearer headers, note bodies,
transcripts, answers, or credential-file contents.

## Network exposure

Health and notification services are designed for numeric loopback binding.
If you publish an endpoint through a reverse proxy, use current TLS, strict
authentication, no request/authorization-header logging, rate limits, and a
firewall that exposes only the intended port. The optional notification
protocol carries no note content but its bearer token is still a secret.

## ChatGPT credentials on a self-hosted VM

OpenAI documents a supported workflow for running an open-source Sign in with
ChatGPT client on a
[self-hosted VM](https://developers.openai.com/siwc/token-sharing-open-source/self-hosted-vms).
Give the VM its own stable host ID, complete the browser flow locally, transfer
the protected credential record over a secure channel, and let the VM own later
refreshes. Do not copy the laptop's host ID onto the VM.

This bridge enforces an owner-only credential directory and file, rejects
symbolic-link credential paths, and serializes rotating-token refreshes. Host
hardening, encrypted backups, secure transfer, account revocation, and access
to the service user remain operator responsibilities. OpenAI notes that
host-specific usage attribution and revocation are not yet available for a
transferred session.

## Data lifecycle

Set independent retention periods for source audio, accepted text, transcripts,
answers, closed conversations, and metadata. Deleting local state does not
delete Light Cloud data, OpenAI data, host backups, or filesystem snapshots.
OpenAI requests set `store=false`; review OpenAI's current data controls for the
remaining service-side handling.

## Reporting vulnerabilities

Use the repository's private GitHub security-advisory feature. Do not open a
public issue containing credentials, note content, private host details, or a
working exploit against a live account.
