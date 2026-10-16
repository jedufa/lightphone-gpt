# Changelog

All notable changes will be recorded here. This project follows semantic
versioning after its first stable release.

## [Unreleased]

- Prepared a simulator-first, source-only open-source candidate.
- Added durable Light Notes intake, local speech transcription adapters,
  sequential conversation processing, guarded answer publication, and optional
  source-note cleanup.
- Added Sign in with ChatGPT authorization and completed-stream-only inference.
- Added `AI READY HH:MM` titles with a configurable IANA timezone.
- Added best-effort continuation past ambiguous publication, bounded recovery
  discovery, and the user-facing `Assistant, reset` command.
- Added deterministic wheel and source-archive generation from the committed
  Git tree.
- Normalized malformed HTTP/0.9-style notifier requests to a bounded HTTP/1.1
  error across Python 3.11, 3.12, and 3.13.
- Replaced the private-review clone placeholder and aligned self-hosted
  authentication guidance with OpenAI's documented VM workflow.
- Added a concise capability matrix, navigation, and conceptual architecture
  artwork to the public README.
- Added public architecture, installation, privacy, security, contribution,
  and third-party attribution documentation.

The project remains alpha. Physical handset behavior, upstream compatibility,
and each operator's account and host configuration require independent
validation.
