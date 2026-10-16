# Contributing

Thank you for helping improve Lightphone GPT Bridge.

1. Use synthetic notes and mock providers. Never attach real recordings,
   transcripts, credentials, account IDs, host details, databases, or logs.
2. Keep the simulator and mock assistant as the default path. Tests and pull
   requests must not contact Light Cloud or OpenAI.
3. Preserve the single-attempt boundary for uncertain external mutations and
   model calls. A green retry test is not sufficient if it can duplicate a
   remote effect.
4. Identify notes by ID. Titles and filenames never establish ownership.
5. Add focused tests for behavioral changes and update public documentation
   when user-visible behavior changes.

Run before submitting:

```bash
uv sync --frozen --extra dev --extra chatgpt
uv run --frozen pytest
uv run --frozen python -m compileall -q src tests
uv run --frozen python -m build
git diff --check
```

Pull requests should explain the trigger, resulting behavior, validation, and
any remaining compatibility or privacy limits. By contributing, you agree that
your contribution is distributed under GPL-3.0-or-later and that you have the
right to submit it under that license.
