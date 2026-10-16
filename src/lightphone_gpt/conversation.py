"""Bounded, local conversation-context construction."""

from __future__ import annotations

from collections.abc import Sequence

from .models import JobRecord, PromptMessage
from .store import Store


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise voice assistant replying on a small e-ink phone. "
    "Answer directly in plain text. Prefer short paragraphs and avoid tables."
)


def build_prompt(
    store: Store,
    job: JobRecord,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_prior_turns: int = 10,
    max_context_chars: int = 24_000,
) -> list[PromptMessage]:
    if job.conversation_id is None or not job.transcript:
        raise ValueError("a model turn needs a conversation and transcript")
    if max_prior_turns < 0 or max_context_chars < 0:
        raise ValueError("context limits cannot be negative")

    turns = store.conversation_turns(
        job.conversation_id, before_queue_seq=job.queue_seq
    )
    chosen: list[tuple[str, str]] = []
    used = len(job.transcript)
    for question, answer in reversed(turns[-max_prior_turns:]):
        cost = len(question) + len(answer)
        if used + cost > max_context_chars:
            break
        chosen.append((question, answer))
        used += cost
    chosen.reverse()

    messages = [PromptMessage("system", system_prompt)]
    for question, answer in chosen:
        messages.append(PromptMessage("user", question))
        messages.append(PromptMessage("assistant", answer))
    messages.append(PromptMessage("user", job.transcript))
    return messages
