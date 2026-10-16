import pytest

from lightphone_gpt.formatting import split_answer
from lightphone_gpt.models import RouteKind
from lightphone_gpt.routing import route_spoken_transcript, route_transcript


def test_prefix_and_normal_question():
    routed = route_transcript("  ASSISTANT, what is next? ")
    assert routed.kind == RouteKind.QUESTION
    assert routed.text == "what is next?"


def test_prefix_must_be_at_start_and_a_complete_word():
    assert route_transcript("Tell my assistant to call").kind == RouteKind.IGNORED
    assert route_transcript("assistantship news").kind == RouteKind.IGNORED
    assert route_transcript("Assistant").kind == RouteKind.IGNORED


@pytest.mark.parametrize(
    "transcript",
    (
        "Uh, Assistant, what is next?",
        "um assistant what is next?",
    ),
)
def test_spoken_prefix_accepts_one_allowlisted_leading_filler(transcript):
    routed = route_spoken_transcript(transcript)
    assert routed.kind is RouteKind.QUESTION
    assert routed.text in {"what is next?", "What is next?"}


def test_spoken_vocative_reconstructs_observed_leading_question():
    routed = route_spoken_transcript(
        "Do you, Assistant, remember what I asked yesterday?"
    )

    assert routed.kind is RouteKind.QUESTION
    assert routed.text == "do you remember what I asked yesterday?"


@pytest.mark.parametrize(
    "transcript",
    (
        "My assistant should remember this",
        "The assistant should remember this",
        "AI assistant, remember this",
        "Tell my assistant to remember this",
        "This is ordinary. Assistant, remember this",
        "Do your assistant settings remember this",
        "Could you, Assistant, remember this",
        "Do you? Assistant, remember this",
        "Uh, my assistant should remember this",
        "hello there Assistant, remember this",
        "so assistant managers need training",
        "hello assistant users need onboarding",
        "hello. Assistant, remember this",
        "Uh. Assistant, remember this",
        "Um! Assistant, remember this",
        "Uh,                         Assistant, remember this",
        "Do                         you, Assistant, remember this",
        "Uh, um, Assistant, remember this",
        "assistantship news",
    ),
)
def test_spoken_prefix_rejects_unapproved_or_incidental_mentions(transcript):
    assert route_spoken_transcript(transcript).kind is RouteKind.IGNORED


def test_spoken_tolerance_does_not_relax_text_routing():
    assert route_transcript("Um, Assistant, what is next?").kind is RouteKind.IGNORED
    assert route_transcript("Do you, Assistant, remember?").kind is RouteKind.IGNORED


def test_spoken_direct_prefix_keeps_deterministic_commands():
    assert route_spoken_transcript("Assistant, continue").kind is RouteKind.CONTINUE
    assert (
        route_spoken_transcript("Um, Assistant, new conversation").kind
        is RouteKind.NEW_CONVERSATION
    )
    routed = route_spoken_transcript("Do you, Assistant, continue?")
    assert routed.kind is RouteKind.QUESTION
    assert routed.text == "do you continue?"


def test_commands_are_exact_after_normalization():
    assert route_transcript("Assistant. Continue!").kind == RouteKind.CONTINUE
    assert (
        route_transcript("Assistant: new   conversation.").kind
        == RouteKind.NEW_CONVERSATION
    )
    assert (
        route_transcript("Assistant, continue with an example").kind
        == RouteKind.QUESTION
    )


def test_reset_is_an_exact_text_and_spoken_command():
    assert route_transcript("Assistant reset").kind is RouteKind.RESET
    assert route_transcript("Assistant, reset.").kind is RouteKind.RESET
    assert route_spoken_transcript("Um, Assistant, reset").kind is RouteKind.RESET
    assert route_transcript("Assistant, reset the timer").kind is RouteKind.QUESTION


def test_answer_sections_are_persistable_phone_sized_chunks():
    text = " ".join(f"word{i}" for i in range(300))
    sections = split_answer(text)
    assert len(sections) > 1
    assert all(len(section) <= 600 for section in sections)
    assert " ".join(sections).split() == text.split()
