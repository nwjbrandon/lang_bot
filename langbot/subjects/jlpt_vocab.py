"""JLPT vocabulary subject: kanji / hiragana / meaning / sentence."""

from typing import List, Sequence

from langbot.models import Entry, Mode
from langbot.subjects.base import Strings, Subject, base_command_aliases

MODE_AUTO = "MODE_AUTO"
MODE_TEST_KANGI = "MODE_TEST_KANGI"
MODE_TEST_HIRAGANA = "MODE_TEST_HIRAGANA"
MODE_TEST_MEANING = "MODE_TEST_MEANING"

FIELDS = ["kangi", "hiragana", "meaning", "sentence"]
FIELD_LABELS = {
    "kangi": "Kangi",
    "hiragana": "Hiragana",
    "meaning": "Meaning",
    "sentence": "Sentence",
}


def _mode(name: str, answer_field: str, title: str, label: str) -> Mode:
    return Mode(
        name=name,
        answer_field=answer_field,
        prompt_fields=[f for f in FIELDS if f != answer_field],
        required_fields=[answer_field],
        title=title,
        label=label,
    )


MODES = [
    _mode(MODE_TEST_KANGI, "kangi", "Choose the correct Kangi", "Test Kangi"),
    _mode(MODE_TEST_HIRAGANA, "hiragana", "Choose the correct Hiragana", "Test Hiragana"),
    _mode(MODE_TEST_MEANING, "meaning", "Choose the correct Meaning", "Test Meaning"),
]


def _fallback_order(entries: Sequence[Entry]) -> List[str]:
    """Prefer testing meaning when meanings are present."""
    if any(entry.get("meaning", "").strip() for entry in entries):
        return [MODE_TEST_MEANING, MODE_TEST_KANGI, MODE_TEST_HIRAGANA]
    return [MODE_TEST_KANGI, MODE_TEST_HIRAGANA, MODE_TEST_MEANING]


STRINGS = Strings(
    empty_data="No vocabulary rows were loaded from the CSV.",
    not_enough_data="Not enough quiz data for the current mode. Please add at least 4 distinct values in one test column.",
    welcome=("👋 Welcome to the JLPT vocab quiz bot\n\nCommands:\n/quiz - start a 4-option quiz\n/mode - change quiz mode\n/reset - reset score and progress\n/stats - see your score\n/help - show help\n\nCurrent mode: {mode}"),
    welcome_quick=[("Quiz", "/quiz"), ("Mode", "/mode")],
    help=("Use /quiz to get a question with 4 answer choices.\nUse /mode to switch what column is being tested.\nWrong answers are tested again after a few questions.\nUse /stats to see your current score.\nUse /reset to reset score and progress."),
    stats_header="📊 Score",
    stats_correct="Correct",
    stats_total="Total",
    stats_accuracy="Accuracy",
    reset_done="🔄 Score and progress have been reset. Use /quiz to continue.",
    mode_prompt="Choose a quiz mode:",
    mode_set="✅ Mode set to: {mode}\nUse /quiz to start.",
    invalid_mode="Invalid mode.",
    no_active_question="No active question. Send /quiz to start a new one.",
    invalid_answer="Invalid answer. Send /quiz to try again.",
    unknown_command="Unknown command. Use /help.",
    non_text="Please send a text command. Use /help.",
    next_label="Next",
    next_display="Next question",
    question_prefix="Question: ",
    given_header="Given:",
    options_header="Options:",
    review_note="Review question",
    score_label="Score",
    progress_label="Progress",
    correct_msg="Correct!",
    incorrect_msg="Not quite.",
    my_answer_label="My answer",
    correct_answer_label="Correct answer",
    full_entry_header="Full entry:",
)


def build_subject() -> Subject:
    return Subject(
        name="jlpt_vocab",
        fields=FIELDS,
        field_labels=FIELD_LABELS,
        csv_columns={"kangi": "Kangi", "hiragana": "Hiragana", "meaning": "Meaning", "sentence": "Sentence"},
        csv_required_columns=["kangi", "hiragana"],
        csv_required_fields=[],
        modes=MODES,
        menu_modes=[
            (MODE_AUTO, "Auto"),
            (MODE_TEST_KANGI, "Test Kangi"),
            (MODE_TEST_HIRAGANA, "Test Hiragana"),
            (MODE_TEST_MEANING, "Test Meaning"),
        ],
        default_mode=MODE_AUTO,
        auto_mode=MODE_AUTO,
        fallback_order=_fallback_order,
        command_aliases=base_command_aliases(),
        strings=STRINGS,
    )
