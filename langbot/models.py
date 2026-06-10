"""Dataclasses shared across the quiz framework."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# An entry is a single CSV row exposed as ``field name -> value``.
Entry = Dict[str, str]


@dataclass(frozen=True)
class Mode:
    """A way of quizzing one field of an entry.

    ``answer_field`` is what the player picks; ``prompt_fields`` are shown as the
    "given" hint. ``required_fields`` must all be non-empty for an entry to be
    usable as the correct answer in this mode.
    """

    name: str
    answer_field: str
    prompt_fields: List[str]
    required_fields: List[str]
    title: str
    label: str
    footer: Optional[str] = None


@dataclass
class Question:
    """A concrete 4-option question generated for a single entry."""

    mode: Mode
    correct: Entry
    options: List[str]
    correct_index: int
    source: str = "cycle"  # "cycle" (fresh) or "review" (spaced-repetition)
    progress_done: int = 0
    progress_total: int = 0

    @property
    def answer_value(self) -> str:
        return self.correct.get(self.mode.answer_field, "").strip()


@dataclass(frozen=True)
class FieldLine:
    """A label/value pair for display."""

    label: str
    value: str


@dataclass
class QuestionView:
    """Renderer-agnostic content of a question message."""

    title: str
    given: List[FieldLine]
    options: List[str]
    is_review: bool
    correct: int
    total: int
    progress_done: int
    progress_total: int
    footer: Optional[str] = None


@dataclass
class ResultView:
    """Renderer-agnostic content of an answer-result message."""

    is_correct: bool
    selected_value: str
    answer_value: str
    full_entry: List[FieldLine]
    correct: int
    total: int
    progress_done: int
    progress_total: int


def option_label(index: int) -> str:
    """Map 0, 1, 2, 3 -> 'A', 'B', 'C', 'D'."""
    return chr(ord("A") + index)


def accuracy(correct: int, total: int) -> float:
    return (correct / total * 100) if total else 0.0


@dataclass
class ModeState:
    """Spaced-repetition state for one mode within a user session."""

    cycle_queue: List[Entry]
    cycle_done: int
    cycle_total: int
    asked_total: int
    reviews: List["Review"] = field(default_factory=list)


@dataclass
class Review:
    row: Entry
    due_at: int
