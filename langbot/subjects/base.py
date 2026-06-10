"""Subject definition: the content + localized strings a quiz needs."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from langbot.models import Entry, Mode


@dataclass(frozen=True)
class Strings:
    """All user-facing text for one subject, in that subject's language."""

    empty_data: str
    not_enough_data: str

    welcome: str  # uses ``{mode}``
    welcome_quick: List[Tuple[str, str]]  # (label, message text)
    help: str

    stats_header: str
    stats_correct: str
    stats_total: str
    stats_accuracy: str

    reset_done: str
    mode_prompt: str
    mode_set: str  # uses ``{mode}``
    invalid_mode: str
    no_active_question: str
    invalid_answer: str
    unknown_command: str
    non_text: str

    next_label: str
    next_display: str

    question_prefix: str
    given_header: str
    options_header: str
    review_note: str
    score_label: str
    progress_label: str

    correct_msg: str
    incorrect_msg: str
    my_answer_label: str
    correct_answer_label: str
    full_entry_header: str


@dataclass
class Subject:
    """Everything a transport-agnostic quiz needs for one language/content set."""

    name: str
    fields: List[str]  # canonical order, used for the "full entry" view
    field_labels: Dict[str, str]

    csv_columns: Dict[str, str]  # field -> expected CSV header
    csv_required_columns: List[str]
    csv_required_fields: List[str]

    modes: List[Mode]  # real (selectable, answerable) modes
    menu_modes: List[Tuple[str, str]]  # (name, label) shown in the mode menu, may include auto
    default_mode: str
    auto_mode: Optional[str]
    fallback_order: Callable[[Sequence[Entry]], List[str]]

    command_aliases: Dict[str, set]
    strings: Strings

    @property
    def valid_mode_names(self) -> set:
        return {name for name, _ in self.menu_modes}

    def mode_label(self, name: str) -> str:
        for menu_name, label in self.menu_modes:
            if menu_name == name:
                return label
        return name


# Shared command aliases (slash + bare word). Subjects can extend with localized aliases.
def base_command_aliases() -> Dict[str, set]:
    return {
        "start": {"/start", "start"},
        "help": {"/help", "help"},
        "quiz": {"/quiz", "quiz"},
        "stats": {"/stats", "stats"},
        "reset": {"/reset", "reset"},
        "mode": {"/mode", "mode"},
    }
