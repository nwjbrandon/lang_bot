"""Quiz-mode resolution and 4-option question generation."""

from typing import TYPE_CHECKING, Dict, List

from langbot.models import Entry, Mode, Question

if TYPE_CHECKING:
    from langbot.subjects.base import Subject

MIN_OPTIONS = 4


class Quiz:
    """Holds the loaded entries for a subject and builds questions from them."""

    def __init__(self, subject: "Subject", entries: List[Entry]):
        if not entries:
            raise ValueError(subject.strings.empty_data)
        self.subject = subject
        self.entries = entries
        self._modes: Dict[str, Mode] = {mode.name: mode for mode in subject.modes}

    def mode(self, name: str) -> Mode:
        return self._modes[name]

    def is_real_mode(self, name: str) -> bool:
        return name in self._modes

    def _distinct_answer_count(self, name: str) -> int:
        answer_field = self._modes[name].answer_field
        values = {entry.get(answer_field, "").strip() for entry in self.entries if entry.get(answer_field, "").strip()}
        return len(values)

    def is_mode_playable(self, name: str) -> bool:
        if name == self.subject.auto_mode:
            return any(self.is_mode_playable(mode_name) for mode_name in self.subject.fallback_order(self.entries))
        if name not in self._modes:
            return False
        return self._distinct_answer_count(name) >= MIN_OPTIONS

    def resolve_mode(self, requested: str) -> str:
        """Return a concrete, playable mode name, falling back when needed."""
        if requested != self.subject.auto_mode and requested in self._modes and self._distinct_answer_count(requested) >= MIN_OPTIONS:
            return requested

        for name in self.subject.fallback_order(self.entries):
            if name in self._modes and self._distinct_answer_count(name) >= MIN_OPTIONS:
                return name

        raise ValueError(self.subject.strings.not_enough_data)

    def mode_entries(self, name: str) -> List[Entry]:
        """Entries usable as the *correct* answer for ``name`` (required fields filled)."""
        mode = self._modes[name]
        return [entry for entry in self.entries if all(entry.get(fieldname, "").strip() for fieldname in mode.required_fields)]

    def _values_to_entries(self, name: str) -> Dict[str, List[Entry]]:
        answer_field = self._modes[name].answer_field
        grouped: Dict[str, List[Entry]] = {}
        for entry in self.entries:
            value = entry.get(answer_field, "").strip()
            if value:
                grouped.setdefault(value, []).append(entry)
        return grouped

    def build_question(self, mode: Mode, correct: Entry, offset: int) -> Question:
        """Build a 4-option question. ``offset`` deterministically rotates options."""
        answer_value = correct.get(mode.answer_field, "").strip()
        distinct_values = list(self._values_to_entries(mode.name).keys())

        if len(distinct_values) < MIN_OPTIONS or answer_value not in distinct_values:
            raise ValueError(self.subject.strings.not_enough_data)

        other_values = [value for value in distinct_values if value != answer_value]
        if len(other_values) < MIN_OPTIONS - 1:
            raise ValueError(self.subject.strings.not_enough_data)

        start = offset % len(other_values)
        distractors = [other_values[(start + i) % len(other_values)] for i in range(MIN_OPTIONS - 1)]

        options = distractors + [answer_value]
        shift = offset % len(options)
        options = options[shift:] + options[:shift]

        return Question(mode=mode, correct=correct, options=options, correct_index=options.index(answer_value))
