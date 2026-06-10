"""Per-user session state and the spaced-repetition quiz controller."""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langbot.models import Entry, ModeState, Question, Review
from langbot.quiz import Quiz

RETRY_MIN_GAP = 3
RETRY_MAX_GAP = 5


@dataclass
class UserSession:
    """All mutable per-user state for one player."""

    mode: str
    stats: Dict[str, int] = field(default_factory=lambda: {"correct": 0, "total": 0})
    mode_states: Dict[str, ModeState] = field(default_factory=dict)
    current_question: Optional[Question] = None

    def reset(self) -> None:
        self.stats = {"correct": 0, "total": 0}
        self.mode_states = {}
        self.current_question = None


def _new_mode_state(rows: List[Entry]) -> ModeState:
    queue = rows[:]
    random.shuffle(queue)
    return ModeState(cycle_queue=queue, cycle_done=0, cycle_total=len(rows), asked_total=0)


class QuizController:
    """Drives question selection and grading against a :class:`Quiz`."""

    def __init__(self, quiz: Quiz):
        self.quiz = quiz

    def _mode_state(self, session: UserSession, mode_name: str, rows: List[Entry]) -> ModeState:
        if mode_name not in session.mode_states:
            session.mode_states[mode_name] = _new_mode_state(rows)
        return session.mode_states[mode_name]

    @staticmethod
    def _refresh_cycle_if_needed(state: ModeState, rows: List[Entry]) -> None:
        if state.cycle_queue:
            return
        queue = rows[:]
        random.shuffle(queue)
        state.cycle_queue = queue
        state.cycle_done = 0
        state.cycle_total = len(rows)

    def _pick_row(self, state: ModeState, rows: List[Entry]) -> Tuple[Entry, str]:
        for idx, review in enumerate(state.reviews):
            if review.due_at <= state.asked_total:
                due = state.reviews.pop(idx)
                state.asked_total += 1
                return due.row, "review"

        self._refresh_cycle_if_needed(state, rows)
        row = state.cycle_queue.pop(0)
        state.cycle_done += 1
        state.asked_total += 1
        return row, "cycle"

    @staticmethod
    def _schedule_retry(state: ModeState, row: Entry) -> None:
        delay = random.randint(RETRY_MIN_GAP, RETRY_MAX_GAP)
        state.reviews.append(Review(row=row, due_at=state.asked_total + delay))

    def next_question(self, session: UserSession) -> Question:
        """Resolve the mode, pick the next row, and build a question."""
        resolved = self.quiz.resolve_mode(session.mode)
        mode = self.quiz.mode(resolved)
        rows = self.quiz.mode_entries(resolved)

        state = self._mode_state(session, resolved, rows)
        row, source = self._pick_row(state, rows)
        question = self.quiz.build_question(mode, row, state.asked_total)
        question.source = source
        question.progress_done = state.cycle_done
        question.progress_total = state.cycle_total

        session.mode = resolved
        session.current_question = question
        return question

    def grade(self, session: UserSession, selected_index: int) -> bool:
        """Score the current question and schedule a retry on a wrong answer."""
        question = session.current_question
        assert question is not None  # callers guard for an active question

        session.stats["total"] += 1
        is_correct = selected_index == question.correct_index

        if is_correct:
            session.stats["correct"] += 1
        else:
            rows = self.quiz.mode_entries(question.mode.name)
            state = self._mode_state(session, question.mode.name, rows)
            self._schedule_retry(state, question.correct)

        return is_correct
