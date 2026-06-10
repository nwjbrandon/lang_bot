"""Build renderer-agnostic views from questions and session stats."""

from typing import TYPE_CHECKING, Dict

from langbot.models import FieldLine, Question, QuestionView, ResultView

if TYPE_CHECKING:
    from langbot.subjects.base import Subject


def _value(question: Question, fieldname: str) -> str:
    return question.correct.get(fieldname, "").strip() or "-"


def question_view(subject: "Subject", question: Question, stats: Dict[str, int]) -> QuestionView:
    given = [FieldLine(subject.field_labels[name], _value(question, name)) for name in question.mode.prompt_fields]
    return QuestionView(
        title=question.mode.title,
        given=given,
        options=question.options,
        is_review=question.source == "review",
        correct=stats["correct"],
        total=stats["total"],
        progress_done=question.progress_done,
        progress_total=question.progress_total,
        footer=question.mode.footer,
    )


def result_view(subject: "Subject", question: Question, stats: Dict[str, int], selected_index: int) -> ResultView:
    options = question.options
    selected = options[selected_index] if 0 <= selected_index < len(options) else ""
    full_entry = [FieldLine(subject.field_labels[name], _value(question, name)) for name in subject.fields]
    return ResultView(
        is_correct=selected_index == question.correct_index,
        selected_value=selected or "-",
        answer_value=question.answer_value or "-",
        full_entry=full_entry,
        correct=stats["correct"],
        total=stats["total"],
        progress_done=question.progress_done,
        progress_total=question.progress_total,
    )
