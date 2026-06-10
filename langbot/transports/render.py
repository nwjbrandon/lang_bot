"""Render question/result views into transport-specific text.

``PlainRenderer`` produces plain text (LINE). ``HtmlRenderer`` produces the
emoji + ``<b>`` formatted text used by Telegram (parse_mode=HTML).
"""

import html
from typing import List

from langbot.models import QuestionView, ResultView, accuracy, option_label
from langbot.subjects.base import Strings


class PlainRenderer:
    def _score_line(self, correct: int, total: int, done: int, of: int, s: Strings) -> str:
        return f"{s.score_label}: {correct}/{total} ({accuracy(correct, total):.1f}%)  {s.progress_label}: {done}/{of}"

    def question_text(self, view: QuestionView, s: Strings) -> str:
        lines: List[str] = [f"{s.question_prefix}{view.title}", ""]

        if view.given:
            lines.append(s.given_header)
            for line in view.given:
                lines.append(f"- {line.label}: {line.value}")
        if view.is_review:
            lines.append(s.review_note)

        lines += ["", s.options_header]
        for idx, option in enumerate(view.options):
            lines.append(f"{option_label(idx)}. {option}")

        lines += ["", self._score_line(view.correct, view.total, view.progress_done, view.progress_total, s)]
        if view.footer:
            lines.append(view.footer)
        return "\n".join(lines)

    def result_text(self, view: ResultView, s: Strings) -> str:
        lines: List[str] = [
            s.correct_msg if view.is_correct else s.incorrect_msg,
            "",
            f"{s.my_answer_label}: {view.selected_value}",
            f"{s.correct_answer_label}: {view.answer_value}",
            "",
            s.full_entry_header,
        ]
        for line in view.full_entry:
            lines.append(f"- {line.label}: {line.value}")

        lines += ["", self._score_line(view.correct, view.total, view.progress_done, view.progress_total, s)]
        return "\n".join(lines)


class HtmlRenderer:
    def _score_line(self, correct: int, total: int, done: int, of: int, s: Strings) -> str:
        return f"<b>{html.escape(s.score_label)}:</b> {correct}/{total} ({accuracy(correct, total):.1f}%)    <b>{html.escape(s.progress_label)}:</b> {done}/{of}"

    def question_text(self, view: QuestionView, s: Strings) -> str:
        e = html.escape
        lines: List[str] = [f"📝 <b>{e(view.title)}</b>", "", f"<b>{e(s.given_header)}</b>"]

        if view.is_review:
            lines.append(f"♻️ <b>{e(s.review_note)}</b>")
        for line in view.given:
            lines.append(f"• <b>{e(line.label)}:</b> {e(line.value)}")

        lines += ["", f"<b>{e(s.options_header)}</b>"]
        for idx, option in enumerate(view.options):
            lines.append(f"{option_label(idx)}. {e(option)}")

        lines += ["", self._score_line(view.correct, view.total, view.progress_done, view.progress_total, s)]
        if view.footer:
            lines.append(e(view.footer))
        return "\n".join(lines)

    def result_text(self, view: ResultView, s: Strings) -> str:
        e = html.escape
        header = f"✅ <b>{e(s.correct_msg)}</b>" if view.is_correct else f"❌ <b>{e(s.incorrect_msg)}</b>"
        lines: List[str] = [
            header,
            "",
            f"<b>{e(s.my_answer_label)}:</b> {e(view.selected_value)}",
            f"<b>{e(s.correct_answer_label)}:</b> {e(view.answer_value)}",
            "",
            f"<b>{e(s.full_entry_header)}</b>",
        ]
        for line in view.full_entry:
            lines.append(f"• <b>{e(line.label)}:</b> {e(line.value)}")

        lines += ["", self._score_line(view.correct, view.total, view.progress_done, view.progress_total, s)]
        return "\n".join(lines)
