"""Transport-agnostic command and action dispatch."""

from typing import List, Optional

from langbot.models import accuracy
from langbot.present import question_view, result_view
from langbot.quiz import Quiz
from langbot.replies import Message, ModeMenuReply, QuestionReply, Reply, ResultReply
from langbot.session import QuizController, UserSession

# Commands (already-parsed intent). Transports map their inputs onto these.
CMD_START = "start"
CMD_HELP = "help"
CMD_QUIZ = "quiz"
CMD_STATS = "stats"
CMD_RESET = "reset"
CMD_MODE = "mode"
COMMANDS = [CMD_START, CMD_HELP, CMD_QUIZ, CMD_STATS, CMD_RESET, CMD_MODE]

# Action callback payloads (button presses).
ACTION_ANSWER_PREFIX = "answer:"
ACTION_NEXT = "next"
ACTION_MODE_PREFIX = "mode:"


class BotEngine:
    """Maps parsed commands and button actions to a list of replies."""

    def __init__(self, quiz: Quiz):
        self.quiz = quiz
        self.subject = quiz.subject
        self.strings = quiz.subject.strings
        self.controller = QuizController(quiz)

    def new_session(self) -> UserSession:
        return UserSession(mode=self.subject.default_mode)

    def parse_command(self, text: str) -> Optional[str]:
        normalized = text.strip().lower()
        for command, aliases in self.subject.command_aliases.items():
            if normalized in aliases:
                return command
        return None

    def command(self, session: UserSession, command: str) -> List[Reply]:
        if command == CMD_START:
            text = self.strings.welcome.format(mode=self._current_mode_label(session))
            return [Message(text, quick_replies=self.strings.welcome_quick)]

        if command == CMD_HELP:
            return [Message(self.strings.help)]

        if command == CMD_QUIZ:
            return self._new_question(session)

        if command == CMD_STATS:
            return [Message(self._stats_text(session))]

        if command == CMD_RESET:
            session.reset()
            return [Message(self.strings.reset_done)]

        if command == CMD_MODE:
            return [ModeMenuReply(self.strings.mode_prompt, list(self.subject.menu_modes))]

        return [Message(self.strings.unknown_command)]

    def action(self, session: UserSession, data: str) -> List[Reply]:
        if data.startswith(ACTION_MODE_PREFIX):
            return self._select_mode(session, data.split(":", 1)[1])

        if data == ACTION_NEXT:
            return self._new_question(session)

        if not data.startswith(ACTION_ANSWER_PREFIX):
            return []

        question = session.current_question
        if not question:
            return [Message(self.strings.no_active_question)]

        try:
            selected_index = int(data.split(":", 1)[1])
        except ValueError:
            return [Message(self.strings.invalid_answer)]

        self.controller.grade(session, selected_index)
        return [ResultReply(result_view(self.subject, question, session.stats, selected_index))]

    # -- helpers ---------------------------------------------------------

    def _new_question(self, session: UserSession) -> List[Reply]:
        try:
            question = self.controller.next_question(session)
        except ValueError:
            return [Message(self.strings.not_enough_data)]
        return [QuestionReply(question_view(self.subject, question, session.stats))]

    def _select_mode(self, session: UserSession, name: str) -> List[Reply]:
        if name not in self.subject.valid_mode_names:
            return [Message(self.strings.invalid_mode)]
        session.mode = name
        try:
            resolved = self.quiz.resolve_mode(name)
        except ValueError:
            return [Message(self.strings.not_enough_data)]
        return [Message(self.strings.mode_set.format(mode=self.subject.mode_label(resolved)))]

    def _current_mode_label(self, session: UserSession) -> str:
        try:
            return self.subject.mode_label(self.quiz.resolve_mode(session.mode))
        except ValueError:
            return self.subject.mode_label(session.mode)

    def _stats_text(self, session: UserSession) -> str:
        stats = session.stats
        s = self.strings
        return f"{s.stats_header}\n{s.stats_correct}: {stats['correct']}\n{s.stats_total}: {stats['total']}\n{s.stats_accuracy}: {accuracy(stats['correct'], stats['total']):.1f}%"
