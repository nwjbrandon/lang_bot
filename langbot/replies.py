"""Transport-agnostic replies produced by the engine.

Each transport renders these into its own message format (LINE quick replies,
Telegram inline keyboards, ...).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Union

from langbot.models import QuestionView, ResultView


@dataclass
class Message:
    """Plain text reply. ``quick_replies`` are (label, text) shortcut buttons
    that transports may surface (LINE) or ignore (Telegram)."""

    text: str
    quick_replies: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class QuestionReply:
    view: QuestionView


@dataclass
class ResultReply:
    view: ResultView


@dataclass
class ModeMenuReply:
    prompt: str
    options: List[Tuple[str, str]]  # (mode name, label)


Reply = Union[Message, QuestionReply, ResultReply, ModeMenuReply]
