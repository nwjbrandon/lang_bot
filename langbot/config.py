"""Environment configuration helpers and shared app wiring."""

import logging
import os

from dotenv import load_dotenv

from langbot.engine import BotEngine
from langbot.loader import load_entries
from langbot.quiz import Quiz
from langbot.subjects.base import Subject

logger = logging.getLogger(__name__)


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def require_env(name: str) -> str:
    value = get_env(name)
    if not value:
        raise RuntimeError(f"Missing {name} environment variable.")
    return value


def build_engine(subject: Subject, csv_path: str, quiz_mode: str = "") -> BotEngine:
    """Load CSV data for ``subject`` and wire up a :class:`BotEngine`.

    ``quiz_mode`` (from ``QUIZ_MODE``), when set, overrides the subject's
    default starting mode after validation.
    """
    if quiz_mode:
        if quiz_mode not in subject.valid_mode_names:
            raise RuntimeError(f"QUIZ_MODE must be one of: {', '.join(sorted(subject.valid_mode_names))}")
        subject.default_mode = quiz_mode

    entries = load_entries(csv_path, subject.csv_columns, subject.csv_required_columns, subject.csv_required_fields)
    quiz = Quiz(subject, entries)
    logger.info("Loaded %d entries for subject '%s'", len(entries), subject.name)
    return BotEngine(quiz)


def configure_logging() -> None:
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)


def load_env_file() -> None:
    load_dotenv()
