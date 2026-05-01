import html
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODE_AUTO = "MODE_AUTO"
MODE_TEST_KANGI = "MODE_TEST_KANGI"
MODE_TEST_HIRAGANA = "MODE_TEST_HIRAGANA"
MODE_TEST_MEANING = "MODE_TEST_MEANING"

MODE_OPTIONS = [
    MODE_AUTO,
    MODE_TEST_KANGI,
    MODE_TEST_HIRAGANA,
    MODE_TEST_MEANING,
]

VALID_MODES = set(MODE_OPTIONS)

CALLBACK_ANSWER_PREFIX = "answer:"
CALLBACK_NEXT = "next"
CALLBACK_MODE_PREFIX = "mode:"

RETRY_MIN_GAP = 3
RETRY_MAX_GAP = 5


@dataclass
class QuizRow:
    kangi: str
    hiragana: str
    meaning: str = ""
    sentence: str = ""


class QuizBotData:
    def __init__(self, rows: List[QuizRow], default_mode: str = MODE_AUTO):
        if not rows:
            raise ValueError("No vocabulary rows were loaded from the CSV.")
        self.rows = rows
        self.default_mode = default_mode

    def has_meaning(self) -> bool:
        return any(r.meaning.strip() for r in self.rows)

    def _mode_priority(self) -> List[str]:
        # Keep meaning first when available, then fall back to other columns.
        if self.has_meaning():
            return [
                MODE_TEST_MEANING,
                MODE_TEST_KANGI,
                MODE_TEST_HIRAGANA,
            ]
        return [
            MODE_TEST_KANGI,
            MODE_TEST_HIRAGANA,
            MODE_TEST_MEANING,
        ]

    def _distinct_answer_values(self, mode: str) -> List[str]:
        answer_key = self._get_mode_config(mode)["answer_key"]
        values = {getattr(row, answer_key).strip() for row in self.rows if getattr(row, answer_key).strip()}
        return list(values)

    def is_mode_playable(self, mode: str) -> bool:
        if mode == MODE_AUTO:
            return any(self.is_mode_playable(m) for m in self._mode_priority())
        return len(self._distinct_answer_values(mode)) >= 4

    def pick_fallback_mode(self) -> Optional[str]:
        for mode in self._mode_priority():
            if self.is_mode_playable(mode):
                return mode
        return None

    def resolve_mode(self, requested_mode: str) -> str:
        if requested_mode != MODE_AUTO and self.is_mode_playable(requested_mode):
            return requested_mode

        fallback = self.pick_fallback_mode()
        if fallback:
            return fallback

        if requested_mode == MODE_AUTO:
            raise ValueError("No quiz mode has at least 4 valid distinct rows.")
        raise ValueError(f"Need at least 4 valid distinct rows for mode: {requested_mode}")

    def _get_mode_config(self, mode: str) -> Dict[str, str]:
        if mode == MODE_TEST_KANGI:
            return {
                "answer_key": "kangi",
                "title": "Choose the correct Kangi",
            }
        if mode == MODE_TEST_HIRAGANA:
            return {
                "answer_key": "hiragana",
                "title": "Choose the correct Hiragana",
            }
        if mode == MODE_TEST_MEANING:
            return {
                "answer_key": "meaning",
                "title": "Choose the correct Meaning",
            }
        raise ValueError(f"Unsupported mode: {mode}")

    def _rows_by_answer_value(self, mode: str) -> Dict[str, List[QuizRow]]:
        config = self._get_mode_config(mode)
        answer_key = config["answer_key"]

        value_to_rows: Dict[str, List[QuizRow]] = {}
        for row in self.rows:
            value = getattr(row, answer_key).strip()
            if not value:
                continue
            value_to_rows.setdefault(value, []).append(row)
        return value_to_rows

    def get_mode_rows(self, mode: str) -> List[QuizRow]:
        answer_key = self._get_mode_config(mode)["answer_key"]
        return [row for row in self.rows if getattr(row, answer_key).strip()]

    def make_question_for_row(self, mode: str, correct: QuizRow, offset: int) -> Dict[str, Any]:
        config = self._get_mode_config(mode)
        answer_key = config["answer_key"]
        value_to_rows = self._rows_by_answer_value(mode)
        answer_value = getattr(correct, answer_key).strip()

        distinct_values = list(value_to_rows.keys())
        if len(distinct_values) < 4:
            raise ValueError(f"Need at least 4 valid distinct rows for mode: {mode}")
        if answer_value not in value_to_rows:
            raise ValueError(f"Answer value is not available for mode: {mode}")

        other_values = [value for value in distinct_values if value != answer_value]
        if len(other_values) < 3:
            raise ValueError(f"Need at least 3 distractors for mode: {mode}")

        start = offset % len(other_values)
        distractors = [other_values[(start + i) % len(other_values)] for i in range(3)]

        options = distractors + [answer_value]
        shift = offset % len(options)
        options = options[shift:] + options[:shift]
        correct_index = options.index(answer_value)

        return {
            "mode": mode,
            "answer_key": answer_key,
            "correct": correct,
            "question_title": config["title"],
            "options": options,
            "correct_index": correct_index,
            "answer_value": answer_value,
        }


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _resolve_csv_files(csv_source: str) -> List[Path]:
    source = Path(csv_source).expanduser()

    if source.is_file():
        return [source]

    if source.is_dir():
        csv_files = sorted(path for path in source.glob("*.csv") if path.is_file())
        if csv_files:
            return csv_files
        raise ValueError(f"No CSV files found in directory: {source}")

    raise ValueError(f"CSV_PATH must be a CSV file or directory: {source}")


def load_rows(csv_path: str) -> List[QuizRow]:
    csv_files = _resolve_csv_files(csv_path)
    df = pd.concat((pd.read_csv(path) for path in csv_files), ignore_index=True)
    normalized = {col.strip().lower(): col for col in df.columns}

    kangi_col = normalized.get("kangi")
    hiragana_col = normalized.get("hiragana")
    meaning_col = normalized.get("meaning")
    sentence_col = normalized.get("sentence")

    if not kangi_col or not hiragana_col:
        raise ValueError("CSV must contain at least 'Kangi' and 'Hiragana' columns.")

    rows: List[QuizRow] = []
    for _, row in df.iterrows():
        kangi = _clean(row.get(kangi_col))
        hiragana = _clean(row.get(hiragana_col))
        meaning = _clean(row.get(meaning_col)) if meaning_col else ""
        sentence = _clean(row.get(sentence_col)) if sentence_col else ""

        if not kangi and not hiragana and not meaning and not sentence:
            continue

        rows.append(
            QuizRow(
                kangi=kangi,
                hiragana=hiragana,
                meaning=meaning,
                sentence=sentence,
            )
        )

    return rows


def get_user_mode(context: ContextTypes.DEFAULT_TYPE, default_mode: str) -> str:
    mode = context.user_data.get("mode", default_mode)
    return mode if mode in VALID_MODES else default_mode


def get_stats(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, int]:
    if "stats" not in context.user_data:
        context.user_data["stats"] = {"correct": 0, "total": 0}
    return context.user_data["stats"]


def get_mode_states(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Dict[str, Any]]:
    if "mode_states" not in context.user_data:
        context.user_data["mode_states"] = {}
    return context.user_data["mode_states"]


def _new_mode_state(rows: List[QuizRow]) -> Dict[str, Any]:
    queue = rows[:]
    random.shuffle(queue)
    return {
        "cycle_queue": queue,
        "cycle_done": 0,
        "cycle_total": len(rows),
        "asked_total": 0,
        "reviews": [],
    }


def get_or_create_mode_state(context: ContextTypes.DEFAULT_TYPE, mode: str, rows: List[QuizRow]) -> Dict[str, Any]:
    mode_states = get_mode_states(context)
    if mode not in mode_states:
        mode_states[mode] = _new_mode_state(rows)
    return mode_states[mode]


def _refresh_cycle_if_needed(state: Dict[str, Any], rows: List[QuizRow]) -> None:
    if state["cycle_queue"]:
        return
    queue = rows[:]
    random.shuffle(queue)
    state["cycle_queue"] = queue
    state["cycle_done"] = 0
    state["cycle_total"] = len(rows)


def pick_next_question_row(state: Dict[str, Any], rows: List[QuizRow]) -> tuple[QuizRow, str]:
    reviews: List[Dict[str, Any]] = state["reviews"]
    for idx, item in enumerate(reviews):
        if item["due_at"] <= state["asked_total"]:
            due = reviews.pop(idx)
            state["asked_total"] += 1
            return due["row"], "review"

    _refresh_cycle_if_needed(state, rows)
    question_row = state["cycle_queue"].pop(0)
    state["cycle_done"] += 1
    state["asked_total"] += 1
    return question_row, "cycle"


def schedule_retry(state: Dict[str, Any], question_row: QuizRow) -> None:
    delay = random.randint(RETRY_MIN_GAP, RETRY_MAX_GAP)
    state["reviews"].append({"row": question_row, "due_at": state["asked_total"] + delay})


def esc(text: str) -> str:
    return html.escape(text or "")


def option_label(idx: int) -> str:
    return chr(ord("A") + idx)


def build_question_text(question: Dict[str, Any], stats: Dict[str, int]) -> str:
    correct: QuizRow = question["correct"]
    answer_key: str = question["answer_key"]
    accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0.0

    field_map = {
        "kangi": correct.kangi,
        "hiragana": correct.hiragana,
        "meaning": correct.meaning,
        "sentence": correct.sentence,
    }

    display_order = ["kangi", "hiragana", "meaning", "sentence"]
    display_labels = {
        "kangi": "Kangi",
        "hiragana": "Hiragana",
        "meaning": "Meaning",
        "sentence": "Sentence",
    }

    lines = [
        f"📝 <b>{esc(question['question_title'])}</b>",
        "",
        "<b>Given:</b>",
    ]

    if question.get("source") == "review":
        lines.append("♻️ <b>Review question</b>")

    for key in display_order:
        if key == answer_key:
            continue
        value = field_map[key].strip() or "-"
        lines.append(f"• <b>{esc(display_labels[key])}:</b> {esc(value)}")

    lines.extend(["", "<b>Options:</b>"])
    for idx, option in enumerate(question["options"]):
        lines.append(f"{option_label(idx)}. {esc(option)}")

    lines.extend(
        [
            "",
            f"<b>Score:</b> {stats['correct']}/{stats['total']} ({accuracy:.1f}%)    <b>Progress:</b> {question['progress_done']}/{question['progress_total']}",
        ]
    )

    return "\n".join(lines)


def build_question_keyboard(options: List[str]) -> InlineKeyboardMarkup:
    keyboard = []
    for idx, _ in enumerate(options):
        keyboard.append([InlineKeyboardButton(option_label(idx), callback_data=f"{CALLBACK_ANSWER_PREFIX}{idx}")])
    return InlineKeyboardMarkup(keyboard)


def build_mode_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(mode, callback_data=f"{CALLBACK_MODE_PREFIX}{mode}")] for mode in MODE_OPTIONS])


async def send_new_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    bot_data: QuizBotData = context.bot_data["quiz_data"]
    requested_mode = get_user_mode(context, bot_data.default_mode)

    try:
        resolved_mode = bot_data.resolve_mode(requested_mode)
        mode_rows = bot_data.get_mode_rows(resolved_mode)
        state = get_or_create_mode_state(context, resolved_mode, mode_rows)
        question_row, source = pick_next_question_row(state, mode_rows)
        question = bot_data.make_question_for_row(resolved_mode, question_row, state["asked_total"])
        question["source"] = source
        question["progress_done"] = state["cycle_done"]
        question["progress_total"] = state["cycle_total"]
    except ValueError:
        message = "Not enough quiz data for the current mode. Please add at least 4 distinct values in one test column."
        if update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text(message)
        else:
            chat = update.effective_chat
            if not chat:
                return
            await context.bot.send_message(chat_id=chat.id, text=message)
        return

    if resolved_mode != requested_mode:
        context.user_data["mode"] = resolved_mode

    if question["mode"] != requested_mode and requested_mode != MODE_AUTO:
        context.user_data["mode"] = question["mode"]
    context.user_data["current_question"] = question

    text = build_question_text(question, get_stats(context))
    keyboard = build_question_keyboard(question["options"])

    if update.callback_query and update.callback_query.message:
        await update.callback_query.message.reply_text(
            text=text,
            reply_markup=keyboard,
            parse_mode="HTML",
        )
    else:
        chat = update.effective_chat
        if not chat:
            return
        await context.bot.send_message(
            chat_id=chat.id,
            text=text,
            reply_markup=keyboard,
            parse_mode="HTML",
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_data: QuizBotData = context.bot_data["quiz_data"]
    resolved = bot_data.resolve_mode(get_user_mode(context, bot_data.default_mode))
    text = f"👋 Welcome to the JLPT vocab quiz bot\n\nCommands:\n/quiz - start a 4-option quiz\n/mode - change quiz mode\n/reset - reset score and progress\n/stats - see your score\n/help - show help\n\nCurrent mode: {resolved}"
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Use /quiz to get a question with 4 answer choices.\nUse /mode to switch what column is being tested.\nWrong answers are tested again after a few questions.\nUse /stats to see your current score.\nUse /reset to reset score and progress."
    )


async def quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_new_question(update, context)


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s = get_stats(context)
    accuracy = (s["correct"] / s["total"] * 100) if s["total"] else 0.0
    await update.message.reply_text(f"📊 Score\nCorrect: {s['correct']}\nTotal: {s['total']}\nAccuracy: {accuracy:.1f}%")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["stats"] = {"correct": 0, "total": 0}
    context.user_data["mode_states"] = {}
    context.user_data.pop("current_question", None)
    await update.message.reply_text("🔄 Score and progress have been reset. Use /quiz to continue.")


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Choose a quiz mode:",
        reply_markup=build_mode_keyboard(),
    )


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return

    await query.answer()
    data = query.data or ""

    if data.startswith(CALLBACK_MODE_PREFIX):
        mode = data.split(":", 1)[1]
        if mode in VALID_MODES:
            context.user_data["mode"] = mode
            bot_data: QuizBotData = context.bot_data["quiz_data"]
            resolved = bot_data.resolve_mode(mode)
            await query.edit_message_text(f"✅ Mode set to: {resolved}\nUse /quiz to start.")
        return

    if data == CALLBACK_NEXT:
        await send_new_question(update, context)
        return

    if not data.startswith(CALLBACK_ANSWER_PREFIX):
        return

    question: Optional[Dict[str, Any]] = context.user_data.get("current_question")
    if not question:
        await query.edit_message_text("No active question. Send /quiz to start a new one.")
        return

    try:
        selected_index = int(data.split(":", 1)[1])
    except ValueError:
        await query.edit_message_text("Invalid answer. Send /quiz to try again.")
        return

    stats_dict = get_stats(context)
    bot_data: QuizBotData = context.bot_data["quiz_data"]
    stats_dict["total"] += 1

    correct_row: QuizRow = question["correct"]
    answer_key: str = question["answer_key"]
    correct_index: int = question["correct_index"]
    is_correct = selected_index == correct_index

    if is_correct:
        stats_dict["correct"] += 1
    else:
        mode = question["mode"]
        mode_rows = bot_data.get_mode_rows(mode)
        mode_state = get_or_create_mode_state(context, mode, mode_rows)
        schedule_retry(mode_state, correct_row)

    selected_value = question["options"][selected_index] if 0 <= selected_index < len(question["options"]) else ""
    answer_value = getattr(correct_row, answer_key).strip()

    result_lines = [
        "✅ <b>Correct!</b>" if is_correct else "❌ <b>Not quite.</b>",
        "",
        f"<b>My answer:</b> {esc(selected_value or '-')}",
        f"<b>Correct answer:</b> {esc(answer_value or '-')}",
        "",
        "<b>Full entry:</b>",
        f"• <b>Kangi:</b> {esc(correct_row.kangi or '-')}",
        f"• <b>Hiragana:</b> {esc(correct_row.hiragana or '-')}",
        f"• <b>Meaning:</b> {esc(correct_row.meaning or '-')}",
        f"• <b>Sentence:</b> {esc(correct_row.sentence or '-')}",
    ]

    accuracy = stats_dict["correct"] / stats_dict["total"] * 100
    result_lines.extend(
        [
            "",
            f"<b>Score:</b> {stats_dict['correct']}/{stats_dict['total']} ({accuracy:.1f}%)    <b>Progress:</b> {question['progress_done']}/{question['progress_total']}",
        ]
    )

    reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("Next question", callback_data=CALLBACK_NEXT)]])

    await query.message.reply_text(
        "\n".join(result_lines),
        reply_markup=reply_markup,
        parse_mode="HTML",
    )


def main() -> None:
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    csv_path = os.getenv("CSV_PATH", "./data/csv").strip()
    default_mode = os.getenv("QUIZ_MODE", MODE_AUTO).strip() or MODE_AUTO

    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable.")
    if not csv_path:
        raise RuntimeError("Missing CSV_PATH environment variable.")
    if default_mode not in VALID_MODES:
        raise RuntimeError(f"QUIZ_MODE must be one of: {', '.join(sorted(VALID_MODES))}")

    rows = load_rows(csv_path)
    quiz_data = QuizBotData(rows, default_mode=default_mode)
    logger.info("Loaded %d vocabulary rows", len(rows))

    application = Application.builder().token(token).build()
    application.bot_data["quiz_data"] = quiz_data

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("quiz", quiz))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("mode", mode_command))
    application.add_handler(CallbackQueryHandler(on_button))

    logger.info("Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
