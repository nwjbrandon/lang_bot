import base64
import hashlib
import hmac
import json
import logging
import os
import random
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

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


def get_stats(user_state: Dict[str, Any]) -> Dict[str, int]:
    if "stats" not in user_state:
        user_state["stats"] = {"correct": 0, "total": 0}
    return user_state["stats"]


def get_mode_states(user_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "mode_states" not in user_state:
        user_state["mode_states"] = {}
    return user_state["mode_states"]


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


def get_or_create_mode_state(user_state: Dict[str, Any], mode: str, rows: List[QuizRow]) -> Dict[str, Any]:
    mode_states = get_mode_states(user_state)
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
        f"Question: {question['question_title']}",
        "",
        "Given:",
    ]

    if question.get("source") == "review":
        lines.append("Review question")

    for key in display_order:
        if key == answer_key:
            continue
        value = field_map[key].strip() or "-"
        lines.append(f"- {display_labels[key]}: {value}")

    lines.extend(["", "Options:"])
    for idx, option in enumerate(question["options"]):
        lines.append(f"{option_label(idx)}. {option}")

    lines.extend(
        [
            "",
            f"Score: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)  Progress: {question['progress_done']}/{question['progress_total']}",
        ]
    )

    return "\n".join(lines)


def build_result_text(question: Dict[str, Any], stats_dict: Dict[str, int], selected_index: int) -> str:
    correct_row: QuizRow = question["correct"]
    answer_key: str = question["answer_key"]
    correct_index: int = question["correct_index"]
    is_correct = selected_index == correct_index

    selected_value = question["options"][selected_index] if 0 <= selected_index < len(question["options"]) else ""
    answer_value = getattr(correct_row, answer_key).strip()

    result_lines = [
        "Correct!" if is_correct else "Not quite.",
        "",
        f"My answer: {selected_value or '-'}",
        f"Correct answer: {answer_value or '-'}",
        "",
        "Full entry:",
        f"- Kangi: {correct_row.kangi or '-'}",
        f"- Hiragana: {correct_row.hiragana or '-'}",
        f"- Meaning: {correct_row.meaning or '-'}",
        f"- Sentence: {correct_row.sentence or '-'}",
    ]

    accuracy = stats_dict["correct"] / stats_dict["total"] * 100
    result_lines.extend(
        [
            "",
            f"Score: {stats_dict['correct']}/{stats_dict['total']} ({accuracy:.1f}%)  Progress: {question['progress_done']}/{question['progress_total']}",
        ]
    )

    return "\n".join(result_lines)


def quick_reply_message(label: str, text: str) -> Dict[str, Any]:
    return {
        "type": "action",
        "action": {
            "type": "message",
            "label": label,
            "text": text,
        },
    }


def quick_reply_postback(label: str, data: str, display_text: str) -> Dict[str, Any]:
    return {
        "type": "action",
        "action": {
            "type": "postback",
            "label": label,
            "data": data,
            "displayText": display_text,
        },
    }


def build_question_message(question: Dict[str, Any], stats: Dict[str, int]) -> Dict[str, Any]:
    quick_items = []
    for idx, _ in enumerate(question["options"]):
        label = option_label(idx)
        quick_items.append(quick_reply_postback(label, f"{CALLBACK_ANSWER_PREFIX}{idx}", label))

    return {
        "type": "text",
        "text": build_question_text(question, stats),
        "quickReply": {"items": quick_items},
    }


def build_next_question_message(text: str) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": text,
        "quickReply": {
            "items": [
                quick_reply_postback("Next", CALLBACK_NEXT, "Next question"),
            ]
        },
    }


def build_mode_message() -> Dict[str, Any]:
    mode_to_label = {
        MODE_AUTO: "Auto",
        MODE_TEST_KANGI: "Test Kangi",
        MODE_TEST_HIRAGANA: "Test Hiragana",
        MODE_TEST_MEANING: "Test Meaning",
    }
    items = [quick_reply_postback(mode_to_label[m], f"{CALLBACK_MODE_PREFIX}{m}", mode_to_label[m]) for m in MODE_OPTIONS]
    return {
        "type": "text",
        "text": "Choose a quiz mode:",
        "quickReply": {"items": items},
    }


def build_welcome_message(current_mode: str) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": (f"Welcome to the JLPT...\n\nCommands:\n/quiz - start a 4-option quiz\n/mode - change quiz mode\n/reset - reset score and progress\n/stats - see your score\n/help - show help\n\nCurrent mode: {current_mode}"),
        "quickReply": {
            "items": [
                quick_reply_message("Quiz", "/quiz"),
                quick_reply_message("Mode", "/mode"),
            ]
        },
    }


class LineClient:
    def __init__(self, channel_access_token: str):
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {channel_access_token}",
            "Content-Type": "application/json",
        }

    def reply(self, reply_token: str, messages: List[Dict[str, Any]]) -> None:
        url = "https://api.line.me/v2/bot/message/reply"
        payload = {
            "replyToken": reply_token,
            "messages": messages,
        }
        response = self.session.post(url, headers=self.headers, json=payload, timeout=10)
        if response.status_code >= 400:
            logger.error("LINE reply failed (%s): %s", response.status_code, response.text)


def verify_signature(channel_secret: str, body: bytes, signature: str) -> bool:
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


class BotRuntime:
    def __init__(self, quiz_data: QuizBotData, line_client: LineClient):
        self.quiz_data = quiz_data
        self.line_client = line_client
        self.user_state: Dict[str, Dict[str, Any]] = {}

    def get_or_create_user_state(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.user_state:
            self.user_state[user_id] = {
                "mode": self.quiz_data.default_mode,
                "stats": {"correct": 0, "total": 0},
                "mode_states": {},
                "current_question": None,
            }
        return self.user_state[user_id]

    def send_new_question(self, user_state: Dict[str, Any]) -> Dict[str, Any]:
        requested_mode = user_state.get("mode", self.quiz_data.default_mode)

        resolved_mode = self.quiz_data.resolve_mode(requested_mode)
        mode_rows = self.quiz_data.get_mode_rows(resolved_mode)
        state = get_or_create_mode_state(user_state, resolved_mode, mode_rows)
        question_row, source = pick_next_question_row(state, mode_rows)
        question = self.quiz_data.make_question_for_row(resolved_mode, question_row, state["asked_total"])
        question["source"] = source
        question["progress_done"] = state["cycle_done"]
        question["progress_total"] = state["cycle_total"]

        if resolved_mode != requested_mode:
            user_state["mode"] = resolved_mode

        if question["mode"] != requested_mode and requested_mode != MODE_AUTO:
            user_state["mode"] = question["mode"]

        user_state["current_question"] = question
        return build_question_message(question, get_stats(user_state))

    def handle_text_command(self, user_state: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
        normalized = text.strip().lower()

        if normalized in {"/start", "start"}:
            resolved = self.quiz_data.resolve_mode(user_state.get("mode", self.quiz_data.default_mode))
            return [build_welcome_message(resolved)]

        if normalized in {"/help", "help"}:
            return [
                {
                    "type": "text",
                    "text": (
                        "Use /quiz to get a question with 4 answer choices.\n"
                        "Use /mode to switch what column is being tested.\n"
                        "Wrong answers are tested again after a few questions.\n"
                        "Use /stats to see your current score.\n"
                        "Use /reset to reset score and progress."
                    ),
                }
            ]

        if normalized in {"/quiz", "quiz"}:
            try:
                return [self.send_new_question(user_state)]
            except ValueError:
                return [
                    {
                        "type": "text",
                        "text": "Not enough quiz data for the current mode. Please add at least 4 distinct values in one test column.",
                    }
                ]

        if normalized in {"/stats", "stats"}:
            s = get_stats(user_state)
            accuracy = (s["correct"] / s["total"] * 100) if s["total"] else 0.0
            return [
                {
                    "type": "text",
                    "text": f"Score\nCorrect: {s['correct']}\nTotal: {s['total']}\nAccuracy: {accuracy:.1f}%",
                }
            ]

        if normalized in {"/reset", "reset"}:
            user_state["stats"] = {"correct": 0, "total": 0}
            user_state["mode_states"] = {}
            user_state["current_question"] = None
            return [
                {
                    "type": "text",
                    "text": "Score and progress have been reset. Use /quiz to continue.",
                }
            ]

        if normalized in {"/mode", "mode"}:
            return [build_mode_message()]

        return [
            {
                "type": "text",
                "text": "Unknown command. Use /help.",
            }
        ]

    def handle_postback(self, user_state: Dict[str, Any], data: str) -> List[Dict[str, Any]]:
        if data.startswith(CALLBACK_MODE_PREFIX):
            mode = data.split(":", 1)[1]
            if mode in VALID_MODES:
                user_state["mode"] = mode
                resolved = self.quiz_data.resolve_mode(mode)
                return [
                    {
                        "type": "text",
                        "text": f"Mode set to: {resolved}\nUse /quiz to start.",
                    }
                ]
            return [{"type": "text", "text": "Invalid mode."}]

        if data == CALLBACK_NEXT:
            try:
                return [self.send_new_question(user_state)]
            except ValueError:
                return [
                    {
                        "type": "text",
                        "text": "Not enough quiz data for the current mode. Please add at least 4 distinct values in one test column.",
                    }
                ]

        if not data.startswith(CALLBACK_ANSWER_PREFIX):
            return []

        question: Optional[Dict[str, Any]] = user_state.get("current_question")
        if not question:
            return [{"type": "text", "text": "No active question. Send /quiz to start a new one."}]

        try:
            selected_index = int(data.split(":", 1)[1])
        except ValueError:
            return [{"type": "text", "text": "Invalid answer. Send /quiz to try again."}]

        stats_dict = get_stats(user_state)
        stats_dict["total"] += 1

        correct_row: QuizRow = question["correct"]
        correct_index: int = question["correct_index"]
        is_correct = selected_index == correct_index

        if is_correct:
            stats_dict["correct"] += 1
        else:
            mode = question["mode"]
            mode_rows = self.quiz_data.get_mode_rows(mode)
            mode_state = get_or_create_mode_state(user_state, mode, mode_rows)
            schedule_retry(mode_state, correct_row)

        result_text = build_result_text(question, stats_dict, selected_index)
        return [build_next_question_message(result_text)]


class LineWebhookHandler(BaseHTTPRequestHandler):
    runtime: BotRuntime
    channel_secret: str

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return

        body = json.dumps({"status": "ok"}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/callback":
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        logger.info("Received LINE webhook: %s", body.decode("utf-8"))
        signature = self.headers.get("X-Line-Signature", "")

        if not verify_signature(self.channel_secret, body, signature):
            logger.warning("Invalid LINE signature")
            self.send_response(HTTPStatus.FORBIDDEN)
            self.end_headers()
            return

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.end_headers()
            return

        events = payload.get("events", [])
        for event in events:
            self._handle_event(event)

        self.send_response(HTTPStatus.OK)
        self.end_headers()

    def _handle_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("type")
        reply_token = event.get("replyToken")
        source = event.get("source", {})
        user_id = source.get("userId") or source.get("groupId") or source.get("roomId")

        if not reply_token or not user_id:
            return

        user_state = self.runtime.get_or_create_user_state(user_id)

        if event_type == "message":
            message = event.get("message", {})
            if message.get("type") != "text":
                self.runtime.line_client.reply(reply_token, [{"type": "text", "text": "Please send a text command. Use /help."}])
                return
            text = message.get("text", "")
            messages = self.runtime.handle_text_command(user_state, text)
            if messages:
                self.runtime.line_client.reply(reply_token, messages)
            return

        if event_type == "postback":
            postback_data = event.get("postback", {}).get("data", "")
            messages = self.runtime.handle_postback(user_state, postback_data)
            if messages:
                self.runtime.line_client.reply(reply_token, messages)
            return

        if event_type == "follow":
            messages = self.runtime.handle_text_command(user_state, "/start")
            self.runtime.line_client.reply(reply_token, messages)


def main() -> None:
    load_dotenv()

    channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
    channel_secret = os.getenv("LINE_CHANNEL_SECRET", "").strip()
    csv_path = os.getenv("CSV_PATH", "./data/csv").strip()
    default_mode = os.getenv("QUIZ_MODE", MODE_AUTO).strip() or MODE_AUTO
    host = os.getenv("LINE_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = int(os.getenv("LINE_PORT", "8000"))

    if not channel_access_token:
        raise RuntimeError("Missing LINE_CHANNEL_ACCESS_TOKEN environment variable.")
    if not channel_secret:
        raise RuntimeError("Missing LINE_CHANNEL_SECRET environment variable.")
    if not csv_path:
        raise RuntimeError("Missing CSV_PATH environment variable.")
    if default_mode not in VALID_MODES:
        raise RuntimeError(f"QUIZ_MODE must be one of: {', '.join(sorted(VALID_MODES))}")

    rows = load_rows(csv_path)
    quiz_data = QuizBotData(rows, default_mode=default_mode)
    logger.info("Loaded %d vocabulary rows", len(rows))

    line_client = LineClient(channel_access_token)
    runtime = BotRuntime(quiz_data, line_client)

    LineWebhookHandler.runtime = runtime
    LineWebhookHandler.channel_secret = channel_secret

    server = ThreadingHTTPServer((host, port), LineWebhookHandler)
    logger.info("LINE bot is listening on http://%s:%d/callback", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
