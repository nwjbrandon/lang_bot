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

MODE_EN_TO_JA = "MODE_EN_TO_JA"
MODE_JA_TO_EN = "MODE_JA_TO_EN"
MODE_OPTIONS = [MODE_EN_TO_JA, MODE_JA_TO_EN]
VALID_MODES = set(MODE_OPTIONS)

CALLBACK_ANSWER_PREFIX = "answer:"
CALLBACK_NEXT = "next"
CALLBACK_MODE_PREFIX = "mode:"

RETRY_MIN_GAP = 3
RETRY_MAX_GAP = 5


@dataclass
class PhraseRow:
    english: str
    japanese: str


class PhraseQuizData:
    def __init__(self, rows: List[PhraseRow], default_mode: str = MODE_EN_TO_JA):
        if not rows:
            raise ValueError("CSVからフレーズを読み込めませんでした。")
        self.rows = rows
        self.default_mode = default_mode

    def _get_mode_config(self, mode: str) -> Dict[str, str]:
        if mode == MODE_EN_TO_JA:
            return {
                "question_key": "english",
                "answer_key": "japanese",
                "question_title": "英語に合う日本語を選んでください",
                "question_label": "英語",
                "answer_label": "日本語",
            }
        if mode == MODE_JA_TO_EN:
            return {
                "question_key": "japanese",
                "answer_key": "english",
                "question_title": "日本語に合う英語を選んでください",
                "question_label": "日本語",
                "answer_label": "英語",
            }
        raise ValueError(f"未対応のモードです: {mode}")

    def _distinct_answer_values(self, mode: str) -> List[str]:
        answer_key = self._get_mode_config(mode)["answer_key"]
        values = {getattr(row, answer_key).strip() for row in self.rows if getattr(row, answer_key).strip()}
        return list(values)

    def is_mode_playable(self, mode: str) -> bool:
        return len(self._distinct_answer_values(mode)) >= 4

    def resolve_mode(self, requested_mode: str) -> str:
        if requested_mode in VALID_MODES and self.is_mode_playable(requested_mode):
            return requested_mode

        for mode in MODE_OPTIONS:
            if self.is_mode_playable(mode):
                return mode

        raise ValueError("クイズ作成に必要な選択肢数が足りません。各列に4つ以上の異なる値を用意してください。")

    def get_mode_rows(self, mode: str) -> List[PhraseRow]:
        config = self._get_mode_config(mode)
        question_key = config["question_key"]
        answer_key = config["answer_key"]
        return [row for row in self.rows if getattr(row, question_key).strip() and getattr(row, answer_key).strip()]

    def _rows_by_answer_value(self, mode: str) -> Dict[str, List[PhraseRow]]:
        answer_key = self._get_mode_config(mode)["answer_key"]
        value_to_rows: Dict[str, List[PhraseRow]] = {}

        for row in self.rows:
            value = getattr(row, answer_key).strip()
            if not value:
                continue
            value_to_rows.setdefault(value, []).append(row)

        return value_to_rows

    def make_question_for_row(self, mode: str, correct: PhraseRow, offset: int) -> Dict[str, Any]:
        config = self._get_mode_config(mode)
        answer_key = config["answer_key"]
        question_key = config["question_key"]
        value_to_rows = self._rows_by_answer_value(mode)
        answer_value = getattr(correct, answer_key).strip()

        distinct_values = list(value_to_rows.keys())
        if len(distinct_values) < 4:
            raise ValueError("4択に必要なデータが不足しています。")

        if answer_value not in value_to_rows:
            raise ValueError("正解候補が見つかりませんでした。")

        other_values = [value for value in distinct_values if value != answer_value]
        if len(other_values) < 3:
            raise ValueError("不正解の選択肢が不足しています。")

        start = offset % len(other_values)
        distractors = [other_values[(start + i) % len(other_values)] for i in range(3)]

        options = distractors + [answer_value]
        shift = offset % len(options)
        options = options[shift:] + options[:shift]

        return {
            "mode": mode,
            "question_title": config["question_title"],
            "question_key": question_key,
            "answer_key": answer_key,
            "question_label": config["question_label"],
            "answer_label": config["answer_label"],
            "correct": correct,
            "options": options,
            "correct_index": options.index(answer_value),
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
        raise ValueError(f"ディレクトリにCSVが見つかりません: {source}")

    raise ValueError(f"CSV_PATHはCSVファイルかディレクトリを指定してください: {source}")


def load_rows(csv_path: str) -> List[PhraseRow]:
    csv_files = _resolve_csv_files(csv_path)
    df = pd.concat((pd.read_csv(path) for path in csv_files), ignore_index=True)
    normalized = {col.strip().lower(): col for col in df.columns}

    english_col = normalized.get("english")
    japanese_col = normalized.get("japanese")
    if not english_col or not japanese_col:
        raise ValueError("CSVには 'English' と 'Japanese' 列が必要です。")

    rows: List[PhraseRow] = []
    for _, row in df.iterrows():
        english = _clean(row.get(english_col))
        japanese = _clean(row.get(japanese_col))
        if not english and not japanese:
            continue
        if not english or not japanese:
            continue
        rows.append(PhraseRow(english=english, japanese=japanese))

    return rows


def get_stats(user_state: Dict[str, Any]) -> Dict[str, int]:
    if "stats" not in user_state:
        user_state["stats"] = {"correct": 0, "total": 0}
    return user_state["stats"]


def get_mode_states(user_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "mode_states" not in user_state:
        user_state["mode_states"] = {}
    return user_state["mode_states"]


def _new_mode_state(rows: List[PhraseRow]) -> Dict[str, Any]:
    queue = rows[:]
    random.shuffle(queue)
    return {
        "cycle_queue": queue,
        "cycle_done": 0,
        "cycle_total": len(rows),
        "asked_total": 0,
        "reviews": [],
    }


def get_or_create_mode_state(user_state: Dict[str, Any], mode: str, rows: List[PhraseRow]) -> Dict[str, Any]:
    mode_states = get_mode_states(user_state)
    if mode not in mode_states:
        mode_states[mode] = _new_mode_state(rows)
    return mode_states[mode]


def _refresh_cycle_if_needed(state: Dict[str, Any], rows: List[PhraseRow]) -> None:
    if state["cycle_queue"]:
        return
    queue = rows[:]
    random.shuffle(queue)
    state["cycle_queue"] = queue
    state["cycle_done"] = 0
    state["cycle_total"] = len(rows)


def pick_next_question_row(state: Dict[str, Any], rows: List[PhraseRow]) -> tuple[PhraseRow, str]:
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


def schedule_retry(state: Dict[str, Any], question_row: PhraseRow) -> None:
    delay = random.randint(RETRY_MIN_GAP, RETRY_MAX_GAP)
    state["reviews"].append({"row": question_row, "due_at": state["asked_total"] + delay})


def option_label(idx: int) -> str:
    return chr(ord("A") + idx)


def build_question_text(question: Dict[str, Any], stats: Dict[str, int]) -> str:
    correct: PhraseRow = question["correct"]
    question_key: str = question["question_key"]
    accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0.0

    prompt_text = getattr(correct, question_key).strip() or "-"

    lines = [
        f"問題: {question['question_title']}",
        "",
        f"{question['question_label']}: {prompt_text}",
    ]

    if question.get("source") == "review":
        lines.append("(復習問題)")

    lines.extend(["", "選択肢:"])
    for idx, option in enumerate(question["options"]):
        lines.append(f"{option_label(idx)}. {option}")

    lines.extend(
        [
            "",
            f"スコア: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)  進捗: {question['progress_done']}/{question['progress_total']}",
            f"答えは「{question['answer_label']}」を選んでください。",
        ]
    )

    return "\n".join(lines)


def build_result_text(question: Dict[str, Any], stats_dict: Dict[str, int], selected_index: int) -> str:
    correct_row: PhraseRow = question["correct"]
    question_key: str = question["question_key"]
    answer_key: str = question["answer_key"]
    correct_index: int = question["correct_index"]
    is_correct = selected_index == correct_index

    selected_value = question["options"][selected_index] if 0 <= selected_index < len(question["options"]) else ""
    answer_value = getattr(correct_row, answer_key).strip()
    prompt_value = getattr(correct_row, question_key).strip()

    result_lines = [
        "正解です!" if is_correct else "不正解です。",
        "",
        f"あなたの回答: {selected_value or '-'}",
        f"正解: {answer_value or '-'}",
        "",
        "対応フレーズ:",
        f"- 英語: {correct_row.english or '-'}",
        f"- 日本語: {correct_row.japanese or '-'}",
        f"- 出題文: {prompt_value or '-'}",
    ]

    accuracy = stats_dict["correct"] / stats_dict["total"] * 100
    result_lines.extend(
        [
            "",
            f"スコア: {stats_dict['correct']}/{stats_dict['total']} ({accuracy:.1f}%)  進捗: {question['progress_done']}/{question['progress_total']}",
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
                quick_reply_postback("次へ", CALLBACK_NEXT, "次の問題"),
            ]
        },
    }


def build_mode_message() -> Dict[str, Any]:
    mode_to_label = {
        MODE_EN_TO_JA: "英語 -> 日本語",
        MODE_JA_TO_EN: "日本語 -> 英語",
    }
    items = [quick_reply_postback(mode_to_label[m], f"{CALLBACK_MODE_PREFIX}{m}", mode_to_label[m]) for m in MODE_OPTIONS]
    return {
        "type": "text",
        "text": "モードを選んでください:",
        "quickReply": {"items": items},
    }


def mode_name(mode: str) -> str:
    names = {
        MODE_EN_TO_JA: "英語 -> 日本語",
        MODE_JA_TO_EN: "日本語 -> 英語",
    }
    return names.get(mode, mode)


def build_welcome_message(current_mode: str) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": (f"フレーズ学習クイズへようこそ!\n\nコマンド:\n/quiz - 4択クイズを開始\n/mode - 出題モードを変更\n/reset - スコアと進捗をリセット\n/stats - 現在の成績を表示\n/help - 使い方を表示\n\n現在のモード: {mode_name(current_mode)}"),
        "quickReply": {
            "items": [
                quick_reply_message("クイズ", "/quiz"),
                quick_reply_message("モード", "/mode"),
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
    def __init__(self, quiz_data: PhraseQuizData, line_client: LineClient):
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

        user_state["mode"] = resolved_mode
        user_state["current_question"] = question
        return build_question_message(question, get_stats(user_state))

    def handle_text_command(self, user_state: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
        normalized = text.strip().lower()

        if normalized in {"/start", "start", "スタート"}:
            resolved = self.quiz_data.resolve_mode(user_state.get("mode", self.quiz_data.default_mode))
            return [build_welcome_message(resolved)]

        if normalized in {"/help", "help", "ヘルプ"}:
            return [
                {
                    "type": "text",
                    "text": ("使い方:\n・/quiz で4択問題を出題します。\n・/mode で出題方向を切り替えます。\n・間違えた問題は数問後に復習として再出題されます。\n・/stats で成績を確認できます。\n・/reset で成績と進捗をリセットします。"),
                }
            ]

        if normalized in {"/quiz", "quiz", "クイズ"}:
            try:
                return [self.send_new_question(user_state)]
            except ValueError as ex:
                return [{"type": "text", "text": f"問題を作れませんでした: {ex}"}]

        if normalized in {"/stats", "stats", "統計"}:
            stats = get_stats(user_state)
            accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0.0
            return [
                {
                    "type": "text",
                    "text": (f"成績\n正解数: {stats['correct']}\n回答数: {stats['total']}\n正答率: {accuracy:.1f}%"),
                }
            ]

        if normalized in {"/reset", "reset", "リセット"}:
            user_state["stats"] = {"correct": 0, "total": 0}
            user_state["mode_states"] = {}
            user_state["current_question"] = None
            return [{"type": "text", "text": "成績と進捗をリセットしました。/quiz で再開できます。"}]

        if normalized in {"/mode", "mode", "モード"}:
            return [build_mode_message()]

        return [{"type": "text", "text": "コマンドが分かりません。/help を入力してください。"}]

    def handle_postback(self, user_state: Dict[str, Any], data: str) -> List[Dict[str, Any]]:
        if data.startswith(CALLBACK_MODE_PREFIX):
            mode = data.split(":", 1)[1]
            if mode in VALID_MODES:
                user_state["mode"] = mode
                resolved = self.quiz_data.resolve_mode(mode)
                return [{"type": "text", "text": f"モードを変更しました: {mode_name(resolved)}\n/quiz で開始できます。"}]
            return [{"type": "text", "text": "無効なモードです。"}]

        if data == CALLBACK_NEXT:
            try:
                return [self.send_new_question(user_state)]
            except ValueError as ex:
                return [{"type": "text", "text": f"問題を作れませんでした: {ex}"}]

        if not data.startswith(CALLBACK_ANSWER_PREFIX):
            return []

        question: Optional[Dict[str, Any]] = user_state.get("current_question")
        if not question:
            return [{"type": "text", "text": "出題中の問題がありません。/quiz で開始してください。"}]

        try:
            selected_index = int(data.split(":", 1)[1])
        except ValueError:
            return [{"type": "text", "text": "回答データが不正です。/quiz で再開してください。"}]

        stats = get_stats(user_state)
        stats["total"] += 1

        correct_row: PhraseRow = question["correct"]
        correct_index: int = question["correct_index"]
        is_correct = selected_index == correct_index

        if is_correct:
            stats["correct"] += 1
        else:
            mode = question["mode"]
            mode_rows = self.quiz_data.get_mode_rows(mode)
            mode_state = get_or_create_mode_state(user_state, mode, mode_rows)
            schedule_retry(mode_state, correct_row)

        result_text = build_result_text(question, stats, selected_index)
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
                self.runtime.line_client.reply(reply_token, [{"type": "text", "text": "テキストで入力してください。/help で使い方を確認できます。"}])
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
    csv_path = os.getenv("CSV_PATH", "./english_japanese_sentences.csv").strip()
    default_mode = os.getenv("QUIZ_MODE", MODE_EN_TO_JA).strip() or MODE_EN_TO_JA
    host = os.getenv("LINE_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = int(os.getenv("LINE_PORT", "8000"))

    if not channel_access_token:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN が設定されていません。")
    if not channel_secret:
        raise RuntimeError("LINE_CHANNEL_SECRET が設定されていません。")
    if not csv_path:
        raise RuntimeError("CSV_PATH が設定されていません。")
    if default_mode not in VALID_MODES:
        raise RuntimeError(f"QUIZ_MODE は次のいずれかを指定してください: {', '.join(sorted(VALID_MODES))}")

    rows = load_rows(csv_path)
    quiz_data = PhraseQuizData(rows, default_mode=default_mode)
    logger.info("Loaded %d phrase rows", len(rows))

    line_client = LineClient(channel_access_token)
    runtime = BotRuntime(quiz_data, line_client)

    LineWebhookHandler.runtime = runtime
    LineWebhookHandler.channel_secret = channel_secret

    server = ThreadingHTTPServer((host, port), LineWebhookHandler)
    logger.info("Phrase LINE bot is listening on http://%s:%d/callback", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
