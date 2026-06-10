"""LINE Messaging API webhook transport."""

import base64
import hashlib
import hmac
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import requests

from langbot.engine import ACTION_NEXT, BotEngine
from langbot.replies import Message, ModeMenuReply, QuestionReply, Reply, ResultReply
from langbot.session import UserSession
from langbot.transports.render import PlainRenderer

logger = logging.getLogger(__name__)

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"


class LineClient:
    def __init__(self, channel_access_token: str):
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {channel_access_token}",
            "Content-Type": "application/json",
        }

    def reply(self, reply_token: str, messages: List[Dict[str, Any]]) -> None:
        payload = {"replyToken": reply_token, "messages": messages}
        response = self.session.post(LINE_REPLY_URL, headers=self.headers, json=payload, timeout=10)
        if response.status_code >= 400:
            logger.error("LINE reply failed (%s): %s", response.status_code, response.text)


def verify_signature(channel_secret: str, body: bytes, signature: str) -> bool:
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def _message_action(label: str, text: str) -> Dict[str, Any]:
    return {"type": "action", "action": {"type": "message", "label": label, "text": text}}


def _postback_action(label: str, data: str, display_text: str) -> Dict[str, Any]:
    return {"type": "action", "action": {"type": "postback", "label": label, "data": data, "displayText": display_text}}


def _text_message(text: str, quick_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    message: Dict[str, Any] = {"type": "text", "text": text}
    if quick_items:
        message["quickReply"] = {"items": quick_items}
    return message


class LineRuntime:
    """Keeps per-user sessions and turns engine replies into LINE messages."""

    def __init__(self, engine: BotEngine, renderer: PlainRenderer, line_client: LineClient):
        self.engine = engine
        self.renderer = renderer
        self.strings = engine.strings
        self.line_client = line_client
        self.sessions: Dict[str, UserSession] = {}

    def session(self, user_id: str) -> UserSession:
        if user_id not in self.sessions:
            self.sessions[user_id] = self.engine.new_session()
        return self.sessions[user_id]

    def handle_text(self, session: UserSession, text: str) -> List[Reply]:
        command = self.engine.parse_command(text)
        if command is None:
            return [Message(self.strings.unknown_command)]
        return self.engine.command(session, command)

    def to_messages(self, replies: List[Reply]) -> List[Dict[str, Any]]:
        return [self._to_message(reply) for reply in replies]

    def _to_message(self, reply: Reply) -> Dict[str, Any]:
        if isinstance(reply, Message):
            quick = [_message_action(label, text) for label, text in reply.quick_replies]
            return _text_message(reply.text, quick)

        if isinstance(reply, QuestionReply):
            text = self.renderer.question_text(reply.view, self.strings)
            quick = [_postback_action(chr(ord("A") + i), f"answer:{i}", chr(ord("A") + i)) for i in range(len(reply.view.options))]
            return _text_message(text, quick)

        if isinstance(reply, ResultReply):
            text = self.renderer.result_text(reply.view, self.strings)
            quick = [_postback_action(self.strings.next_label, ACTION_NEXT, self.strings.next_display)]
            return _text_message(text, quick)

        if isinstance(reply, ModeMenuReply):
            quick = [_postback_action(label, f"mode:{name}", label) for name, label in reply.options]
            return _text_message(reply.prompt, quick)

        raise TypeError(f"Unsupported reply type: {type(reply)!r}")


class LineWebhookHandler(BaseHTTPRequestHandler):
    runtime: LineRuntime
    channel_secret: str

    def log_message(self, *args: Any) -> None:  # quieter default access logging
        logger.debug("%s - %s", self.address_string(), args)

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

        for event in payload.get("events", []):
            self._handle_event(event)

        self.send_response(HTTPStatus.OK)
        self.end_headers()

    def _reply(self, reply_token: str, replies: List[Reply]) -> None:
        if replies:
            self.runtime.line_client.reply(reply_token, self.runtime.to_messages(replies))

    def _handle_event(self, event: Dict[str, Any]) -> None:
        reply_token = event.get("replyToken")
        source = event.get("source", {})
        user_id = source.get("userId") or source.get("groupId") or source.get("roomId")
        if not reply_token or not user_id:
            return

        session = self.runtime.session(user_id)
        event_type = event.get("type")

        if event_type == "message":
            message = event.get("message", {})
            if message.get("type") != "text":
                self._reply(reply_token, [Message(self.runtime.strings.non_text)])
                return
            self._reply(reply_token, self.runtime.handle_text(session, message.get("text", "")))

        elif event_type == "postback":
            data = event.get("postback", {}).get("data", "")
            self._reply(reply_token, self.runtime.engine.action(session, data))

        elif event_type == "follow":
            self._reply(reply_token, self.runtime.handle_text(session, "/start"))


def run_line(
    engine: BotEngine,
    renderer: PlainRenderer,
    channel_access_token: str,
    channel_secret: str,
    host: str,
    port: int,
) -> None:
    line_client = LineClient(channel_access_token)
    LineWebhookHandler.runtime = LineRuntime(engine, renderer, line_client)
    LineWebhookHandler.channel_secret = channel_secret

    server = ThreadingHTTPServer((host, port), LineWebhookHandler)
    logger.info("LINE bot listening on http://%s:%d/callback", host, port)
    server.serve_forever()
