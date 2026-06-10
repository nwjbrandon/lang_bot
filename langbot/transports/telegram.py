"""Telegram (long-polling) transport."""

import logging
from typing import List

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram import Message as TgMessage
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from langbot.engine import ACTION_NEXT, COMMANDS, BotEngine
from langbot.models import option_label
from langbot.replies import Message, ModeMenuReply, QuestionReply, Reply, ResultReply
from langbot.session import UserSession
from langbot.transports.render import HtmlRenderer

logger = logging.getLogger(__name__)


def _session(engine: BotEngine, context: ContextTypes.DEFAULT_TYPE) -> UserSession:
    session = context.user_data.get("session")
    if session is None:
        session = engine.new_session()
        context.user_data["session"] = session
    return session


async def _send(message: TgMessage, replies: List[Reply], renderer: HtmlRenderer, strings) -> None:
    for reply in replies:
        if isinstance(reply, Message):
            await message.reply_text(reply.text)

        elif isinstance(reply, QuestionReply):
            keyboard = [[InlineKeyboardButton(option_label(i), callback_data=f"answer:{i}")] for i in range(len(reply.view.options))]
            await message.reply_text(renderer.question_text(reply.view, strings), reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")

        elif isinstance(reply, ResultReply):
            keyboard = [[InlineKeyboardButton(strings.next_label, callback_data=ACTION_NEXT)]]
            await message.reply_text(renderer.result_text(reply.view, strings), reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")

        elif isinstance(reply, ModeMenuReply):
            keyboard = [[InlineKeyboardButton(label, callback_data=f"mode:{name}")] for name, label in reply.options]
            await message.reply_text(reply.prompt, reply_markup=InlineKeyboardMarkup(keyboard))


def _make_command_handler(engine: BotEngine, renderer: HtmlRenderer, command: str):
    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None:
            return
        session = _session(engine, context)
        await _send(update.message, engine.command(session, command), renderer, engine.strings)

    return handler


def _make_callback_handler(engine: BotEngine, renderer: HtmlRenderer):
    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.message:
            return
        await query.answer()
        session = _session(engine, context)
        await _send(query.message, engine.action(session, query.data or ""), renderer, engine.strings)

    return handler


def run_telegram(engine: BotEngine, renderer: HtmlRenderer, token: str) -> None:
    application = Application.builder().token(token).build()

    for command in COMMANDS:
        application.add_handler(CommandHandler(command, _make_command_handler(engine, renderer, command)))
    application.add_handler(CallbackQueryHandler(_make_callback_handler(engine, renderer)))

    logger.info("Telegram bot starting (long polling)...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
