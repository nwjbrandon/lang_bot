"""JLPT vocab quiz over Telegram (long polling)."""

from langbot.config import build_engine, configure_logging, get_env, load_env_file, require_env
from langbot.subjects.jlpt_vocab import build_subject
from langbot.transports.render import HtmlRenderer
from langbot.transports.telegram import run_telegram


def main() -> None:
    configure_logging()
    load_env_file()

    engine = build_engine(
        build_subject(),
        csv_path=get_env("CSV_PATH", "./data/jp/csv"),
        quiz_mode=get_env("QUIZ_MODE"),
    )
    run_telegram(engine, HtmlRenderer(), token=require_env("TELEGRAM_BOT_TOKEN"))


if __name__ == "__main__":
    main()
