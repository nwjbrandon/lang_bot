"""JLPT vocab quiz over the LINE Messaging API."""

from langbot.config import build_engine, configure_logging, get_env, load_env_file, require_env
from langbot.subjects.jlpt_vocab import build_subject
from langbot.transports.line import run_line
from langbot.transports.render import PlainRenderer


def main() -> None:
    configure_logging()
    load_env_file()

    engine = build_engine(
        build_subject(),
        csv_path=get_env("CSV_PATH", "./data/jp/csv"),
        quiz_mode=get_env("QUIZ_MODE"),
    )
    run_line(
        engine,
        PlainRenderer(),
        channel_access_token=require_env("LINE_CHANNEL_ACCESS_TOKEN"),
        channel_secret=require_env("LINE_CHANNEL_SECRET"),
        host=get_env("LINE_HOST", "0.0.0.0") or "0.0.0.0",
        port=int(get_env("LINE_PORT", "8000") or "8000"),
    )


if __name__ == "__main__":
    main()
