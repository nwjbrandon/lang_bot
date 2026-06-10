"""English <-> Japanese phrase subject."""

from typing import List, Sequence

from langbot.models import Entry, Mode
from langbot.subjects.base import Strings, Subject, base_command_aliases

MODE_EN_TO_JA = "MODE_EN_TO_JA"
MODE_JA_TO_EN = "MODE_JA_TO_EN"

FIELDS = ["english", "japanese"]
FIELD_LABELS = {"english": "英語", "japanese": "日本語"}

MODES = [
    Mode(
        name=MODE_EN_TO_JA,
        answer_field="japanese",
        prompt_fields=["english"],
        required_fields=["english", "japanese"],
        title="英語に合う日本語を選んでください",
        label="英語 -> 日本語",
        footer="答えは「日本語」を選んでください。",
    ),
    Mode(
        name=MODE_JA_TO_EN,
        answer_field="english",
        prompt_fields=["japanese"],
        required_fields=["english", "japanese"],
        title="日本語に合う英語を選んでください",
        label="日本語 -> 英語",
        footer="答えは「英語」を選んでください。",
    ),
]


def _fallback_order(_entries: Sequence[Entry]) -> List[str]:
    return [MODE_EN_TO_JA, MODE_JA_TO_EN]


STRINGS = Strings(
    empty_data="CSVからフレーズを読み込めませんでした。",
    not_enough_data="クイズ作成に必要なデータが不足しています。各列に4つ以上の異なる値を用意してください。",
    welcome=("フレーズ学習クイズへようこそ!\n\nコマンド:\n/quiz - 4択クイズを開始\n/mode - 出題モードを変更\n/reset - スコアと進捗をリセット\n/stats - 現在の成績を表示\n/help - 使い方を表示\n\n現在のモード: {mode}"),
    welcome_quick=[("クイズ", "/quiz"), ("モード", "/mode")],
    help=("使い方:\n・/quiz で4択問題を出題します。\n・/mode で出題方向を切り替えます。\n・間違えた問題は数問後に復習として再出題されます。\n・/stats で成績を確認できます。\n・/reset で成績と進捗をリセットします。"),
    stats_header="成績",
    stats_correct="正解数",
    stats_total="回答数",
    stats_accuracy="正答率",
    reset_done="成績と進捗をリセットしました。/quiz で再開できます。",
    mode_prompt="モードを選んでください:",
    mode_set="モードを変更しました: {mode}\n/quiz で開始できます。",
    invalid_mode="無効なモードです。",
    no_active_question="出題中の問題がありません。/quiz で開始してください。",
    invalid_answer="回答データが不正です。/quiz で再開してください。",
    unknown_command="コマンドが分かりません。/help を入力してください。",
    non_text="テキストで入力してください。/help で使い方を確認できます。",
    next_label="次へ",
    next_display="次の問題",
    question_prefix="問題: ",
    given_header="出題:",
    options_header="選択肢:",
    review_note="(復習問題)",
    score_label="スコア",
    progress_label="進捗",
    correct_msg="正解です!",
    incorrect_msg="不正解です。",
    my_answer_label="あなたの回答",
    correct_answer_label="正解",
    full_entry_header="対応フレーズ:",
)


def _command_aliases():
    aliases = base_command_aliases()
    aliases["start"].add("スタート")
    aliases["help"].add("ヘルプ")
    aliases["quiz"].add("クイズ")
    aliases["stats"].add("統計")
    aliases["reset"].add("リセット")
    aliases["mode"].add("モード")
    return aliases


def build_subject() -> Subject:
    return Subject(
        name="en_ja_phrases",
        fields=FIELDS,
        field_labels=FIELD_LABELS,
        csv_columns={"english": "English", "japanese": "Japanese"},
        csv_required_columns=["english", "japanese"],
        csv_required_fields=["english", "japanese"],
        modes=MODES,
        menu_modes=[(MODE_EN_TO_JA, "英語 -> 日本語"), (MODE_JA_TO_EN, "日本語 -> 英語")],
        default_mode=MODE_EN_TO_JA,
        auto_mode=None,
        fallback_order=_fallback_order,
        command_aliases=_command_aliases(),
        strings=STRINGS,
    )
