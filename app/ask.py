"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保って質問する

ask() はツールも会話履歴も持たない単発の「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
--chat では同じチャットを使って会話履歴を保つ。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    # TODO(第1回 演習): response.usage_metadata から
    #   入力トークン数と出力トークン数を表示してみよう（Copilotに頼んでよい）

    return response.text


def interactive_chat() -> None:
    """会話履歴を保ちながら、入力が続く間モデルに質問する。"""
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)
    print("対話モードです。空行または exit で終了します。")

    while True:
        question = input("質問> ").strip()
        if not question or question.lower() == "exit":
            break

        response = chat.send_message(question)
        print(response.text)
        usage = response.usage_metadata
        input_tokens = (
            usage.prompt_token_count
            if usage and usage.prompt_token_count is not None
            else "不明"
        )
        output_tokens = (
            usage.candidates_token_count
            if usage and usage.candidates_token_count is not None
            else "不明"
        )
        print(f"[tokens] 入力: {input_tokens} / 出力: {output_tokens}")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って対話する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            interactive_chat()
        else:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
