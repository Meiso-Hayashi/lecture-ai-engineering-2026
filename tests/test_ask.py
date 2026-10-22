from types import SimpleNamespace
from unittest.mock import Mock

from app import ask as ask_module
from app.config import MODEL


def test_interactive_chat_preserves_history_and_displays_usage(monkeypatch, capsys):
    chat = SimpleNamespace(
        send_message=Mock(
            side_effect=[
                SimpleNamespace(
                    text="1つ目の回答",
                    usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=8),
                ),
                SimpleNamespace(
                    text="2つ目の回答",
                    usage_metadata=SimpleNamespace(prompt_token_count=21, candidates_token_count=5),
                ),
            ]
        )
    )
    create_chat = Mock(return_value=chat)
    monkeypatch.setattr(
        ask_module.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(chats=SimpleNamespace(create=create_chat)),
    )
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    questions = iter(["最初の質問", "続きの質問", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))

    ask_module.interactive_chat()

    create_chat.assert_called_once_with(model=MODEL)
    assert [call.args[0] for call in chat.send_message.call_args_list] == [
        "最初の質問",
        "続きの質問",
    ]
    output = capsys.readouterr().out
    assert "1つ目の回答" in output
    assert "2つ目の回答" in output
    assert "[tokens] 入力: 12 / 出力: 8" in output
    assert "[tokens] 入力: 21 / 出力: 5" in output


def test_chat_option_starts_interactive_mode_without_changing_single_question(
    monkeypatch, capsys
):
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat"])
    monkeypatch.setattr(ask_module, "interactive_chat", lambda: print("chat started"))

    ask_module.main()

    output = capsys.readouterr().out
    assert "[model]" in output
    assert "chat started" in output
    assert "[質問]" not in output


def test_interactive_chat_exits_on_exit(monkeypatch):
    send_message = Mock()
    chat = SimpleNamespace(send_message=send_message)
    create_chat = Mock(return_value=chat)
    monkeypatch.setattr(
        ask_module.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(chats=SimpleNamespace(create=create_chat)),
    )
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("builtins.input", lambda _: "exit")

    ask_module.interactive_chat()

    send_message.assert_not_called()
