"""
Tests for CLI interrupt handling — the paths the piped smoke tests can't reach.

  - harness REPL: Ctrl-C at the prompt exits cleanly; Ctrl-C mid-generation
    cancels that answer and re-prompts.
  - rag CLI: Ctrl-C becomes a clean exit code 130.

input() and the provider are faked, so no terminal, model, or server is needed.
"""

from harness.cli import run_repl


class _Recorder:
    """A fake LLMProvider that records prompts and follows a scripted behaviour.

    *behaviours* is applied one entry per generate() call, in order:
      "ok"        -> return a canned answer
      "interrupt" -> raise KeyboardInterrupt (simulating Ctrl-C mid-generation)
    """

    def __init__(self, behaviours: list[str]):
        self._behaviours = list(behaviours)
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.prompts.append(prompt)
        behaviour = self._behaviours.pop(0) if self._behaviours else "ok"
        if behaviour == "interrupt":
            raise KeyboardInterrupt
        return f"answer to {prompt!r}"


def _scripted_input(lines: list[str]):
    """Return a fake input() that yields each line then raises EOFError."""
    it = iter(lines)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return fake_input


def test_ctrl_c_at_prompt_exits_cleanly(monkeypatch, capsys):
    def fake_input(prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", fake_input)
    provider = _Recorder(behaviours=[])

    # Must return normally — a leaked KeyboardInterrupt would fail the test.
    run_repl(provider, max_tokens=10, label="Test")

    out = capsys.readouterr().out
    assert "Bye!" in out
    assert provider.prompts == []  # generation never attempted


def test_ctrl_c_during_generation_cancels_and_reprompts(monkeypatch, capsys):
    # First prompt's generation is interrupted; the loop must survive and serve
    # the second prompt; the third read hits EOF and ends the loop.
    monkeypatch.setattr("builtins.input", _scripted_input(["first", "second"]))
    provider = _Recorder(behaviours=["interrupt", "ok"])

    run_repl(provider, max_tokens=10, label="Test")

    out = capsys.readouterr().out
    assert "[cancelled]" in out                  # first generation was cancelled
    assert "answer to 'second'" in out           # loop re-prompted and served the next
    assert provider.prompts == ["first", "second"]


def test_rag_ctrl_c_returns_exit_code_130(monkeypatch, capsys):
    import rag.__main__ as rag_main

    def boom() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(rag_main, "main", boom)

    code = rag_main._run()

    assert code == 130
    assert "Interrupted" in capsys.readouterr().err
