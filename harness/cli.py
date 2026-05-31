"""
Interactive CLI for the LLM harness.

Run with:  python -m harness [options]

Pick a provider with --provider:
  toy     the hand-built GPT model (default; needs a trained checkpoint)
  ollama  a local Ollama model, e.g. --provider ollama --model qwen2.5-coder:14b

Type a prompt and press Enter to generate. Empty line, "quit", Ctrl-C, or Ctrl-D exits.
"""

import argparse
import os
import sys

from harness.provider import LLMProvider
from harness.toy_provider import ToyLLMProvider
from harness.ollama_provider import OllamaProvider


def _enable_line_editing() -> None:
    """Enable arrow keys, line editing, and persistent history for input().

    Importing readline activates cursor movement (left/right), in-session
    history (up/down), and emacs-style editing for the built-in input().
    History is also persisted across sessions. Degrades gracefully on
    platforms where readline is unavailable (e.g. vanilla Windows).
    """
    try:
        import atexit
        import readline
    except ImportError:
        return  # line editing unavailable; input() still works

    history_file = os.path.expanduser("~/.toyllm_history")
    try:
        readline.read_history_file(history_file)
    except OSError:
        pass  # no history yet (first run) or unreadable
    atexit.register(readline.write_history_file, history_file)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Interactive text generation through the LLMProvider interface.",
    )
    parser.add_argument(
        "--provider",
        choices=["toy", "ollama"],
        default="toy",
        help="Which LLM provider to use (default: toy).",
    )

    # Toy-model options
    parser.add_argument(
        "--checkpoint",
        default="data/checkpoint.pt",
        help="Toy model checkpoint (default: data/checkpoint.pt).",
    )
    parser.add_argument(
        "--vocab",
        default="data/vocab.json",
        help="Toy model vocabulary (default: data/vocab.json).",
    )

    # Ollama options
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:14b",
        help="Ollama model name (default: qwen2.5-coder:14b).",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434).",
    )

    # Shared sampling options
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Max new tokens to generate per prompt (default: 200).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8). Lower = more deterministic.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling cutoff (default: 40). 0 = disabled.",
    )
    return parser


def build_provider(args: argparse.Namespace) -> LLMProvider:
    """Construct the provider selected by --provider.

    Returns an object satisfying the LLMProvider interface, so the rest of the
    CLI never needs to know which concrete provider it is talking to.
    """
    top_k = args.top_k if args.top_k > 0 else None

    if args.provider == "ollama":
        return OllamaProvider(
            model=args.model,
            host=args.host,
            temperature=args.temperature,
            top_k=top_k,
        )

    # Default: the hand-built toy model.
    return ToyLLMProvider.from_checkpoint(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        temperature=args.temperature,
        top_k=top_k,
    )


def run_repl(provider: LLMProvider, max_tokens: int, label: str) -> None:
    """Run the interactive prompt loop against *provider*.

    Extracted from main() so the Ctrl-C / EOF handling is unit-testable with a
    fake provider and a monkeypatched input() — no model, server, or terminal.
    """
    print(f"{label} — type a prompt and press Enter. Empty line or 'quit' to exit.")
    print(f"(generating up to {max_tokens} tokens per prompt)\n")

    while True:
        try:
            prompt = input("prompt> ")
        except EOFError:                  # Ctrl-D
            print()
            break
        except KeyboardInterrupt:         # Ctrl-C at the prompt → exit cleanly
            print("\nBye!")
            break

        if prompt.strip() == "" or prompt.strip().lower() == "quit":
            break

        try:
            output = provider.generate(prompt, max_tokens=max_tokens)
        except KeyboardInterrupt:         # Ctrl-C while generating → cancel, re-prompt
            print("\n[cancelled]")
            continue
        except (ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        print(output)
        print()


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        provider = build_provider(args)
    except FileNotFoundError as exc:
        print(
            f"Could not load the toy model: {exc}\n"
            "Train it first with:  python -m model.train",
            file=sys.stderr,
        )
        sys.exit(1)

    _enable_line_editing()

    label = f"Ollama ({args.model})" if args.provider == "ollama" else "Toy LLM"
    run_repl(provider, args.max_tokens, label)
