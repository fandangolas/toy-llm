"""
Interactive CLI for the toy LLM harness.

Run with:  python -m harness [options]

Type a prompt and press Enter to generate a continuation.
Press Enter on an empty line, type "quit", or send EOF (Ctrl-D) to exit.
"""

import argparse
import sys

from harness.toy_provider import ToyLLMProvider


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Interactive text generation using the toy GPT model.",
    )
    parser.add_argument(
        "--checkpoint",
        default="data/checkpoint.pt",
        help="Path to the model checkpoint (default: data/checkpoint.pt).",
    )
    parser.add_argument(
        "--vocab",
        default="data/vocab.json",
        help="Path to the vocabulary file (default: data/vocab.json).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Number of new characters to generate per prompt (default: 200).",
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


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    top_k = args.top_k if args.top_k > 0 else None

    try:
        provider = ToyLLMProvider.from_checkpoint(
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab,
            temperature=args.temperature,
            top_k=top_k,
        )
    except FileNotFoundError as exc:
        print(
            f"Could not load model: {exc}\n"
            "Train the model first with:  python -m model.train",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Toy LLM — type a prompt and press Enter. Empty line or 'quit' to exit.")
    print(f"(generating {args.max_tokens} characters per prompt)\n")

    while True:
        try:
            prompt = input("prompt> ")
        except EOFError:
            print()
            break

        if prompt.strip() == "" or prompt.strip().lower() == "quit":
            break

        try:
            continuation = provider.generate(prompt, max_tokens=args.max_tokens)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        print(continuation)
        print()
