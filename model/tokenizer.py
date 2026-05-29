import json


class CharTokenizer:
    def __init__(self, text: str | None = None):
        self._chars: list[str] = []
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}
        if text is not None:
            self.build_vocab(text)

    def build_vocab(self, text: str) -> None:
        self._chars = sorted(set(text))
        self._stoi = {ch: i for i, ch in enumerate(self._chars)}
        self._itos = {i: ch for i, ch in enumerate(self._chars)}

    def encode(self, text: str) -> list[int]:
        return [self._stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._itos[t] for t in tokens)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"chars": self._chars}, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.build_vocab("".join(data["chars"]))

    @property
    def vocab_size(self) -> int:
        return len(self._chars)
