import dataclasses
import typing
import pathlib
import json

@dataclasses.dataclass
class Config:
    vocab_size: int = 30000
    min_freq: int = 2
    special_tokens: typing.List[str] = dataclasses.field(default_factory=lambda: [
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "[SEP]",
        "[MASK]"
    ])
    max_word_len: int = 100
    min_subword_len: int = 2
    lowercase: bool = True
    normalize_unicode: bool = True
    remove_accents: bool = False
    clean_text: bool = True
    merge_method: str = "bpe" # "bpe" or "wordpiece"

    def save(self, path: pathlib.Path):
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: pathlib.Path) -> "Config":
        with open(path / "config.json", "r", encoding="utf-8") as f:
            return cls(**json.load(f))