"""Inspectable synthetic text fixtures and train-only vocabulary, introduced in 38."""

PAD = 0
UNK = 1

TEXT_SPLITS = {
    "train": [
        ("good service", 1), ("bright team", 1), ("helpful service today", 1),
        ("friendly team today", 1), ("good calm visit", 1),
        ("bright service today", 1), ("helpful team", 1), ("friendly visit", 1),
        ("good team today", 1), ("bright calm visit", 1),
        ("helpful calm service", 1), ("friendly service today", 1),
        ("bad service", 0), ("dull team", 0), ("rude service today", 0),
        ("slow team today", 0), ("bad calm visit", 0),
        ("dull service today", 0), ("rude team", 0), ("slow visit", 0),
        ("bad team today", 0), ("dull calm visit", 0),
        ("rude calm service", 0), ("slow service today", 0),
    ],
    "validation": [
        ("good visit today", 1), ("bright service", 1),
        ("helpful weekend team", 1), ("friendly calm team", 1),
        ("bad visit today", 0), ("dull service", 0),
        ("rude weekend team", 0), ("slow calm team", 0),
    ],
    "test": [
        ("good evening team", 1), ("bright visit today", 1),
        ("helpful visit today", 1), ("friendly calm service", 1),
        ("bad evening team", 0), ("dull visit today", 0),
        ("rude visit today", 0), ("slow calm service", 0),
    ],
}


def tokenize(text: str) -> list[str]:
    """A deliberately limited tokenizer: lowercase, then split whitespace."""
    return text.lower().split()


def make_vocab(training_texts: list[str]) -> dict[str, int]:
    words = sorted({word for text in training_texts for word in tokenize(text)})
    return {"<pad>": PAD, "<unk>": UNK, **{
        word: index for index, word in enumerate(words, start=2)
    }}


def encode(text: str, vocabulary: dict[str, int]) -> list[int]:
    return [vocabulary.get(word, UNK) for word in tokenize(text)]
