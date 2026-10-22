"""Run from the course root: python -m lessons.38-text-vocabulary.train."""
from course.text_data import PAD, UNK, TEXT_SPLITS, encode, make_vocab, tokenize


def main():
    train = TEXT_SPLITS["train"]
    vocabulary = make_vocab([text for text, _ in train])
    print("SYNTHETIC fixture: no claim about real language.")
    for text, label in train:
        print(f"train label={label}: {text!r}")
    print("Train-only vocabulary:", vocabulary)
    for split in ("validation", "test"):
        unknown = sorted({
            word for text, _ in TEXT_SPLITS[split]
            for word in tokenize(text) if word not in vocabulary
        })
        print(f"{split} out-of-vocabulary tokens: {unknown}")
    assert "weekend" not in vocabulary and "evening" not in vocabulary
    assert PAD != UNK
    assert encode("GOOD unfamiliar", vocabulary) == [vocabulary["good"], UNK]
    assert encode("   ", vocabulary) == []
    assert vocabulary == make_vocab(list(reversed([text for text, _ in train])))
    print("Encoded new input:", encode("helpful unfamiliar service", vocabulary))
    print("Contracts passed: deterministic IDs, distinct PAD/UNK, no vocabulary leakage.")


if __name__ == "__main__":
    main()
