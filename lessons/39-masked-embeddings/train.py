"""Variable-length collation and padding-invariant embedding means."""
import torch
from torch import nn

from course.text_data import PAD, TEXT_SPLITS, encode, make_vocab


def collate(texts, vocabulary, extra_padding=0):
    sequences = [encode(text, vocabulary) for text in texts]
    if not sequences or any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("A batch must contain nonempty token sequences.")
    if extra_padding < 0:
        raise ValueError("extra_padding must be nonnegative.")
    width = max(map(len, sequences)) + extra_padding
    ids = torch.full((len(sequences), width), PAD, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return ids


def masked_mean(embedded, ids):
    mask = ids.ne(PAD).unsqueeze(-1)
    counts = mask.sum(dim=1)
    if (counts == 0).any().item():
        raise ValueError("Cannot pool an all-padding sequence.")
    return (embedded * mask).sum(dim=1) / counts


def main():
    torch.manual_seed(39)
    torch.set_num_threads(1)
    vocabulary = make_vocab([text for text, _ in TEXT_SPLITS["train"]])
    texts = ["good service", "helpful calm service", "unfamiliar"]
    ids = collate(texts, vocabulary)
    embedding = nn.Embedding(len(vocabulary), 4, padding_idx=PAD)
    embedded = embedding(ids)
    pooled = masked_mean(embedded, ids)
    longer_ids = collate(texts, vocabulary, extra_padding=4)
    longer_mean = masked_mean(embedding(longer_ids), longer_ids)
    torch.testing.assert_close(pooled, longer_mean)
    assert ids.dtype == torch.long and pooled.shape == (3, 4)
    for bad in ([], ["   "]):
        try:
            collate(bad, vocabulary)
        except ValueError as error:
            print("Expected rejection:", error)
        else:
            raise AssertionError("Empty input was accepted.")
    try:
        masked_mean(embedding(torch.zeros(1, 2, dtype=torch.long)),
                    torch.zeros(1, 2, dtype=torch.long))
    except ValueError as error:
        print("Expected all-padding rejection:", error)
    else:
        raise AssertionError("All-padding pooling was accepted.")
    pooled.square().sum().backward()
    assert embedding.weight.grad[PAD].eq(0).all().item()
    assert embedding.weight.grad[1].abs().sum().item() > 0
    print("IDs:\n", ids)
    print("IDs -> embeddings -> mean:", ids.shape, embedded.shape, pooled.shape)
    print("Masked mean:\n", pooled.detach())
    print("Extra-padding invariance and PAD/UNK gradient checks passed.")


if __name__ == "__main__":
    main()
