"""A complete, intentionally tiny synthetic bag-of-tokens classifier."""
import torch
from torch import nn

from course.text_data import PAD, UNK, TEXT_SPLITS, encode, make_vocab


def collate(rows, vocabulary):
    sequences = [encode(text, vocabulary) for text, _ in rows]
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Every example must contain at least one token.")
    ids = torch.full((len(rows), max(map(len, sequences))), PAD, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, :len(sequence)] = torch.tensor(sequence)
    labels = torch.tensor([label for _, label in rows], dtype=torch.float32)
    return ids, labels


class MeanClassifier(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 8, padding_idx=PAD)
        self.output = nn.Linear(8, 1)
        with torch.no_grad():
            self.embedding.weight[UNK].zero_()

    def forward(self, ids):
        mask = ids.ne(PAD).unsqueeze(-1)
        counts = mask.sum(1)
        if (counts == 0).any().item():
            raise ValueError("Cannot classify an all-padding sequence.")
        mean = (self.embedding(ids) * mask).sum(1) / counts
        return self.output(mean).squeeze(-1)


@torch.no_grad()
def evaluate(model, batch):
    model.eval()
    ids, labels = batch
    logits = model(ids)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    accuracy = (logits.ge(0) == labels.bool()).float().mean().item()
    return loss, accuracy


def main():
    torch.manual_seed(40)
    torch.set_num_threads(1)
    vocabulary = make_vocab([text for text, _ in TEXT_SPLITS["train"]])
    train = collate(TEXT_SPLITS["train"], vocabulary)
    validation = collate(TEXT_SPLITS["validation"], vocabulary)
    model = MeanClassifier(len(vocabulary))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(train[0]), train[1])
        loss.backward()
        optimizer.step()
        validation_loss, _ = evaluate(model, validation)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    print("SYNTHETIC 24/8/8 examples; not a real NLP benchmark.")
    print(f"Validation-selected epoch: {best_epoch}; vocabulary size: {len(vocabulary)}")
    for name, batch in (("train", train), ("validation", validation)):
        loss, accuracy = evaluate(model, batch)
        print(f"{name}: BCE={loss:.4f}, accuracy={accuracy:.3f}")
    test = collate(TEXT_SPLITS["test"], vocabulary)
    loss, accuracy = evaluate(model, test)
    print(f"Final test (not used for selection): BCE={loss:.4f}, accuracy={accuracy:.3f}")
    model.eval()
    with torch.no_grad():
        padded = nn.functional.pad(test[0], (0, 5), value=PAD)
        reference = model(test[0])
        torch.testing.assert_close(reference, model(padded))
        padding_row = model.embedding.weight[PAD].clone()
        model.embedding.weight[PAD].fill_(1000.)
        torch.testing.assert_close(reference, model(padded))
        model.embedding.weight[PAD].copy_(padding_row)
    print("Padding-length/perturbation invariance passed. UNK starts at zero; no semantic fallback.")


if __name__ == "__main__":
    main()
