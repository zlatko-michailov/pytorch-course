"""Same split and optimizer-update budget, different order inductive biases."""
import math

import torch
from torch import nn


TEMPLATES = {
    "train": ["{a} chases {b}", "today {a} chases {b}", "{a} chases {b} quietly",
              "today quietly {a} chases {b}", "{a} today chases {b} quietly",
              "today {a} quietly chases {b} today"],
    "validation": ["quietly {a} chases {b}", "{a} chases {b} today quietly"],
    "test": ["{a} quietly chases {b}", "quietly today {a} chases {b}"],
}


def rows(split):
    return [(template.format(a=a, b=b), label)
            for template in TEMPLATES[split]
            for a, b, label in (("cat", "dog", 1.), ("dog", "cat", 0.))]


def batch(examples, vocabulary):
    tokens = [[vocabulary.get(word, 1) for word in text.split()] for text, _ in examples]
    ids = torch.zeros(len(tokens), max(map(len, tokens)), dtype=torch.long)
    for i, sequence in enumerate(tokens):
        ids[i, :len(sequence)] = torch.tensor(sequence)
    return ids, torch.tensor([label for _, label in examples])


def positions(length, width):
    p = torch.arange(length).float().unsqueeze(1)
    f = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.) / width))
    table = torch.zeros(length, width)
    table[:, 0::2], table[:, 1::2] = (p * f).sin(), (p * f).cos()
    return table


class Classifier(nn.Module):
    def __init__(self, vocabulary_size, attention):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(vocabulary_size, 16, padding_idx=0)
        self.output = nn.Linear(16, 1)
        if attention:
            self.norm1 = nn.LayerNorm(16)
            self.qkv = nn.Linear(16, 48)
            self.mix = nn.Linear(16, 16)
            self.norm2 = nn.LayerNorm(16)
            self.ff = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))

    def forward(self, ids):
        valid = ids.ne(0)
        if not valid.any(1).all().item():
            raise ValueError("All-padding input is invalid.")
        x = self.embedding(ids)
        if self.attention:
            batch_size, length, width = x.shape
            x = x + positions(length, width)
            q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
            q, k, v = [t.reshape(batch_size, length, 2, 8).transpose(1, 2)
                       for t in (q, k, v)]
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(8)
            scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
            mixed = (scores.softmax(-1) @ v).transpose(1, 2).contiguous().reshape_as(x)
            x = x + self.mix(mixed)
            x = x + self.ff(self.norm2(x))
        mean = (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        return self.output(mean).squeeze(-1)


@torch.no_grad()
def metrics(model, data):
    model.eval()
    logits = model(data[0])
    loss = nn.functional.binary_cross_entropy_with_logits(logits, data[1]).item()
    accuracy = (logits.ge(0) == data[1].bool()).float().mean().item()
    return loss, accuracy


def fit(model, train, validation):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    best_loss, state, selected_step = float("inf"), None, 0
    for step in range(1, 241):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(train[0]), train[1])
        loss.backward()
        if step == 1 and model.attention:
            assert model.qkv.weight.grad.abs().sum().item() > 0
        optimizer.step()
        val_loss, _ = metrics(model, validation)
        if val_loss < best_loss:
            best_loss, selected_step = val_loss, step
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    return selected_step


def main():
    torch.set_num_threads(1)
    vocabulary = {"<pad>": 0, "<unk>": 1}
    words = sorted({word for text, _ in rows("train") for word in text.split()})
    vocabulary.update({word: i for i, word in enumerate(words, start=2)})
    train, validation = [batch(rows(split), vocabulary) for split in ("train", "validation")]
    models = {}
    print("SYNTHETIC role/order task; templates, not independent natural-language documents.")
    print("Both models: same data, BCE, Adam lr=.02, 240 full-batch updates, validation BCE selection.")
    for name, attention in (("mean", False), ("attention+positions", True)):
        torch.manual_seed(46)
        model = Classifier(len(vocabulary), attention)
        selected_step = fit(model, train, validation)
        models[name] = model
        print(f"{name}: parameters={sum(p.numel() for p in model.parameters())}, "
              f"selected step={selected_step}, train={metrics(model, train)}, "
              f"validation={metrics(model, validation)}")
    # Neither final test is available to fit() or to the checkpoint-selection rule.
    test = batch(rows("test"), vocabulary)
    for name, model in models.items():
        print(f"Final test {name}: BCE,accuracy={metrics(model, test)}")
        with torch.no_grad():
            padded = nn.functional.pad(test[0], (0, 4), value=0)
            reference = model(test[0])
            torch.testing.assert_close(reference, model(padded), atol=1e-6, rtol=1e-5)
            padding_row = model.embedding.weight[0].clone()
            model.embedding.weight[0].fill_(1000.)
            torch.testing.assert_close(reference, model(padded), atol=1e-6, rtol=1e-5)
            model.embedding.weight[0].copy_(padding_row)
    with torch.no_grad():
        paired = models["mean"](test[0]).reshape(-1, 2)
        torch.testing.assert_close(paired[:, 0], paired[:, 1], atol=1e-6, rtol=1e-5)
    print("Padding and mean-collision checks passed; attention gains are not guaranteed.")
    print("Equal update budget is NOT equal parameter count, compute cost, or statistical evidence.")


if __name__ == "__main__":
    main()
