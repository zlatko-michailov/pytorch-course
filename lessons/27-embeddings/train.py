"""Compare one-hot inputs and learned category lookups on synthetic records."""

import argparse

import torch
from torch import nn
from torch.nn import functional as F


class EmbeddingRegressor(nn.Module):
    def __init__(self, vocabulary_size, dimension):
        super().__init__()
        self.table = nn.Embedding(vocabulary_size + 1, dimension)
        with torch.no_grad():
            self.table.weight[0].zero_()
        self.head = nn.Sequential(nn.Linear(dimension + 1, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, numeric, ids):
        return self.head(torch.cat((numeric, self.table(ids)), dim=1))


class OneHotRegressor(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.head = nn.Sequential(
            nn.Linear(vocabulary_size + 1, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, numeric, ids):
        one_hot = F.one_hot(ids, self.vocabulary_size + 1)[:, 1:].float()
        return self.head(torch.cat((numeric, one_hot), dim=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dim", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.embedding_dim < 1 or args.steps < 1:
        parser.error("embedding-dim and steps must be positive.")
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(270)
    raw = 4 * torch.rand(480, 1, generator=generator) - 2
    names = [f"type-{i % 8}" for i in range(480)]
    for i in range(320, 480, 10):
        names[i] = "new-type"
    offsets = {f"type-{i}": (i - 3.5) * 0.6 for i in range(8)}
    offset = torch.tensor([offsets.get(name, 0.0) for name in names])[:, None]
    target = 0.7 * raw + offset + 0.2 * raw * offset
    target += 0.08 * torch.randn(target.shape, generator=generator)
    train, validation, test = slice(0, 320), slice(320, 400), slice(400, 480)
    vocabulary = {name: i + 1 for i, name in enumerate(sorted(set(names[train])))}
    ids = torch.tensor([vocabulary.get(name, 0) for name in names], dtype=torch.long)
    mean, scale = raw[train].mean(0), raw[train].std(0, correction=0)
    numeric = (raw - mean) / scale
    assert ids[train].min() >= 1 and (ids[validation] == 0).any()
    assert numeric.shape == target.shape == (480, 1)

    probe = nn.Embedding(3, 2)
    probe(torch.tensor([1, 2, 1])).sum().backward()
    assert torch.equal(probe.weight.grad, torch.tensor([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]]))
    print("SYNTHETIC category regression; IDs are labels, not ordered measurements")
    for name in ("one-hot", "embedding"):
        torch.manual_seed(27)
        model = (
            OneHotRegressor(len(vocabulary))
            if name == "one-hot"
            else EmbeddingRegressor(len(vocabulary), args.embedding_dim)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
        for _ in range(args.steps):
            optimizer.zero_grad()
            loss = F.mse_loss(model(numeric[train], ids[train]), target[train])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(numeric, ids)
        print(f"{name}: parameters={sum(p.numel() for p in model.parameters())}")
        for label, rows in (("train", train), ("validation", validation), ("test", test)):
            print(f"  {label:10s} MAE={(predictions[rows] - target[rows]).abs().mean():.3f}")
        for i in (320, 321):
            print(
                f"  {names[i]} id={ids[i].item()} input={raw[i].item():+.2f} "
                f"prediction={predictions[i].item():+.2f} target={target[i].item():+.2f}"
            )
        if name == "embedding":
            dense_lookup = F.one_hot(ids, len(vocabulary) + 1).float() @ model.table.weight
            assert torch.allclose(dense_lookup, model.table(ids))
            assert torch.equal(model.table.weight[0], torch.zeros(args.embedding_dim))
            print(f"  lookup == one_hot @ table; table shape={tuple(model.table.weight.shape)}")
            print(f"  first two learned rows={model.table.weight[1:3].detach().tolist()}")


if __name__ == "__main__":
    main()
