"""Synthetic tabular regression with training-only missing-value preprocessing."""

import argparse

import torch
from torch import nn
from torch.nn import functional as F


def make_data(missing_rate):
    generator = torch.Generator().manual_seed(260)
    numeric = torch.rand(480, 2, generator=generator) * torch.tensor([10.0, 20.0])
    categories = ["small", "medium", "large"]
    names = [categories[i % 3] for i in range(len(numeric))]
    for i in range(300, len(names), 13):
        names[i] = "prototype"
    effects = {"small": -2.0, "medium": 0.0, "large": 2.0, "prototype": 0.0}
    target = (
        8.0
        + 1.4 * numeric[:, 0]
        + 0.3 * numeric[:, 1]
        + torch.tensor([effects[name] for name in names])
        + 0.15 * torch.randn(480, generator=generator)
    )
    missing = torch.rand(numeric.shape, generator=generator) < missing_rate
    return numeric.masked_fill(missing, float("nan")), names, target[:, None]


def fit_preprocessor(numeric, names):
    observed = ~torch.isnan(numeric)
    if not observed.any(dim=0).all():
        raise ValueError("Each numerical column needs an observed training value.")
    mean = torch.where(observed, numeric, 0.0).sum(0) / observed.sum(0)
    filled = torch.where(observed, numeric, mean)
    scale = filled.std(0, correction=0).clamp_min(1e-6)
    vocabulary = {name: i + 1 for i, name in enumerate(sorted(set(names)))}
    return mean, scale, vocabulary


def transform(numeric, names, state):
    mean, scale, vocabulary = state
    assert numeric.ndim == 2 and numeric.shape == (len(names), len(mean))
    missing = torch.isnan(numeric)
    scaled = (torch.where(missing, mean, numeric) - mean) / scale
    ids = torch.tensor([vocabulary.get(name, 0) for name in names])
    # Dropping the UNK column makes an unseen category an all-zero category vector.
    categorical = F.one_hot(ids, num_classes=len(vocabulary) + 1)[:, 1:].float()
    features = torch.cat((scaled, missing.float(), categorical), dim=1)
    assert torch.isfinite(features).all()
    return features, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-rate", type=float, default=0.18)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    if not 0 <= args.missing_rate < 1 or args.steps < 1:
        parser.error("Use 0 <= missing-rate < 1 and steps >= 1.")
    torch.set_num_threads(1)
    torch.manual_seed(26)
    raw, names, target = make_data(args.missing_rate)
    train, validation, test = slice(0, 300), slice(300, 390), slice(390, 480)
    state = fit_preprocessor(raw[train], names[train])
    features, ids = transform(raw, names, state)
    assert torch.allclose(features[train, :2].mean(0), torch.zeros(2), atol=1e-6)
    probe, probe_ids = transform(torch.tensor([[float("nan"), 10.0]]), ["new"], state)
    assert probe_ids.item() == 0 and probe[0, 0].item() == 0
    assert probe[0, 2].item() == 1 and probe[0, 4:].sum().item() == 0
    assert target.shape == (len(features), 1)

    model = nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = F.mse_loss(model(features[train]), target[train])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(features)
    constant = target[train].mean()
    print("SYNTHETIC repair duration; CPU; fixed split, no real-world claims")
    print(f"Training mean={state[0].tolist()}, scale={state[1].tolist()}")
    print(f"Training vocabulary={state[2]}; UNK=0; features={tuple(features.shape)}")
    for label, rows in (("train", train), ("validation", validation), ("test", test)):
        mae = (prediction[rows] - target[rows]).abs().mean().item()
        baseline = (constant - target[rows]).abs().mean().item()
        print(f"{label:10s} MAE={mae:.3f} min; train-mean baseline={baseline:.3f}")
    for i in (300, 301, 313):
        print(
            f"row={i} raw={raw[i].tolist()} category={names[i]} id={ids[i].item()} "
            f"prediction={prediction[i].item():.2f} actual={target[i].item():.2f}"
        )


if __name__ == "__main__":
    main()
