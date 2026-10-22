"""Honest synthetic tabular comparison, including a small CART regression tree."""

import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Node:
    value: float
    feature: int | None = None
    threshold: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None


def fit_tree(x, y, max_depth, min_leaf=12):
    """Greedy squared-error CART: all distinct adjacent-value midpoints."""
    node = Node(y.mean().item())
    n = len(y)
    if max_depth == 0 or n < 2 * min_leaf:
        return node
    parent_error = ((y - y.mean()) ** 2).sum().item()
    best_error, best_feature, best_threshold = parent_error, None, None
    counts = torch.arange(1, n, dtype=x.dtype)
    for feature in range(x.shape[1]):
        order = torch.argsort(x[:, feature], stable=True)
        values, sorted_y = x[order, feature], y[order]
        sums = sorted_y.cumsum(0)
        squares = sorted_y.square().cumsum(0)
        left_error = squares[:-1] - sums[:-1].square() / counts
        right_error = squares[-1] - squares[:-1]
        right_error -= (sums[-1] - sums[:-1]).square() / (n - counts)
        valid = (values[:-1] < values[1:]) & (counts >= min_leaf) & (n - counts >= min_leaf)
        errors = (left_error + right_error).masked_fill(~valid, float("inf"))
        position = errors.argmin().item()
        error = errors[position].item()
        if error < best_error - 1e-6:
            best_error = error
            best_feature = feature
            best_threshold = ((values[position] + values[position + 1]) / 2).item()
    if best_feature is not None:
        left = x[:, best_feature] <= best_threshold
        node.feature, node.threshold = best_feature, best_threshold
        node.left = fit_tree(x[left], y[left], max_depth - 1, min_leaf)
        node.right = fit_tree(x[~left], y[~left], max_depth - 1, min_leaf)
    return node


def predict_tree(node, x):
    result = torch.full((len(x),), node.value, dtype=x.dtype)
    if node.feature is not None:
        left = x[:, node.feature] <= node.threshold
        result[left] = predict_tree(node.left, x[left])
        result[~left] = predict_tree(node.right, x[~left])
    return result


def fit_network(x, y, width):
    torch.manual_seed(29)
    model = (
        nn.Linear(x.shape[1], 1)
        if width == 0
        else nn.Sequential(nn.Linear(x.shape[1], width), nn.ReLU(), nn.Linear(width, 1))
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    for _ in range(350):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", type=float, default=0.12)
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    if args.noise < 0:
        parser.error("noise must be nonnegative.")
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(290)
    numeric = 4 * torch.rand(600, 2, generator=generator) - 2
    names = [("A", "B", "C")[i % 3] for i in range(600)]
    category_effect = torch.tensor([{"A": -1.0, "B": 0.0, "C": 1.0}[s] for s in names])
    y = torch.where(numeric[:, 0] > 0, 3.0, -3.0)
    y += 2 * (numeric[:, 1] > 0.5).float() + category_effect
    y += args.noise * torch.randn(600, generator=generator)
    permutation = torch.randperm(600, generator=generator)
    train, validation, test = permutation[:360], permutation[360:480], permutation[480:]
    vocabulary = {name: i + 1 for i, name in enumerate(sorted({names[i] for i in train.tolist()}))}
    ids = torch.tensor([vocabulary.get(name, 0) for name in names])
    mean = numeric[train].mean(0)
    scale = numeric[train].std(0, correction=0)
    x = torch.cat(
        ((numeric - mean) / scale, F.one_hot(ids, len(vocabulary) + 1)[:, 1:].float()), dim=1
    )
    assert x.shape == (600, 5) and y.shape == (600,)
    # A constant feature must remain a leaf; an exact step must be learnable.
    constant_tree = fit_tree(torch.zeros(8, 1), torch.arange(8).float(), 3, min_leaf=2)
    assert constant_tree.feature is None
    step_x = torch.arange(8).float()[:, None]
    step_y = (step_x[:, 0] >= 4).float()
    assert torch.equal(predict_tree(fit_tree(step_x, step_y, 1, 2), step_x), step_y)

    print("SYNTHETIC piecewise regression; same features, rows, targets and metric for every model")
    finalists = []
    for family, options in (("linear", (0,)), ("MLP", (8, 24)), ("CART", (2, 4, 6))):
        candidates = []
        for option in options:
            if family == "CART":
                model = fit_tree(x[train], y[train], option)
                with torch.no_grad():
                    prediction = predict_tree(model, x)
            else:
                model = fit_network(x[train], y[train], option)
                with torch.no_grad():
                    prediction = model(x).squeeze(1)
            train_mae = (prediction[train] - y[train]).abs().mean().item()
            validation_mae = (prediction[validation] - y[validation]).abs().mean().item()
            print(f"{family:6s} setting={option:2d}: train MAE={train_mae:.3f}, val MAE={validation_mae:.3f}")
            candidates.append((validation_mae, family, option, prediction))
        finalists.append(min(candidates, key=lambda candidate: candidate[0]))
    winner = min(finalists, key=lambda candidate: candidate[0])
    constant = y[train].mean()
    print(f"Train-mean validation MAE={(constant - y[validation]).abs().mean():.3f}")
    print(f"LOCKED by validation: {winner[1]}, setting={winner[2]}")
    first = validation[0].item()
    for _, family, option, prediction in finalists:
        print(
            f"Validation row={first}: {family}({option}) prediction={prediction[first]:+.2f}, "
            f"actual={y[first]:+.2f}"
        )
    if args.final_test:
        print("SEALED TEST opened after selection; all three predeclared family finalists:")
        for _, family, option, prediction in finalists:
            print(f"  {family}({option}) test MAE={(prediction[test] - y[test]).abs().mean():.3f}")
        print(f"  Train-mean test MAE={(constant - y[test]).abs().mean():.3f}")
    else:
        print("Test metrics withheld. Add --final-test only after freezing your experiment.")


if __name__ == "__main__":
    main()
