"""Expanding-window validation, horizon errors, then one optional sealed test."""

import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F


WINDOW, HORIZON = 48, 6
FOLDS = ((240, 288), (288, 336), (336, 384))
TEST_START, SERIES_LENGTH = 384, 480
CANDIDATES = ("persistence", "seasonal naive", "linear window", "window MLP")


def make_series():
    generator = torch.Generator().manual_seed(340)
    t = torch.arange(SERIES_LENGTH, dtype=torch.float32)
    return (
        20 + 0.025 * t + 3 * torch.sin(2 * math.pi * t / 24)
        + 0.8 * torch.cos(2 * math.pi * t / 12)
        + 0.15 * torch.randn(SERIES_LENGTH, generator=generator)
    )


def windows(series, origins):
    x = torch.stack([series[t - WINDOW:t] for t in origins.tolist()])
    y = torch.stack([series[t:t + HORIZON] for t in origins.tolist()])
    assert (origins >= WINDOW).all() and (origins + HORIZON <= len(series)).all()
    assert x.shape == (len(origins), WINDOW) and y.shape == (len(origins), HORIZON)
    return x, y


def fit_candidate(name, observed, steps):
    """The argument is a prefix, so the fitter cannot inspect later values."""
    if name in ("persistence", "seasonal naive"):
        return None
    mean, scale = observed.mean(), observed.std(correction=0).clamp_min(1e-6)
    train_origins = torch.arange(WINDOW, len(observed) - HORIZON + 1)
    assert train_origins[-1] + HORIZON <= len(observed)
    x, y = windows((observed - mean) / scale, train_origins)
    torch.manual_seed(37)
    model = (
        nn.Linear(WINDOW, HORIZON)
        if name == "linear window"
        else nn.Sequential(nn.Linear(WINDOW, 24), nn.ReLU(), nn.Linear(24, HORIZON))
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, mean, scale


def predict(name, fitted, x):
    if name == "persistence":
        return x[:, -1:].expand(-1, HORIZON)
    if name == "seasonal naive":
        return x[:, -24:-24 + HORIZON]
    model, mean, scale = fitted
    with torch.no_grad():
        prediction = model((x - mean) / scale) * scale + mean
    assert prediction.shape == (len(x), HORIZON) and torch.isfinite(prediction).all()
    return prediction


def evaluate_block(series, start, stop, name, steps):
    # Six-step spacing avoids duplicate target timestamps within a block.
    origins = torch.arange(start, stop - HORIZON + 1, HORIZON)
    assert (origins + HORIZON <= stop).all()
    fitted = fit_candidate(name, series[:start], steps)
    x, y = windows(series, origins)
    prediction = predict(name, fitted, x)
    return (prediction - y).abs(), prediction, y, origins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive.")
    torch.set_num_threads(1)
    series = make_series()
    assert FOLDS[-1][1] == TEST_START
    assert all(left[1] == right[0] for left, right in zip(FOLDS, FOLDS[1:]))
    changed = series.clone()
    changed[240:] += 1000
    assert torch.equal(series[240 - WINDOW:240], changed[240 - WINDOW:240])
    errors = {name: [] for name in CANDIDATES}
    print("SYNTHETIC expanding-window backtest; fixed recipes, no test-driven choices")
    print("Origin stride=6, horizon=6; shared history still makes errors dependent.")
    for start, stop in FOLDS:
        print(f"\nRefit using [0,{start}); evaluate target blocks inside [{start},{stop})")
        for name in CANDIDATES:
            error, _, _, origins = evaluate_block(series, start, stop, name, args.steps)
            errors[name].append(error)
            print(f"  {name:14s} origins={len(origins)} MAE={error.mean():.3f}")
    scores = {}
    print("\nAggregate validation: each origin/horizon pair receives equal weight")
    for name in CANDIDATES:
        error = torch.cat(errors[name], dim=0)
        scores[name] = error.mean().item()
        print(f"{name:14s} MAE={error.mean():.3f}; horizon MAE={error.mean(0).tolist()}")
    winner = min(scores, key=scores.get)
    print(f"LOCKED recipe using validation only: {winner}")
    if not args.final_test:
        print("SEALED TEST [384,480) not evaluated. Freeze choices, then add --final-test.")
        return
    error, prediction, target, origins = evaluate_block(
        series, TEST_START, SERIES_LENGTH, winner, args.steps
    )
    print(f"\nFINAL TEST: refit selected recipe on [0,{TEST_START}); no test refits")
    print(f"origins={len(origins)}, MAE={error.mean():.3f}; horizon MAE={error.mean(0).tolist()}")
    print(f"origin={origins[0].item()} prediction={prediction[0].tolist()}")
    print(f"origin={origins[0].item()} actual    ={target[0].tolist()}")
    print("Do not reopen model selection after this result; test labels never enter fitting.")


if __name__ == "__main__":
    main()
