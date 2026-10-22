"""Compare window MLP, temporal Conv1d, and baselines without future inputs."""

import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F


def make_series():
    generator = torch.Generator().manual_seed(340)
    t = torch.arange(360, dtype=torch.float32)
    return (
        20 + 0.025 * t + 3 * torch.sin(2 * math.pi * t / 24)
        + 0.8 * torch.cos(2 * math.pi * t / 12)
        + 0.15 * torch.randn(360, generator=generator)
    )


def make_windows(series, start, stop, window, horizon=6):
    origins = torch.arange(max(start, window), stop - horizon + 1)
    x = torch.stack([series[t - window:t] for t in origins.tolist()])
    y = torch.stack([series[t:t + horizon] for t in origins.tolist()])
    assert x.shape == (len(origins), window)
    assert y.shape == (len(origins), horizon)
    assert (origins + horizon <= stop).all()
    return x, y, origins


class TemporalConv(nn.Module):
    def __init__(self, window, horizon):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * (window - 8), horizon),
        )

    def forward(self, x):
        return self.network(x.unsqueeze(1))


def fit(model, train_x, train_y, validation_x, validation_y, steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_score, best_state, best_step = float("inf"), None, None
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        loss = F.mse_loss(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                score = (model(validation_x) - validation_y).abs().mean().item()
            if score < best_score:
                best_score, best_step = score, step
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return best_step


def forecasts(models, x, mean, scale):
    with torch.no_grad():
        predictions = {name: model(x) * scale + mean for name, model in models.items()}
    raw_x = x * scale + mean
    predictions["persistence"] = raw_x[:, -1:].expand(-1, 6)
    predictions["seasonal naive"] = raw_x[:, -24:-18]
    assert all(p.shape == (len(x), 6) and torch.isfinite(p).all() for p in predictions.values())
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    if not 24 <= args.window < 200 or args.steps < 1:
        parser.error("Use 24 <= window < 200 and steps >= 1.")
    torch.set_num_threads(1)
    series = make_series()
    mean, scale = series[:240].mean(), series[:240].std(correction=0)
    scaled = (series - mean) / scale
    train_x, train_y, train_origins = make_windows(scaled, 0, 240, args.window)
    validation_x, validation_y, validation_origins = make_windows(scaled, 240, 300, args.window)
    assert train_origins[-1] + 6 <= 240 and validation_origins[0].item() == 240
    torch.manual_seed(36)
    models = {
        "window MLP": nn.Sequential(nn.Linear(args.window, 32), nn.ReLU(), nn.Linear(32, 6)),
        "temporal Conv1d": TemporalConv(args.window, 6),
    }
    print("SYNTHETIC series; both learned models consume only the past window")
    print(f"MLP input={tuple(train_x.shape)}; Conv1d input={tuple(train_x.unsqueeze(1).shape)}")
    print(f"Conv lengths: {args.window} -> {args.window - 4} -> {args.window - 8}")
    for name, model in models.items():
        step = fit(model, train_x, train_y, validation_x, validation_y, args.steps)
        print(f"{name}: parameters={sum(p.numel() for p in model.parameters())}, validation checkpoint={step}")
    predictions = forecasts(models, validation_x, mean, scale)
    actual = validation_y * scale + mean
    scores = {}
    for name, prediction in predictions.items():
        error = (prediction - actual).abs()
        scores[name] = error.mean().item()
        print(f"validation {name:16s} MAE={error.mean():.3f}, by horizon={error.mean(0).tolist()}")
    winner = min(scores, key=scores.get)
    print(f"LOCKED using validation: {winner}")
    print(f"First validation prediction={predictions[winner][0].tolist()}")
    print(f"First validation actual    ={actual[0].tolist()}")
    if args.final_test:
        test_x, test_y, _ = make_windows(scaled, 300, 360, args.window)
        actual = test_y * scale + mean
        for name, prediction in forecasts(models, test_x, mean, scale).items():
            print(f"test {name:16s} MAE={(prediction - actual).abs().mean():.3f}")
        print(f"Selection remains {winner}; do not switch based on test results.")
    else:
        print("Test withheld. --final-test reports the frozen comparison.")


if __name__ == "__main__":
    main()
