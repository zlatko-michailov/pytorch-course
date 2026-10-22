"""Persistence and recursive seasonal-naive forecasts on identical origins."""

import argparse
import math

import torch


def make_series():
    generator = torch.Generator().manual_seed(340)
    t = torch.arange(360, dtype=torch.float32)
    return (
        20 + 0.025 * t + 3 * torch.sin(2 * math.pi * t / 24)
        + 0.8 * torch.cos(2 * math.pi * t / 12)
        + 0.15 * torch.randn(360, generator=generator)
    )


def comparison(series, start, stop, horizon, season_length, train_mean):
    origins = torch.arange(start, stop - horizon + 1)
    x = torch.stack([series[t - 48:t] for t in origins.tolist()])
    target = torch.stack([series[t:t + horizon] for t in origins.tolist()])
    seasonal_positions = torch.arange(horizon) % season_length
    seasonal = x[:, -season_length:][:, seasonal_positions]
    predictions = {
        "train mean": torch.full_like(target, train_mean.item()),
        "persistence": x[:, -1:].expand(-1, horizon),
        "seasonal naive": seasonal,
    }
    source_indices = origins[:, None] - season_length + seasonal_positions
    assert (source_indices < origins[:, None]).all()
    assert torch.equal(seasonal, series[source_indices])
    assert all(prediction.shape == target.shape for prediction in predictions.values())
    assert (origins + horizon <= stop).all()
    return target, predictions, origins


def report(label, target, predictions):
    scores = {}
    for name, prediction in predictions.items():
        error = (prediction - target).abs()
        scores[name] = error.mean().item()
        print(f"{label} {name:14s} MAE={error.mean():.3f}; by horizon={error.mean(0).tolist()}")
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season-length", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.season_length <= 48 or not 1 <= args.horizon <= 60:
        parser.error("Use 1 <= season-length <= 48 and 1 <= horizon <= 60.")
    torch.set_num_threads(1)
    series = make_series()
    train_mean = series[:240].mean()
    target, predictions, origins = comparison(
        series, 240, 300, args.horizon, args.season_length, train_mean
    )
    print("SYNTHETIC series; no optimizer and no fitted seasonal period")
    print(f"Same validation origins for all baselines: {origins[0]}..{origins[-1]}, n={len(origins)}")
    scores = report("validation", target, predictions)
    winner = min(scores, key=scores.get)
    print(f"LOCKED using validation only: {winner}")
    print(f"First origin={origins[0].item()}, actual={target[0].tolist()}")
    for name, prediction in predictions.items():
        print(f"  {name}: {prediction[0].tolist()}")
    if args.final_test:
        target, predictions, origins = comparison(
            series, 300, 360, args.horizon, args.season_length, train_mean
        )
        print(f"SEALED TEST opened; chosen baseline remains {winner}")
        report("test", target, predictions)
    else:
        print("Test withheld; use --final-test only after choosing the baseline specification.")


if __name__ == "__main__":
    main()
