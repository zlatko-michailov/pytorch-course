"""Turn an explicitly synthetic series into leakage-safe forecast windows."""

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


def make_windows(series, start, stop, window, horizon):
    origins = torch.arange(max(start, window), stop - horizon + 1)
    if not len(origins):
        raise ValueError("The split must contain at least one complete target horizon.")
    x = torch.stack([series[t - window:t] for t in origins.tolist()])
    y = torch.stack([series[t:t + horizon] for t in origins.tolist()])
    assert (origins - window >= 0).all()
    assert (origins + horizon <= stop).all()
    assert x.shape == (len(origins), window) and y.shape == (len(origins), horizon)
    return x, y, origins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.window < 200 or not 1 <= args.horizon <= 60:
        parser.error("Use 1 <= window < 200 and 1 <= horizon <= 60.")
    torch.set_num_threads(1)
    torch.manual_seed(34)
    series = make_series()
    train_end, validation_end = 240, 300
    mean = series[:train_end].mean()
    scale = series[:train_end].std(correction=0).clamp_min(1e-6)
    scaled = (series - mean) / scale
    train_x, train_y, train_origins = make_windows(scaled, 0, train_end, args.window, args.horizon)
    validation_x, validation_y, validation_origins = make_windows(
        scaled, train_end, validation_end, args.window, args.horizon
    )
    assert train_origins[-1] + args.horizon <= train_end
    assert validation_origins[0].item() == train_end
    origin = validation_origins[0].item()
    changed_future = series.clone()
    changed_future[origin:] += 1000
    assert torch.equal(series[origin - args.window:origin], changed_future[origin - args.window:origin])

    model = nn.Linear(args.window, args.horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(250):
        optimizer.zero_grad()
        loss = F.mse_loss(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    print("SYNTHETIC series; origin t means first unknown target is series[t]")
    print(f"Training-only mean={mean:.3f}, scale={scale:.3f}")
    print(f"train X={tuple(train_x.shape)}, Y={tuple(train_y.shape)}")
    print(f"validation X={tuple(validation_x.shape)}, Y={tuple(validation_y.shape)}")
    print(
        f"First validation origin={origin}: inputs=[{origin - args.window},{origin}), "
        f"targets=[{origin},{origin + args.horizon})"
    )
    splits = [("validation", validation_x, validation_y)]
    if args.final_test:
        test_x, test_y, _ = make_windows(
            scaled, validation_end, len(series), args.window, args.horizon
        )
        splits.append(("test", test_x, test_y))
    with torch.no_grad():
        for label, x, y in splits:
            prediction = model(x) * scale + mean
            actual = y * scale + mean
            absolute_error = (prediction - actual).abs()
            print(f"{label} MAE={absolute_error.mean():.3f}; horizon MAE={absolute_error.mean(0).tolist()}")
            print(f"  first prediction={prediction[0].tolist()}")
            print(f"  first actual    ={actual[0].tolist()}")
    if not args.final_test:
        print("Test withheld; --final-test opens it after your design is frozen.")


if __name__ == "__main__":
    main()
