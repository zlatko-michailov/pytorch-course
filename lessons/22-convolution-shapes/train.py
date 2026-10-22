import argparse

import torch
from torch import nn

from course.image_models import cnn
from course.image_training import mlp


def main():
    parser = argparse.ArgumentParser(description="Session 22: self-contained convolution shape demo")
    parser.add_argument("--smoke", action="store_true",
                        help="Same deterministic shape demo; no external data in either mode")
    parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2200)
    images = torch.arange(2 * 28 * 28, dtype=torch.float32).reshape(2, 1, 28, 28) / 784
    model = cnn()
    current = images
    print("input", tuple(current.shape))
    for layer in model:
        current = layer(current)
        print(type(layer).__name__, tuple(current.shape))
    assert current.shape == (2, 10)
    assert sum(p.numel() for p in model.parameters()) == 9098
    for name, candidate in [("MLP", mlp()), ("CNN", model)]:
        print(name, "parameters:", sum(p.numel() for p in candidate.parameters()))
    kernel = nn.Conv2d(1, 1, 3, bias=False)
    with torch.no_grad():
        kernel.weight.fill_(1.0)
        actual = kernel(images[:1])
    expected = images[:1, :, :3, :3].sum()
    torch.testing.assert_close(actual[0, 0, 0, 0], expected)
    assert actual.shape == (1, 1, 26, 26)
    print("Top-left convolution is the sum of its local 3x3 patch:", expected.item())


if __name__ == "__main__":
    main()
