"""Explicit batched multi-head attention, with two independent references."""
import math

import torch
from torch import nn


def split_heads(projected, heads):
    batch, tokens, width = projected.shape
    if width % heads:
        raise ValueError("Model width must be divisible by number of heads.")
    return projected.reshape(batch, tokens, heads, width // heads).transpose(1, 2)


def main():
    torch.manual_seed(43)
    torch.set_num_threads(1)
    batch, tokens, width, heads = 2, 3, 8, 2
    depth = width // heads
    x = torch.randn(batch, tokens, width)
    projections = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(3)])
    output_projection = nn.Linear(width, width, bias=False)
    q, k, v = [split_heads(layer(x), heads) for layer in projections]
    assert q.shape == k.shape == v.shape == (batch, heads, tokens, depth)
    scores = q @ k.transpose(-2, -1) / math.sqrt(depth)
    assert scores.shape == (batch, heads, tokens, tokens)
    weights = scores.softmax(-1)
    attended = weights @ v
    reference = torch.empty_like(attended)
    for b in range(batch):
        for h in range(heads):
            one_scores = q[b, h] @ k[b, h].T / math.sqrt(depth)
            reference[b, h] = one_scores.softmax(-1) @ v[b, h]
    torch.testing.assert_close(attended, reference)
    builtin = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    torch.testing.assert_close(attended, builtin)
    merged = attended.transpose(1, 2).contiguous().reshape(batch, tokens, width)
    result = output_projection(merged)
    assert result.shape == x.shape
    result.square().mean().backward()
    for layer in [*projections, output_projection]:
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all().item()
        assert layer.weight.grad.abs().sum().item() > 0
    print("X:", tuple(x.shape), "Q/K/V per head:", tuple(q.shape))
    print("scores/weights:", tuple(scores.shape), "head values:", tuple(attended.shape))
    print("merged/output:", tuple(merged.shape), tuple(result.shape))
    print("Loop reference, PyTorch reference, and all projection-gradient checks passed.")


if __name__ == "__main__":
    main()
