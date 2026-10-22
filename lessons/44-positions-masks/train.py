"""Positions, key padding, and causal visibility without all-masked-row NaNs."""
import math

import torch


def positions(length, width):
    if width % 2:
        raise ValueError("This sinusoidal implementation requires even width.")
    index = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
    table = torch.zeros(length, width)
    table[:, 0::2] = torch.sin(index * frequency)
    table[:, 1::2] = torch.cos(index * frequency)
    return table


def attend(x, valid, causal=False):
    if x.ndim != 3 or valid.shape != x.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("Expected x [B,T,D] and bool valid [B,T].")
    if not valid.any(dim=1).all().item():
        raise ValueError("Every sequence needs at least one real token.")
    _, length, width = x.shape
    z = x + positions(length, width)
    scores = z @ z.transpose(-2, -1) / math.sqrt(width)
    allowed = valid[:, None, :].expand(-1, length, -1)
    if causal:
        allowed = allowed & torch.ones(length, length, dtype=torch.bool).tril()
    has_key = allowed.any(dim=-1, keepdim=True)
    masked_scores = scores.masked_fill(~allowed, float("-inf"))
    # A leading padding query can have no visible key under a causal mask.
    safe_scores = torch.where(has_key, masked_scores, torch.zeros_like(scores))
    weights = safe_scores.softmax(-1) * allowed
    result = (weights @ z) * valid.unsqueeze(-1)
    return result, weights


def main():
    torch.manual_seed(44)
    torch.set_num_threads(1)
    x = torch.randn(2, 5, 4)
    valid = torch.tensor([[True, True, True, False, False],
                          [False, True, True, True, False]])
    noisy_padding = x.clone()
    noisy_padding[~valid] = 1000.0
    for causal in (False, True):
        output, weights = attend(x, valid, causal)
        changed, _ = attend(noisy_padding, valid, causal)
        torch.testing.assert_close(output, changed)
        assert torch.isfinite(output).all().item()
        assert torch.isfinite(weights).all().item()
        assert output[~valid].eq(0).all().item()
        key_mask = ~valid[:, None, :].expand_as(weights)
        assert weights[key_mask].eq(0).all().item()
        if causal:
            assert weights.triu(diagonal=1).eq(0).all().item()
        extended = torch.cat([x, torch.randn(2, 3, 4)], dim=1)
        extended_valid = torch.cat([valid, torch.zeros(2, 3, dtype=torch.bool)], dim=1)
        extended_output, _ = attend(extended, extended_valid, causal)
        torch.testing.assert_close(output, extended_output[:, :5])
        print(f"causal={causal}: finite, zero padding influence, right-padding invariance")
    future_changed = x.clone()
    future_changed[:, 2] += 3
    causal_before, _ = attend(x, valid, True)
    causal_after, _ = attend(future_changed, valid, True)
    torch.testing.assert_close(causal_before[:, :2], causal_after[:, :2])
    bidirectional_before, _ = attend(x, valid, False)
    bidirectional_after, _ = attend(future_changed, valid, False)
    assert not torch.allclose(bidirectional_before[:, :2], bidirectional_after[:, :2])
    try:
        attend(x[:1], torch.zeros(1, 5, dtype=torch.bool), True)
    except ValueError as error:
        print("Expected rejection:", error)
    else:
        raise AssertionError("All-padding sequence was accepted.")
    print("Future-token isolation passed for causal attention; bidirectional can see it.")


if __name__ == "__main__":
    main()
