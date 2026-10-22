"""An inspectable pre-normalization encoder block, not a framework wrapper."""
import math

import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, width=8, heads=2):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads.")
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(),
                                         nn.Linear(2 * width, width))

    def forward(self, x, valid):
        batch, length, width = x.shape
        if valid.shape != (batch, length) or valid.dtype != torch.bool:
            raise ValueError("valid must be a bool [B,T] mask.")
        if not valid.any(1).all().item():
            raise ValueError("Every sequence needs a real token.")
        q, k, v = self.qkv(self.attention_norm(x)).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, length, self.heads, width // self.heads)
                   .transpose(1, 2) for t in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(width // self.heads)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        weights = scores.softmax(-1)
        mixed = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        x = x + self.output(mixed)
        x = x + self.feedforward(self.feedforward_norm(x))
        return x * valid.unsqueeze(-1)


def main():
    torch.manual_seed(45)
    torch.set_num_threads(1)
    block = EncoderBlock()
    head = nn.Linear(8, 1)
    x = torch.randn(4, 5, 8)
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0],
                          [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    labels = torch.tensor([0., 1., 0., 1.])
    optimizer = torch.optim.Adam(list(block.parameters()) + list(head.parameters()), lr=0.02)
    losses = []
    for step in range(60):
        optimizer.zero_grad(set_to_none=True)
        states = block(x, valid)
        pooled = states.sum(1) / valid.sum(1, keepdim=True)
        logits = head(pooled).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        if step == 0:
            for name, parameter in block.named_parameters():
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all().item(), name
                assert parameter.grad.abs().sum().item() > 0, name
        optimizer.step()
        losses.append(loss.item())
    with torch.no_grad():
        changed = x.clone()
        changed[~valid] = 999
        torch.testing.assert_close(block(x, valid), block(changed, valid))
        assert block(x, valid)[~valid].eq(0).all().item()
    print(f"Four-example wiring check: BCE {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0]
    print("All block parameters received finite nonzero gradients; padding invariance passed.")
    print("This is a wiring/overfitting test, NOT a generalization result.")


if __name__ == "__main__":
    main()
