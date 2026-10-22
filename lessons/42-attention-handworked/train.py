"""Two-token scaled dot-product attention, checked against scalar arithmetic."""
import math

import torch


def main():
    torch.set_num_threads(1)
    # X is the identity: these projections make Q/K identity and V easy to read.
    x = torch.eye(2)
    w_q = torch.eye(2)
    w_k = torch.eye(2)
    w_v = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    q = (x @ w_q).requires_grad_()
    k = (x @ w_k).requires_grad_()
    v = (x @ w_v).requires_grad_()
    scores = q @ k.T / math.sqrt(q.shape[-1])
    weights = scores.softmax(dim=-1)
    output = weights @ v
    a = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
    b = 1 - a
    hand_weights = torch.tensor([[a, b], [b, a]])
    hand_output = torch.tensor([[2 * a, 4 * b], [2 * b, 4 * a]])
    torch.testing.assert_close(weights, hand_weights)
    torch.testing.assert_close(output, hand_output)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2))
    output[0, 0].backward()
    for name, tensor in (("Q", q), ("K", k), ("V", v)):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all().item()
        assert tensor.grad.abs().sum().item() > 0
        print(f"{name}:\n{tensor.detach()}\ngradient:\n{tensor.grad}")
    print("Scaled scores:\n", scores.detach())
    print("Row-normalized weights:\n", weights.detach())
    print("Weighted values:\n", output.detach())
    print(f"Scalar weights: a={a:.6f}, b={b:.6f}; manual and autograd checks passed.")


if __name__ == "__main__":
    main()
