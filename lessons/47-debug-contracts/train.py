"""Expected bugs are isolated demonstrations; the default script finishes cleanly."""
import torch
from torch import nn


def checked_bce(logits, targets):
    if logits.shape != targets.shape:
        raise ValueError(f"BCE shapes must match exactly: {logits.shape} vs {targets.shape}.")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("Binary logits and binary targets must be floating point.")
    if not ((targets >= 0) & (targets <= 1)).all().item():
        raise ValueError("Binary targets must be in [0, 1].")
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def main():
    torch.set_num_threads(1)
    predictions = torch.tensor([[1.], [2.], [3.]], requires_grad=True)
    targets = torch.tensor([1., 2., 3.])
    wrong_errors = predictions - targets
    wrong_loss = wrong_errors.square().mean()
    repaired_loss = (predictions.squeeze(-1) - targets).square().mean()
    assert wrong_errors.shape == (3, 3)
    assert wrong_loss.item() > 0 and repaired_loss.item() == 0
    print("Silent regression bug:", tuple(wrong_errors.shape), "wrong MSE", wrong_loss.item())
    print("Repaired aligned MSE:", repaired_loss.item())
    logits = torch.tensor([[0.5], [-0.5], [1.0]], requires_grad=True)
    try:
        checked_bce(logits, targets.clamp(max=1))
    except ValueError as error:
        print("Expected shape-contract rejection:", error)
    else:
        raise AssertionError("Shape mismatch escaped the contract.")
    binary_targets = torch.tensor([1, 0, 1], dtype=torch.long)
    try:
        checked_bce(logits.squeeze(-1), binary_targets)
    except TypeError as error:
        print("Expected BCE dtype rejection:", error)
    else:
        raise AssertionError("Integer binary targets escaped the contract.")
    loss = checked_bce(logits.squeeze(-1), binary_targets.float())
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all().item()
    embedding = nn.Embedding(5, 3)
    try:
        embedding(torch.tensor([1., 2.]))
    except RuntimeError as error:
        print("Expected embedding dtype failure:", str(error).splitlines()[0])
    else:
        raise AssertionError("Float embedding indices unexpectedly succeeded.")
    assert embedding(torch.tensor([1, 2], dtype=torch.long)).shape == (2, 3)
    class_logits = torch.tensor([[2., -1., 0.], [-1., 2., 0.]])
    class_ids = torch.tensor([0, 1], dtype=torch.long)
    ce = nn.functional.cross_entropy(class_logits, class_ids)
    singleton = torch.ones(1, 1)
    assert singleton.squeeze(-1).shape == (1,)
    assert singleton.squeeze().shape == ()
    print(f"Repaired BCE={loss.item():.4f}; CE with integer class IDs={ce.item():.4f}")
    print("All expected bugs observed; repaired shape, gradient, and dtype contracts passed.")


if __name__ == "__main__":
    main()
