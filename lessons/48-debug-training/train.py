"""Gradient accumulation, unstable step sizes, and train/eval are separate bugs."""
import torch
from torch import nn


def gradient_demo():
    x = torch.tensor([1., 2., 3., 4.])
    y = 2 * x
    w = nn.Parameter(torch.tensor(0.))
    ((w * x - y).square().mean()).backward()
    fresh = w.grad.clone()
    ((w * x - y).square().mean()).backward()
    assert torch.equal(w.grad, 2 * fresh)
    print(f"Forgot zero_grad: fresh={fresh.item():.1f}, stale+fresh={w.grad.item():.1f}")
    w.grad = None
    for micro_x, micro_y in zip(x.chunk(2), y.chunk(2)):
        ((w * micro_x - micro_y).square().mean() / 2).backward()
    torch.testing.assert_close(w.grad, fresh)
    print("Intentional two-microbatch accumulation equals the full-batch gradient.")


def trajectory(lr):
    w = nn.Parameter(torch.tensor(0.))
    optimizer = torch.optim.SGD([w], lr=lr)
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = (w - 3).square()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses


def mode_demo():
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(0.5), nn.Linear(4, 1))
    x = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 10
    model.train()
    with torch.no_grad():
        before = model[0].running_mean.clone()
        train_a, train_b = model(x), model(x)
    assert not torch.equal(train_a, train_b)
    assert not torch.equal(before, model[0].running_mean)
    print("no_grad alone: dropout still random and BatchNorm still updates.")
    model.eval()
    before = model[0].running_mean.clone()
    eval_a, eval_b = model(x), model(x)
    torch.testing.assert_close(eval_a, eval_b, rtol=0, atol=0)
    assert torch.equal(before, model[0].running_mean)
    assert eval_a.requires_grad
    try:
        assert model.training, "Training resumed while model was still in eval mode."
    except AssertionError as error:
        print("Expected mode-contract failure:", error)
    else:
        raise AssertionError("The demonstration did not leave eval mode active.")
    model.train()
    assert model.training and model[0].training and model[1].training
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    optimizer.step()
    print("Repair: restore train() before the next optimization step.")


def main():
    torch.manual_seed(48)
    torch.set_num_threads(1)
    gradient_demo()
    unstable, stable = trajectory(1.1), trajectory(0.1)
    assert unstable[-1] > unstable[0]
    assert stable[-1] < stable[0]
    print(f"lr=1.1: loss {unstable[0]:.3f}->{unstable[-1]:.3f}; "
          f"lr=.1: {stable[0]:.3f}->{stable[-1]:.3f}")
    mode_demo()
    print("All bug demonstrations and repaired training checks passed.")


if __name__ == "__main__":
    main()
