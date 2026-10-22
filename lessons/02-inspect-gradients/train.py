import torch

inputs = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
targets = 2 * inputs + 1
weight = torch.tensor(0.0, requires_grad=True)
bias = torch.tensor(0.0, requires_grad=True)

errors = weight * inputs + bias - targets
loss = errors.square().mean()
loss.backward()

with torch.no_grad():
    manual_weight_gradient = (2 * errors * inputs).mean()
    manual_bias_gradient = (2 * errors).mean()

print("Autograd:", weight.grad.item(), bias.grad.item())
print("By formula:", manual_weight_gradient.item(), manual_bias_gradient.item())
torch.testing.assert_close(weight.grad, manual_weight_gradient)
torch.testing.assert_close(bias.grad, manual_bias_gradient)

second_loss = ((weight * inputs + bias - targets) ** 2).mean()
second_loss.backward()
print("Without clearing:", weight.grad.item(), bias.grad.item())

weight.grad = None
bias.grad = None
fresh_loss = ((weight * inputs + bias - targets) ** 2).mean()
fresh_loss.backward()
print("After clearing:", weight.grad.item(), bias.grad.item())

for learning_rate in [0.01, 0.1, 0.5, 0.6]:
    w = 0.0
    b = 0.0
    for step in range(10):
        w -= learning_rate * 4 * (w - 2)
        b -= learning_rate * 2 * (b - 1)
    mse = 2 * (w - 2) ** 2 + (b - 1) ** 2
    print(f"lr={learning_rate}: w={w:.4f}, b={b:.4f}, MSE={mse:.6f}")
