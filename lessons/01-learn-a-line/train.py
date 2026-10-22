import torch

learning_rate = 0.1
steps = 60

inputs = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
targets = torch.tensor([-3.0, -1.0, 1.0, 3.0, 5.0], dtype=torch.float32)

weight = torch.tensor(0.0, requires_grad=True)
bias = torch.tensor(0.0, requires_grad=True)

for step in range(steps):
    weight.grad = None
    bias.grad = None

    predictions = weight * inputs + bias
    loss = torch.mean((predictions - targets) ** 2)
    loss.backward()

    if step == 0 or (step + 1) % 10 == 0:
        print(f"Before update {step + 1:02d}: MSE={loss.item():.6f}")

    with torch.no_grad():
        weight -= learning_rate * weight.grad
        bias -= learning_rate * bias.grad

with torch.no_grad():
    final_loss = torch.mean((weight * inputs + bias - targets) ** 2)
    new_inputs = torch.tensor([3.0, 4.0], dtype=torch.float32)
    new_predictions = weight * new_inputs + bias

print(f"\nLearned weight: {weight.item():.6f}")
print(f"Learned bias:   {bias.item():.6f}")
print(f"Final training MSE: {final_loss.item():.3e}")
print(f"New inputs: {new_inputs.tolist()}")
print(f"New predictions: {new_predictions.tolist()}")
