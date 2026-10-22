import torch
from torch import nn

torch.manual_seed(7)
inputs = torch.randn(120, 2) * torch.tensor([1.0, 1000.0])
targets = inputs @ torch.tensor([[2.0], [-0.003]]) + 1
targets += 0.1 * torch.randn(120, 1)
train_x, validation_x = inputs[:90], inputs[90:]
train_y, validation_y = targets[:90], targets[90:]
mean = train_x.mean(dim=0)
scale = train_x.std(dim=0, correction=0)
scale = torch.where(scale == 0, torch.ones_like(scale), scale)
normalized_train = (train_x - mean) / scale
normalized_validation = (validation_x - mean) / scale
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(120):
    optimizer.zero_grad(set_to_none=True)
    loss = ((model(normalized_train) - train_y) ** 2).mean()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    validation_mse = ((model(normalized_validation) - validation_y) ** 2).mean()
    raw_weight = model.weight / scale
    raw_bias = model.bias - (raw_weight * mean).sum()

print("Training mean:", mean.tolist(), "scale:", scale.tolist())
print("Normalized training means:", normalized_train.mean(dim=0).tolist())
print("Coefficients in original units:", raw_weight.tolist(), raw_bias.item())
print("Validation MSE:", validation_mse.item())
