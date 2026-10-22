import torch
from torch import nn

torch.manual_seed(7)
inputs = torch.randn(100, 2)
coefficients = torch.tensor([[2.0], [-3.0]])
targets = inputs @ coefficients + 1 + 0.1 * torch.randn(100, 1)
train_x, validation_x = inputs[:80], inputs[80:]
train_y, validation_y = targets[:80], targets[80:]
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.MSELoss()

for step in range(120):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_function(model(train_x), train_y)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    validation_mse = loss_function(model(validation_x), validation_y)
    example = torch.tensor([[0.5, -1.0]])
    prediction = model(example)

print("Input/weight/output shapes:", train_x.shape, model.weight.shape, prediction.shape)
print("Learned coefficients:", model.weight.tolist(), "bias:", model.bias.item())
print("Validation MSE:", validation_mse.item())
print("Prediction for [0.5, -1.0]:", prediction.item())
