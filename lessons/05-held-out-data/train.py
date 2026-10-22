import torch
from torch import nn

torch.manual_seed(7)
inputs = torch.linspace(-2, 2, 120).unsqueeze(1)
targets = 2 * inputs + 1 + 0.3 * torch.randn_like(inputs)
indices = torch.randperm(len(inputs))
train_ids = indices[:80]
validation_ids = indices[80:100]
test_ids = indices[100:]
train_x, train_y = inputs[train_ids], targets[train_ids]
validation_x, validation_y = inputs[validation_ids], targets[validation_ids]

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.MSELoss()
baseline = train_y.mean()

for step in range(120):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_function(model(train_x), train_y)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    training_mse = loss_function(model(train_x), train_y)
    validation_mse = loss_function(model(validation_x), validation_y)
    baseline_validation_mse = ((baseline - validation_y) ** 2).mean()

print("Training examples:", len(train_ids), "validation:", len(validation_ids))
print("Training MSE:", training_mse.item())
print("Validation MSE:", validation_mse.item())
print("Constant validation MSE:", baseline_validation_mse.item())
print("Reserved test examples:", len(test_ids), "(not scored)")
