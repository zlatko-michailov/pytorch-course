from copy import deepcopy
import torch
from torch import nn

torch.manual_seed(7)
torch.set_num_threads(1)
inputs = torch.linspace(-2, 2, 200).unsqueeze(1)
targets = torch.sin(3 * inputs) + 0.35 * torch.randn_like(inputs)
order = torch.randperm(len(inputs))
train_x, train_y = inputs[order[:32]], targets[order[:32]]
validation_x, validation_y = inputs[order[32:116]], targets[order[32:116]]
test_x, test_y = inputs[order[116:]], targets[order[116:]]
model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
best_loss = float("inf")
best_state = deepcopy(model.state_dict())
best_epoch = 0
stale_epochs = 0
patience = 40

for epoch in range(800):
    optimizer.zero_grad(set_to_none=True)
    loss = ((model(train_x) - train_y) ** 2).mean()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        validation_loss = ((model(validation_x) - validation_y) ** 2).mean().item()
    if validation_loss < best_loss - 0.00001:
        best_loss = validation_loss
        best_state = deepcopy(model.state_dict())
        best_epoch = epoch + 1
        stale_epochs = 0
    else:
        stale_epochs += 1
    if stale_epochs >= patience:
        break

model.load_state_dict(best_state)
with torch.no_grad():
    test_mse = ((model(test_x) - test_y) ** 2).mean().item()
    baseline_mse = ((train_y.mean() - test_y) ** 2).mean().item()
print("Updates run:", epoch + 1, "selected epoch:", best_epoch)
print("Best validation MSE:", best_loss)
print("Final test MSE:", test_mse, "constant test MSE:", baseline_mse)
