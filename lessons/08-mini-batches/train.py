import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(7)
inputs = torch.randn(100, 2)
targets = inputs @ torch.tensor([[2.0], [-3.0]]) + 1
targets += 0.1 * torch.randn(100, 1)
dataset = TensorDataset(inputs[:80], targets[:80])
loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
loss_function = nn.MSELoss()

for epoch in range(30):
    squared_error_sum = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        squared_error_sum += loss.item() * len(batch_x)
    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: online training MSE={squared_error_sum / len(dataset):.6f}")

with torch.no_grad():
    validation_mse = loss_function(model(inputs[80:]), targets[80:])
print("Validation MSE:", validation_mse.item())
