import torch
from torch import nn

torch.manual_seed(7)
inputs = torch.linspace(-2, 2, 80).unsqueeze(1)
signal = 2 * inputs + 1
targets = signal + 0.3 * torch.randn_like(signal)
constant_prediction = targets.mean()
baseline_mse = ((constant_prediction - targets) ** 2).mean()

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.MSELoss()

for step in range(120):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_function(model(inputs), targets)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    predictions = model(inputs)
    fitted_mse = loss_function(predictions, targets)
    signal_mse = loss_function(signal, targets)

print("Constant baseline MSE:", baseline_mse.item())
print("Known signal versus noisy observations MSE:", signal_mse.item())
print("Fitted training MSE:", fitted_mse.item())
print("Learned weight and bias:", model.weight.item(), model.bias.item())
