import torch
from torch import nn

inputs = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
targets = 2 * inputs + 1
model = nn.Linear(1, 1)
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(60):
    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    loss = loss_function(predictions, targets)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    final_loss = loss_function(model(inputs), targets)
    new_predictions = model(torch.tensor([[3.0], [4.0]]))

print("Weight shape:", tuple(model.weight.shape))
print("Bias shape:", tuple(model.bias.shape))
print("Weight:", model.weight.item(), "bias:", model.bias.item())
print("Final MSE:", final_loss.item())
print("New predictions:", new_predictions.tolist())
