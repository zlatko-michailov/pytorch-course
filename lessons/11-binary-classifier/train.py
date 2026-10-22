import torch
from torch import nn

torch.manual_seed(7)
inputs = torch.randn(360, 2)
targets = (2 * inputs[:, 0] - inputs[:, 1] + 0.3 > 0).float().unsqueeze(1)
train_x, train_y = inputs[:240], targets[:240]
validation_x, validation_y = inputs[240:300], targets[240:300]
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.BCEWithLogitsLoss()

for step in range(240):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_function(model(train_x), train_y)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    logits = model(validation_x)
    predictions = (logits >= 0).float()
    accuracy = (predictions == validation_y).float().mean()
    validation_loss = loss_function(logits, validation_y)

print("Validation loss:", validation_loss.item())
print("Validation accuracy:", accuracy.item())
print("Learned weight:", model.weight.tolist(), "bias:", model.bias.item())
print("Reserved test rows:", len(inputs[300:]), "(not scored)")
