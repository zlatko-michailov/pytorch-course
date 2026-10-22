import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, inputs):
        return self.layers(inputs)


torch.manual_seed(7)
torch.set_num_threads(1)
inputs = torch.randn(400, 2)
targets = (inputs[:, 0] * inputs[:, 1] > 0).float().unsqueeze(1)
train_x, train_y = inputs[:240], targets[:240]
validation_x, validation_y = inputs[240:320], targets[240:320]
model = Classifier()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.BCEWithLogitsLoss()
for step in range(1000):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_function(model(train_x), train_y)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    accuracy = ((model(validation_x) >= 0) == validation_y.bool()).float().mean()
    corners = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    corner_probabilities = torch.sigmoid(model(corners))
print("Validation accuracy:", accuracy.item())
print("Corner probabilities:", corner_probabilities.flatten().tolist())
print("Registered parameter names:", list(model.state_dict()))
