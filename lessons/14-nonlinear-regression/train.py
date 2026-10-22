import torch
from torch import nn

torch.manual_seed(7)
torch.set_num_threads(1)
inputs = torch.linspace(-2, 2, 120).unsqueeze(1)
targets = inputs.square()
order = torch.randperm(len(inputs))
train_x, train_y = inputs[order[:80]], targets[order[:80]]
validation_x, validation_y = inputs[order[80:100]], targets[order[80:100]]
models = {
    "affine": nn.Linear(1, 1),
    "nonlinear": nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)),
}
for name, model in models.items():
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    for step in range(800):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(train_x) - train_y) ** 2).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        validation_mse = ((model(validation_x) - validation_y) ** 2).mean()
        predictions = model(torch.tensor([[-1.5], [0.0], [1.5]]))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(name, "parameters:", parameter_count, "validation MSE:", validation_mse.item())
    print("Predictions at -1.5, 0, 1.5:", predictions.flatten().tolist())
print("Final 20 test rows are reserved.")
