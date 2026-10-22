import torch
from torch import nn

torch.manual_seed(7)
torch.set_num_threads(1)
inputs = torch.linspace(-2, 2, 160).unsqueeze(1)
targets = torch.sin(3 * inputs) + 0.35 * torch.randn_like(inputs)
order = torch.randperm(len(inputs))
train_x, train_y = inputs[order[:16]], targets[order[:16]]
validation_x, validation_y = inputs[order[16:120]], targets[order[16:120]]

for width in [2, 64]:
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in range(1200):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(train_x) - train_y) ** 2).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_mse = ((model(train_x) - train_y) ** 2).mean().item()
        validation_mse = ((model(validation_x) - validation_y) ** 2).mean().item()
    count = sum(parameter.numel() for parameter in model.parameters())
    print(f"width={width}, parameters={count}, training MSE={train_mse:.4f}, validation MSE={validation_mse:.4f}")
print("Final 40 test rows remain reserved.")
