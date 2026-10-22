from pathlib import Path
import torch
from torch import nn

torch.manual_seed(7)
inputs = torch.randn(180, 2) * torch.tensor([1.0, 1000.0])
targets = inputs @ torch.tensor([[2.0], [-0.003]]) + 1
targets += 0.1 * torch.randn(180, 1)
train_x, train_y = inputs[:120], targets[:120]
validation_x, validation_y = inputs[120:150], targets[120:150]
test_x, test_y = inputs[150:], targets[150:]
mean = train_x.mean(dim=0)
scale = train_x.std(dim=0, correction=0)
scale = torch.where(scale == 0, torch.ones_like(scale), scale)
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(160):
    optimizer.zero_grad(set_to_none=True)
    loss = ((model((train_x - mean) / scale) - train_y) ** 2).mean()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    print("Validation MSE:", ((model((validation_x - mean) / scale) - validation_y) ** 2).mean().item())
    print("Final test MSE:", ((model((test_x - mean) / scale) - test_y) ** 2).mean().item())
    print("Constant test MSE:", ((train_y.mean() - test_y) ** 2).mean().item())

folder = Path(__file__).resolve().parents[2] / "artifacts" / "09-save-and-load"
folder.mkdir(parents=True, exist_ok=True)
checkpoint = {"state": model.state_dict(), "mean": mean, "scale": scale, "features": ["reading_a", "reading_b"]}
torch.save(checkpoint, folder / "model.pt")
restored = nn.Linear(2, 1)
loaded = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
restored.load_state_dict(loaded["state"])
with torch.no_grad():
    expected = model((test_x - mean) / scale)
    actual = restored((test_x - loaded["mean"]) / loaded["scale"])
torch.testing.assert_close(actual, expected)
print("Saved and verified:", folder / "model.pt")
