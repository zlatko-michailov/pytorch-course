import torch
from torch import nn

torch.manual_seed(7)
inputs = torch.randn(600, 2)
targets = (inputs[:, 0] + 0.2 * inputs[:, 1] > 1.3).float().unsqueeze(1)
train_x, train_y = inputs[:400], targets[:400]
validation_x, validation_y = inputs[400:500], targets[400:500]
positive_count = train_y.sum(dim=0)
if not (positive_count > 0).all() or not (positive_count < len(train_y)).all():
    raise ValueError("Both classes must be present in the training data.")
positive_weight = (len(train_y) - positive_count) / positive_count
print("Training positive fraction:", train_y.mean().item())
print("Always-negative validation accuracy:", (validation_y == 0).float().mean().item())

for name, weighting in [("ordinary", torch.ones(1)), ("weighted", positive_weight)]:
    torch.manual_seed(17)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weighting)
    for step in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predicted = model(validation_x) >= 0
        actual = validation_y.bool()
        tp = (predicted & actual).sum().item()
        fp = (predicted & ~actual).sum().item()
        fn = (~predicted & actual).sum().item()
        accuracy = (predicted == actual).float().mean().item()
    recall = tp / (tp + fn) if tp + fn else float("nan")
    print(f"{name}: accuracy={accuracy:.3f}, recall={recall:.3f}, FP={fp}, FN={fn}")
print("Final 100 test rows remain reserved.")
