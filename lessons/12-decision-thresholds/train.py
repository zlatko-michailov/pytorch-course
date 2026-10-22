import torch
from torch import nn

torch.manual_seed(7)
inputs = torch.randn(360, 2)
targets = (2 * inputs[:, 0] - inputs[:, 1] + 0.3 > 0).float().unsqueeze(1)
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.BCEWithLogitsLoss()
for step in range(240):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_function(model(inputs[:240]), targets[:240])
    loss.backward()
    optimizer.step()

with torch.no_grad():
    probabilities = torch.sigmoid(model(inputs[240:300]))
actual_positive = targets[240:300].bool()
for threshold in [0.2, 0.5, 0.8]:
    predicted_positive = probabilities >= threshold
    tp = (predicted_positive & actual_positive).sum().item()
    fp = (predicted_positive & ~actual_positive).sum().item()
    fn = (~predicted_positive & actual_positive).sum().item()
    tn = (~predicted_positive & ~actual_positive).sum().item()
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    cost = 5 * fn + fp
    print(f"threshold={threshold}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  precision={precision:.3f}, recall={recall:.3f}, assumed cost={cost}")
