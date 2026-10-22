import torch
from torch import nn

logits = torch.tensor([[-2.0], [0.0], [2.0]])
targets = torch.tensor([[0.0], [1.0], [1.0]])
probabilities = torch.sigmoid(logits)
decisions = probabilities >= 0.5
loss_function = nn.BCEWithLogitsLoss(reduction="none")
losses = loss_function(logits, targets)

print("Logits:", logits.flatten().tolist())
print("Probabilities:", probabilities.flatten().tolist())
print("Decisions:", decisions.flatten().tolist())
print("Per-example losses:", losses.flatten().tolist())
print("Mean loss:", losses.mean().item())

extreme_logits = torch.tensor([[-1000.0], [1000.0]])
extreme_targets = torch.tensor([[1.0], [0.0]])
stable_losses = loss_function(extreme_logits, extreme_targets)
print("Confidently wrong, but finite:", stable_losses.flatten().tolist())
assert torch.isfinite(stable_losses).all()
