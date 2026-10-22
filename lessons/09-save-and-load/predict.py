from pathlib import Path
import torch
from torch import nn

checkpoint_path = Path(__file__).resolve().parents[2] / "artifacts" / "09-save-and-load" / "model.pt"
if not checkpoint_path.is_file():
    raise FileNotFoundError("Run lessons/09-save-and-load/train.py before predict.py.")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model = nn.Linear(2, 1)
model.load_state_dict(checkpoint["state"])
inputs = torch.tensor([[0.25, 500.0], [1.0, -1000.0]])
if inputs.ndim != 2 or inputs.shape[1] != len(checkpoint["features"]) or not torch.isfinite(inputs).all():
    raise ValueError("Expected finite rows with features: " + ", ".join(checkpoint["features"]))
with torch.no_grad():
    predictions = model((inputs - checkpoint["mean"]) / checkpoint["scale"])
print("Feature order:", checkpoint["features"])
print("Predictions:", predictions.tolist())
