"""Small explicit loops, first explained in session 19."""

from pathlib import Path

import torch
from torch import nn

from course.fashion_data import CLASSES, PREPROCESS, SPLIT, loader


def mlp():
    return nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(784, 64),
                         nn.ReLU(), nn.Linear(64, 10))


def train_epoch(model, batches, optimizer, augment=None):
    model.train()
    total_loss, count = 0.0, 0
    for images, labels in batches:
        if augment is not None:
            images = augment(images)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        assert logits.shape == (len(labels), 10) and labels.dtype == torch.long
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        count += len(labels)
    return total_loss / count


def evaluate(model, batches):
    model.eval()
    total_loss, predictions, targets = 0.0, [], []
    with torch.no_grad():
        for images, labels in batches:
            logits = model(images)
            total_loss += nn.functional.cross_entropy(logits, labels).item() * len(labels)
            predictions.append(logits.argmax(dim=1))
            targets.append(labels)
    predicted, actual = torch.cat(predictions), torch.cat(targets)
    return {"loss": total_loss / len(actual),
            "accuracy": (predicted == actual).float().mean().item(),
            "predicted": predicted, "actual": actual}


def fit(model, train, val, epochs, learning_rate=0.05, augment=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    batches = loader(train, shuffle=True)
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        train_loss = train_epoch(model, batches, optimizer, augment)
        result = evaluate(model, loader(val))
        print(f"epoch={epoch + 1} train_CE={train_loss:.4f} "
              f"val_CE={result['loss']:.4f} val_accuracy={result['accuracy']:.3f}")
        if result["loss"] < best_loss:
            best_loss = result["loss"]
            best_state = {name: value.detach().clone()
                          for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return evaluate(model, loader(val))


def artifact(args, name):
    prefix = "smoke-" if args.smoke else "fashion-"
    return Path(".artifacts/vision") / f"{prefix}{name}.pt"


def save_model(path, model, architecture, args, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "task": "fashion-mnist", "architecture": architecture,
               "state": model.state_dict(), "classes": list(CLASSES),
               "preprocess": dict(PREPROCESS), "split": SPLIT,
               "split_seed": 1801, "initialization_seed": 1900,
               "smoke": args.smoke,
               "training": {"epochs": args.epochs, "max_train": args.max_train,
                            "max_val": args.max_val}, **extra}
    torch.save(payload, path)
    print(f"saved {path}")


def load_model(path, builder, expected_smoke):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["schema"] != 1 or payload["task"] != "fashion-mnist"
            or payload["preprocess"] != PREPROCESS or payload["classes"] != CLASSES
            or payload["split"] != SPLIT or payload["smoke"] != expected_smoke):
        raise ValueError("Checkpoint does not match this lesson's data contract")
    model = builder()
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    return model, payload
