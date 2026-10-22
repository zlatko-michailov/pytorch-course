"""Leakage, misleading accuracy, and memorization with completed repairs."""
import torch
from torch import nn


def scaling_demo():
    train = torch.tensor([0., 1., 2.])
    validation = torch.tensor([100., 101.])
    leaked_mean = torch.cat([train, validation]).mean()
    changed_leaked_mean = torch.cat([train, validation + 1000]).mean()
    assert not torch.equal(leaked_mean, changed_leaked_mean)
    mean = train.mean()
    scale = train.std(unbiased=False).clamp_min(1e-6)
    train_scaled = (train - mean) / scale
    validation_scaled = (validation - mean) / scale
    torch.testing.assert_close(train_scaled.mean(), torch.tensor(0.), atol=1e-6, rtol=0)
    assert torch.equal(mean, train.mean())
    print(f"Leak: all-data mean={leaked_mean:.2f}; changed heldout mean={changed_leaked_mean:.2f}")
    print(f"Repair: train-only mean={mean:.2f}, scale={scale:.3f}; "
          f"validation z-scores={validation_scaled.tolist()}")
    print("A large heldout shift is evidence to investigate, not permission to refit on it.")


def binary_metrics(predictions, targets):
    tp = (predictions & targets).sum().item()
    fn = (~predictions & targets).sum().item()
    tn = (~predictions & ~targets).sum().item()
    fp = (predictions & ~targets).sum().item()
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {"accuracy": (tp + tn) / len(targets), "positive_recall": recall,
            "balanced_accuracy": balanced, "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def memorization_demo():
    generator = torch.Generator().manual_seed(49)
    features = torch.eye(40)
    balanced = torch.tensor([0.] * 10 + [1.] * 10)
    train_y = balanced[torch.randperm(20, generator=generator)]
    validation_y = balanced[torch.randperm(20, generator=generator)]
    model = nn.Linear(40, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.15)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features[:20]).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_pred = model(features[:20]).squeeze(-1).ge(0)
        validation_pred = model(features[20:]).squeeze(-1).ge(0)
    train_metrics = binary_metrics(train_pred, train_y.bool())
    validation_metrics = binary_metrics(validation_pred, validation_y.bool())
    assert train_metrics["accuracy"] == 1
    assert validation_metrics["accuracy"] == 0.5
    print("Random-label ID lookup train:", train_metrics)
    print("Disjoint-ID diagnostic validation:", validation_metrics)
    print("Repair the evaluation: hold out IDs; remove ID features; use meaningful predictors.")
    print("No learner can recover an independent random label from unseen IDs.")


def main():
    torch.set_num_threads(1)
    scaling_demo()
    targets = torch.tensor([False] * 95 + [True] * 5)
    majority = binary_metrics(torch.zeros(100, dtype=torch.bool), targets)
    assert majority["accuracy"] == 0.95 and majority["positive_recall"] == 0
    assert majority["balanced_accuracy"] == 0.5
    print("Misleading all-negative classifier:", majority)
    print("Repair the report: include class counts, confusion counts, recall, and a baseline.")
    memorization_demo()
    print("All evaluation-debugging contracts passed; no deployment test was tuned.")


if __name__ == "__main__":
    main()
