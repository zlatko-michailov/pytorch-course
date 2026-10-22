"""Explicit stage controls and loops, introduced in session 31."""

import torch
from torch import nn

from course.fashion_data import loader
from course.transfer_contract import preprocess


def configure(model, fine_tune=False):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)
    if fine_tune:
        for parameter in model.layer4.parameters():
            parameter.requires_grad_(True)
    model.eval()


def training_mode(model, fine_tune=False):
    model.eval()
    model.fc.train()
    if fine_tune:
        model.layer4.train()
        for module in model.layer4.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_frozen(before, model, fine_tune=False):
    for name, value in model.state_dict().items():
        allowed = name.startswith("fc.") or (fine_tune and name.startswith("layer4."))
        is_stat = name.endswith(("running_mean", "running_var", "num_batches_tracked"))
        if not allowed or is_stat:
            assert torch.equal(before[name], value), f"Unexpected frozen-state change: {name}"


def transfer_evaluate(model, dataset, smoke):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader(dataset, batch_size=8):
            logits = model(preprocess(images, smoke))
            assert logits.shape == (len(labels), 10)
            total_loss += nn.functional.cross_entropy(logits, labels).item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)
    return {"loss": total_loss / count, "accuracy": correct / count}


def train_stage(model, train, val, args, fine_tune=False):
    configure(model, fine_tune)
    before = snapshot(model)
    if fine_tune:
        groups = [{"params": list(model.layer4.parameters()), "lr": 0.001},
                  {"params": list(model.fc.parameters()), "lr": 0.01}]
    else:
        groups = [{"params": list(model.fc.parameters()), "lr": 0.05}]
    selected = [p for group in groups for p in group["params"]]
    assert len({id(p) for p in selected}) == len(selected)
    assert {id(p) for p in selected} == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer = torch.optim.SGD(groups, weight_decay=1e-4)
    batches = loader(train, shuffle=True, seed=3100, batch_size=8)
    best_loss, best_state = float("inf"), None
    for epoch in range(args.epochs):
        training_mode(model, fine_tune)
        for images, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(preprocess(images, args.smoke))
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        result = transfer_evaluate(model, val, args.smoke)
        print(f"stage={'fine-tune' if fine_tune else 'head'} epoch={epoch + 1} "
              f"val_CE={result['loss']:.4f} val_accuracy={result['accuracy']:.3f}")
        if result["loss"] < best_loss:
            best_loss, best_state = result["loss"], snapshot(model)
    model.load_state_dict(best_state)
    model.eval()
    assert_frozen(before, model, fine_tune)
    assert any(not torch.equal(before[name], value)
               for name, value in model.state_dict().items() if name.startswith("fc."))
    if fine_tune:
        assert any(not torch.equal(before[name], value)
                   for name, value in model.state_dict().items()
                   if name.startswith("layer4.") and name.endswith("weight"))
    print("Frozen parameters AND BatchNorm running buffers unchanged")
