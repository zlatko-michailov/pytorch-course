"""A synthetic repeated-entity example: row holdouts are not new-entity holdouts."""

import argparse

import torch
from torch import nn
from torch.nn import functional as F


def run_split(label, indices, numeric, entities, target, include_id):
    train, validation, test = indices
    train_entities = set(entities[train].tolist())
    validation_entities = set(entities[validation].tolist())
    test_entities = set(entities[test].tolist())
    if label == "group split":
        assert train_entities.isdisjoint(validation_entities | test_entities)
        assert validation_entities.isdisjoint(test_entities)
    vocabulary = {entity: i + 1 for i, entity in enumerate(sorted(train_entities))}
    ids = torch.tensor([vocabulary.get(entity, 0) for entity in entities.tolist()])
    mean = numeric[train].mean(0)
    scale = numeric[train].std(0, correction=0).clamp_min(1e-6)
    features = (numeric - mean) / scale
    if include_id:
        known = F.one_hot(ids, len(vocabulary) + 1)[:, 1:].float()
        features = torch.cat((features, known), dim=1)
    torch.manual_seed(28)
    model = nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.7)
    for _ in range(600):
        optimizer.zero_grad()
        loss = F.mse_loss(model(features[train]), target[train])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(features)
    print(f"\n{label}: train entities={len(train_entities)}, features={features.shape[1]}")
    print(
        f"Entity overlap train/validation={len(train_entities & validation_entities)}, "
        f"train/test={len(train_entities & test_entities)}"
    )
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        mae = (prediction[rows] - target[rows]).abs().mean().item()
        unknown = (ids[rows] == 0).float().mean().item()
        baseline = (target[train].mean() - target[rows]).abs().mean().item()
        print(f"{name:10s} MAE={mae:.3f}; UNK={unknown:.0%}; mean baseline={baseline:.3f}")
    i = validation[0].item()
    print(
        f"Validation example entity={entities[i].item()}, encoded id={ids[i].item()}, "
        f"prediction={prediction[i].item():+.2f}, actual={target[i].item():+.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-id", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(280)
    entities = torch.arange(60).repeat_interleave(8)
    numeric = torch.randn(480, 1, generator=generator)
    offsets = 3 * torch.randn(60, 1, generator=generator)
    target = 0.7 * numeric + offsets[entities]
    target += 0.08 * torch.randn(480, 1, generator=generator)
    permutation = torch.randperm(480, generator=generator)
    row_split = (permutation[:288], permutation[288:384], permutation[384:])
    group_order = torch.randperm(60, generator=generator)
    group_split = tuple(
        torch.where(torch.isin(entities, groups))[0]
        for groups in (group_order[:36], group_order[36:48], group_order[48:])
    )
    assert all(len(rows) == count for rows, count in zip(group_split, (288, 96, 96)))
    assert len(set(entities[row_split[0]].tolist())) == 60
    print("SYNTHETIC entities: eight readings share an unobserved entity offset")
    print(f"Include entity ID={not args.without_id}; both experiments use the same raw rows")
    run_split("row split", row_split, numeric, entities, target, not args.without_id)
    run_split("group split", group_split, numeric, entities, target, not args.without_id)


if __name__ == "__main__":
    main()
