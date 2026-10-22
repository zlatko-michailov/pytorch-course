# Session 28 — Split the entity, not just its rows

[Course map](../../CURRICULUM.md) · Project 7: structured records

## The question has changed

[Session 27](../27-embeddings/README.md) treated records as independent examples.
Now each machine contributes eight readings.
If deployment means predicting on **new machines**, does a random row holdout
measure the right generalization?

This lesson constructs an explicit **synthetic** failure case.
Machine identity predicts a persistent hidden offset.
A row split lets the learner estimate that offset from other readings of the
same machine.
A group split prevents that information from crossing the evaluation boundary.
The difference is about the prediction question, not a magical better splitter.

You already know numerical scaling, category vocabularies, UNK handling, affine
regression, and train/validation/test roles.
New concepts are groups, entity overlap, deployment-aligned holdouts, and fitting
preprocessing separately inside each experiment.

## Run the entire example

From the course root:

```bash
.venv/bin/python lessons/28-group-splits/train.py
```

The complete runnable source is [train.py](train.py).
No files or earlier runs are required, and no datasets are downloaded.
The program needs Python and PyTorch only.
A missing-NumPy warning from PyTorch is harmless here: no NumPy API is used.
One CPU thread reduces overhead on these small tensors; do not infer a timing
claim about larger workloads.

## 1. Generate a repeated-entity problem

There are 60 entities, each with eight measurements, for 480 rows.
`torch.arange(60).repeat_interleave(8)` creates their row-level identifiers:

```text
0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,...
```

Each entity has a normally distributed latent offset.
Its target is:

```text
target = 0.7 * measurement + entity_offset + small independent noise
```

The model sees measurement and, by default, an entity-ID category.
It does not see the latent offset.
The identifier is predictive only because training may contain previous examples
from that entity.
There is no mathematical way to infer an independently sampled unseen entity's
offset from its arbitrary identifier.

The goal stated for the **correct** experiment is new-entity prediction.
If the goal instead allowed earlier labeled readings from known entities, an
entity-aware model might be appropriate.
For a future-reading task, a chronological split would also be necessary;
random readings from the future could still be unavailable at prediction time.

## 2. Construct two different evaluation questions

The row split permutes all 480 row indices:

```text
288 training rows, 96 validation rows, 96 test rows
```

Many held-out rows share an entity with training rows.
In this seeded example, training contains all 60 entities.
Validation contains 48 of those entities and test contains 52.
High overlap is expected, not a bug in the permutation.

The group split first permutes the **60 entity IDs**:

```text
36 training entities, 12 validation entities, 12 test entities
```

`torch.isin(entities, groups)` identifies all rows belonging to a partition's IDs.
`torch.where(mask)[0]` extracts those row indices.
Each entity contributes exactly eight rows, so row counts again become
288, 96, and 96.
The two experiments use the same generated population, but intentionally assign
different rows to their respective partitions.
They answer different questions; this is not a same-holdout architecture contest.

Three assertions check that all pairs of group-ID sets are disjoint.
Disjoint **row indices** alone would not be sufficient.
Every ordinary row holdout already has disjoint rows.

## 3. Fit a separate pipeline inside each experiment

`run_split()` receives only the split indices and the raw generated tensors.
It builds its own:

1. Training-only numeric mean and scale.
2. Training-only vocabulary of entity IDs.
3. Transformed feature matrix using that fixed vocabulary.
4. Fresh affine model and optimizer.

The numeric measurement is always available.
Entity identity is one-hot encoded, not used as an ordered number.
Unknown entities map to ID zero and then to an all-zero category vector, as in
Session 26.
The group-split validation and test sets therefore have **100% unknown IDs**.
It would be leakage to build their vocabulary from all entities and learn an
entity's effect from held-out labels.
An allocated but untrained embedding row would not solve the missing evidence.

Both models use the same fixed 600-update SGD recipe.
There is no validation-driven search or test-driven adjustment.
An affine model is sufficient to learn the measurement slope and known-entity
offsets, making the leakage mechanism easy to identify.
Adding a deeper network would not create information about unseen random offsets.

## 4. Follow the contracts

| Object | Shape / role |
|---|---|
| `entities` | `(480,)`, int64 group identifiers |
| `numeric` | `(480,1)`, float32 available measurement |
| `target` | `(480,1)`, float32 synthetic response |
| split indices | Three disjoint int64 row-index tensors |
| row-split features | `(480,61)`: numeric + 60 known IDs |
| group-split features | `(480,37)`: numeric + 36 known IDs |
| predictions | `(480,1)` |

Different vocabulary sizes reflect different training populations.
The encoded ID zero is not a newly learned entity effect.
The training-target-mean baseline is independently fitted for each split.
MAE is reported in the target's synthetic units.

## 5. Interpret the contrast

A reference run gives:

| Experiment | Validation MAE | Test MAE | Held-out unknown IDs |
|---|---:|---:|---:|
| Random rows | 0.073 | 0.062 | 0% |
| Disjoint entities | 2.157 | 2.310 | 100% |

The group split did not damage a previously valid new-entity model.
It exposed that the row split measured an easier, different use case.
Its poor performance is close to the roughly `2.3` mean-baseline error.
For the first group-validation example, a prediction near `0.01` misses an actual
value around `-1.97`: the missing entity effect matters.

The low row-split training error and validation error are perfectly compatible
with this failure.
Do not choose the row split because its metric is prettier.
Choose a split that represents the intended incoming population and information.
Synthetic offsets and limited samples make these numbers illustrative, not
real-world performance estimates.

## 6. Remove the identity shortcut

Before executing, predict what happens if the learner cannot see entity IDs:

```bash
.venv/bin/python lessons/28-group-splits/train.py --without-id
```

The model now receives only one scaled numerical feature in both experiments.
It should no longer memorize known-entity offsets.
Expect the row-split advantage to shrink sharply and its error to increase.
The exact two errors need not match because the held-out entities differ.
Inspect the printed overlap counts: removing a feature does not change the split.
Conversely, grouping rows does not by itself remove proxy identity features.

Real proxies include account numbers, site-specific codes, repeated images,
household membership, and near-duplicate recordings.
Decide which grouping level matters: patient, hospital, session, customer, or
several levels together.
If both time and entity novelty matter, neither a row split nor a single group
split alone may be adequate.

## Checkpoint questions

1. Does “no duplicate row indices” prove no group leakage?
2. Why must the category vocabulary be fitted again for the group experiment?
3. Is entity identity always an illegitimate feature?
4. Can a more expressive network recover these unseen entity offsets?

<details>
<summary>Explained solutions</summary>

1. No. Two distinct rows can share the same entity, hidden context, or near-
   duplicate content. Check the intended unit of generalization.
2. The training population changed. Preprocessing must represent information
   available to that experiment, not inherit another split's fitted state.
3. No. It may be available and useful for previously observed entities. The
   evaluation must then reflect what past labels and timestamps are available.
4. Not from arbitrary IDs and independent offsets. More capacity cannot recover
   information absent from all available features; collecting meaningful entity
   descriptors or previous permitted observations would change the problem.

</details>

## Design limits

There are only 60 equal-sized groups, no time axis, and independent synthetic
group effects.
A realistic study needs enough representative groups and an uncertainty analysis
that does not treat correlated rows as independent samples.
This script prints both fixed test diagnostics to reveal the failure mode;
do not tune a split or architecture from those reports.
Next, [Session 29](../29-tree-baselines/README.md) holds data and splits fixed
while comparing an affine model, an MLP, and a real small regression tree.
