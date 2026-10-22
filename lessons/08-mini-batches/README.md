# Session 8: an epoch can contain many updates

Project 2, lesson 5 of 6.

**Prerequisite:** [Session 7](../07-training-only-scaling/README.md).
**New:** a dataset interface, a data loader, mini-batches, shuffling, epochs,
and correctly weighted loss aggregation.

## Why not always use all examples?

An entire dataset may not fit in memory, and a small batch can provide a useful
gradient estimate much sooner than a full pass. Mini-batch gradients fluctuate:
the learning process is no longer exactly Session 2's deterministic full-batch
recurrence.

An **epoch** visits the training dataset once. With 80 examples and batch size 16,
one epoch contains five parameter updates. Thirty epochs therefore contain 150
updates. Comparing "30 epochs" and "30 updates" would be misleading.

We return to already similarly scaled features to isolate batching. Previous
training-only preprocessing still applies when needed; fitting it independently
inside each batch would change the problem.

## Run and read

[Complete source](train.py)

```sh
.venv/bin/python lessons/08-mini-batches/train.py
```

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 2 | Import layers/losses. |
| 3 | Import dataset and batching utilities. |
| 5 | Seed data generation, initialization, and loader shuffling. |
| 6 | Create independent two-feature inputs. |
| 7 | Generate the signal. |
| 8 | Add observation noise. |
| 9 | `TensorDataset` pairs corresponding rows and provides indexed access and length. It does not shuffle or train. |
| 10 | `DataLoader` groups examples into batches and shuffles their order each epoch. Its default worker count is zero: no multiprocessing is needed here. |
| 11 | Construct the affine model. |
| 12 | Configure SGD at a smaller rate for noisy batch gradients. |
| 13 | Construct mean-reduced MSE. |
| 15 | Iterate over complete passes through the dataset. |
| 16 | Reset a Python accumulator for this epoch's observed squared-error sum. |
| 17 | Iterate over paired batches; unpack each pair into feature and target tensors. |
| 18 | Clear gradients before **every batch update**, not just each epoch. |
| 19 | Compute a batch mean loss. |
| 20 | Differentiate that batch loss. |
| 21 | Update parameters immediately after this batch. |
| 22 | Multiply its mean by its example count before summing. With one target per row, this recovers the batch squared-error sum. |
| 23 | Select sparse progress output. |
| 24 | Divide the epoch total by the total example count. |
| 26 | Disable tracking for the final validation measurement. |
| 27 | Evaluate all 20 validation rows with the final model. |
| 28 | Report validation MSE. |

The logged training loss is an **online** average: different batches were
predicted at different parameter values. It is not exactly the final model's
loss over the whole training set. A separate no-gradient pass would measure that.

## The uneven-last-batch issue

Change batch size to 18: the last batch contains eight examples, not eighteen.
An unweighted average of batch means would overrepresent those eight examples.
Multiplying by actual batch size fixes this for our one-target MSE.

For multiple output elements or masked targets, the proper denominator must
match what the loss averages. "Always multiply by batch size" is not a universal
metric-aggregation rule.

## Experiments

1. Compare batch sizes 1, 16, and 80 at fixed epoch count. Record both updates and
   examples processed; the optimization budgets are different.
2. Use batch size 18 and manually verify the weighted-average denominator.
3. Set `shuffle=False`. Will every synthetic task fail? No. Discuss situations
   where sorted examples make fixed-order updates particularly problematic.
4. Clear gradients only once per epoch as an intentional bug. Explain why later
   updates now use sums of gradients evaluated at different parameter values.

## Checkpoint

Which object owns examples, which forms batches, and which updates parameters?
Does `shuffle=True` shuffle validation labels separately from inputs? How many
updates occur when 81 examples use batch size 16 without dropping the last batch?

<details>
<summary>Worked answers</summary>

Dataset, loader, and optimizer have distinct jobs. Shuffling uses example indices,
so paired features and labels remain aligned. There are six updates for 81
examples, with a final batch of one. `drop_last=True` would change both the visit
count and which examples contribute, so it is not used to conceal that batch.

</details>

[Next: save, load, and infer](../09-save-and-load/README.md).
[Course map](../../CURRICULUM.md).
