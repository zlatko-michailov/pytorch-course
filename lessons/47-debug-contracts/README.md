# Project 47 / Session 47 — Turn shape and dtype assumptions into contracts

## Prerequisites

Complete sessions 1–46.
Review broadcasting, binary BCE, multiclass cross-entropy, and embedding indices.
You should distinguish tensor dimensions from the semantic meaning of each axis.

**Review:** subtraction, reductions, `.squeeze(-1)`, floating targets, integer IDs.
**New practice:** isolate a deliberate bug, predict its symptom, assert the repair.
Every example has a completed repair in the same source file.
The default command is **not** a broken exercise waiting for you to implement a fix.

## The debugging problem

An exception is often easier to diagnose than a plausible wrong scalar loss.
Broadcasting can compare every prediction with every target instead of comparing matching examples.
The resulting loss is finite and differentiable.
Optimization can happily improve the wrong objective.

Dtype problems are different.
An embedding consumes category indices.
BCE consumes floating binary targets.
Cross-entropy in its class-index form consumes integer class IDs.
Casting everything to float or everything to long is not a general repair.

## Run

From the course root:

```bash
.venv/bin/python -m lessons.47-debug-contracts.train
```

Complete worked source: [train.py](train.py).
The program prints expected failures, verifies repaired results, and exits with status zero.
Unexpected exceptions are not caught broadly.
No files, previous artifacts, external datasets, or additional packages are needed.

## Bug 1: a regression loss that compares the wrong axes

Predictions have shape `[3,1]`.
Targets have shape `[3]`.
Both contain the values 1, 2, and 3 in the intended example order.
The correct MSE should therefore be zero.

Broadcasting aligns dimensions from the right.
For subtraction, `[3]` behaves as `[1,3]`.
Combining `[3,1]` and `[1,3]` yields `[3,3]`.
Rows correspond to predictions and columns to all targets.

The error matrix is:

```text
[[ 0, -1, -2],
 [ 1,  0, -1],
 [ 2,  1,  0]]
```

Squaring and averaging all nine entries gives about 1.333333.
That scalar is mathematically valid for the wrong comparison.
Its finite value should not reassure you.

The completed repair uses `predictions.squeeze(-1)`.
Both operands then have shape `[3]`.
Only corresponding examples are subtracted.
The repaired MSE is exactly zero for this fixture.

The script asserts both the buggy shape and the repaired value.
This establishes that the demonstration actually exercised the intended error.
It does not merely print advice about avoiding broadcasting.

## Why not use unrestricted squeeze?

For a batch of one, a tensor `[1,1]` has two singleton axes.
`.squeeze()` removes both and returns a scalar.
`.squeeze(-1)` removes only the output-feature axis and leaves shape `[1]`.
The script checks that distinction explicitly.
Batch-size-one inference should preserve the same semantic batch contract.

## Bug 2: binary loss boundary

`checked_bce(logits, targets)` is a small explicit application contract.
First, shapes must match exactly.
We do not rely on the library to broadcast or reject a mismatch differently across versions.
Second, both tensors must be floating point.
Third, target values must be within `[0,1]`.

The demonstration first supplies `[3,1]` logits and `[3]` targets.
It catches the expected `ValueError` and prints the reason.
Then it supplies aligned logits but integer binary targets.
It catches the expected `TypeError`.

The completed repair is meaningful, not arbitrary:

```text
logits.squeeze(-1): [3]
binary_targets.float(): [3], values still 0 or 1
```

The repaired BCE backpropagates into the original logits.
Their gradients must exist and be finite.
This confirms that fixing the input contract did not accidentally detach the learning path.

The helper is a teaching boundary, not a full validation library.
For example, it does not implement device policy or all probability/label schemes.
Keep contracts specific to the model and objective you actually intend.

## Bug 3: embedding inputs are not measurements

The script tries to index an embedding with `[1.0, 2.0]`.
Those values have floating dtype even though they look like whole numbers.
PyTorch rejects that lookup with `RuntimeError`.
Only that expected exception type is caught.

The repair creates `torch.tensor([1,2], dtype=torch.long)`.
These are category IDs, and lookup produces shape `[2,3]`.
Do not repair an arbitrary fractional feature by truncating it to an embedding index.
A valid cast requires a valid semantic interpretation.

## The multiclass contrast

The final example has class logits `[2,3]`.
Two examples each have three unnormalized class scores.
Class IDs are `[0,1]` with dtype `torch.long`.
`cross_entropy` consumes these without a sigmoid or softmax preprocessing step.

This is intentionally different from binary BCE's floating `[B]` labels.
Cross-entropy also supports probability-target forms, but that is not our contract here.
Name the target representation before choosing the loss and dtype.

## Controlled experiments

### A. Increase batch size while keeping the shape bug

Predict: `[B,1] - [B]` creates `[B,B]` pairwise errors.
Compute grows quadratically in B and the objective is still wrong.
Batch size one may hide the bug, so test more than one batch size.

### B. Cast embedding IDs with a fractional example

Try discussing `[1.7] -> long` before running any cast.
Predict: truncation changes the value rather than recovering a legitimate category.
Reject the malformed input upstream instead of treating casting as validation.

### C. Pass NaN as a BCE target

Predict: the range contract rejects it because ordered comparisons with NaN are false.
This is a useful consequence of validation before entering the loss.
It is not a complete substitute for checking finite model outputs.

### D. Replace `.squeeze(-1)` with `.squeeze()`

Use the included singleton example.
Predict: an extra batch-axis collapse occurs.
Look for a contract failure, not just whether one particular library call accepts a scalar.

## Checkpoint questions

<details>
<summary>Why can training loss decrease while this broadcasting bug remains?</summary>

Autograd differentiates the tensor expression you wrote.
It has no knowledge that columns were supposed to represent matched examples.
An optimizer can improve the incorrect all-pairs objective normally.
</details>

<details>
<summary>Should every shape mismatch be fixed with reshape?</summary>

No. First identify each axis and which values should correspond.
A reshape that preserves element count can still scramble example/feature meaning.
Use explicit, justified axis operations and assert the resulting contract.
</details>

<details>
<summary>Why do the expected failures not hide unexpected defects?</summary>

Each try block contains one intentionally invalid operation.
Each catch names its expected exception type, and an else branch fails if no exception occurs.
Unrelated failures outside those narrow demonstrations remain visible.
</details>

## Design judgment

Log shapes, dtypes, ranges, and small example values at model/loss boundaries.
A finite loss is necessary but not sufficient evidence of correctness.
Next, investigate training state bugs that preserve all these shapes.
