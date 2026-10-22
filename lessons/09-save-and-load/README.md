# Session 9: a model must survive its training process

Project 2, lesson 6 of 6.

**Prerequisite:** [Session 8](../08-mini-batches/README.md).
**New:** state dictionaries, local artifacts, safe loading, an inference-only
program, and final evaluation after the design is settled.

## What counts as the saved model?

For our raw-input predictor, the function is

```text
raw input -> fixed feature order -> training-only normalization -> affine layer
```

Save the feature contract and preprocessing **with** the learned parameters.
Saving only a weight matrix would leave the raw-input function ambiguous.

`state_dict()` maps parameter/buffer names to tensors. It does not serialize the
Python architecture. The inference program constructs the same architecture
and loads that mapping.

## Run two separate programs

[Training source](train.py), [inference source](predict.py).

```sh
.venv/bin/python lessons/09-save-and-load/train.py
.venv/bin/python lessons/09-save-and-load/predict.py
```

The second command does not train, download anything, or inspect target values.
If no checkpoint exists it raises an explicit error instead of pretending to infer.
Both programs locate artifacts relative to the course, not the shell's current
directory. Training writes `artifacts/09-save-and-load/model.pt`.

This reference freezes rate 0.1 and 160 updates, reports validation, then evaluates
the final test partition once. Do not keep changing the design in response to
the test number; create a new final evaluation protocol if you begin development
again. Reproducibly rerunning a fixed reference is not an independent new test.

## Training source, every nonblank line

| Line | Meaning |
|---|---|
| 1 | Import standard-library `Path`, an object-oriented filesystem-path API. |
| 2 | Import PyTorch. |
| 3 | Import layer types. |
| 5 | Seed the synthetic reference experiment. |
| 6 | Generate differently scaled input features. |
| 7 | Generate the affine signal. |
| 8 | Add independent observation noise. |
| 9 | Assign 120 independent rows to training. |
| 10 | Assign the next 30 to validation. |
| 11 | Reserve the last 30 for final testing. |
| 12 | Fit feature means on training only. |
| 13 | Fit feature scales on training only. |
| 14 | Define constant-feature handling explicitly. |
| 15 | Construct the architecture. |
| 16 | Construct the optimizer. |
| 18 | Repeat a fixed, already selected number of full-batch updates. |
| 19 | Clear gradients. |
| 20 | Normalize training rows and compute MSE. |
| 21 | Calculate gradients. |
| 22 | Update parameters. |
| 24 | Disable recording for evaluation. |
| 25 | Report validation performance with training-fitted preprocessing. |
| 26 | Measure the frozen reference on the final test data. |
| 27 | Compare a training-fitted constant on those same test rows. |
| 29 | `__file__` is this script's path; resolve it, take the course ancestor, and join artifact path components using `/`. |
| 30 | Create that specific artifact folder if needed; existing folders are not deleted. |
| 31 | Build a plain dictionary of tensors, the layer state, and a feature-name list. |
| 32 | Write it to a local checkpoint. No third-party upload occurs. |
| 33 | Construct a fresh layer, initially unrelated to the trained one. |
| 34 | Load on CPU, restricting deserialization to the `weights_only`-compatible representation. |
| 35 | Copy saved parameters into the reconstructed architecture. |
| 36 | Disable tracking for a round-trip check. |
| 37 | Calculate predictions using the original function. |
| 38 | Calculate predictions using the reloaded parameters and preprocessing. |
| 39 | Raise an error if the two disagree. |
| 40 | Report the saved checkpoint's path. |

Only load checkpoints you trust. `weights_only=True` is a safer loading interface,
not permission to accept arbitrary untrusted artifacts.

## Inference source, every nonblank line

| Line | Meaning |
|---|---|
| 1 | Import path operations. |
| 2 | Import PyTorch. |
| 3 | Import the architecture class. |
| 5 | Locate the course checkpoint deterministically. |
| 6 | Check the required artifact exists. |
| 7 | Raise a specific actionable error if it does not. |
| 8 | Load tensor/simple metadata on CPU with restricted deserialization. |
| 9 | Rebuild the architecture without training it. |
| 10 | Restore its learned values. |
| 11 | Define two raw-input examples in saved feature order. |
| 12 | Check rank, feature count, and finiteness. `or` short-circuits; a bad rank is rejected before indexing its second dimension. |
| 13 | Reject a violated contract rather than returning a plausible number. |
| 14 | Disable graph tracking. |
| 15 | Normalize with saved training statistics and run inference. |
| 16 | Show the feature-order contract. |
| 17 | Display predictions without changing any parameters. |

## Experiments and checkpoint

Change only the inference inputs and predict the direction of the outputs.
Temporarily omit normalization: explain why equal-shaped code now predicts the
wrong function. Move the checkpoint aside, run inference, and inspect the
actionable failure; restore it afterward.

Why does a fresh `nn.Linear(2,1)` not know the learned rule? Why is preprocessing
in the artifact? Does an optimizer need to be constructed for inference?

<details>
<summary>Worked answers</summary>

Construction defines the architecture but initializes new weights. Loading
restores the fitted state. Preprocessing is part of the raw-input function, and
an inference-only process needs neither gradients nor an optimizer. Resuming
training would be a different contract and might also require optimizer state.

</details>

**Project 2 complete:** we can fit noisy multifeature regression, compare fairly,
keep data roles distinct, batch updates, and reproduce inference.
[Next: classification scores](../10-logits-and-probabilities/README.md).
[Course map](../../CURRICULUM.md).
