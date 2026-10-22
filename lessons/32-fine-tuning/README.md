# Project 8, Session 32: adapt the last block without erasing the representation

[Course curriculum](../../CURRICULUM.md) · [Complete train.py](train.py)

## Prerequisites and new concepts

Review Session 31's head-only optimizer boundary and frozen BatchNorm buffers.
Review Session 17's validation selection and Session 30's preprocessing contract.
No tabular-specific concept is required.

**New:** staged fine-tuning, selective unfreezing, optimizer parameter groups,
discriminative learning rates, and rejecting an adaptation that loses to its
head-only starting point.

## The problem

A frozen representation can be useful but not quite appropriate for clothes.
Fine-tuning allows selected feature weights to adapt.
Updating every early edge detector at the same aggressive rate as the random
classification head risks overwriting broadly useful structure.
The appropriate amount of adaptation depends on domain similarity, data, and budget.

We unfreeze only `layer4`, ResNet-18's final residual-block group, plus `fc`.
Earlier layers stay frozen. Running BatchNorm statistics stay fixed everywhere.
The block learns at 0.001 while the already trained head learns at 0.01.
These are conservative teaching choices, not universal optimal values.

## Run from the course root

```sh
.venv/bin/python -m lessons.32-fine-tuning.train --smoke --epochs 1
```

The fixture command independently trains a head, then fine-tunes its tiny final
block. It does not depend on an existing artifact.
The `[N,3,224,224]` preprocessing and staged procedure are real code, but random
fixture features are not pretrained ImageNet features.

For the real path, first complete the head stage from Session 31:

```sh
.venv/bin/python -m lessons.32-fine-tuning.train --data data/fashion --epochs 3 --max-train 512 --max-val 128
```

The default input is `.artifacts/transfer/fashion-head.pt`.
To name it explicitly:

```sh
.venv/bin/python -m lessons.32-fine-tuning.train --data data/fashion --checkpoint .artifacts/transfer/fashion-head.pt --epochs 3 --max-train 512 --max-val 128
```

Use the same subset limits as your head-only run.
This path needs the compatible optional torchvision environment and offline
Fashion-MNIST files described in [Session 30](../30-transfer-contract/README.md).
It does **not** need the original V1 weight file again: the head checkpoint
contains the whole backbone.
Missing checkpoints are errors, not permission to initialize a random “pretrained”
model. The program performs no download and does not open official test files.

## The two parameter groups

The shared stage code constructs:

```python
groups = [
    {"params": list(model.layer4.parameters()), "lr": 0.001},
    {"params": list(model.fc.parameters()), "lr": 0.01},
]
optimizer = torch.optim.SGD(groups, weight_decay=1e-4)
```

This is a list of dictionaries. Each dictionary gives a parameter iterable and
settings overriding optimizer defaults for just that group.
`list(...)` materializes the parameter generator so we can inspect membership.
The two sets must be disjoint; one parameter must not receive two updates.
We assert identity-based disjointness and exact coverage of all trainable tensors.

Do not pass `model.parameters()` as an additional third group.
It would include duplicates plus frozen early layers.
Changing `requires_grad` after constructing an optimizer does not automatically
rebuild the optimizer's parameter membership.
We deliberately construct a fresh optimizer at the stage boundary.
Our SGD uses no momentum, so we are not silently throwing away an assumed
momentum history; an optimizer with momentum would require an explicit policy.

## Complete code and execution trace

Read all of [train.py](train.py), with the full shared
[transfer_training.py](../../course/transfer_training.py) taught in Session 31,
and [transfer_contract.py](../../course/transfer_contract.py) taught in Session 30.

1. Parse offline options and build the same development split.
2. Smoke mode explicitly constructs a fixture and runs its head stage.
3. Real mode calls `load_transfer` to reconstruct the complete adapted ResNet,
   requiring metadata stage `"head"`.
4. The stored provenance remains attached to the adapted model.
   Fine-tuning does not magically change which original weights we started from.
5. Measure the starting head checkpoint's validation CE with `transfer_evaluate`.
6. Call `train_stage(...,fine_tune=True)`.
   This is a keyword argument selecting the concrete branch, not a new trainer.
7. `configure` first freezes and clears gradients everywhere, then enables only
   `layer4` and `fc`. Old trainable flags are not assumed to survive loading.
8. `training_mode` starts from `model.eval()`, enables last-block training behavior,
   then sets every last-block BatchNorm2d back to eval.
9. Each training image goes through the unchanged pretrained input transform.
   Cross-entropy still consumes `[B,10]` logits and `[B]` integer labels.
10. Each epoch evaluates validation and snapshots a better state.
11. The selected fine-tuned state is restored.
    Early-layer state and every BatchNorm running buffer must be bitwise unchanged.
12. Assertions require both a head tensor and a last-block weight to have changed.
13. Measure the resulting validation CE and print it beside the starting value.
14. If fine-tuning did not improve CE, print an explicit recommendation to prefer
    the head checkpoint. No test score is involved.
15. Save the fine-tuned candidate as `fashion-finetuned.pt` or
    `smoke-finetuned.pt`, separately from the preserved head checkpoint.

The candidate is saved even if it loses, so you can inspect and compare it.
That file name means “a fine-tuned candidate,” not “the selected deployment model.”
In Session 33, explicitly load `fashion-head.pt` if the head-only stage won.
Using the later stage automatically is not validation-based model selection.

## BatchNorm policy, stated precisely

In the unfreezed block, convolution weights and BatchNorm scale/bias can learn.
BatchNorm in eval mode still applies a differentiable affine transformation using
stored running statistics. Evaluation mode does not disable those gradients.
Its running mean, running variance, and tracked-batch counter do not update.
This policy is useful for very small batches, where newly estimated statistics
can be noisy, but it is not necessarily best for a large domain shift.
Adapting running statistics would be a separate declared experiment.

## Controlled edits

1. Set the block learning rate equal to the head's 0.01.
   Predict larger feature changes and a greater risk of validation degradation.
2. Unfreeze all parameters without changing the groups.
   Predict the exact trainable-membership assertion fails before training.
3. Let BatchNorm run in training mode.
   Predict frozen-buffer assertions fail; you changed more than feature gradients.
4. Increase only the fine-tuning epoch budget.
   Predict possible lower training loss but no guaranteed validation improvement.
   Keep the starting head checkpoint for a fair fallback.

## Checkpoint questions

**Why use a smaller rate for the reused feature block?**
<details>
<summary>Solution</summary>
It already encodes a useful representation. Smaller steps permit adaptation while
reducing destructive changes. The head and block have different starting roles;
their sensible learning rates need not match.
</details>

**Does the code guarantee fine-tuning is better than head-only training?**
<details>
<summary>Solution</summary>
No. It reports both validation losses and preserves both artifacts.
If the fine-tuned candidate loses, prefer the simpler head-only checkpoint or form
another validation hypothesis. Do not ask official test accuracy to decide.
</details>

**Does eval-mode BatchNorm prevent its affine parameters learning?**
<details>
<summary>Solution</summary>
No. Mode selects statistics behavior. The scale and bias still participate in
autograd when their requires-grad flags are enabled.
</details>

## Design judgment and limitations

One last-block boundary and two learning rates are a tractable experiment, not
a full fine-tuning search. Additional flexibility increases variance and compute.
Selection across many such trials can overfit validation too.
The final artifact should reflect an explicit decision, with its original source
and preprocessing carried into inference.
