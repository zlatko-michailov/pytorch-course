# Project 8, Session 31: train a new head while the backbone truly stays fixed

[Course curriculum](../../CURRICULUM.md) · [Complete train.py](train.py)

## Prerequisites and new ideas

Review Session 30's pretrained input contract, local weights, and new head.
Review Session 25's independent controls for mode and autograd.
Review Sessions 17 and 19's validation-based state selection.

**New:** freezing parameters, optimizer membership, BatchNorm buffers, a head-only
training stage, and assertions that a frozen representation did not drift.
We introduce BatchNorm's mode-dependent state before relying on it.

## The problem

With limited labeled data, a random replacement head needs training, but updating
the whole pretrained network immediately may destroy useful features.
A frozen-backbone stage learns only how to combine existing features.
“Frozen” must mean more than “the backbone parameters have no gradients.”

BatchNorm maintains per-channel running mean and variance during training.
These are **buffers**, not parameters, and change without optimizer updates.
Calling `model.train()` after setting `requires_grad=False` can still change them.
That would change the representation even though every backbone weight was fixed.

## Run from the course root

```sh
.venv/bin/python -m lessons.31-frozen-head.train --smoke --epochs 1
```

This uses random TinyFixture features and synthetic stripes.
It verifies freezing including BatchNorm state, optimizer updates, and saving.
It does not test the utility of ImageNet features.

After supplying the optional packages, official local V1 weights, trusted digest,
and Fashion-MNIST files described in [Session 30](../30-transfer-contract/README.md):

```sh
.venv/bin/python -m lessons.31-frozen-head.train --data data/fashion --weights assets/resnet18-f37072fd.pth --weights-sha256 "$RESNET18_SHA256" --epochs 3 --max-train 512 --max-val 128
```

This explicitly limited real experiment is intentionally smaller than Project 6's
default: a 224×224 ResNet is much more expensive on CPU.
To compare quality fairly with Project 6, rerun baselines on the same limits or
declare and justify the different data/compute budgets.
Omit limits only if you deliberately want the full 54,000/6,000 development split.

The output is `.artifacts/transfer/fashion-head.pt` or `smoke-head.pt`.
No official test data is read and no network call occurs.
Missing optional packages or weights do not trigger a random substitute.

## Shapes and the exact optimization boundary

```text
raw batch:       [B,1,28,28]
preprocessed:    [B,3,224,224]
ResNet features: [B,512]             TinyFixture features: [B,8]
new head logits: [B,10]
targets:         [B], torch.long
loss:            scalar cross-entropy
```

Only `fc.weight` and `fc.bias` enter SGD in the head-only stage.
The optimizer cannot accidentally apply weight decay to frozen backbone weights
because those weights are not in its parameter groups.
The feature extractor runs in evaluation mode; the head runs in training mode.
Our head is Linear, so its train/eval numerical behavior is currently identical,
but this distinction remains correct if a mode-dependent head is added later.

## Complete sources

The full entry point is [train.py](train.py).
The complete new shared loop is
[transfer_training.py](../../course/transfer_training.py), explained below.
Model construction, preprocessing, and serialization remain explicit in
[transfer_contract.py](../../course/transfer_contract.py), taught in Session 30.
There is no training framework, automatic download, or hidden fit callback.

## Entry point walkthrough

`transfer_arguments` reads the declared offline contract.
`partitions(args)` reconstructs the same split and applies explicit prefix limits.
`build_initial(args)` strictly loads the local pretrained model and replaces its
head, or explicitly constructs the fixture.
`train_stage(...)` configures the head-only boundary, trains, and restores the best
validation-loss epoch.
`save_transfer(...,"head")` persists the full adapted state and its interpretation.
It saves the entire backbone as well as the head: inference should not depend on
finding the original ImageNet file again.

## Every function in `transfer_training.py`

### `configure(model, fine_tune=False)`

First iterate `model.parameters()` and call `requires_grad_(False)` on each.
The underscore denotes in-place mutation of the flag.
`parameter.grad = None` clears stale gradients from a previous stage.
Then enable gradients on `model.fc.parameters()`.
The optional fine-tune branch enables `model.layer4` too; Session 32 uses it.
`model.eval()` establishes a safe mode baseline but does not freeze by itself.
Trainable flags and module mode are intentionally managed separately.

### `training_mode(model, fine_tune=False)`

`model.eval()` recursively selects evaluation behavior for all children.
`model.fc.train()` then selects head training behavior only.
Head-only training never calls `model.train()` on the whole backbone.
The future fine-tune branch calls `layer4.train()`, then immediately sets all
`nn.BatchNorm2d` children there back to eval.
`.modules()` recursively yields nested modules.
`isinstance(module,nn.BatchNorm2d)` checks type, like C# `is` for this concrete layer.
For small-data transfer we keep running statistics fixed in **both** stages.
In the fine-tuned block its affine scale/bias may still receive gradients in eval.

### `snapshot` and `assert_frozen`

`snapshot` copies every `state_dict` tensor with `detach().clone()`.
Unlike `parameters()`, `state_dict()` includes BatchNorm running buffers and its
integer `num_batches_tracked` counter.
We must copy, not keep aliases to values that later training could change.

`assert_frozen` visits every current state entry.
String `startswith("fc.")` identifies head state.
When fine-tuning, `layer4.` parameters may also change.
`endswith(("running_mean","running_var","num_batches_tracked"))` accepts a tuple
of possible suffixes. These buffer names must remain unchanged even in layer4.
For every prohibited change, `torch.equal` checks exact state equality and the
assertion reports the offending state name.
This is a behavioral check, stronger than merely printing requires-grad flags.

### `transfer_evaluate`

It selects eval mode and no-grad, uses deterministic preprocessing, and iterates
the dataset in batches of eight.
Each batch is transformed immediately before forward computation; we do not store
all 60,000 expanded 224×224 RGB images in memory.
The helper checks `[B,10]`, accumulates CE weighted by batch size, counts correct
argmax decisions, and divides totals by the number of examples.
No training update or random augmentation occurs.

### `train_stage`

It calls `configure`, then takes the before-training snapshot.
Head-only groups contain exactly `fc.parameters()` with learning rate 0.05.
Fine-tuning groups and their different rates are explained in Session 32.
The flattened `selected` list contains all optimizer parameter references.
`id(p)` is the Python object's identity, not a numerical tensor value.
Two set assertions verify no duplicates and exact equality with the trainable set.
This catches forgotten head parameters and accidental whole-backbone membership.

`torch.optim.SGD(groups,weight_decay=1e-4)` creates a fresh stage optimizer.
The training loader shuffles with its own seed 3100 and batch size eight.
For each epoch we reapply `training_mode`, because validation leaves the model in
eval mode. Every batch clears gradients, preprocesses, computes raw logits and
CE, backpropagates, and updates only optimizer members.
There is no `no_grad` around the head: it must receive gradients.

After each epoch, evaluation reports held-out CE and accuracy.
Improving CE triggers a full snapshot; the best epoch is restored at the end.
This is fixed-budget best-state selection, not exact training resumption.
`assert_frozen` checks the final selected state against the initial snapshot.
Another assertion checks that at least one head tensor actually changed.
The fine-tune branch also verifies a last-block weight changed.
Together these detect both “nothing trained” and “too much trained.”

## Why frozen features still participate in forward computation

We still compute the backbone output for each batch; freezing saves gradients and
updates, not the entire forward-pass cost.
Our input images do not require gradients and the backbone parameters do not
either, so autograd can avoid recording those frozen operations.
The trainable head operates on resulting features and builds its own graph.
Caching fixed features is a possible optimization, but would need care if training
augmentation later changes the input on each pass.

## Controlled edits

1. Replace `training_mode` with `model.train()`.
   Predict the frozen-state assertion fails on BatchNorm buffers, even with no
   backbone gradients.
2. Forget to enable head gradients.
   Predict backward fails or the parameter-membership assertion exposes the error,
   not a successful “frozen head training” run.
3. Add backbone tensors to the optimizer while leaving them frozen.
   Predict the membership assertion rejects the mismatch before training.
4. Remove `.clone()` from the snapshot.
   Predict the before/after comparison may become meaningless due to shared storage.

## Checkpoint questions

**Does `requires_grad_(False)` stop running-mean updates?**
<details>
<summary>Solution</summary>
No. Running statistics are buffers changed by BatchNorm's training forward pass.
Evaluation mode stops that behavior; gradient flags control a different mechanism.
</details>

**Can a frozen pretrained backbone still be a bad choice?**
<details>
<summary>Solution</summary>
Yes. Features may not separate the new domain's classes. Head-only training can
only recombine existing features; validation and comparison with a small scratch
CNN decide whether that restriction is useful.
</details>

## Design judgment and limitations

A head-only stage is a cheap, controlled baseline for reuse.
It is not proof that all useful adaptation is in the last layer.
Keep its selected checkpoint: Session 32's fine-tuning must earn its extra
complexity by improving validation, not by being the next item in a recipe.
