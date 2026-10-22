# Project 5, Session 19: ten logits, one correct class

[Course curriculum](../../CURRICULUM.md) · [Complete train.py](train.py)

## Prerequisites and new ideas

Review Session 18's `[N,1,28,28]` image contract and stable split.
Review binary logits and `BCEWithLogitsLoss` in Sessions 10–11.
Review the MLP, SGD, and validation selection in Sessions 14–17.

**New:** mutually exclusive multiclass classification, one raw logit per class,
integer class-index targets, cross-entropy, and `argmax` predictions.
This is not ten unrelated binary classifications: each image has exactly one label.

## The motivating decision

The previous binary classifier could answer “is this class A?”
Fashion-MNIST asks “which one of ten classes?”
We need a score for every candidate, and a loss that compares the correct score
with all its competitors.

Our MLP is deliberately small:

```text
[N,1,28,28] -> Flatten -> [N,784]
 -> Linear(784,64) -> [N,64] -> ReLU
 -> Linear(64,10) -> [N,10] raw logits
```

Targets are `[N]` `torch.long`, not `[N,10]` one-hot floats.
For a single example with class index `y` and logits `z`,
`CE(z,y) = log(sum(exp(z_j))) - z_y`.
PyTorch computes this stably; do not implement the naive exponentials yourself.
There is no softmax layer before the loss.

## Run from the course root

```sh
.venv/bin/python -m lessons.19-multiclass-training.train --smoke --epochs 1
```

The smoke fixture uses the exact ten-logit loss and optimizer loop but synthetic
stripes, not clothes. Its accuracy has no benchmark meaning.

```sh
.venv/bin/python -m lessons.19-multiclass-training.train --data data/fashion --epochs 3
```

Real data must be the local official IDX files described in
[Session 18](../18-images-flat/README.md). No network operation occurs.
The default uses all 54,000 training and 6,000 validation examples.
For a limited CPU experiment, explicitly add `--max-train 4096 --max-val 1024`;
use those same limits for any comparison and report the reduced-data experiment.
Do not call a subset result a full-dataset benchmark.

The script writes `.artifacts/vision/fashion-mlp.pt` in real mode, or
`.artifacts/vision/smoke-mlp.pt` in smoke mode. They cannot be substituted silently.
It does not read official test data.

## Complete sources and walkthrough

[train.py](train.py) is the full entry point.
[image_training.py](../../course/image_training.py) is the full shared loop and
serialization source, introduced completely below.
[fashion_data.py](../../course/fashion_data.py) was explained in Session 18.

### The entry point, operation by operation

`arguments(...).parse_args()` reads the common offline-data options.
`set_num_threads(1)` bounds CPU concurrency.
`manual_seed(1900)` fixes parameter initialization; the split and batch shuffling
use separate generators, so reconstructing the split does not consume that seed.

`partitions(args)` returns training and validation datasets.
`mlp()` constructs the network; it has 50,890 trainable numbers.
`train.tensors[0][:4]` selects four images from the feature tensor inside our
`TensorDataset`; `tensors[1]` holds corresponding class labels.
We access that storage here for a tiny contract check, not as a required Dataset API.

`scores = model(...)` calls `nn.Module.__call__`, which invokes the model forward
operation. Prefer this over calling `.forward()` directly.
`cross_entropy(scores, labels)` consumes `[4,10]` and `[4]`.
The assertion compares it with `cross_entropy(scores+7, labels)`.
Adding the same scalar to every candidate score changes neither softmax nor CE.
`torch.allclose(..., atol=1e-6)` permits floating-point rounding.
This check is a numerical contract check, not a proof that the model learned.

`fit(...)` performs the epochs and restores the best validation-loss state.
`save_model(...)` stores that state with its interpretation, not a Python model
object whose class definition would need unsafe unpickling.

### Every function in the shared training helper

`mlp()` returns an `nn.Sequential`: modules run in the listed order.
`nn.Flatten(start_dim=1)` preserves the batch axis.
`nn.Linear(784,64)` learns a weight matrix `[64,784]` and a bias `[64]`.
`nn.ReLU()` applies `max(0,x)` elementwise.
`nn.Linear(64,10)` learns ten competing scores.
The absence of a final sigmoid or softmax is intentional.

`train_epoch(model, batches, optimizer, augment=None)` contains the familiar loop.
`None` means “no object,” analogous to a nullable reference.
`model.train()` sets training behavior for mode-dependent modules introduced later.
The optional callable `augment` is unused until Session 24; if present, only
training batches pass through it.
`zero_grad(set_to_none=True)` clears accumulated gradients before the forward pass.
The assertion checks score shape and integer-target dtype on every batch.
`loss.backward()` differentiates CE; `optimizer.step()` updates parameters.
`total_loss += loss.item() * len(labels)` accumulates a sum over examples.
We divide by the example count, not by the number of batches: a final short batch
must not get the same total weight as a full batch.

`evaluate(model, batches)` calls `model.eval()` and enters `torch.no_grad()`.
These are distinct controls; later we will demonstrate their difference.
No backward operation or optimizer update occurs.
`logits.argmax(dim=1)` chooses one column index per row, returning `[N]`.
`predictions` and `targets` are Python lists of tensors.
`.append(...)` adds one batch; `torch.cat(...)` concatenates along axis 0.
The result is a dictionary containing CE, accuracy, predictions, and actual labels.
`(predicted == actual)` is an elementwise boolean tensor, converted to float and
averaged to obtain accuracy.

`fit(...)` constructs SGD with learning rate 0.05 and weight decay `1e-4`.
It constructs the shuffled loader once, so its generator advances across epochs.
Each epoch trains, then measures validation CE and accuracy.
`float("inf")` initializes a value larger than any finite CE.
Whenever validation CE improves, the dictionary comprehension
`{name: value.detach().clone() for ...}` copies every state tensor.
Without `.clone()`, later training could mutate the supposedly “best” state.
`.detach()` excludes the copied state from an autograd graph.
This is a fixed short epoch budget with best-state restoration, not patience-based
early stopping. We reuse Session 17's validation-selection principle.
`load_state_dict(best_state)` restores the selected epoch, then we evaluate it.

`artifact(args,name)` builds a relative `Path` with a real/smoke-specific prefix.
`save_model` creates its parent directory using `mkdir(parents=True, exist_ok=True)`.
Its payload records architecture, weights, class order, preprocessing, split,
initialization seed, smoke status, and explicit data limits.
`**extra` accepts additional keyword metadata; later CNN lessons use it.
`torch.save(payload,path)` writes only our known dictionary of data and tensors.

`load_model` uses `torch.load(..., map_location="cpu", weights_only=True)`.
It checks schema, task, class mapping, preprocessing, split, and smoke status.
It then calls a supplied `builder()` and loads weights with `strict=True`.
A builder is an ordinary Python function passed as a value, like a C# delegate.
The model is returned in evaluation mode along with its payload.
Session 21 makes this reconstruction the main subject.
Only load artifacts from trusted sources: `weights_only=True` narrows supported
objects but does not make every hostile or enormous file safe.

## Controlled edits with predictions

1. Insert `softmax(dim=1)` before CE. Predict no shape exception, but the loss now
   treats bounded probabilities as logits, changing gradients and the objective.
   Remove it: “the program accepted it” is not enough.
2. Change the output width to 9. Predict failure when the target equals 9, and a
   shape assertion before that. The class mapping has ten entries.
3. Compare one epoch with five, holding split, limits, and seed fixed.
   Predict training loss may improve; validation need not improve monotonically.
   The saved state is chosen by validation CE, not the final epoch number.

## Checkpoint questions

**Do we need softmax to choose the predicted class?**
<details>
<summary>Solution</summary>
No. Softmax preserves ordering, so logits.argmax(dim=1) gives the same class.
Softmax is useful when displaying normalized scores, but those scores are not
automatically calibrated probabilities.
</details>

**Why not use the binary loss from Session 11 with ten outputs?**
<details>
<summary>Solution</summary>
Independent binary losses express ten yes/no labels and can allow multiple
positives. This task has exactly one class per example, so multiclass CE expresses
the appropriate competition and uses one integer target.
</details>

**Why choose the best epoch by CE rather than training accuracy?**
<details>
<summary>Solution</summary>
Training accuracy measures fitted examples and ignores how strongly wrong scores
are ranked. Validation CE measures held-out predictive loss. It still has selection
bias after repeated experimentation, which is why the test remains reserved.
</details>

## Judgment and limits

This MLP is an explicit baseline, not a claim that dense layers are the best image
architecture. An optimizer and a fixed seed do not eliminate experimental variance.
The next session inspects *which* errors hide inside aggregate accuracy.
