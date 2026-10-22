# Project 6, Session 24: change training images without changing their labels

[Course curriculum](../../CURRICULUM.md) · [Complete train.py](train.py)

## Prerequisites and new ideas

Review Session 22's local image features and Session 23's controlled comparison.
Review Session 7's rule that evaluation preprocessing must not learn from holdouts.

**New:** augmentation as a label-preserving hypothesis, random cropping and
horizontal flipping, a callable closure with private RNG state, and train-only
transforms. Validation and test still receive deterministic fixed preprocessing.
No new model layers or loss are introduced.

## The problem

A small translation should usually not turn a Shirt into a Bag.
If training images show objects in nearly identical positions, a network may rely
too much on those positions. Augmentation presents varied versions of the same
training examples, encouraging less brittle features.
It is not free extra independently collected data.

Here we allow up to two pixels of translation and optional horizontal reflection.
These are plausible hypotheses for Fashion-MNIST's labels, not universal rules:
flipping a directional sign or a medical image may change its meaning.
Even on clothes, cropping can erase useful details and harm performance.

## Run from the course root

```sh
.venv/bin/python -m lessons.24-augmentation.train --smoke --epochs 1
```

This checks shapes, range, train-only routing, and unchanged validation storage.
The synthetic stripe scores are not evidence that real clothing benefits.

```sh
.venv/bin/python -m lessons.24-augmentation.train --data data/fashion --epochs 3
```

Optional explicitly limited comparison:

```sh
.venv/bin/python -m lessons.24-augmentation.train --data data/fashion --epochs 3 --max-train 4096 --max-val 1024
```

Use the same limits as Session 23 for a comparable development experiment.
Offline data acquisition, file names, and provenance are documented in
[Session 18](../18-images-flat/README.md).
No torchvision, image decoder, or package download is needed.
Only the real official training files are required; the official test is not read.

## Complete sources

Read [train.py](train.py) and the entire small helper
[image_augmentation.py](../../course/image_augmentation.py).
The unchanged CNN is in [image_models.py](../../course/image_models.py).
The optional augmentation argument in
[image_training.py](../../course/image_training.py) was introduced in Session 19.
This session explains the helper completely.

## The transformation, line by line

`from torch.nn import functional as F` gives the functional-operations module a
short local alias. `F.pad` is a function, not a layer with learned parameters.
`make_augment(seed=2400)` is a factory for a callable.
It creates a private `torch.Generator().manual_seed(seed)`.

The nested `def augment(images):` can refer to the outer `generator`.
Python retains that reference after `make_augment` returns.
This is a **closure**, similar to a C# lambda capturing a local variable.
Its RNG advances on each call. Creating a new factory restarts the seeded sequence.
The final `return augment` returns the function, not its result.

`F.pad(images,(2,2,2,2),value=-1.0)` adds two columns on each side and two rows
on top/bottom. The ordering is left, right, top, bottom.
`[N,1,28,28]` becomes `[N,1,32,32]`.
The fill value -1 means black under our fixed byte-to-[-1,1] normalization.
Using 0 here would add middle gray, not black.

`torch.randint(0,5,(len(images),2),generator=generator)` samples integer crop starts.
Each example gets `(top,left)` values in 0 through 4 inclusive.
A center start of 2 reproduces the untranslated original.
Starts 0 or 4 shift its visible content by two pixels in opposite directions.

`offsets.tolist()` converts the small integer table into Python rows.
`enumerate(...)` yields an example position plus its two offsets.
`padded[index,:,top:top+28,left:left+28]` selects one `[1,28,28]` crop.
The crop is a view into padded storage, not a modified original training image.
`torch.rand((),generator=generator).item()` samples a scalar uniform draw.
If it is below 0.5, `crop.flip(dims=(-1,))` reverses the last axis (width).
`(-1,)` is a one-element tuple; the comma matters.
Horizontal flipping does not reverse height or class indices.
`crops.append(crop)` collects image tensors, and `torch.stack(crops)` creates a
new `[N,1,28,28]` batch. Labels are never changed.

## Follow the entry point

`original = val.tensors[0].clone()` stores an independent validation-pixel snapshot.
`sample = train.tensors[0][:8]` chooses a training probe.
A separately created augmenter transforms it for a demonstration.
Assertions check identical shape, bounded pixel range, and an actual change.
The demonstration does not consume the later training augmenter's private RNG.

The loop runs `enabled=False` and then `True`.
Before each model construction, the global parameter seed is reset to 1900.
Each `fit` call gets the same shuffled training order and validation split.
The expression `make_augment() if enabled else None` selects a callable or no
transformation. This is a Python conditional expression.

Inside `train_epoch`, augmentation occurs before the model forward pass, only
when that callable exists. Evaluation has no augmentation parameter.
Therefore both arms are evaluated on the identical untransformed validation set.
The final `torch.equal` assertion also checks original storage was not mutated.
It would not alone detect evaluation-time transformations on temporary copies;
the separate `evaluate` code path supplies that architectural guarantee.

The augmented arm is saved as `fashion-augmented-cnn.pt` or
`smoke-augmented-cnn.pt`, recording its augmentation policy and p=0 dropout.
Saving that arm is not an assertion that it beat the baseline.
Read validation results before deciding to use it.

## Controlled edits

1. Change padding and offset ranges to allow a ten-pixel shift.
   Predict more missing content and possibly worse validation performance.
   Bigger variation is not always better regularization.
2. Replace horizontal flip with vertical flip.
   Predict upside-down clothing; explain why the label-preservation assumption
   and deployment plausibility deserve rechecking.
3. Reset the generator inside `augment` on every call.
   Predict repeated offset patterns across batches, reducing transformation variety.
4. Transform validation batches too.
   Predict a noisier and changed evaluation target, no longer an apples-to-apples
   estimate of performance on the original validation distribution. Restore it.

## Checkpoint questions

**Does augmentation increase the number of independent labeled observations?**
<details>
<summary>Solution</summary>
No. Multiple transformed versions share the original content and label.
They encode a chosen invariance assumption; they do not replace additional sampling.
</details>

**Why use a private augmentation generator?**
<details>
<summary>Solution</summary>
It makes transformations reproducible without changing the batch shuffle sequence
or consuming parameter-initialization randomness. This reduces avoidable confounds.
</details>

**Should inference randomly flip each new image?**
<details>
<summary>Solution</summary>
Not in this pipeline. Its inference contract is deterministic normalization only.
Test-time augmentation is another deliberate procedure with aggregation and cost;
it must be specified and validated rather than accidentally inherited from training.
</details>

## Design judgment and limitations

The readable per-image Python loop is adequate for a small CPU lesson, not a
high-throughput augmentation system. A fixed seed does not measure uncertainty.
One short comparison can support a follow-up hypothesis, not a universal claim
that augmentation helps all clothes or all image tasks.
