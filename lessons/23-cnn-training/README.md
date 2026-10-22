# Project 6, Session 23: compare the CNN with the existing MLP

[Course curriculum](../../CURRICULUM.md) · [Complete train.py](train.py)

## Prerequisites and new ideas

Review Session 22's shapes, kernels, channels, and parameter sharing.
Review Sessions 18–20's data contract, CE, and validation analysis.
The optimization loop is unchanged from Session 19.

**New:** a controlled architecture comparison and the distinction between equal
training budgets, equal parameter counts, and equal optimization opportunity.
The CNN is a hypothesis about images, not an automatic upgrade.

## The question

Does adding locality and shared features improve held-out predictions under a
small, declared CPU training budget?
To answer that, compare on the same task rather than swapping datasets, data
fractions, normalization, learning rates, and architectures simultaneously.

We fit the old MLP and the small CNN in the same process.
Both see the same `[N,1,28,28]` inputs and `[N]` labels.
Both return `[N,10]` raw logits and minimize multiclass CE.
Both use the same train/validation indices and batch-order generator.
No augmentation or active dropout is involved yet.

## Run from the course root

```sh
.venv/bin/python -m lessons.23-cnn-training.train --smoke --epochs 1
```

This runs both real code paths on synthetic stripes.
It verifies model and loop compatibility, not a Fashion-MNIST ranking.

```sh
.venv/bin/python -m lessons.23-cnn-training.train --data data/fashion --epochs 3
```

For a deliberately reduced real experiment:

```sh
.venv/bin/python -m lessons.23-cnn-training.train --data data/fashion --epochs 3 --max-train 4096 --max-val 1024
```

Keep the chosen subset limits consistent throughout Project 6.
The offline official file layout and license provenance are in
[Session 18](../18-images-flat/README.md). Missing real files fail explicitly.
No torchvision, NumPy, or plotting dependency is needed.

The script writes `.artifacts/vision/fashion-cnn.pt` or `smoke-cnn.pt`.
It does not overwrite the earlier MLP artifact.
It opens no official test files.

## Complete code and walkthrough

The entire entry point is [train.py](train.py).
The model factories are [image_models.py](../../course/image_models.py) and
[image_training.py](../../course/image_training.py).
The latter also contains the explicit `fit` and evaluation loops explained in
Session 19; no trainer framework has been inserted.

`partitions(args)` runs once, returning the exact two datasets used by both models.
The list `[("mlp",mlp),("cnn",cnn)]` holds pairs of a printable name and a callable.
Unlike `mlp()`, the expression `mlp` does not construct a network yet.
It passes a function object, like a C# delegate or a C++ function pointer.
`for name, builder in ...` unpacks each pair.

`torch.manual_seed(1900)` is reset before each construction.
The models have different shapes, so this does not give corresponding identical
parameters; it fixes each initialization reproducibly.
`model = builder()` calls the current factory.
The parameter count is computed from actual tensors, not copied from a claim.
The CNN has 9,098 parameters and the MLP 50,890.

`fit(model,train,val,args.epochs)` uses:

- SGD with learning rate 0.05 and weight decay `1e-4`;
- batch size 32, training shuffled with a private generator seeded 1900;
- unshuffled validation with no optimization;
- a fixed epoch budget and best-validation-CE state restoration.

Each call constructs its own loader with the same shuffle seed.
Thus model-specific random operations cannot alter the batch order.
The returned dictionary describes the selected state, not necessarily the last epoch.
`result["loss"]` is the selected validation CE.

Only the CNN branch calls `save_model`.
Its architecture tag is `cnn-8-16`, with `dropout=0.0` recorded as metadata.
The base payload also stores classes, preprocessing, split, and subset limits.
Saving the selected state makes later inference reconstruction possible.
No criterion here depends on a test score.

## Make a small comparison record

For your real run, record both:

1. Training CE over epochs: is optimization making progress?
2. Validation CE over epochs: does held-out predictive loss improve?
3. Validation accuracy: is score improvement reflected in class choices?
4. Parameter count and wall-clock budget: what did the change cost?
5. Major per-class confusions: did one class improve at another's expense?

Use Session 20's confusion computation on CNN predictions if aggregate scores
suggest a useful follow-up. Do not assume a lower CE guarantees fewer errors on
every individual class.
CE rewards the whole score distribution; accuracy only checks the largest score.

## What is and is not controlled

Holding the optimizer fixed isolates a practical architecture substitution.
It does not prove either architecture has been optimally tuned.
An equal epoch budget is an equal pass-count budget, not equal FLOPs or wall time.
The CNN performs many spatial operations despite having fewer learned numbers.
The models also have different inductive biases and effective capacities.

If the CNN loses after three epochs, first check training curves and input shapes.
It might optimize more slowly at this learning rate.
If both fit training poorly, undertraining can hide differences in generalization.
If both fit training well but validation differs, capacity and inductive bias become
more plausible explanations.
Form one specific next experiment rather than declaring a universal winner.

## Controlled edits and expected outcomes

1. Increase only the epoch budget for both architectures.
   Predict more optimization opportunity; validation can flatten or worsen.
   Record each best epoch rather than comparing only the final epoch.
2. Change both learning rates in the shared helper from 0.05 to 0.01.
   Predict slower initial changes, not necessarily worse final generalization.
   Restore the helper afterward to preserve later lesson defaults.
3. Train on 64 examples with a fixed validation set.
   Predict large variance and possible memorization; neither model's result is
   representative of the full 54,000-example training partition.
4. Reorder pixels consistently at every location for all examples.
   Predict an MLP can still treat the inputs as features, but the CNN's locality
   assumption becomes much less appropriate.

## Checkpoint questions

**Does resetting the seed make the comparison statistically conclusive?**
<details>
<summary>Solution</summary>
No. It makes one paired setup reproducible. Multiple predeclared seeds and
uncertainty summaries are needed for a stronger estimate of a small difference.
</details>

**Why not pick the better model using official test accuracy now?**
<details>
<summary>Solution</summary>
That would turn the official test into an architecture-selection set.
We compare on validation and defer the Project 6 final evaluation until choices
about augmentation and dropout are fixed.
</details>

**Can the smaller CNN be slower?**
<details>
<summary>Solution</summary>
Yes. Parameter count measures learned storage, not how often weights are reused.
Convolutions apply their kernels at many spatial positions.
</details>

## Judgment and limitations

The printed comparison is an experiment under a budget, not a leaderboard result.
Our image size and class taxonomy are narrow.
In Session 24 we change the training distribution while keeping validation fixed;
that is a different hypothesis from merely changing the architecture.
