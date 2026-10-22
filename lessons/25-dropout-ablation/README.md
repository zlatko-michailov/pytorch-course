# Project 6, Session 25: dropout, execution modes, and a fair ablation

[Course curriculum](../../CURRICULUM.md) · [Complete train.py](train.py)

## Prerequisites and genuinely new ideas

Review Sessions 16–17's overfitting, weight decay, and validation selection.
Review Sessions 22–24's CNN, fixed split, and train-only augmentation.

**New:** stochastic activation masking, inverted-dropout scaling, `train()` versus
`eval()` versus `no_grad()`, and a one-factor dropout ablation.
The loss, labels, and image preprocessing remain unchanged.

## The problem

When a model overfits, one hypothesis is that its prediction relies too strongly
on specific combinations of hidden activations.
Dropout randomly masks activations during training, forcing updates under varied
subsets of features. It is not a universal cure for poor generalization.
It can slow fitting or worsen results, particularly with a small network.

We put dropout after flattening the CNN features and before the ten-class head.
For probability `p`, each activation is zeroed with probability p during training.
Retained activations are multiplied by `1/(1-p)`.
This keeps each activation's expected value unchanged, not its variance.
In evaluation mode dropout is the identity; there is no additional scaling then.

## Run from the course root

```sh
.venv/bin/python -m lessons.25-dropout-ablation.train --smoke --epochs 1
```

This demonstrates execution modes and runs two synthetic training arms.
It is a pipeline check, not an estimate of real regularization benefit.
To exercise the explicitly synthetic final-test branch:

```sh
.venv/bin/python -m lessons.25-dropout-ablation.train --smoke --epochs 1 --final-test
```

For the real offline comparison:

```sh
.venv/bin/python -m lessons.25-dropout-ablation.train --data data/fashion --epochs 3
```

Keep any `--max-train` and `--max-val` limits equal to prior Project 6 runs.
The official data layout and provenance are in
[Session 18](../18-images-flat/README.md).
Only after you fix this experimental configuration, optionally run once:

```sh
.venv/bin/python -m lessons.25-dropout-ablation.train --data data/fashion --epochs 3 --final-test
```

That command evaluates only the validation-selected dropout arm on official test.
Default runs do not open the official test files.
Do not keep changing choices after reading the final score.

## First understand the three independent controls

| Control | What it does | What it does not do |
|---|---|---|
| `model.train()` | Selects training behavior of dropout/BatchNorm | Does not run a training loop |
| `model.eval()` | Selects evaluation behavior of those modules | Does not disable gradients or freeze parameters |
| `torch.no_grad()` | Prevents recording an autograd graph in its scope | Does not disable dropout |
| `parameter.requires_grad_(False)` | Excludes that parameter from gradient calculation | Does not set module mode |

The last row will matter especially for transfer learning.
Always reason about mode and differentiation separately.
Neither `eval()` nor `no_grad()` restores a previously selected parameter state;
checkpoint loading and mode selection are separate operations.

## Complete sources and detailed operations

Read [train.py](train.py), [image_models.py](../../course/image_models.py), and
the existing [image_training.py](../../course/image_training.py).
No trainer framework or new external package is involved.

The demonstration starts with a seed and `nn.Dropout(0.5)`.
`ones = torch.ones(256,requires_grad=True)` creates a differentiable probe vector.
`demonstration.train()` selects stochastic behavior.
Inside `no_grad`, its output still contains zeros and twos, not all ones.
The assertion deliberately proves that disabling gradients did not disable dropout.
It does not require every random mask to have exactly 128 zeros.

Next `demonstration.eval()` selects the identity operation.
Outside `no_grad`, the result still has `requires_grad=True`.
The equality and requires-grad assertion prove that evaluation mode is not an
autograd switch. A model in eval mode can still support input gradients.

`partitions(args)` returns the fixed train/validation data.
The ablation loops over probabilities `[0.0,0.4]`.
`manual_seed(1900)` resets initialization before each model is built.
The dropout module has no parameters, so inserting a different probability does
not alter the learned tensor shapes or initial weights.
The batch shuffle has its own generator, so dropout's random draws do not reorder
later training examples.

`fit` uses the same SGD, learning rate, weight decay, batch size, and epoch budget
in both arms. Each arm restores its best validation-loss epoch.
The augmentation option is intentionally not passed: **both arms have none**.
This isolates dropout rather than combining it with Session 24's intervention.
If augmentation won your real comparison, run a separate declared experiment
holding that policy fixed while varying only dropout.

`best_loss` begins at infinity.
When a returned validation loss is smaller, the code stores that model and p.
These are Python references, but no more training occurs on a model after its arm,
so retaining that reference is sufficient here.
The `selected-cnn` artifact records the selected dropout probability, class
mapping, input contract, split, and `augmentation="none"`.
It is named `smoke-selected-cnn.pt` or `fashion-selected-cnn.pt`.

Only the explicit `--final-test` branch constructs the test dataset.
`evaluate` sets eval mode and no-grad, so dropout is inactive for final scoring.
Neither arm is selected using test outcomes.
The smoke test partition is different synthetic data, not the official benchmark.

## Interpreting an ablation

An ablation changes one ingredient to estimate its effect under stated conditions.
Compare validation CE and accuracy, not only training loss.
The p=0.4 arm trains on masked features, whereas validation uses unmasked features;
training and validation losses are therefore not purely a generalization gap on
identical computations.
A high dropout rate may simply underfit within a short budget.
Repeat a promising real experiment with several declared seeds before treating a
small difference as reliable.

## Controlled edits with predictions

1. Change p=0.4 to p=0.9.
   Predict fewer retained features and noisier gradients, often much slower fitting.
   Retained features are multiplied by ten, not left unchanged.
2. Remove `model.eval()` from evaluation.
   Predict random validation scores because the previous training mode remains.
   `no_grad()` alone does not fix this.
3. Remove only `no_grad()` while keeping eval.
   Predict deterministic dropout behavior but unnecessary graph recording.
   Evaluation does not automatically update weights even if graphs are recorded.
4. Add augmentation to only the p=0.4 arm.
   Predict an uninterpretable one-factor comparison: two interventions changed.

## Checkpoint questions

**Does dropout remove weights from the saved model?**
<details>
<summary>Solution</summary>
No. It masks intermediate activations during training and has no learned weights
of its own. All dense-head weights remain in the checkpoint and are used in eval.
</details>

**Why are retained activations divided by `1-p` during training?**
<details>
<summary>Solution</summary>
An activation x survives with probability 1-p, so the expected scaled output is
(1-p)*x/(1-p)=x. This is inverted dropout; evaluation can then be the identity.
</details>

**Does this choose the best of all Project 6 configurations?**
<details>
<summary>Solution</summary>
No. It selects between p=0 and p=0.4 with no augmentation at this budget.
Combining independently promising ingredients requires another validation
experiment; their effects need not add.
</details>

## Judgment and limits

Fashion-MNIST's official test has already been available for Project 5; any
cross-project test-informed choices weaken a pristine final-test interpretation.
A fresh untouched holdout is stronger for consequential deployment.
Dropout is one tool alongside data quality, capacity control, weight decay,
augmentation, and early stopping—not a required feature of every neural network.
