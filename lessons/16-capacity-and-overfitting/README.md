# Session 16: fitting observations is not the same as learning a useful rule

Project 4, lesson 3 of 4.

**Prerequisite:** [Session 15](../15-hidden-classifier/README.md).
**New:** capacity experiments, the training/validation gap, and Adam.

## A problem with more flexibility than data

We fit a noisy sinusoid from only 16 training measurements, with separate
validation examples. Compare a narrow network with a wide one on exactly those
same rows. Both have two hidden layers and ReLU.

**Underfitting** means the fitted predictor fails to capture relevant structure.
Insufficient capacity is one cause, but poor optimization or inadequate features
can also produce high training error.

**Overfitting** means development fits particularities of the sample that do
not transfer. Very low training error with substantially worse validation error
is evidence, not a mathematical proof, of that problem.

A larger network can improve validation performance and still overfit relative
to an earlier training point. Neither "large bad" nor "small good" is a rule.

## A second optimizer, motivated but not magical

Adam maintains moving averages of gradients and their squared magnitudes and
uses them to adapt per-parameter updates (with bias corrections). Unlike plain
SGD, it is not just `parameter -= lr*gradient`.

We use its standard defaults with a stated learning rate to make these tiny
nonlinear fits practical. It still requires sensible scaling and a suitable rate,
and cannot repair leakage or a model incapable of representing the task.

## Run and inspect

[Full source](train.py)

```sh
.venv/bin/python lessons/16-capacity-and-overfitting/train.py
```

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 2 | Import layers and activations. |
| 4 | Seed the data and split. |
| 5 | Limit tiny-operation overhead to one CPU thread. |
| 6 | Make scalar feature rows. |
| 7 | Generate a nonlinear signal with independent noise. |
| 8 | Shuffle row indices once. |
| 9 | Choose only 16 rows for training. |
| 10 | Choose 104 disjoint validation rows, leaving 40 reserved. |
| 12 | Compare two widths while holding the dataset fixed. |
| 13 | Use a reproducible initialization for each candidate; different widths still imply different parameter tensors. |
| 14 | Construct two hidden layers and an unrestricted scalar regression output. |
| 15 | Construct Adam over all registered parameters. |
| 16 | Use the same number of updates per candidate. |
| 17 | Clear gradients. |
| 18 | Calculate training MSE. |
| 19 | Calculate gradients. |
| 20 | Apply Adam's stateful update. |
| 21 | Disable recording for final measurements. |
| 22 | Measure the final predictor on its training rows. |
| 23 | Measure the same predictor on validation rows. |
| 24 | Count trainable scalars. |
| 25 | Report capacity and both scores together. |
| 26 | Preserve the final test partition. |

Unlike an online mini-batch average, these scores describe one fixed final
parameter state. Record them together before drawing conclusions.

## Experiments

1. Compare 100 and 1,200 updates at fixed width. Can training improve while
   validation worsens? Inspect the curve, not just the two endpoints.
2. Increase training count from 16 to 64 while preserving disjoint partitions.
   Does additional data reduce the gap for the same width?
3. Remove observation noise. Distinguish fitting noise from approximating the
   smooth underlying function.
4. Replace Adam with SGD. A poor fit at the unchanged rate/budget is not by itself
   evidence that the model lacks capacity; the optimizer's dynamics changed.

## Diagnose before prescribing

High training and validation loss: first check implementation, feature scale,
optimization, and whether the architecture can fit a tiny subset.
Low training but high validation loss: examine sample size, leakage-free splits,
regularization, distribution mismatch, and sampling variation.

Validation itself is a finite sample. Repeated architecture searches can overfit
it, even though no validation gradients are calculated.

## Checkpoint

Does a wide model's lower training loss prove its superiority? Is a gap always
caused by too many layers? Is Adam's learning rate interchangeable with SGD's?

<details>
<summary>Worked answers</summary>

Training loss tests fit, not future usefulness. A gap can also reflect distribution
differences, small samples, or a bad evaluation procedure. Adam scales updates
using stateful statistics, so equal numeric learning rates are not equal updates.
Choose interventions based on a discriminating check.

</details>

[Next: regularization and stopping](../17-regularization-and-stopping/README.md).
[Course map](../../CURRICULUM.md).
