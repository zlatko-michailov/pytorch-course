# Session 5: give data three separate jobs

Project 2, lesson 2 of 6.

**Prerequisite:** [Session 4](../04-noise-and-baselines/README.md).
**New:** training, validation, and test roles; random index splits; leakage.

## A score is only meaningful relative to a question

Training data answers "which parameters reduce error on these examples?"
Validation data answers "which of our development choices appears promising on
unseen examples?" Test data provides a final estimate **after** development
choices have been fixed.

Validation influences us even when it has no autograd graph. If we try learning
rates and keep the one with the best score, the validation set helped choose the
result. Repeating that process on the test set makes it another validation set.

For our synthetic, independently noisy measurements drawn from the same simple
operating range, a random split is reasonable. It is not a universal recipe:
future forecasting needs chronological splits, and repeated records from the
same entity may need group-based splits.

## Run the complete example

[Full source](train.py)

```sh
.venv/bin/python lessons/05-held-out-data/train.py
```

The final print reports the **count**, not a score, of the reserved test examples.
We will keep that boundary through preprocessing and training changes, then
demonstrate final evaluation and persistence in Session 9.

## Every nonblank line

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 2 | Import layers and losses. |
| 4 | Fix the synthetic experiment's random seed. |
| 5 | Make 120 scalar inputs as a `[120,1]` matrix. |
| 6 | Generate independent noisy observations. |
| 7 | `randperm` returns every index once in random order; `len(inputs)` is the first dimension's size. |
| 8 | Python slicing `[:80]` takes the first 80 shuffled indices. |
| 9 | `[80:100]` takes the next 20, excluding index position 100. |
| 10 | `[100:]` reserves the remaining 20. |
| 11 | Tensor indexing selects paired training rows. Python tuple unpacking assigns two names from the two expressions. |
| 12 | Select paired validation rows without changing their correspondence. |
| 14 | Create an affine model. |
| 15 | Set up SGD. |
| 16 | Construct MSE loss. |
| 17 | Fit the constant baseline **only** from training targets. |
| 19 | Repeat training. |
| 20 | Clear gradients. |
| 21 | Compute loss from training rows only. |
| 22 | Backpropagate that training loss. |
| 23 | Update learned parameters. |
| 25 | Disable graph recording for reporting. |
| 26 | Measure training error after all updates. |
| 27 | Measure error on rows that never supplied gradients. |
| 28 | Evaluate the training-fitted constant on the **same validation rows** as the network. |
| 30 | Show the split sizes. |
| 31 | Report fitting performance. |
| 32 | Report validation performance. |
| 33 | Report the comparable baseline score. |
| 34 | Confirm the test data is reserved, without inspecting its performance. |

The two scores do not have to obey `training < validation` in every small sample.
They average different noise realizations; a chance reversal is possible.
We care about patterns and a defensible procedure, not a memorized inequality.

## What leakage would look like

Computing the constant using **all** targets would let validation/test values
affect a fitted component. Next, scaling all inputs before splitting would let
unseen rows influence preprocessing. A less obvious example is using a field
recorded only after the event we want to predict.

No-gradient evaluation prevents parameter differentiation, not information leaks.
The distinction matters even with very simple models.

## Experiments

1. Train on 10 examples instead of 80, leaving disjoint validation/test slices.
   Across fixed, honestly reported seeds, does coefficient variability grow?
2. Try learning rates `0.01` and `0.1` and compare **validation** MSE after 120
   updates. Explain why you should not inspect test scores to choose.
3. Replace the random split with the first 80 sorted inputs for training.
   You changed the prediction question: the held-out range now requires
   extrapolation rather than interpolation. Interpret the score accordingly.

## Checkpoint

Can validation examples change a chosen model without backpropagation? Is a
random split always appropriate? Where must the baseline constant be fitted?

<details>
<summary>Worked answers</summary>

Choosing architecture, rate, stopping time, or a threshold from validation scores
uses those examples indirectly. Split design must match deployment information
and dependencies. Fit the baseline on training data, then evaluate it on the
same validation/test rows as competitors.

</details>

**Design habit:** write down the split rule and the intended future use before
looking at model scores. [Next: more features](../06-multiple-features/README.md).
[Course map](../../CURRICULUM.md).
