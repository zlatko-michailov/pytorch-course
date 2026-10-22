# Session 11: train a binary classifier

Project 3, lesson 2 of 4.

**Prerequisite:** [Session 10](../10-logits-and-probabilities/README.md).
**New:** learning a decision boundary, converting labels to the required type,
classification accuracy versus cross-entropy.

## A deliberately inspectable decision

Our synthetic label is 1 when `2*x0 - x1 + 0.3 > 0`, otherwise 0.
One affine unit can represent this separating boundary. The model emits raw
logits; the loss encourages their signs to agree with labels.

The zero-logit boundary is the line `w0*x0 + w1*x1 + b = 0`. Multiplying all
coefficients by a positive constant preserves that boundary but changes the
scores and loss. Consequently recovering the exact generating coefficients is
not the correct success criterion.

For perfectly separable data, unregularized logistic-loss fitting can continue
increasing weight magnitudes even after classification accuracy is perfect.
Loss and accuracy measure different properties.

## Run and read

[Full source](train.py)

```sh
.venv/bin/python lessons/11-binary-classifier/train.py
```

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 2 | Import layers/losses. |
| 4 | Seed the data and initialization. |
| 5 | Create independent two-feature samples. |
| 6 | Compute the Boolean rule, convert labels to floating point, then insert the output-feature axis. `inputs[:,0]` selects every row's first column. |
| 7 | Set aside the first 240 rows for fitting. |
| 8 | Set aside the next 60 for development. The final 60 remain reserved. |
| 9 | Create a two-input, one-logit layer. |
| 10 | Configure SGD. |
| 11 | Use numerically stable binary cross-entropy over raw logits. |
| 13 | Repeat a fixed training budget. |
| 14 | Clear gradients. |
| 15 | Compute **training** loss on raw outputs. |
| 16 | Calculate gradients. |
| 17 | Update parameters. |
| 19 | Disable recording for evaluation. |
| 20 | Compute validation logits. |
| 21 | Threshold at logit zero, equivalent to probability 0.5, then convert to the labels' type. |
| 22 | Compare predictions with labels, convert Booleans to 0/1 floats, and average. |
| 23 | Evaluate validation loss without thresholding. |
| 25 | Report the smooth score-based objective on validation. |
| 26 | Report the fraction of correct decisions. |
| 27 | Inspect learned decision-boundary coefficients. |
| 28 | Confirm the remaining test data is not used for development. |

No sigmoid is necessary to threshold at zero. We would apply sigmoid to display
probability-like scores, never before this combined training loss.

## Accuracy is not a training objective here

Correct decisions all contribute the same amount to accuracy, whether their
logits are barely on the right side or highly confident. Cross-entropy also
cares about confidence, and supplies a useful gradient before and after a
thresholded decision changes.

Neither metric tells the full story under class imbalance or asymmetric costs.
The next two sessions separate those issues before adding nonlinear capacity.

## Experiments

1. Change training updates from 240 to 1. Inspect both validation loss and accuracy.
2. Train longer. Does accuracy stop changing while loss keeps changing?
3. Multiply learned weights and bias by two inside `no_grad` after training.
   The boundary and decisions remain the same, but the loss may change.
4. Replace the generating rule with `inputs[:,0]*inputs[:,1] > 0`.
   A single line cannot express this XOR-like relation; remember that failure
   for Session 15 rather than trying arbitrary rates forever.

## Checkpoint

Why are labels floating point for this loss? Why is zero the default logit
threshold? Why can multiple weight scales give identical decisions?

<details>
<summary>Worked answers</summary>

The binary cross-entropy interface expects a floating target with matching
output shape. Sigmoid maps zero to 0.5. Positive rescaling preserves the sign
of every logit, but changes probabilities and penalties. This is why coefficient
equality is not the measure of classifier adequacy.

</details>

[Next: decision thresholds](../12-decision-thresholds/README.md).
[Course map](../../CURRICULUM.md).
