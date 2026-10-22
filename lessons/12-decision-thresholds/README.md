# Session 12: a threshold encodes a decision policy

Project 3, lesson 3 of 4.

**Prerequisite:** [Session 11](../11-binary-classifier/README.md).
**New:** confusion counts, precision, recall, undefined metrics, and error costs.

## Keep the trained function fixed

Suppose missing an actual positive is five times as costly as a false alarm.
Accuracy alone treats those errors equally. We will reuse the same model and
change only the probability threshold on validation data.

```text
TP: predicted positive, actually positive
FP: predicted positive, actually negative
FN: predicted negative, actually positive
TN: predicted negative, actually negative

precision = TP / (TP+FP)
recall    = TP / (TP+FN)
```

Precision asks how reliable positive predictions are; recall asks how many actual
positives were found. When a denominator is zero the corresponding fraction is
undefined, not automatically perfect or zero. This lesson prints `nan` explicitly.

The illustrative cost `5*FN + FP` is a stated assumption, not a discovered
property of the data. Deployment costs and class prevalence must be understood
before treating such a cost as an optimization target.

## Run and read

[Full source](train.py)

```sh
.venv/bin/python lessons/12-decision-thresholds/train.py
```

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 2 | Import model/loss classes. |
| 4 | Reproduce the preceding synthetic example. |
| 5 | Generate feature rows. |
| 6 | Generate binary floating labels in `[N,1]` form. |
| 7 | Construct the same affine classifier. |
| 8 | Configure SGD. |
| 9 | Construct the combined binary loss. |
| 10 | Train for the same fixed budget. |
| 11 | Clear gradients. |
| 12 | Use only training rows to calculate loss. |
| 13 | Differentiate. |
| 14 | Update. |
| 16 | Disable tracking for validation scores. |
| 17 | Convert validation logits to probabilities for threshold interpretation. |
| 18 | Convert actual labels to Boolean values. |
| 19 | Consider three decision policies without retraining. |
| 20 | Apply each threshold to the same validation scores. |
| 21 | Count true positives using elementwise Boolean `&`. |
| 22 | Count false positives; `~` inverts each Boolean. |
| 23 | Count false negatives. |
| 24 | Count true negatives. |
| 25 | Calculate precision only when its denominator exists; otherwise return the explicitly undefined float `nan`. Python's conditional expression is `a if condition else b`. |
| 26 | Calculate recall under the same explicit convention. |
| 27 | Apply the stated relative error cost. |
| 28 | Print confusion counts. |
| 29 | Print metrics and the assumed policy cost. |

For Boolean tensors, `&` and `~` operate elementwise. Python `and` would try to
interpret an entire multi-element tensor as a single truth value and is not the
same operation.

## What threshold movement guarantees

Lowering the threshold on fixed scores can only add positive predictions.
It cannot reduce TP or FP, and cannot increase FN or TN. Recall therefore cannot
decrease where defined. Precision is not guaranteed to change monotonically:
the newly included examples may have a different positive rate.

Threshold choice is a hyperparameter choice. Make it on validation data, then
freeze it before final test evaluation. A saved classifier's decision policy
should be saved along with its scoring function.

## Experiments

1. Include thresholds `0.0` and `1.0`. Inspect the all-positive and all-negative
   policies and identify undefined precision.
2. Change the FN cost from 5 to 1, then 20. Which of the three policies would you
   select on validation? Do not claim the best among three is globally optimal.
3. Print total count `tp+fp+fn+tn` and verify it equals the validation size.
4. Repeat on a different validation sample without tuning on test data.
   Small samples produce unstable policy estimates.

## Checkpoint

Is threshold selection part of training? Can it use validation without
backpropagation? Does a lower threshold necessarily improve precision?

<details>
<summary>Worked answers</summary>

It is model-development/policy selection even though no neural parameters change.
Validation can guide this choice without gradients. Lower thresholds increase
the set of positive predictions, but their accuracy depends on which examples
join that set. The precision tradeoff is empirical, not a universal monotone law.

</details>

[Next: class imbalance](../13-imbalanced-decisions/README.md).
[Course map](../../CURRICULUM.md).
