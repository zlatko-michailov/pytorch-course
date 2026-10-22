# Session 13: accuracy can reward doing nothing useful

Project 3, lesson 4 of 4.

**Prerequisite:** [Session 12](../12-decision-thresholds/README.md).
**New:** rare positives, a majority-class baseline, and class-weighted training.

## A high score can conceal a useless policy

We change the synthetic label rule so positives are relatively rare. Predicting
"negative" for every example can produce high accuracy and exactly zero recall.
That is an important baseline to report, not a competent detector.

We compare ordinary binary cross-entropy with its `pos_weight` option:

```text
weighted loss = -[a*y*log(p) + (1-y)*log(1-p)]
a = training_negative_count / training_positive_count
```

This counts the positive contribution more heavily. It does not create new
positive examples, repair missing labels, or guarantee an optimal deployed policy.
Class weights come from training data; a business cost may suggest a different
weight than a count ratio.

## Run and read

[Full source](train.py)

```sh
.venv/bin/python lessons/13-imbalanced-decisions/train.py
```

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 2 | Import layers/losses. |
| 4 | Seed the synthetic examples. |
| 5 | Generate independent two-feature rows. |
| 6 | Create a higher-threshold, rarer-positive rule. |
| 7 | Split training features and labels. |
| 8 | Split validation features and labels; reserve the final 100. |
| 9 | Count positives in training, retaining a one-output vector shape. |
| 10 | Check that neither class is absent. |
| 11 | Raise an explicit error rather than dividing by zero or silently inventing a class weight. |
| 12 | Calculate the training-count ratio. |
| 13 | Report training prevalence. |
| 14 | Report the always-negative baseline on validation. |
| 16 | Iterate over ordinary and weighted losses; each pair unpacks into a label and weight tensor. |
| 17 | Reset the initialization seed, giving both models identical starting parameters. |
| 18 | Construct the affine scorer. |
| 19 | Use the same optimizer/rate for both. |
| 20 | Configure the positive term's weight; a weight of one is ordinary BCE. |
| 21 | Use the same update budget for both. |
| 22 | Clear gradients. |
| 23 | Calculate the specified training objective. |
| 24 | Differentiate. |
| 25 | Update. |
| 26 | Evaluate without recording. |
| 27 | Use the same default logit threshold for this controlled comparison. |
| 28 | Convert validation labels to Boolean values. |
| 29 | Count detected positives. |
| 30 | Count false alarms. |
| 31 | Count missed positives. |
| 32 | Calculate accuracy. |
| 33 | Calculate recall with an explicit undefined convention. |
| 34 | Report both success and error counts, not accuracy alone. |
| 35 | Confirm test rows were not used for this comparison. |

## Weighting and thresholding are related, not identical APIs

Both can shift the error tradeoff. Thresholding changes only the policy on fixed
scores; class weighting changes the fitted scoring function.

Under idealized unrestricted fitting, weighted BCE's optimum is
`q = a*p / (a*p + 1-p)`, not the original conditional probability `p`.
Therefore the sigmoid output of a weighted model should not casually be described
as a calibrated probability of the original event.

Choose weighting and a threshold on validation using a metric aligned with use.
Do not compare weighted and unweighted *training loss numbers* as if they were
the same objective.

## Experiments

1. Increase the label threshold from 1.3 to 2.0. Inspect class counts before
   trusting any score. A small validation set may contain very few positives.
2. Set the positive weight to one for both runs. With the same initialization
   and data, their results should agree.
3. Freeze the ordinary scorer and try Session 12's lower thresholds. Compare
   this intervention with retraining a weighted scorer.
4. State a constraint such as "recall at least 0.9" and discuss what validation
   size is needed before confidently making such a claim.

## Checkpoint

Why can 90% accuracy be poor? Does the test set determine class weights? Does
weighted BCE guarantee calibrated probabilities?

<details>
<summary>Worked answers</summary>

If 90% of examples are negative, predicting negative always reaches 90% accuracy
while finding no positives. Fitted class counts come from training, and policy
choices from validation. Weighted BCE changes the objective, so its output
probabilities require careful interpretation/calibration.

</details>

**Project 3 complete:** output, loss, metric, and threshold each have a separate
job. [Next: nonlinearity](../14-nonlinear-regression/README.md).
[Course map](../../CURRICULUM.md).
