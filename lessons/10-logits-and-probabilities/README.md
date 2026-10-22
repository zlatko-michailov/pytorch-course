# Session 10: scores, probabilities, and decisions are different

Project 3, lesson 1 of 4.

**Prerequisite:** [Session 9](../09-save-and-load/README.md).
**Review:** model outputs and loss. **New:** binary labels, logits, sigmoid,
binary cross-entropy, and numerical stability.

## Change the output's meaning, not just its shape

For regression we predict a real-valued measurement. For a binary decision we
want to distinguish labels 0 and 1. An affine output remains unbounded; we call
that output a **logit** or raw score `z`.

```text
p = sigmoid(z) = 1 / (1 + exp(-z))
z = log(p / (1-p))
```

The inverse expression explains "logit": it is log-odds. A score of zero maps to
0.5, positive scores to values above 0.5, and negative scores below 0.5.

A threshold then maps the score/probability to a Boolean decision. A model can
keep the same scores while a user changes its decision threshold.

## A loss for binary labels

Binary cross-entropy is

```text
L = -[y*log(p) + (1-y)*log(1-p)]
```

If `y=1`, it penalizes assigning small probability to class 1; if `y=0`, it
penalizes assigning large probability to class 1.

`BCEWithLogitsLoss` combines this transformation and loss in a numerically stable
way. **Feed it raw logits, not sigmoid probabilities.** Applying sigmoid first
would transform probabilities a second time and give the loss the wrong contract.

Output shape `[N,1]` and floating targets `[N,1]` must match. Later, multiclass
cross-entropy will use a different target interface.

## Run and read

[Full source](train.py)

```sh
.venv/bin/python lessons/10-logits-and-probabilities/train.py
```

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 2 | Import loss classes. |
| 4 | Define three illustrative scores as a `[3,1]` tensor. |
| 5 | Define matching floating binary labels. |
| 6 | Apply elementwise sigmoid for interpretation, not as input to the combined loss. |
| 7 | Compare each probability with the declared decision threshold; the result has Boolean type. |
| 8 | Request one loss per example instead of the default mean reduction. |
| 9 | Calculate losses from **logits** and labels. |
| 11 | Flatten only for concise printing. |
| 12 | Print the corresponding sigmoid values. |
| 13 | Print thresholded decisions. |
| 14 | Print individual losses. |
| 15 | Calculate their mean explicitly. |
| 17 | Define extremely confident scores. |
| 18 | Give them the opposite labels to demonstrate a large penalty. |
| 19 | Calculate the stable combined loss. |
| 20 | Print the large but finite results. |
| 21 | Assert finiteness; a failed numerical contract raises an error. |

The stable implementation avoids directly calculating `log(0)` after a sigmoid
has rounded to exactly zero or one.

## Experiments

1. Negate the three scores without changing targets. Predict each loss change.
2. Change only the threshold to 0.8. Which decisions change? The losses do not:
   the threshold is not part of this training loss.
3. Make the extreme predictions confidently correct. Their losses approach zero.
4. Compare a score of `0.8` to a probability of `0.8`; calculate the sigmoid to
   see why these are not interchangeable numbers.

## Checkpoint

Does a sigmoid output guarantee calibration? Is thresholding differentiable?
Why train with the combined loss instead of thresholded decisions?

<details>
<summary>Worked answers</summary>

A sigmoid constrains the range but does not guarantee that predictions of 0.8
are correct 80% of the time. Thresholding is a discontinuous decision rule with
no useful ordinary derivative for this training loop. The smooth score-based
loss supplies a gradient while preserving the score/probability distinction.

</details>

**Design habit:** specify the output representation and the loss together.
[Next: train a classifier](../11-binary-classifier/README.md).
[Course map](../../CURRICULUM.md).
