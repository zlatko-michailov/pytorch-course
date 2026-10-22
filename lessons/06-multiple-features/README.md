# Session 6: predict from more than one feature

Project 2, lesson 3 of 6.

**Prerequisite:** [Session 5](../05-held-out-data/README.md).
**Review:** train/validation separation and `Linear`. **New:** feature vectors,
matrix multiplication, coefficients, and feature availability.

## From a line to an affine function of two measurements

Suppose two measurements help predict a calibrated reading:

```text
y = 2*x0 - 3*x1 + 1 + noise
```

Each example now has two features. We still predict one scalar, so the model is
`nn.Linear(2, 1)`: two weights and one bias. We have not introduced hidden layers.

For row-major examples `X` with shape `[N,2]`, the expression is
`X @ W.T + b`, where the layer stores `W` with shape `[1,2]`.
Our generating coefficient column has shape `[2,1]`, hence `X @ coefficients`
directly produces `[N,1]`.

The matrix multiplication contracts the feature dimension:

```text
[N, 2] @ [2, 1] -> [N, 1]
```

`*` would multiply elements, not sum weighted features. Using a familiar operator
with the wrong tensor semantics is a common source of plausible-looking bugs.

## Run and read

[Complete source](train.py)

```sh
.venv/bin/python lessons/06-multiple-features/train.py
```

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 2 | Import layers and losses. |
| 4 | Seed the experiment. |
| 5 | Generate 100 independent two-feature examples from a standard normal distribution. |
| 6 | Store the known two coefficients as a column. |
| 7 | Multiply input rows by that column, then add bias and independent noise. |
| 8 | Split feature rows into training and validation. Contiguous slicing is acceptable here because rows were sampled independently, not sorted by time/entity. |
| 9 | Split targets at the same row boundary. |
| 10 | Construct a two-input, one-output layer. |
| 11 | Configure SGD over all three learned scalars. |
| 12 | Construct MSE loss. |
| 14 | Repeat 120 updates. |
| 15 | Clear old gradients. |
| 16 | Calculate training predictions and loss. |
| 17 | Differentiate with respect to both weights and bias. |
| 18 | Apply the update. |
| 20 | Disable recording for reporting. |
| 21 | Measure error on untouched validation rows. |
| 22 | Construct one new example as `[1,2]`, not `[2,1]`. |
| 23 | Predict its one scalar output. |
| 25 | Display shapes to make the contract visible. |
| 26 | Display the learned coefficient row and scalar bias. |
| 27 | Display validation loss. |
| 28 | Display the example prediction, near `2*0.5 - 3*(-1) + 1 = 5`. |

The row split differs from the previous random-permutation example only because
these rows are already independently randomized. It is not permission to split
time series this way.

## What a coefficient does and does not tell you

Holding the other feature fixed, one additional unit of `x0` changes the model's
prediction by its first weight. Units matter: coefficients are not directly
comparable when features have different units.

Nor do fitted coefficients automatically imply causation. Correlated measurements,
omitted factors, or a changed data-generating process can alter interpretation.
We control all factors here only because the experiment is synthetic.

## Experiments

1. Replace the generating coefficient `-3` with `0`. Can the learned second weight
   become small without removing the feature from the architecture?
2. Set `inputs[:, 1] = inputs[:, 0]` before generating targets. Now only the sum
   of the coefficients is identifiable. Multiple weight pairs make the same
   predictions; recovering the original pair is not an appropriate requirement.
3. Swap the feature order in the new example only. The model will not know:
   equal-shaped tensors can still violate the semantic input contract.

## Checkpoint

How many parameters are learned? Why does `nn.Linear(2,1)` store a `[1,2]` weight?
What is the output shape for one versus 80 examples? Can a measurement recorded
after a prediction time be a valid input?

<details>
<summary>Worked answers</summary>

There are two weights and one bias. The stored row corresponds to one output's
weights over two input features. Outputs are `[1,1]` and `[80,1]`. A future
measurement unavailable at inference time leaks information even if the split
and tensor shapes otherwise look correct.

</details>

**Design habit:** record feature meanings, order, units, and availability, not
just tensor sizes. [Next: scaling](../07-training-only-scaling/README.md).
[Course map](../../CURRICULUM.md).
