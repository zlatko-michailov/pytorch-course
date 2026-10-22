# Session 14: add nonlinearity only after identifying the limitation

Project 4, lesson 1 of 4.

**Prerequisite:** [Session 13](../13-imbalanced-decisions/README.md).
**Review:** regression, validation, and parameter counts.
**New:** hidden features, `Sequential`, ReLU, and representation capacity.

## A line cannot represent a parabola

We return to regression with the noiseless target `y=x^2`. An affine model is now
deliberately misspecified. Making SGD run longer cannot turn a line into a curve.

Stacking affine layers without an activation does not solve this:

```text
h = A*x + a
y = B*h + b = (B*A)*x + (B*a+b)
```

The composition is still affine. An intermediate **nonlinear activation** changes
what can be represented. ReLU is `max(0,z)`, applied elementwise. A hidden layer
learns 16 affine features, ReLU bends them, and the final layer combines them.

The final output is not passed through ReLU: a regression output should not be
restricted without a reason grounded in the task.

## Run the complete comparison

[Full source](train.py)

```sh
.venv/bin/python lessons/14-nonlinear-regression/train.py
```

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 2 | Import layer classes. |
| 4 | Seed initialization and the split. |
| 5 | Use one CPU thread for tiny operations, avoiding multithreading overhead; this is not a model change. |
| 6 | Construct scalar inputs as `[120,1]`. |
| 7 | Define the nonlinear target. |
| 8 | Shuffle row indices once. |
| 9 | Select training rows. |
| 10 | Select validation rows; leave the final 20 reserved. |
| 11 | Begin a Python dictionary of named candidate models. |
| 12 | Include the previous affine baseline. |
| 13 | `Sequential` registers layers and calls them in order: `[N,1] -> [N,16] -> [N,16] -> [N,1]`. |
| 14 | Close the dictionary literal. |
| 15 | `.items()` yields each name/model pair. |
| 16 | Construct a fresh optimizer for each model's own parameters. |
| 17 | Give each candidate the same update budget. |
| 18 | Clear gradients. |
| 19 | Compute training MSE. |
| 20 | Differentiate through the entire composition. |
| 21 | Update all registered parameters. |
| 22 | Disable recording for evaluation. |
| 23 | Measure validation error. |
| 24 | Inspect three informative input locations. |
| 25 | Count learned scalars using `.numel()`; the generator expression visits every parameter tensor. |
| 26 | Report capacity and validation score. |
| 27 | Inspect the predicted curve at three points. |
| 28 | Explicitly preserve the test partition. |

The affine model has two parameters. The hidden model has
`16*1 + 16 + 1*16 + 1 = 49`. More parameters permit a different family of functions
but do not automatically guarantee better optimization or better future error.

## What ReLU contributes

For scalar input each unit has a threshold where `w*x+b=0`.
A weighted sum of these bent functions forms a piecewise-linear approximation.
It need not reproduce an exact parabola everywhere to approximate it within the
training range.

ReLU's derivative is one on its positive side and zero on its negative side;
PyTorch chooses zero at exactly zero. A unit inactive on every training example
can receive no useful gradient through that path. Nonlinearity introduces
optimization considerations as well as capacity.

## Experiments

1. Remove `nn.ReLU()` but leave both affine layers. Predict the limitation before
   running; extra affine parameters do not make a nonlinear function.
2. Change width 16 to 2 and then 32. Track validation MSE and parameter count.
   Do not inspect test error to pick a width.
3. Predict at `x=5`. A good interpolation score is not an extrapolation promise.
4. Add noise as in Session 4. Explain why reducing training error toward zero
   may now cease to be a sensible goal.

## Checkpoint

Why is an activation essential between these layers? What exactly is hidden:
unobservable magic, or intermediate features? Is the output layer always activated?

<details>
<summary>Worked answers</summary>

Without an activation the composition collapses to an affine function.
Hidden means an intermediate representation rather than a supplied target.
The output transformation follows the task: raw regression values here,
raw logits for BCE earlier, not an arbitrary universal final activation.

</details>

[Next: nonlinear classification](../15-hidden-classifier/README.md).
[Course map](../../CURRICULUM.md).
