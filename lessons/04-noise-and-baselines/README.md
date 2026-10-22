# Session 4: imperfect measurements and a baseline

Project 2, lesson 1 of 6.

**Prerequisite:** [Session 3](../03-layers-and-optimizers/README.md).
**Review:** affine models, MSE, SGD. **New:** synthetic noise, random seeds,
constant baselines, and the distinction between a latent signal and observations.

## Why fitting a perfect line was too easy

A sensor rarely reports the exact quantity we want. We will generate observations
as `target = 2*x + 1 + noise`, with independent zero-mean Gaussian noise of standard
deviation `0.3`. The model sees observations, not the separate noise values.

The underlying signal is still linear. A line cannot fit every noisy observation
exactly, nor should exact interpolation be our goal.

For independent future noise, the expected MSE of the true signal is the noise
variance, `0.3^2 = 0.09`. A finite sample need not give exactly `0.09`, and a fitted
line can improve **training** loss slightly by fitting accidental sample variation.

We are simulating this mechanism, not claiming all real errors are Gaussian,
independent, zero-mean, or constant in scale.

## Build a baseline before celebrating

If inputs contained no useful information, what should we predict?
Under squared-error loss, the best constant on training data is the training
target mean. Differentiating `mean((c-y)^2)` with respect to `c` gives
`2*(c-mean(y))`, which vanishes at `c=mean(y)`.

Beating this constant means something different from beating the untrained
network. An untrained network is not a serious predictive baseline.

## Run the complete example

[Full source](train.py)

```sh
.venv/bin/python lessons/04-noise-and-baselines/train.py
```

## Every nonblank line

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 2 | Import its layer/loss namespace. |
| 4 | Seed PyTorch's random generator so repeated runs use the same synthetic noise and initialization. |
| 5 | Make 80 evenly spaced values; `.unsqueeze(1)` inserts an axis, turning `[80]` into `[80,1]`. |
| 6 | Compute the noiseless signal, available only because we designed this experiment. |
| 7 | `randn_like` returns independent standard-normal samples with the signal's shape/type; scale them and add them to the signal. |
| 8 | Compute the best training-set constant under MSE. |
| 9 | Measure its training MSE. Scalar broadcasting applies the same constant to every example. |
| 11 | Construct the one-input affine model. |
| 12 | Register its parameters with SGD. |
| 13 | Construct MSE loss. |
| 15 | Perform 120 updates. |
| 16 | Clear parameter gradients. |
| 17 | Compute predictions and MSE in one expression. |
| 18 | Calculate fresh gradients. |
| 19 | Update the parameters. |
| 21 | Disable tracking for measurements. |
| 22 | Compute final predictions. |
| 23 | Measure the fitted line against the noisy observations. |
| 24 | Measure the known generating signal against those same observations. This is a teaching diagnostic unavailable on ordinary real data. |
| 26 | Print the constant baseline's loss. |
| 27 | Print the finite sample's average squared noise. |
| 28 | Print the fitted model's training loss. |
| 29 | Show learned coefficients; they need not be exactly 2 and 1. |

The model has two parameters despite having 80 examples. Increasing the example
count does not increase its capacity.

## Experiments

1. Change noise scale `0.3` to `0.0`. Predict the final loss and coefficients.
2. Change it to `1.0`. Predict the signal-versus-observation MSE *in expectation*,
   not an exact observed number. Do longer training runs remove independent noise?
3. Change the seed and record three runs. Are coefficient differences evidence of
   a bug, or can noisy samples imply different best-fitting lines?
4. Replace `targets.mean()` with `0.0`. Why is this a weaker constant baseline?

Do not select a noise seed because it makes your model look better.

## Checkpoint

Why can training loss be below the true signal's loss? Which quantity is
approximately `0.09`, and why not exactly? Does beating the mean establish that
future sensor predictions will work?

<details>
<summary>Worked answers</summary>

Fitting adjusts the line to this finite sample, including accidental noise
correlations. The average squared noise estimates its variance but varies with
sampling. A training score does not test new measurements; next session creates
separate data roles before we use scores for model decisions.

</details>

**Design habit:** define a meaningful baseline and understand the observation
mechanism before enlarging a network.
[Next: held-out data](../05-held-out-data/README.md). [Course map](../../CURRICULUM.md).
