# Session 34 — A forecast is indexed by what was known when

[Course map](../../CURRICULUM.md) · Project 9: predict a sequence

## Problem, prior knowledge, and new concepts

We return to regression, but observations now have an order.
Our **synthetic** series combines a trend, repeating cycles, and small noise.
At a forecast origin, predict the next six values using only already observed
history.
Randomly shuffling overlapping windows into train and test would answer the
wrong question and can expose future information.

You already know affine/MLP regression, tensor shapes, train-only scaling, and
held-out evaluation.
Convolutions from earlier image lessons will become useful later in this project.
This session introduces a window length, forecast origin, forecast horizon,
chronological target boundaries, and a direct multi-output predictor.
We begin with an affine window model so indexing is the main new difficulty.

## Run the complete exercise

From the course root:

```bash
.venv/bin/python lessons/34-forecast-windows/train.py
```

The full independent source is [train.py](train.py).
It creates the series in memory; there are no downloads or earlier artifacts.
Only Python and PyTorch are used.
The existing PyTorch build may warn about missing NumPy, which this program never
calls.
The script uses one CPU thread to avoid tiny-operation thread-pool overhead,
without making a general speed claim.

The default command reports validation results and withholds test metrics.
After freezing the window, horizon, and training recipe:

```bash
.venv/bin/python lessons/34-forecast-windows/train.py --final-test
```

Do not use repeated final-test runs to choose settings.
No checkpoint file is written; this is a fresh deterministic run each time.

## 1. State the observation process

The 360 points follow this explicitly artificial rule:

```text
value[t] = 20 + 0.025*t
           + 3*sin(2*pi*t/24)
           + 0.8*cos(2*pi*t/12)
           + noise[t]
```

Noise has standard deviation `0.15`.
One integer step represents one observation interval; we do not pretend these
are real hourly sales or sensor readings.
The learner receives past values, not the sine formula or future timestamps.
All observations are assumed to become available immediately after their step.
Delayed reporting, revisions, and unavailable contemporaneous features would
require a different input-availability contract.

The chronological raw regions are:

```text
training       [0,240)
validation     [240,300)
sealed test    [300,360)
```

Intervals use an inclusive start and exclusive end.
The splits describe **target timestamps**, not merely where a window starts.

## 2. Define “origin” without an off-by-one ambiguity

Here origin `t` means the **first unknown target index**.
The last available observation is `value[t-1]`.
Other forecasting texts sometimes call that last observed timestamp the origin;
always check the convention before translating formulas.

For window length `W=24` and horizon `H=6`:

```text
input  = series[t-24:t]   # indices t-24,...,t-1
target = series[t:t+6]    # indices t,...,t+5
```

At the first validation origin `t=240`:

```text
input timestamps: 216,...,239
target timestamps: 240,...,245
```

Input and target share no timestamp **within one example**.
The model produces all six predictions at once.
It does not receive the actual first target to predict the second target.
This is direct multi-horizon prediction, not recursive prediction with teacher
forcing.

## 3. Build windows without crossing target boundaries

`make_windows(series,start,stop,W,H)` creates origins:

```python
torch.arange(max(start, W), stop - H + 1)
```

`torch.arange` excludes its endpoint, so the last allowed origin is `stop-H`.
Its last target index is `stop-1`.
For training, origins are `24,...,234`: 211 examples.
For validation, origins are `240,...,294`: 55 examples.
Origins `235,...,239` are deliberately not training examples because their target
horizons would cross into validation.

Using training history as the first validation window's context is legitimate:
those observations are already known at origin 240.
Likewise, later validation forecasts can use earlier validation observations once
their timestamps have passed.
The model parameters and scaler are not refitted from those observations in this
lesson.
That is rolling input refresh, not permission to train on future targets.

The window function asserts shape, nonnegative history indices, and complete
targets inside the requested region.
A separate probe changes every value at and after origin 240 and verifies that
its past input slice remains unchanged.
It protects the input-availability idea, not every possible leakage mechanism.

## 4. Fit scaling once, using the training prefix

The mean and population standard deviation come from `series[:240]`.
Every input and target uses:

```text
scaled = (raw - training_mean) / training_scale
```

The affine model predicts scaled targets.
Reporting reverses this transformation before calculating MAE:

```text
raw_prediction = scaled_prediction * training_scale + training_mean
```

Fitting this scaler from the whole 360-point series would reveal future levels.
Using all training values, including training targets, to fit a fixed scaler is
legitimate: the final trained model is deployed only after that prefix is observed.
That does not imply the scaler was available at every historical training origin
for an online model; this is an offline-fit, future-deployment experiment.

## 5. Trace the complete model and training loop

| Tensor | Default shape |
|---|---|
| raw series | `(360,)` |
| training inputs | `(211,24)` |
| training targets | `(211,6)` |
| validation inputs | `(55,24)` |
| validation targets | `(55,6)` |
| model predictions | `(B,6)` |

`nn.Linear(24,6)` has `24*6 + 6 = 150` parameters.
Each horizon has its own weight vector over the same available history.
It is a multi-output affine regression, not six sequential model calls.
The full-batch loop makes 250 Adam updates on training MSE.
Settings are fixed; validation is reported rather than used for early stopping.
`model.eval()` and `torch.no_grad()` mark inference explicitly.

MAE averages absolute errors in the original synthetic units.
`absolute_error.mean(0)` leaves the horizon axis intact, giving one MAE for each
lead time.
Horizon column zero means the first next observation, conventionally “horizon 1.”

## 6. Inspect a reference run

One run gives training mean about `23.007` and scale about `2.688`.
Validation MAE is about `0.148`.
The first predicted six values are approximately:

```text
prediction: 26.73, 27.52, 28.09, 28.32, 28.46, 28.39
actual:     26.81, 27.51, 28.10, 28.10, 28.29, 28.19
```

The optional fixed-recipe test MAE is around `0.144`.
These are illustrative synthetic results, not tolerances or performance promises.
This regular low-noise pattern happens to suit an affine combination of lags.
A strong result here does not imply neural forecasting superiority.
Next session adds meaningful forecasting baselines, not just a global constant.

## 7. Predict the outcome of controlled experiments

```bash
.venv/bin/python lessons/34-forecast-windows/train.py --window 1
.venv/bin/python lessons/34-forecast-windows/train.py --window 48
.venv/bin/python lessons/34-forecast-windows/train.py --horizon 12
```

With one input, the model cannot directly inspect a whole seasonal shape;
expect difficulty distinguishing rising and falling phases at the same level.
Window 48 includes two cycles but also adds parameters and changes the number
of training windows; improvement is not guaranteed.
Horizon 12 changes the task and reduces the number of complete validation origins.
Do not treat its aggregate MAE as an apples-to-apples architecture comparison
against a six-horizon run.
Compare horizon-specific errors and the printed origin/shape contracts first.
Use validation, not final-test output, to explore.

## Checkpoint questions

1. Why is a training origin of 239 invalid for a six-step target and cutoff 240?
2. Can a validation input contain a training observation?
3. Why does shuffling windows before splitting risk leakage?
4. Is every validation-window error an independent sample?

<details>
<summary>Explained solutions</summary>

1. Its targets are indices 239 through 244, so five targets lie after training.
   All target indices, not just the origin, must satisfy the boundary.
2. Yes. Historical context known by the forecast origin is available input.
   Reusing known history is different from fitting on held-out future targets.
3. Adjacent windows share most inputs and many targets. A training window may
   contain timestamps that are future relative to a randomly held-out origin.
4. No. Adjacent windows share history and target timestamps. Even nonoverlapping
   targets can have serial dependence. Do not calculate naive independent-sample
   confidence claims from the window count.

</details>

## Limitations and next step

The model has fixed-length complete history, evenly spaced observations, no
external covariates, no missingness, and immediate observation availability.
The single chronological validation period covers only one synthetic regime.
In [Session 35](../35-forecast-baselines/README.md), compare persistence and a
seasonal-naive forecast on exactly aligned origins and horizons.
