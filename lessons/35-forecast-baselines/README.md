# Session 35 — Beat a forecast, not an arbitrary constant

[Course map](../../CURRICULUM.md) · Project 9: predict a sequence

## Problem and prerequisites

The window model from [Session 34](../34-forecast-windows/README.md) predicts a
highly regular synthetic series.
A global-mean baseline ignores almost everything already known at the origin.
Before adding neural complexity, compare forecasts that exploit recency and
seasonality without any learned parameters.

This remains an explicitly **synthetic** exercise, not a real forecasting study.
You already know chronological target splits, windows, forecast origin `t`,
direct horizons, and MAE.
New concepts are persistence, seasonal naive prediction, recursive seasonal
reuse for horizons longer than one season, and horizon-aligned comparisons.
There is deliberately no training optimizer in this `train.py`.
A baseline that requires no gradient updates is still a complete model.

## Run the complete comparison

From the course root:

```bash
.venv/bin/python lessons/35-forecast-baselines/train.py
```

The full independent source is [train.py](train.py).
Only Python and PyTorch are needed; there are no downloads, saved files, or
earlier runs to execute.
The script uses the same 360-step synthetic formula, seed, and chronological
boundaries as Session 34.
A PyTorch warning about missing NumPy does not affect this torch-only code.
One CPU thread avoids overhead in tiny operations; it is not a benchmark claim.

The default command selects a baseline using validation and does not open test.
After specifying the period and horizon without consulting test results:

```bash
.venv/bin/python lessons/35-forecast-baselines/train.py --final-test
```

Test metrics for all three predeclared baselines are diagnostic.
The printed selected baseline remains the one chosen by validation.
Do not switch winners or change the period after reading those test diagnostics.

## 1. Keep the information set explicit

Origin `t` means the first unknown value is `series[t]`.
History ends at `series[t-1]`.
Each origin receives the last 48 observations and predicts the next six.
Training raw observations are `[0,240)`.
Validation target timestamps are `[240,300)` and test timestamps `[300,360)`.
Every compared forecast uses the identical origin array and target matrix.

`comparison()` builds:

```text
X: (N,48)  — available raw history
Y: (N,H)   — true future values, used only for evaluation
```

At the default horizon, validation origins are `240,...,294`, giving 55 rows.
There is no separate seasonal-baseline subset that silently drops difficult
origins.
The required seasonal context is already present in the 48-step input.
Comparing methods on different timestamps can make rankings meaningless.

## 2. Training-mean baseline

The constant prediction is the mean of `series[:240]`, approximately `23.007`.
Repeat that one value for every origin and horizon.
Do not recompute it using validation or test values.

This baseline is cheap and useful for checking scale, but it ignores both the
rising level and the repeating cycle.
It is not the strongest sensible competitor for this problem.
There is no numerical scaling here: these baselines operate directly in the
original units and do not need gradients or standardized optimization inputs.
Avoid introducing fitted state when a method does not require it.

## 3. Persistence: the most recent value

Persistence predicts:

```text
prediction[t,h] = series[t-1], for h = 0,...,H-1
```

The code uses `x[:, -1:].expand(-1, H)`.
`-1:` retains a length-one axis, so broadcasting produces shape `(N,H)`.
`expand` is a view that repeats values logically; the source is not future data.
This forecast is often competitive when a level changes slowly.

It cannot anticipate the rise and fall of a cycle.
Longer horizons often hurt, but a particular finite horizon curve need not be
strictly increasing.
Seasonality, noise, and the chosen origin distribution all matter.

## 4. Seasonal naive: copy the corresponding previous-season value

For a known season length `P=24` and `H<=P`:

```text
prediction[t,h] = series[t+h-P]
```

Each source index is strictly smaller than `t`.
This is **not** `series[t+h-1]`: that expression would use actual future values
for every horizon after the first.
The latter is a collection of one-step forecasts with later information, not a
six-step forecast issued at one origin.

For horizons longer than a season, the naive forecast must recurse.
It repeats the last observed cycle:

```text
prediction[t,h] = series[t-P + (h % P)]
```

`%` is the remainder operator.
Once the forecast reaches the next season, it copies its own repeated pattern,
not a newly observed target.
The source computes these indices and asserts every one is before its origin.
It also checks that indexing the raw series gives the same seasonal matrix as
indexing the history window.

The period 24 is supplied by the known **synthetic design**, not fitted using the
test series.
For real data, period selection is a modeling decision requiring domain knowledge
or training/validation evidence.

## 5. Compare exactly aligned errors

For absolute-error matrix `E` with shape `(N,H)`:

```text
overall MAE  = E.mean()
horizon MAE  = E.mean(0)    # shape (H,)
```

This weights every origin/horizon pair equally.
The same target timestamp may appear at several horizons from different origins.
That is a deliberate forecast-level metric, not a count of independent outcomes.
If business costs weight later horizons more heavily, define those weights before
selection and apply them identically to every model.

The default synthetic reference validation scores are approximately:

| Forecast | Overall MAE |
|---|---:|
| Training mean | 4.019 |
| Persistence | 1.553 |
| Seasonal naive | 0.600 |

Persistence's horizon errors rise roughly from `0.52` to `2.47`.
Seasonal naive stays around `0.60` across the six horizons.
The generator's trend rises `24*0.025 = 0.6` per season, explaining much of this
systematic lag.
This is an interpretation of the known generator, not a correction fitted from
test residuals.

At origin 240 the actual first value is about `26.81`.
Persistence starts at about `25.98`; seasonal naive starts at `26.37`.
The script prints complete forecast vectors so you can inspect alignment.
An optional frozen test gives approximately `1.731` persistence MAE and `0.630`
seasonal MAE; exact decimals can vary slightly by version.

## 6. Predict before changing the specification

```bash
.venv/bin/python lessons/35-forecast-baselines/train.py --season-length 12
.venv/bin/python lessons/35-forecast-baselines/train.py --horizon 1
.venv/bin/python lessons/35-forecast-baselines/train.py --horizon 30
```

Period 12 matches the smaller cosine component but not the dominant sine.
Predict substantially worse seasonal-naive validation error.
For horizon one, persistence should be more competitive because it only projects
one step forward.
Horizon 30 forces the recursive seasonal behavior.
The contract assertion must still show all source indices before each origin.
Its number of complete target windows and task difficulty change, so do not
compare its overall MAE with horizon six as though only the architecture changed.

Use validation only during these exercises; omit `--final-test`.
If you experiment with a trend-adjusted seasonal baseline, estimate adjustments
from permitted training history rather than subtracting the measured test error.
Predeclare that new recipe before its final evaluation.

## Checkpoint questions

1. Is seasonal naive a “trained network” with a zero learning rate?
2. Why is `series[t+h-1]` an invalid persistence baseline for `h>0`?
3. What happens at horizon 25 when `P=24`?
4. Can a method win overall while losing at a particular horizon?

<details>
<summary>Explained solutions</summary>

1. No. It is a deterministic forecasting rule, with period as a supplied
   hyperparameter. Its usefulness does not depend on having trainable weights.
2. The forecast was issued at `t`; later actual values are not yet observed.
   Reusing them changes the information set and makes comparison dishonest.
3. Zero-based `h=24` returns to the first value of the last observed season.
   It does not read `series[t]`, even though that timestamp exists in the stored
   offline evaluation tensor.
4. Yes. Overall MAE averages a vector of horizon errors. Report that vector to
   reveal tradeoffs that a single scalar hides.

</details>

## Limitations and next step

No forecast here accounts for holidays, structural changes, irregular time steps,
uncertainty, or delayed observations.
Persistence can beat seasonal naive when seasonality is weak or changes.
There is no universal winner or guarantee a neural model improves either rule.
In [Session 36](../36-temporal-convolution/README.md), compare a window MLP and a
temporal convolution against both baselines on the same forecast examples.
