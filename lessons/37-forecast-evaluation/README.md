# Session 37 — Evaluate the forecasting procedure, not one lucky window

[Course map](../../CURRICULUM.md) · Project 9: predict a sequence

## Problem and prerequisites

A model may fit one chronological holdout well and fail at another forecast time.
We need to test a **procedure**: what data it trains on, when it is refitted, which
input history is available, and how its horizon errors are scored.
That procedure must be chosen before inspecting final test performance.

This remains an explicitly **synthetic** teaching problem.
The longer series has the same trend-plus-cycles formula, now over 480 steps.
Nothing here estimates real sales, demand, or sensor accuracy.
You already know window regression, training-only scaling, persistence, seasonal
naive, MLPs, and chronological splits.
The new ideas are expanding-window backtesting, rolling forecast origins,
aggregate horizon metrics, recipe selection, and a sealed final evaluation.

## Run validation first

From the course root:

```bash
.venv/bin/python lessons/37-forecast-evaluation/train.py
```

The complete runnable source is [train.py](train.py).
There are no downloads, saved checkpoints, earlier-run dependencies, or packages
beyond Python and PyTorch.
A NumPy initialization warning from the installed PyTorch build is harmless here;
the program uses no NumPy API.
One CPU thread avoids overhead on tiny tensors, not a timing guarantee.

The default command performs validation only.
After freezing candidate recipes, update budget, metric, horizon, and refit policy:

```bash
.venv/bin/python lessons/37-forecast-evaluation/train.py --final-test
```

That command deterministically repeats validation, locks its winner, refits that
recipe on the full development prefix, and opens the final test once conceptually.
There is no disk lock preventing another execution.
The sealing is an experimental discipline reinforced by an explicit CLI gate,
not cryptographic access control.
Do not change settings after reading final test output and still call the same
test an untouched estimate.

## 1. Write the chronological protocol before modeling

Window length is 48 and horizon is six.
Origin `t` means history ends at `t-1` and targets are `t,...,t+5`.
The candidate recipes are fixed:

1. Persistence.
2. Seasonal naive, period 24.
3. An affine 48-lag, six-output model.
4. A 48-lag MLP with 24 hidden ReLU units.

Both learned recipes receive 200 full-batch Adam updates at learning rate `0.015`.
There is no fold-level early stopping or checkpoint selection.
The fixed update budget is part of the recipe being evaluated.
This isolates evaluation design rather than introducing another search loop.

The backtest schedule is:

| Fold | Fit from observed prefix | Score target timestamps |
|---|---|---|
| 1 | `[0,240)` | `[240,288)` |
| 2 | `[0,288)` | `[288,336)` |
| 3 | `[0,336)` | `[336,384)` |
| Final | `[0,384)` after selection | sealed `[384,480)` |

The training prefix expands as time advances.
Earlier validation observations become legitimate training history at a later
fold cutoff.
That is allowed because the procedure simulates successive historical deployments.
It would be wrong to use fold 3's future observations when fitting fold 1.
All candidate recipes follow the same schedule.

## 2. Separate refit cutoffs from individual forecast origins

At cutoff 240, fit once using only `[0,240)`.
Issue forecasts at origins `240,246,252,...,282`.
Every forecast has six targets entirely inside `[240,288)`.
There are eight origins per validation fold.

At origin 246, actual observations through 245 have become known.
They may enter its 48-step input window.
They do **not** update the model weights or scaler within that fold.
The next refit occurs only at cutoff 288.
Input refresh and parameter refitting are distinct operations.

The final test follows the same policy:
fit once at cutoff 384, then issue forecasts at
`384,390,...,474`.
That gives 16 final test origins.
Later test inputs may include earlier test observations already revealed by time.
No test labels enter optimization, scaler fitting, or recipe selection.
If deployment never observes those later true values, this would be the wrong
protocol; you would need a fixed-origin long-horizon or recursive evaluation.

## 3. Make the fitting API enforce an information boundary

`fit_candidate(name, observed, steps)` receives `series[:cutoff]`, not the complete
series and an easily forgotten mask.
The model fitter therefore cannot inspect later values.
Its training scaler is fitted from that prefix alone.
Its training origins end at `cutoff-HORIZON`, so every training target stays
strictly before the cutoff.

At each fold, learned models initialize from the same stated seed and fit from
scratch.
There is no optimizer-state transfer from a later fold to an earlier fold.
The returned state is `(model, mean, scale)`.
`predict()` scales only available input history and inverts predicted targets.
Baselines return no fitted model because their deterministic rules need none.

`windows()` constructs `(N,48)` inputs and `(N,6)` targets.
`evaluate_block()` fits from the permitted prefix, creates aligned evaluation
windows, obtains predictions, and returns an absolute-error matrix.
Targets are passed to the metric, not to the predictor.
A focused perturbation assertion checks that changing values from origin 240
onward cannot change its input history.
Boundary assertions also check complete target horizons and adjacent fold blocks.

## 4. Understand the aggregation

For each candidate, concatenate the three `(8,6)` validation error matrices:

```text
E has shape (24,6)
overall MAE = E.mean()
horizon MAE = E.mean(0)
```

Every origin/horizon pair has equal weight.
Horizon zero in the tensor is the first future observation, usually called
horizon 1 in a report.
The six horizon MAEs expose failures hidden by an overall scalar.
Here equal fold lengths happen to make unweighted averaging of fold MAEs agree
with concatenation.
Concatenation remains correctly weighted if a later design has different origin
counts.
Do not silently average unequal fold means as though their sample counts match.

The recipe with lowest aggregate **validation** MAE wins.
No final-test error appears in that decision.
Final refitting on all `[0,384)` observations is legitimate because those
observations are available at the final deployment cutoff.
The refit must keep the selected architecture, update budget, preprocessing rule,
and other hyperparameters fixed.

## 5. Nonoverlapping targets do not imply independence

The stride is six, equal to the horizon.
Unlike Sessions 34–36, each target timestamp appears once per evaluation block.
This simplifies interpretation and avoids duplicated target timestamps.
It does **not** create independent forecast errors.

Input windows overlap heavily.
The same fitted parameters generate several forecasts.
The series has common cycles and a trend.
Training prefixes overlap across folds.
Even disjoint target blocks can share a regime or a shock.
Therefore 24 validation origins are not 24 independent experimental replications.

Do not calculate a naive standard error by dividing an error standard deviation
by the square root of all window/horizon entries.
For a real uncertainty analysis, consider a justified time/block resampling
scheme, enough independent series, or a design that explicitly models dependence.
This lesson deliberately reports descriptive errors, not unjustified confidence
intervals or claims of statistical significance.

## 6. Inspect the reference backtest

Approximate aggregate validation MAEs are:

```text
persistence     1.687
seasonal naive  0.606
linear window   0.127
window MLP      0.143
```

The affine window recipe wins this data and budget.
Its fold MAEs are roughly `0.135`, `0.123`, and `0.125`.
That consistency is more informative than one unusually good period, but three
related folds still cover a very restricted synthetic process.

For the affine model, horizon MAEs are approximately:

```text
0.119, 0.106, 0.140, 0.114, 0.127, 0.157
```

They need not increase monotonically in a finite sample.
Seasonal naive's approximately `0.6` bias again matches the synthetic per-season
trend increment.
The fact that a linear lag model wins is acceptable: architectural complexity
should answer evidence, not dictate it.

After selection and final refitting, one reference test MAE is about `0.153`.
Its first forecast at origin 384 starts near `30.36,31.20,31.50`, versus actual
values near `30.24,30.95,31.33`.
The script reports only the selected recipe's final test metrics.
Minor numerical differences across versions are expected.
Neither this result nor the validation ranking guarantees a real-world advantage.

## 7. A validation-only experiment

Before running, predict how a smaller optimization budget changes model rankings:

```bash
.venv/bin/python lessons/37-forecast-evaluation/train.py --steps 20
```

Persistence and seasonal naive should remain unchanged because they do not train.
The learned models may underfit and their relative ranking may change.
That is a comparison of **recipes with different budgets**, not proof that an
architecture family is universally weak.
Keep test closed during the exercise.

As a source-edit study, change the forecast-origin stride from six to one.
Predict the increased evaluation count and duplicated target timestamps before
running.
Aggregate errors may change because the origin distribution changes.
Do not claim that multiplying overlapping forecasts multiplies independent
evidence.
If you change refit frequency, explicitly list what data each new fit can use.

## Checkpoint questions

1. Can fold 2 fit on fold 1's target observations?
2. Why fit scaling separately at each cutoff?
3. Does the final test allow fitting on its earlier true targets?
4. What should happen if the selected model disappoints on final test?

<details>
<summary>Explained solutions</summary>

1. Yes. At cutoff 288 those observations are historical and available. This is
   expanding-window backtesting, not a permanent fixed-training holdout.
2. A fold must reproduce information available at its historical deployment.
   A scaler fitted at cutoff 384 would reveal future levels to earlier folds.
3. This particular protocol permits those values as subsequent input context,
   but does not use them for weight updates, scaler fitting, or model selection.
   A refitting-within-test protocol could be evaluated, but must be specified
   beforehand and simulated honestly rather than improvised after results.
4. Report the failure and investigate. Redesign requires fresh untouched evidence
   for an honest final estimate; the old test has become development feedback.

</details>

## Design limits and readiness check

Only one short synthetic series and one regular regime are represented.
There are no missing intervals, revised observations, covariates, multiple series,
or probabilistic forecast intervals.
The metric treats all units and horizons equally, which may not match real costs.
You are ready to leave this project when you can write an availability-aware
origin table, compare meaningful baselines on identical targets, explain every
refit and fitted statistic, and defend a model choice without using final test
results as tuning feedback.
