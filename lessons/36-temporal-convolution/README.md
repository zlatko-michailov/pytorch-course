# Session 36 — Reuse local patterns along time

[Course map](../../CURRICULUM.md) · Project 9: predict a sequence

## Problem and prerequisites

A window MLP treats every lag as a separate input position.
A temporal convolution instead applies the same small learned filter at many
positions in an available history.
That introduces locality and weight sharing, like an image convolution but along
one time axis.
Whether those assumptions help is an empirical question.

We continue the explicitly **synthetic** trend-plus-seasonality problem.
You already know chronological windows, scaling, persistence/seasonal forecasts,
MLPs, ReLU, convolution channels, and validation-based checkpoint selection.
New concepts are `(batch,channel,time)` layout, `Conv1d`, temporal receptive
fields, and the distinction between a past-only window and a streaming causal
convolution.
No recurrent network is an unexplained prerequisite.

## Run the complete exercise

From the course root:

```bash
.venv/bin/python lessons/36-temporal-convolution/train.py
```

The full standalone source is [train.py](train.py).
It creates the data again; no prior script or file is required.
It needs only Python and PyTorch, without downloads or additional packages.
A missing-NumPy warning from PyTorch does not affect these tensor-only operations.
One CPU thread reduces overhead for this small demonstration, not a universal
performance recommendation.

Test remains withheld until:

```bash
.venv/bin/python lessons/36-temporal-convolution/train.py --final-test
```

Run that only after freezing the architecture, training budget, and input window.
The deterministic rerun selects checkpoints using validation, then prints the
predeclared test comparison.
Do not change the selected model based on those diagnostics.

## 1. Freeze data, information, and metrics

The series has 360 observations from the same synthetic formula as Sessions 34–35.
The target boundaries remain training `[0,240)`, validation `[240,300)`, and test
`[300,360)`.
The horizon is fixed at six.
Both learned models receive the same 48-step history for every forecast.
Persistence and seasonal naive use that history too.

For origin `t`:

```text
available input = series[t-48:t]
future target   = series[t:t+6]
```

Training has 187 complete examples, origins `48,...,234`.
Validation has 55 examples, origins `240,...,294`.
The mean and scale are fitted on raw `series[:240]` once, then reused everywhere.
Both numerical targets and inputs are standardized by that same fitted state.
All reported MAEs are inverted back to original units.

No model receives the true first future target to predict later targets.
The complete six-output forecast is issued at one origin.
The source checks target boundaries, output shapes, and finite predictions.
Validation context can contain earlier validation observations already revealed
at that origin, without refitting the model.

## 2. Start with the window MLP

The first model is:

```text
(B,48) -> Linear(48,32) -> ReLU -> Linear(32,6) -> (B,6)
```

It has `48*32 + 32 + 32*6 + 6 = 1766` parameters.
Every hidden unit can combine any of the 48 lags.
It does not share a local pattern detector across positions.
That flexibility can be useful, but might need more data for less structured
time-series problems.

This MLP is the immediate architectural comparator for the convolution.
Do not attribute differences against Session 34's affine model solely to
nonlinearity: that earlier run also uses a different window, seed, update budget,
and number of training examples.
A strict ablation must freeze all of those choices.

## 3. Add channels and apply `Conv1d`

`TemporalConv.forward()` calls `x.unsqueeze(1)`.
It inserts a channel dimension without changing the underlying time ordering:

```text
(B,48) -> (B,1,48)
```

`Conv1d(1,8,kernel_size=5)` learns eight local filters.
Each filter has five weights per input channel plus a bias.
There is no padding and stride is one, so time length shrinks by four:

```text
(B,1,48) -> Conv1d -> (B,8,44) -> ReLU
```

The second convolution mixes all eight input channels:

```text
(B,8,44) -> Conv1d(8,8,5) -> (B,8,40) -> ReLU
```

`Flatten()` keeps the batch axis and combines channel/time axes:

```text
(B,8,40) -> (B,320) -> Linear(320,6) -> (B,6)
```

The complete convolutional model has 2302 parameters at default window length.
It is **not** automatically smaller than the MLP.
The dense head preserves time-position information but adds many parameters.
A pooled head would change that assumption and should be a separate experiment.

## 4. Receptive field versus forecast availability

With two stride-one width-five filters, a final convolutional position sees:

```text
5 + (5-1) = 9 original time positions
```

The dense output head combines all final positions, so a forecast can depend on
the entire 48-step past window.
Nine is the local convolutional receptive field, not the entire predictor's
history limit.

Do these filters “see the future”?
They see later positions **within the already observed input window**.
All such positions are earlier than the forecast origin.
That is valid for issuing a forecast after collecting the complete history.
There are no target values in the convolution input.

This is not a streaming network that emits a prediction at every intermediate
input position.
For streaming per-timestep outputs, a convolution using later input positions
would violate causality at those intermediate positions.
You would need explicit left-only padding or other causal construction and a
matching target alignment.
Calling a past-only window model “causal” must not obscure that distinction.

## 5. Train and select checkpoints fairly

`fit()` uses 160 full-batch Adam updates at learning rate `0.01`.
Every ten updates, and at the final update, it evaluates validation MAE.
The best state dictionary is cloned with `detach().clone()`.
Storing a live state dictionary without cloning could let later updates overwrite
the intended best checkpoint.
The function restores the selected state and returns its step number.

The MLP and convolution use the same budget, examples, scaled targets, and
checkpoint metric.
That is a controlled small comparison, not proof both have reached their best
possible performance.
The candidate set also includes persistence and seasonal naive.
The overall recipe is selected by validation MAE, including those non-neural
methods.
There is no rule that a learned model must win.

## 6. Read the output

Approximate default validation MAEs are:

| Model | MAE |
|---|---:|
| Window MLP | 0.227 |
| Temporal Conv1d | 0.304 |
| Persistence | 1.553 |
| Seasonal naive | 0.600 |

The MLP wins this particular small run.
For the first validation origin its forecast starts near `26.73,27.60,27.96`,
while actual values start near `26.81,27.51,28.10`.
Inspect the six horizon MAEs too: later horizons are not guaranteed to be worse
on every finite sample.

The optional frozen test reports roughly `0.373` MLP MAE and `0.494` convolution
MAE, versus `0.630` seasonal naive.
These observations do not establish real-world superiority.
The generator is regular and low-noise, and the test covers only a short period.
Minor numerical/version differences can alter final decimals.

## 7. Predict and test two changes

```bash
.venv/bin/python lessons/36-temporal-convolution/train.py --window 24
.venv/bin/python lessons/36-temporal-convolution/train.py --steps 20
```

With window 24, convolution lengths become `24 -> 20 -> 16`.
The dense head and parameter count shrink, but history also shrinks.
Predict that parameter count falls; do not assume validation error must improve.
Twenty steps changes optimization opportunity, not the data distribution.
Expect poorer fitting or a different validation-selected checkpoint.
Compare only validation during exploration; omit `--final-test`.

As a source-edit experiment, replace both kernels with width three.
Before running, derive new intermediate lengths, local receptive field, and
the corresponding head input size.
Changing only the kernels while leaving `8*(window-8)` in the head should fail:
shape arithmetic is part of the architecture design.

## Checkpoint questions

1. Why is `(B,48,1)` the wrong layout for this model?
2. Does a local receptive field of nine restrict the head to nine observations?
3. Why can ordinary valid convolutions be acceptable here but not for per-position
   streaming outputs?
4. Why is a convolution losing to an MLP a useful result?

<details>
<summary>Explained solutions</summary>

1. `Conv1d` interprets axis one as channels and axis two as time. That layout would
   claim 48 channels and only one time position, contradicting the declared layer.
2. No. The final dense head combines many local features spanning the whole past.
3. Every window position is observed before the forecast is issued. A streaming
   output at an earlier internal position would not yet know later positions.
4. Local sharing is an architectural hypothesis, not a law. Validation can reject
   it under this dataset, budget, and head design without any implementation bug.

</details>

## Limitations and next step

Fixed windows omit long-term state; valid convolutions discard edge positions;
the flattened head ties weights to specific lags.
No probabilistic uncertainty, irregular sampling, or external features are modeled.
The short single holdout is fragile.
[Session 37](../37-forecast-evaluation/README.md) evaluates fixed recipes over
several expanding training prefixes before opening one sealed final test.
