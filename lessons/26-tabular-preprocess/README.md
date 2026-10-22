# Session 26 — Missing values are part of the input contract

[Course map](../../CURRICULUM.md) · Project 7: structured records

## Problem and prerequisites

An image arrives as a regular array. A structured record may instead contain
measurements, missing fields, and category names.
Our **synthetic** task predicts repair duration, in minutes, from two measurements
and a machine type known **before** repair begins.
No result here measures real maintenance performance.

You already know affine regression, MSE, train/validation/test roles, train-only
scaling, and the ordinary PyTorch optimization loop.
This session adds imputation, missingness indicators, a categorical vocabulary,
and a defined behavior for categories absent from training.
We deliberately return to `nn.Linear`: changing the input representation does
not automatically justify a deeper model.

## Run the complete exercise

From the course root, with the existing environment:

```bash
.venv/bin/python lessons/26-tabular-preprocess/train.py
```

The complete, independently runnable source is [train.py](train.py).
It uses only Python and PyTorch, creates data in memory, and downloads nothing.
There are no input files, saved checkpoints, or prior scripts to run.
The standard-library `argparse` module reads the optional numeric flags below.
`main()` groups the executable steps; its `if __name__ == "__main__"` guard runs
them for both script execution and Python's `-m` module execution, but not on import.
`F` is just the imported alias for `torch.nn.functional`, used for one-hot encoding
and MSE.
If your shell already activates this environment, `python` can replace
`.venv/bin/python` in every command on this page.
The script sets one CPU thread to avoid thread-pool overhead for tiny matrices;
that is an execution choice, not a speed benchmark or a general tuning rule.
An installed PyTorch build may warn that NumPy is unavailable.
This exercise never calls NumPy and does not require installing it.

## 1. Decide what exists when a prediction is made

The two numerical columns are a pre-repair load measurement and machine age.
The category is an equipment type from an existing record.
The target is the duration revealed after the repair.
The generator knows the complete numerical values when it creates that target,
then hides some inputs from the learner.
That models incomplete measurement, not a model secretly receiving the originals.

Do **not** add final invoice amount, completed repair codes, or a post-repair
diagnostic result just because those columns correlate with duration.
They would be unavailable at the stated prediction time.
For real data, write an availability timestamp for each feature before modeling.

`make_data()` creates 480 independent synthetic records.
Rows `[0,300)` train the model, `[300,390)` validate it, and `[390,480)` test it.
Python slices exclude the right endpoint.
The first 300 rows contain three types.
Some later rows contain `prototype`, deliberately absent from the training set.
This is a controlled cold-start example, not a claim of identical distributions.

## 2. Learn preprocessing from training rows only

`fit_preprocessor(raw[train], names[train])` returns three objects:

1. An observed-value mean for each numerical column.
2. A scale for each column after filling training missing values.
3. A string-to-integer vocabulary built from training category names.

`torch.isnan()` produces a Boolean mask.
`torch.where(condition, a, b)` selects `a` where the condition is true and `b`
elsewhere; here it replaces missing values before arithmetic uses them.
The observed mean is `sum(observed values) / number observed`, not a mean of
zero-filled values.
If an entire training column is missing, the script raises `ValueError`.
Inventing a numeric mean in that case would conceal a data-design decision.

The standard deviation uses `correction=0`: divide squared deviations by the
training count, not by count minus one.
`clamp_min(1e-6)` protects constant columns from division by zero.
The scale describes the **imputed** training column, which is the actual input.
Neither validation nor test values alter these fitted statistics.
At deployment, reuse this state; never recompute it from one incoming batch.

## 3. Represent missingness instead of silently erasing it

For each numerical column we supply:

```text
scaled value = (imputed value - training mean) / training scale
missing flag = 1 if the original field was missing, otherwise 0
```

A missing measurement therefore becomes a zero scaled value **plus** a one flag.
An observed measurement equal to the mean becomes zero **plus** a zero flag.
Those records are distinguishable.
The flag lets a model learn a missingness association, when training supports one.
It does not reconstruct a lost measurement or establish a causal explanation.

Here missingness is sampled independently of the true values.
The flag may add little genuine signal.
In real systems, missingness can reflect a different workflow or a faulty sensor;
that association may change after deployment.

## 4. Categories are names, not magnitudes

The sorted training vocabulary is:

```text
large -> 1, medium -> 2, small -> 3; unknown -> 0
```

Feeding these integers directly to a linear regressor would impose an unjustified
ordered relationship between types.
`F.one_hot()` instead turns a known type into one active column.
We discard the reserved unknown column.
Thus an unknown name becomes `[0,0,0]`, while every known name activates its column.
`dictionary.get(name, 0)` performs this explicit fallback without an exception.
We do not add the new name to the trained vocabulary during inference.

This fallback is a representation contract, **not** learned knowledge about
prototypes.
The model never trains on unknown-category rows.
In particular, a bias plus all known one-hot columns has redundant parameters;
an all-zero category vector is not guaranteed to produce the average type effect.
Cold-start predictions can therefore be poor even when known-type predictions
are excellent.
A production design might use reference coding, constrain category effects,
pool rare types into a trained “other” bucket, or reject unsupported inputs.
Those choices require training evidence and validation, not a guessed UNK effect.

## 5. Trace every shape through the source

| Object | Shape / dtype | Meaning |
|---|---|---|
| `raw` | `(480,2)`, float32 | Numerical fields, possibly NaN |
| `names` | Python list of 480 strings | Equipment types |
| `target` | `(480,1)`, float32 | Duration in minutes |
| `mean`, `scale` | `(2,)`, float32 | Fitted training statistics |
| `ids` | `(480,)`, int64 | Vocabulary lookup results |
| `features` | `(480,7)`, float32 | 2 scaled + 2 flags + 3 category columns |
| `prediction` | `(480,1)`, float32 | One predicted duration per row |

`transform()` never fits anything: it consumes the already fitted state.
Its assertions check row/column alignment and finite transformed inputs.
A focused probe checks that an unseen name maps to zero, a missing numerical
value becomes zero after scaling, and its missing flag stays one.
The training-feature mean assertion checks the numerical preprocessing contract.

The remaining code creates `nn.Linear(7,1)` and runs 500 full-batch SGD updates.
MSE trains the model; MAE reports error in interpretable minutes.
The mean baseline always predicts the **training target mean**.
All model settings are fixed in this exercise before the test report is printed.
No selection or early stopping uses that report.

## 6. Read a run rather than admiring one number

One reference CPU run gives approximately:

```text
train      MAE=0.881 min; train-mean baseline=3.860
validation MAE=1.183 min; train-mean baseline=3.744
test       MAE=0.856 min; train-mean baseline=3.594
```

For known `medium`, row 301 predicts about `24.04` for an actual `24.01`.
For unknown `prototype`, row 313 predicts about `9.51` for an actual `13.87`.
The script also prints an example with a missing measurement.
Inspect these cases: an aggregate metric can conceal cold-start failures.
Small numerical differences across PyTorch versions are normal.
These values are observations from this generator, not guaranteed thresholds.

## 7. Experiments: predict first, then inspect validation

```bash
.venv/bin/python lessons/26-tabular-preprocess/train.py --missing-rate 0
.venv/bin/python lessons/26-tabular-preprocess/train.py --missing-rate 0.5
.venv/bin/python lessons/26-tabular-preprocess/train.py --steps 5
```

Before running, predict that removing missingness helps known-type rows, increasing
missingness removes useful information, and five updates may underfit.
The exact ranking is not guaranteed on a small finite validation set.
Changing missing rate changes the observed-input problem, not the hidden target
rule or train/validation/test boundaries.
Record validation MAE and individual predictions, not just training loss.
Use the automatically printed test only as a fixed-recipe demonstration;
do not select an experiment by repeatedly consulting it.

## Checkpoint questions

1. Why is filling with the mean of all 480 rows leakage?
2. Why keep a missing flag if a missing value already becomes zero?
3. Does UNK handling mean the model understands a new equipment type?
4. What must accompany model weights if this pipeline is saved?

<details>
<summary>Explained solutions</summary>

1. Held-out inputs would change the fitted training transformation. A deployed
   model cannot assume access to its future evaluation population.
2. Zero after scaling can also mean an observed value equal to the training mean.
   The flag preserves that distinction, although it may not improve accuracy.
3. No. It defines a valid numerical input. An unknown type has no directly learned
   effect, and this encoding can extrapolate badly.
4. Numerical column order, missing-value convention, means, scales, vocabulary,
   unknown behavior, and architecture must accompany the state dictionary.
   We do not write an incomplete inference artifact in this exercise.

</details>

## Limitations and the next decision

Mean imputation assumes a simple useful placeholder; it is not a probabilistic
missing-data model.
An unconstrained affine model can predict a negative duration outside its range.
The generator has no repeated entities, temporal drift, or real operational costs.
Keep this lesson's preprocessing boundary when changing the model.
Next, [Session 27](../27-embeddings/README.md) replaces a category vector with a
learned parameter lookup, without changing the requirement for train-only state.
