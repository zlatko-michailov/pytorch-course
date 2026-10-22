# Project 5, Session 20: understand the errors behind an accuracy

[Course curriculum](../../CURRICULUM.md) · [Complete runnable train.py](train.py)

## Prerequisites and genuinely new material

Review Session 19's ten logits, integer labels, and validation-selected MLP.
Review Session 13's binary confusion counts and the effect of class imbalance.
The data reader and split are unchanged from Session 18.

**New:** a ten-by-ten confusion matrix, class support, multiclass recall,
validation-index alignment, and concrete inspection of mistakes.
No new architecture or training objective is introduced.

## The problem

An overall accuracy can hide a model that mostly confuses Shirts with T-shirts,
or almost never recognizes Coats. Those failure modes suggest different actions.
We need counts whose meaning we can verify, not just a colorful plot.
Our console-only analysis needs no NumPy, plotting package, or scikit-learn.

A confusion matrix `M` uses:

```text
M[actual_class, predicted_class] += 1
```

Rows are actual classes; columns are predictions. Other libraries sometimes use
the transpose, so never omit the axis convention.
The diagonal counts correct predictions.
A row sum is the number of examples with that actual class, called its support.
Recall for class `c` is `M[c,c] / sum(M[c,:])`.

## Run from the course root

For an independent tensor/loop check:

```sh
.venv/bin/python -m lessons.20-confusion-errors.train --smoke --epochs 1
```

Smoke explicitly trains a fresh small MLP on synthetic stripes and analyzes its
synthetic validation mistakes. It does not require an earlier artifact and does
not establish a clothing error pattern.

For real error analysis, first run Session 19 with your fixed configuration:

```sh
.venv/bin/python -m lessons.19-multiclass-training.train --data data/fashion --epochs 3
.venv/bin/python -m lessons.20-confusion-errors.train --data data/fashion
```

The real analysis loads `.artifacts/vision/fashion-mlp.pt`; it does not silently
train a replacement if the checkpoint is missing.
Use the same `--max-train` and `--max-val` limits if you limited Session 19.
The validation prefix is stable, but analyzing a different prefix is a different
reported evaluation. The saved checkpoint records the original limits.

Only official **training** IDX files are opened to reconstruct training/validation.
Official test files remain unopened. Offline data and provenance instructions are
in [Session 18](../18-images-flat/README.md).

## Walk through the entire analysis

The full entry point is [train.py](train.py), with explicit evaluation in
[image_training.py](../../course/image_training.py), explained in Session 19.
The new `confusion(actual,predicted)` function has one expression:

```python
torch.bincount(10 * actual + predicted, minlength=100).reshape(10, 10)
```

1. `actual` and `predicted` are equally sized `[V]` integer tensors.
2. Encode a pair `(a,p)` as `10*a+p`, an integer between 0 and 99.
3. `torch.bincount` counts each encoded value, without a Python loop over examples.
4. `minlength=100` preserves zero-count cells, even if class 9 was never predicted.
5. `reshape(10,10)` turns the count vector into rows of ten possible predictions.
   A pair `(2,7)` lands at flat index 27 and matrix row 2, column 7.

`partitions(args)` reconstructs exactly the same deterministic split.
In smoke mode, `mlp()` and `fit(...,1)` produce a self-contained fixture model.
In real mode, `load_model(...)` checks the checkpoint contract and restores it.
The architecture tag must also be `mlp-784-64-10`.

`evaluate(model,loader(val))` returns predicted and actual classes in dataset order.
The loader does not shuffle. That fact becomes important when we inspect images.
`matrix.sum()` must equal the number of validation examples.
`matrix.trace()` sums the diagonal and must equal the explicit correct count.
These assertions detect orientation/count bugs that a plausible accuracy can hide.

`for index, name in enumerate(CLASSES):` pairs every class index with its name.
`matrix[index].sum().item()` obtains its support as a Python number.
The conditional expression computes recall only if support is nonzero.
Otherwise it prints `None`, meaning undefined, not a perfect or zero recall.
Tiny smoke subsets need not contain every class in validation.

`result["actual"] != result["predicted"]` produces a boolean mistake mask.
`.nonzero()` returns positions; `.flatten()[:5]` keeps the first five.
For each position, `.tolist()` makes ordinary Python values for a console loop.
Indexing `val[index]` is valid because the evaluation loader preserved order.
If you shuffled evaluation, those positions would no longer identify these images.

`val[index][0]` selects the image from `(image,label)`.
Another `[0]` selects its single channel, yielding `[28,28]`.
`[::4,::4]` selects every fourth row and column: a crude `[7,7]` thumbnail.
The nested string joins print `#` for normalized pixels above zero and `.` otherwise.
`"\n".join(...)` places rows on separate lines.
This is an intentionally lossy diagnostic display, not input preprocessing.
For difficult cases, inspect the original 28-by-28 image in an approved viewer.

## Interpret the output carefully

Record the largest off-diagonal cells, not just the largest rows.
A class with many examples can have more errors but better recall than a rare class.
Compute row-normalized proportions when comparing class-specific error rates.
If needed, precision for class `c` uses `M[c,c] / sum(M[:,c])`, a column sum.
Its denominator asks how often predictions of `c` were correct.
Do not interchange precision with recall.

A Shirt predicted as T-shirt/top may be visually ambiguous even to a person.
Label definitions, resolution, and grayscale conversion can impose irreducible
confusion. Inspect evidence before prescribing a bigger model.

## Controlled edits

1. Swap `actual` and `predicted` in `confusion`.
   Predict the matrix transposes; its sum and diagonal do not change.
   The printed “recall” would become precision if row labels were not corrected.
2. Use `--max-val 10` in smoke mode.
   Predict some zero-support classes; `None` prevents division by zero.
3. Replace `minlength=100` with no minimum.
   Predict reshape can fail if the highest encoded pair never appears.
4. Shuffle the validation loader, then inspect indices as before.
   Predict image/name misalignment even if aggregate matrix counts remain correct.
   Restore non-shuffled evaluation.

## Checkpoint questions

**Can a diagonal total prove the matrix orientation is correct?**
<details>
<summary>Solution</summary>
No. Transposing preserves the diagonal and total. The encoding formula and known
small examples establish the axis convention; the assertions check other invariants.
</details>

**If we inspect these errors and change the model, is validation now test data?**
<details>
<summary>Solution</summary>
No. We are using validation for its intended role: development decisions.
Repeated decisions can overfit it, so the separate test must remain untouched until
we fix the configuration. Renaming validation “test” would not remove selection bias.
</details>

**Why print undefined recall rather than zero for an absent class?**
<details>
<summary>Solution</summary>
With no actual examples we learned nothing about that class's recall.
Zero would assert measured failure; one would assert measured perfection.
Neither follows from a zero denominator.
</details>

## Design judgment and limitations

These counts describe this split and this label taxonomy, not arbitrary deployment.
Small supports make rates unstable. An ASCII thumbnail may hide important details.
Use error analysis to form a specific hypothesis; change one factor and return to
validation. Session 21 packages the chosen MLP before a deliberately final test.
