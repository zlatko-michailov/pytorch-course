# Session 29 — Make the neural network earn its complexity

[Course map](../../CURRICULUM.md) · Project 7: structured records

## Problem and prerequisites

A neural network is not the default winner on structured records.
Our **synthetic regression** target changes abruptly at two numerical thresholds
and also depends on a category.
We compare a meaningful affine model, two MLP widths, and small CART regression
trees on exactly the same transformed features and row partitions.
The point is a defensible comparison, not manufacturing a neural victory.

You already know MLP regression, one-hot categories, train-only scaling, and
deployment-aligned splitting from Sessions 26–28.
The new ideas are recursive feature splits, squared-error impurity, leaf means,
depth/minimum-leaf regularization, and selection using validation only.
This lesson implements its tree transparently in PyTorch and Python.
It does **not** install scikit-learn, claim to implement a random forest, or replace
a mature production tree library.

## Run, select, and only then open test

From the course root:

```bash
.venv/bin/python lessons/29-tree-baselines/train.py
```

This trains every predeclared candidate, prints training/validation results,
selects one finalist per family, then locks the overall validation winner.
Test metrics remain withheld.
After freezing your experiment:

```bash
.venv/bin/python lessons/29-tree-baselines/train.py --final-test
```

This deterministically repeats selection and reports the three already selected
family finalists on the same sealed test rows.
Those extra family test diagnostics are a predeclared comparison, not permission
to switch the selected model.
If test evidence prompts a redesign, obtain a new untouched evaluation set.

The complete self-contained program is [train.py](train.py).
It needs only Python and PyTorch; no downloads, previous artifacts, or extra
packages are required.
An optional warning about absent NumPy is unrelated to any API this script uses.
One CPU thread limits tiny-operation overhead, not a general performance claim.

## 1. The data-generating rule

There are 600 independent rows with two numerical fields in `[-2,2)` and category
`A`, `B`, or `C`.
The hidden rule is:

```text
target = (-3 if x0 <= 0 else +3)
         + (2 if x1 > 0.5 else 0)
         + category_effect + Gaussian noise
```

The category effects are `A:-1`, `B:0`, and `C:+1`.
Default noise standard deviation is `0.12`.
The target is observed after prediction; neither the target nor the rule's
threshold indicators are included as extra model inputs.

A seeded row permutation assigns 360 training, 120 validation, and 120 test rows.
Unlike Session 28, the generator has no repeated entities.
A row split is appropriate for this specific independent-record question.
Training rows alone determine numeric mean/scale and the categorical vocabulary.
Every model receives two scaled numeric columns and three one-hot columns.
Scaling is unnecessary for a tree's ordering-based splits but useful for the
gradient-trained competitors; using the identical representation keeps the
comparison easy to audit.

## 2. Why the affine baseline is meaningful

The affine model is `nn.Linear(5,1)`, not a deliberately broken model.
It can learn numeric trends, an intercept, and category offsets.
It cannot represent a discontinuous jump merely by changing its slope.
The training-target-mean predictor provides a still simpler reference.

The MLP adds a ReLU hidden layer of width 8 or 24.
Each gradient-trained candidate uses the same fixed 350 Adam updates.
All are optimized on training MSE and ranked on validation MAE.
The fixed budget is an experimental design choice, not proof that every optimizer
has converged or that computational budgets are equivalent.
A broader search might improve any family, but would consume more validation
information and should be planned before test evaluation.

## 3. Derive the CART split criterion

A regression-tree leaf predicts the mean of the training targets that reach it.
For targets `y1,...,yn`, that mean minimizes:

```text
SSE = sum((yi - leaf_value)^2)
```

At a node, try rules of the form `feature_j <= threshold`.
Choose the admissible rule with the smallest:

```text
left_SSE + right_SSE
```

For each feature, sort the node's training values.
Only midpoints between **different adjacent values** can change the partition.
Splitting between equal values is meaningless.
Both children must contain at least 12 training rows.
No validation or test row determines a threshold or a leaf mean.

The implementation computes candidate errors efficiently with prefix sums:

```text
SSE(group) = sum(y^2) - sum(y)^2 / count
```

After sorting once for a feature, cumulative sums provide every left and right
group's sufficient statistics.
`masked_fill(~valid, infinity)` removes forbidden candidate cuts.
`argmin()` picks the best remaining position.
Feature iteration and stable sorting give a deterministic tie convention.

## 4. Walk through the recursive implementation

`Node` is a Python `dataclass`: it declares named fields and generates a simple
constructor.
Each node stores its mean prediction and, optionally, a split and two children.
`feature is None` identifies a leaf.
The `int | None` annotation means an integer or the absence of one;
it is a type annotation, not a tensor operation.

`fit_tree(x,y,max_depth,min_leaf)` first constructs the leaf fallback.
It returns that leaf when depth is exhausted, fewer than two minimum-sized leaves
can be formed, or no candidate improves training SSE.
Otherwise it stores the best feature and threshold, partitions **training** rows,
and recursively fits left and right children with one less remaining depth.
Thus depth four means at most four split decisions along a root-to-leaf path.
It does not require every branch to reach that depth.

`predict_tree(node,x)` starts with the node mean for each incoming row.
At an internal node it applies the stored threshold and recurses into the
appropriate child.
Prediction never refits thresholds or looks at targets.
The script checks two focused behaviors: a constant input stays a leaf, and a
single exact step is representable by one split.

This is a genuine greedy depth-limited CART-style regression tree.
It is not merely a stump labeled “tree,” nor an ensemble labeled “forest.”
It omits cost-complexity pruning, missing-value routing, categorical subset
search, histogram acceleration, sample weights, serialization, and production
robustness.
Its educational objective is a transparent non-neural competitor.

## 5. Keep shape and information contracts identical

| Object | Contract |
|---|---|
| `numeric` | `(600,2)`, float32 raw measurements |
| transformed `x` | `(600,5)`, finite float32 |
| `y` | `(600,)`, float32 regression targets |
| network output | `(B,1)`, explicitly squeezed to `(B,)` |
| tree output | `(B,)`, leaf means |
| split indices | The same three tensors for every candidate |
| selection metric | Validation MAE; smaller is better |

The explicit squeeze prevents accidental `(B,1)` versus `(B,)` broadcasting.
The source computes predictions without gradients for reporting.
It may calculate held-out feature predictions, but no test target metric is
accessed until the `--final-test` branch.
The vocabulary/scaler stay fixed throughout inference.

## 6. Read a representative result

Approximate validation MAE on the default synthetic data:

```text
training-mean       3.036
affine              1.275
MLP width 8         0.495
MLP width 24        0.341
CART depth 2        0.740
CART depth 4        0.143
CART depth 6        0.146
```

Depth four wins by validation; depth six's lower training error is not a reason
to prefer it.
For a printed validation row with actual value about `1.96`, the affine model
predicts `-0.50`, the selected MLP `1.85`, and the selected tree `2.01`.
After selection is frozen, one reference final test reports roughly `1.350`,
`0.281`, and `0.119` for affine, selected MLP, and selected CART respectively.
Do not replace the validation decision with whichever looks best on test.
These are illustrative observations with small version-dependent differences.

The non-neural win matches the generator's axis-aligned step structure.
A smooth target, oblique boundary, extrapolation problem, larger dataset, or
different noise process could reverse the ranking.
A tree predicts piecewise constants and cannot extrapolate a trend outside its
observed leaves in the same way an affine model can.

## 7. Planned experiment

Predict how additional observation noise changes the gap before running:

```bash
.venv/bin/python lessons/29-tree-baselines/train.py --noise 0.8
```

Compare training and **validation** errors, especially depths four and six.
Deeper partitions can start fitting noise, so more depth is not guaranteed to help.
The command changes the problem's noise level, not just the optimizer.
Do not append `--final-test` while choosing noise, capacity, or learning settings.
As a source-edit exercise, replace the step rule with a smooth affine rule and
predict that the affine model becomes a stronger competitor.
Keep the same features, split indices, and metric for all families in that study.

## Checkpoint questions

1. Why do leaf predictions use means when ranking models by MAE?
2. Why reject a split producing one very small child?
3. Is depth chosen using held-out rows the same as fitting thresholds on them?
4. Why is a tree victory a successful outcome for this course?

<details>
<summary>Explained solutions</summary>

1. This implementation trains trees with squared error, whose optimal constant
   is the mean. MAE is the separately chosen evaluation metric. A tree trained
   with absolute-error impurity would need different fitting logic, often medians.
2. Tiny leaves can memorize noise and provide unstable estimates. Minimum leaf
   size regularizes the fitted training partition.
3. No. Validation selects among recipes fitted exclusively on training data.
   Thresholds and leaf means remain training-derived. Extensive repeated recipe
   search can still overfit validation and must be controlled.
4. The goal is to defend a useful model, not to prefer a neural architecture
   regardless of evidence. A simpler suitable model is a valid design decision.

</details>
