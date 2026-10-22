# Project 51 / Session 51 — Establish a complete baseline

## Prerequisites

Complete session 50 and review logistic logits, train-only scaling, and categorical encodings.
You should be able to name every inference field and explain why unit IDs are excluded.
You should also distinguish null measurements from omitted schema fields.

**Review:** binary linear models, imputation, one-hot categories, BCE.
**New integration:** an exact persisted-feature contract for the capstone.
This is a completed baseline, not an instruction to implement one before proceeding.
The synthetic-data and non-safety limitations from session 50 still apply.

## Attempt independently first

Choose a missing-value policy and write the precise feature order.
Decide what an unseen category should do.
Count the parameters in your proposed baseline.
Specify a trivial predictor that provides a score sanity check.
Then compare your design with the working reference below.

The exercise is optional preparation.
The provided script and helper already contain the full working solution.
You can run them immediately without creating any missing code or artifact.

## Run

From the course root:

```bash
.venv/bin/python -m lessons.51-capstone-baseline.train
```

Session source: [train.py](train.py).
Shared implementation:
[`course/capstone.py`](../../course/capstone.py).
Focus on `fit_preprocessor`, `transform`, `tensor_batch`, and `train_candidate`.
Generation and input validation were introduced in session 50.

No earlier checkpoint is required.
The script deterministically regenerates training and validation data.
It does not request the final-test partition.
It does not write a release or secretly train during an import.

## Start with a trivial baseline

Let r be the fraction of positive training labels.
A constant predictor always returns probability r.
Its logit is `log(r/(1-r))`.
PyTorch's `torch.logit` performs this inverse-sigmoid calculation.

The script builds constant validation logits from the **training** prevalence.
It reports validation BCE for that predictor.
The verified value was approximately 0.6511.
Validation prevalence is not used to fit the constant.

This is a sanity baseline, not one of the two release candidates.
If the learned baseline cannot improve on it, inspect the feature/label pipeline first.
Accuracy alone would obscure whether probabilities provide useful information.

## Fit preprocessing only on training

`fit_preprocessor(training_rows)` first validates every input record.
It then processes each numeric field in the schema's declared order.
Null values are excluded when fitting mean and standard deviation.
If a field has no observed training values, fitting raises a specific error.

The mean uses `math.fsum(observed) / len(observed)`.
`fsum` accumulates Python floats with improved summation accuracy.
The population variance uses the average squared distance from that mean.
The scale is its square root, lower-bounded at `1e-6`.

The reference training means are approximately:

```text
temperature_c: 69.2769
vibration_mm_s: 2.0338
load_pct:      53.4762
```

They are fitted data, not hard-coded normalization constants in the inference path.
The returned dictionary also stores scales, categories, version, and feature names.
It contains only ordinary Python types that can accompany tensor weights in a safe checkpoint.

Category strings are collected from training and sorted.
The first saved category is the reserved `<unknown>` slot.
The remaining fitted order is compressor, fan, pump.
Validation and inference never extend this list.

## Transform one input record

First validate the exact schema.
For each of the three numeric fields:

1. Record whether the value is null.
2. Replace null with that field's fitted training mean.
3. Subtract the same mean and divide by its fitted training scale.

A missing field's standardized value is therefore zero.
Its missingness indicator is one.
An observed value equal to the training mean has standardized value zero but indicator zero.
The model can distinguish these situations.

Then map machine type to its fitted category index, or to index zero if unknown.
Construct one one-hot feature per saved category.
An unknown category is a real input with an explicit fallback slot.
It is not silently treated as a known machine type.

## The exact feature order

The model receives ten float32 columns:

```text
0 temperature_c:z
1 vibration_mm_s:z
2 load_pct:z
3 temperature_c:missing
4 vibration_mm_s:missing
5 load_pct:missing
6 machine_type:<unknown>
7 machine_type:compressor
8 machine_type:fan
9 machine_type:pump
```

The order is stored as `feature_names`.
The transform checks its output width and finite values.
Training features have shape `[720,10]`.
Training labels have shape `[720]`.
Validation has the same feature width with 240 rows.

`tensor_batch` separates records from targets.
It never includes `unit_id` in the feature vector.
That ID is useful for split design but would otherwise invite memorization.

## The completed logistic baseline

The model is `nn.Linear(10,1)`.
It has ten weights and one bias: 11 trainable parameters.
Its output is an unbounded binary logit.
The training loss uses `.squeeze(-1)` to align `[B]` logits with `[B]` floating labels.

The unknown one-hot column never appears during training.
Its first-layer coefficients are explicitly initialized to zero.
Their contribution therefore remains neutral rather than arbitrary at first unseen-category use.
That is a fallback policy, not learned evidence about a prototype machine.

`train_candidate` resets the model seed to 5054.
It uses Adam lr 0.025, weight decay 0.001, and 160 full-batch updates.
Every update zeroes gradients, computes BCE, backpropagates, checks finite gradients, and steps.
There is no test-driven stopping condition.

Weight decay regularizes coefficients during optimization.
The reported BCE excludes that regularization contribution.
Do not confuse the optimization penalty with the predictive loss used for comparison.

## Evaluate the reference result

`evaluate` enters evaluation mode and disables gradient recording.
It returns BCE, accuracy, positive recall, balanced accuracy, support, and confusion counts.
The fixed decision boundary is probability 0.5, equivalent to logit zero.
The threshold is not tuned on validation here.

In the verified CPU run:

```text
logistic train BCE:       0.4792
logistic validation BCE:  0.5813
validation accuracy:     0.7333
validation recall:       0.4706
validation balanced acc: 0.6740
```

This is development evidence, not a final-score announcement.
Recall is noticeably lower than accuracy might suggest.
The small dataset and correlated readings limit how precisely we can interpret those ratios.
Do not change the final threshold because you prefer a test-set recall later.

## A preprocessing-state check

The script takes a deep copy of the fitted preprocessing dictionary.
`copy.deepcopy` also copies nested lists, unlike a shallow top-level dictionary copy.
It changes one validation input and transforms it.
The fitted preprocessing must remain identical to the saved copy.

This is a focused data-flow contract.
New requests can change feature values but cannot refit means, scales, or vocabulary.
The final checkpoint will store this exact mapping in session 53.

## Controlled experiments

### A. Remove the missingness indicators

Predict: null and observed-at-the-mean values become indistinguishable.
Update feature names and input width together in a development-only branch.
Whether that distinction helps must be judged on validation.

### B. Fit preprocessing on validation as well

Predict: saved means/scales change even though training rows are unchanged.
That violates the contract and is not a legitimate model improvement.
Use session 49's counterfactual reasoning to diagnose it.

### C. Introduce a new machine type at inference

Predict: it maps to the unknown column without changing feature width.
Its unknown-category coefficient is neutral, but numeric and missingness effects still contribute.
Do not interpret this fallback as evidence of domain generalization.

## Checkpoint questions

<details>
<summary>Why can imputed values be standardized to zero without hiding missingness?</summary>

Zero is the fitted numeric mean after standardization.
The separate missing indicator tells the network whether zero came from imputation or observation.
That policy and feature order must remain fixed at inference.
</details>

<details>
<summary>Why is the train-prevalence constant not fitted from validation labels?</summary>

It is a fitted baseline, so its parameter belongs to training.
Using validation labels to set its probability would make the comparison protocol inconsistent.
Validation scores compare frozen training procedures.
</details>

<details>
<summary>Does lower training BCE justify replacing the baseline?</summary>

No. A larger candidate can fit training noise more easily.
The next session motivates a nonlinear candidate, but session 53 selects using validation only.
The baseline remains eligible to win.
</details>

## Design judgment

An 11-parameter model is a serious reference point on a small tabular task.
Its biggest virtues are inspectable data flow, low variance, and cheap inference.
The next candidate must earn its extra complexity with development evidence.
