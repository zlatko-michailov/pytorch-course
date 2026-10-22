# Project 49 / Session 49 — A good-looking score can still be wrong

## Prerequisites

Complete sessions 47–48.
Review train-only scaling, group splits, binary confusion counts, and overfitting.
You should distinguish training success from evidence about unseen inputs.

**Review:** fitted preprocessing, class imbalance, parameter memorization.
**New practice:** counterfactual leakage checks, explicit metric repairs, disjoint-ID diagnostics.
Every diagnostic includes runnable buggy behavior and a completed repaired interpretation.
The default script does not leave an unfinished evaluation pipeline.

## The evaluation problem

Three different mistakes can all produce reassuring numbers:

1. Let held-out inputs influence fitted preprocessing.
2. Report only accuracy on a heavily imbalanced task.
3. Evaluate a model on identities it has memorized.

None is repaired by a larger network.
The right response starts by identifying which claim the experiment can actually support.

## Run

From the course root:

```bash
.venv/bin/python -m lessons.49-debug-evaluation.train
```

Complete source: [train.py](train.py).
All data is tiny, synthetic, and created in memory.
The default program finishes with explicit assertions.
No external dataset, model download, NumPy, or scikit-learn is used.

## Diagnostic 1: held-out values changing fitted state

Training values are `[0,1,2]`.
Validation values are `[100,101]`.
The training mean is 1.
The mean of the combined arrays is 40.8.

The buggy demonstration fits that combined mean.
It then shifts validation by another 1000 and fits the combined mean again.
The fitted value changes to 440.8 even though the training data is unchanged.
That counterfactual exposes the dependency on held-out inputs.

The repair fits mean and population standard deviation from training only.
`std(unbiased=False)` uses division by N, appropriate for this explicit fitted transform.
The `clamp_min(1e-6)` protects a constant training feature from zero division.
Both training and validation use those same fitted numbers.

Training z-scores have mean zero.
Validation z-scores are very large, approximately 121 and 122.
This is not a preprocessing failure by itself.
It is a visible distribution shift that all-data fitting would partly conceal.

Do not “fix” shifted validation by refitting the scaler on it.
Investigate the data collection, intended deployment distribution, and feature units.
A new preprocessing design requires a new development/evaluation protocol.
The function's fitted state must remain a training-data artifact.

## Diagnostic 2: 95% accuracy with zero positive recall

The target contains 95 negatives and 5 positives.
An all-negative predictor gets 95 examples correct.
Its accuracy is 0.95.
It detects none of the five positive examples.

`binary_metrics` uses boolean tensors:

```text
TP = predicted positive AND actually positive
FN = predicted negative AND actually positive
TN = predicted negative AND actually negative
FP = predicted positive AND actually negative
```

The operators `&` and `~` are elementwise boolean conjunction and inversion.
Summing counts True entries.
`.item()` converts each scalar count to an ordinary Python number.

Positive recall is `TP/(TP+FN)`.
Specificity is `TN/(TN+FP)`.
Balanced accuracy is their average.
Here recall is 0, specificity is 1, and balanced accuracy is 0.5.

If one actual class is absent, its denominator is zero.
The helper returns `None` for the undefined quantity instead of inventing a reassuring zero or one.
A real report should state why that metric is unavailable.
Always include support counts so readers can interpret ratios.

The repair is a more honest report, not a magic new classifier.
Choosing a threshold or class weighting would require validation and a declared cost model.
Do not choose those after inspecting a final test.

## Diagnostic 3: perfect memorization with no transferable signal

There are 40 distinct synthetic IDs.
`torch.eye(40)` encodes each ID as its own one-hot vector.
The first 20 IDs form training.
The other 20 form diagnostic validation.

Each partition receives a separately shuffled balanced set of random labels.
`torch.Generator().manual_seed(49)` controls only that fixture shuffle.
The labels have no relationship to the ID or any meaningful predictor.
They are intentionally impossible to infer for a previously unseen ID.

The model is `nn.Linear(40,1,bias=False)`.
All weights start at zero.
Each training ID activates one independent coefficient.
Adam can push that coefficient toward its arbitrary training label.
After 100 updates, training accuracy is 100%.

Validation IDs activate different, never-updated coefficients.
Those weights remain zero.
The logit-zero decision predicts the same class for every validation example.
Because validation is balanced, accuracy and balanced accuracy are 50%.
This result is checked explicitly.

The validation data is a diagnostic partition, not a repeatedly tuned deployment test.
The example proves a failure of the available signal, not a universal numerical bound for every finite sample.
The balanced construction makes the no-transfer result exact here.

## What is the completed repair?

First, use disjoint identities when deployment concerns new identities.
Second, report held-out performance instead of substituting training accuracy.
Third, remove an identifier that serves only as a lookup key and seek meaningful measurements.

The script completes the first two repairs and demonstrates why the third cannot be faked.
There are no meaningful predictors in this random-label fixture.
No honest repaired neural architecture can manufacture that missing information.
Saying so is part of a complete engineering answer, not a placeholder.

## Controlled experiments

### A. Use all IDs for training and evaluation

Predict: apparent performance becomes excellent.
Interpretation: the model can answer previously memorized ID queries.
That says nothing about deployment on new IDs.

### B. Change prevalence from 95/5 to 99/1

Predict: all-negative accuracy rises while positive recall remains zero.
Balanced accuracy remains 0.5 when both classes are present.
More impressive accuracy can represent the same useless detection behavior.

### C. Shift only validation further

Predict: the train-only fitted mean and scale remain unchanged.
Only transformed validation values change.
A counterfactual that alters fitted state reveals a data-flow leak.

### D. Increase the ID model's depth

Predict: training memorization is still easy.
There is still no relationship that supports new random-label IDs.
Additional capacity cannot solve missing information.

## Checkpoint questions

<details>
<summary>Is using validation repeatedly for model selection also a kind of overfitting?</summary>

Yes. A large adaptive search can overfit validation even without training on it directly.
Limit and record development choices, then use an untouched final test for the frozen choice.
The capstone makes that boundary explicit.
</details>

<details>
<summary>Should a 95% accurate all-negative model always be rejected?</summary>

Judge against the actual use and costs, not one universal threshold.
If detecting positives is the purpose, zero positive recall is a decisive failure.
The problem is hiding that fact behind a single aggregate number.
</details>

<details>
<summary>Why not repair the random-label task by including labels as features?</summary>

That would leak the answer and violate the inference contract.
Only features available when making the real prediction are admissible.
The absence of valid signal should be reported rather than bypassed.
</details>

## Design judgment

Evaluate the intended deployment claim, not the easiest score to improve.
The next five sessions assemble a completed capstone with these boundaries preserved.
