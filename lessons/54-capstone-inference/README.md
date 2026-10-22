# Project 54 / Session 54 — Predict new inputs from the frozen release

## Prerequisites

Complete sessions 50–53.
Run session 53 once so its release checkpoint exists.
Review schema validation, unknown categories, missing-value preprocessing, and evaluation mode.
You should be able to explain why inference must not refit a scaler or vocabulary.

**Review:** safe state-dictionary loading, no-gradient prediction, sigmoid probabilities.
**New integration:** an independent JSON/Python prediction interface with explicit rejection paths.
The completed interface is available now.
It is not a placeholder API that asks you to supply model-loading code.

## Attempt independently first

Sketch a function taking raw records and returning predictions.
List the invalid inputs it must reject.
Specify what happens if the release file is missing.
Decide how to expose unknown categories and missing numeric measurements to the caller.
Then compare your design with the full implementation.

The correct missing-checkpoint behavior is **not** to train secretly.
It is also not to return a constant, cached example, or fabricated success-shaped probability.
Inference has a narrower responsibility than development.

## Run the default demonstration

From the course root:

```bash
.venv/bin/python -m lessons.54-capstone-inference.train
```

If necessary, first run:

```bash
.venv/bin/python -m lessons.53-capstone-selection.train
```

Session source: [train.py](train.py).
Prediction and loading source:
[`course/capstone.py`](../../course/capstone.py).
Focus on `predict_records`, `load_checkpoint`, `transform`, and `validate_record`.
They are the same completed helper used in the development sessions.

Missing `release.pt` produces a nonzero exit with the exact prerequisite command.
No data is generated, no optimizer is constructed, and no final test is evaluated.
The numeric course runner can run 53 then 54 without special orchestration.
All defaults use CPU PyTorch and standard-library Python only.

## Supply a genuinely new input

The CLI accepts one JSON object:

```bash
.venv/bin/python -m lessons.54-capstone-inference.train --input-json '{"temperature_c":82,"vibration_mm_s":2.6,"load_pct":75,"machine_type":"pump"}'
```

It also accepts a JSON array of 1 through 128 objects:

```bash
.venv/bin/python -m lessons.54-capstone-inference.train --input-json '[{"temperature_c":null,"vibration_mm_s":1.4,"load_pct":45,"machine_type":"prototype"},{"temperature_c":92,"vibration_mm_s":3.7,"load_pct":88,"machine_type":"compressor"}]'
```

These are new inference records, not a lookup into the generated training fixture.
The interface validates and transforms their actual supplied values.
JSON `null` becomes Python `None`.
JSON numeric values become Python integers or floats.
Malformed JSON is explicitly rejected by the standard decoder rather than converted into a fallback.

### The small Python CLI wrapper

`argparse.ArgumentParser` is the standard-library command-line parser.
`add_argument("--input-json", ...)` declares an optional named string argument.
`parse_args()` returns an object whose `input_json` attribute is that string or None.
The wrapper distinguishes absence from an explicitly supplied empty string.
`json.loads` parses supplied text; it does not evaluate Python source code.
`isinstance(inputs, dict)` normalizes a single object into a one-element list.
Other containers still pass through the prediction function's list/length validation.
The final `json.dumps(..., indent=2)` prints ordinary result dictionaries as readable JSON.

## Python entry point

The same interface can be called from code started at the course root:

```python
from course.capstone import RELEASE_PATH, predict_records

records = [{
    "temperature_c": 82.0,
    "vibration_mm_s": 2.6,
    "load_pct": 75.0,
    "machine_type": "pump",
}]
predictions = predict_records(RELEASE_PATH, records)
print(predictions)
```

The function expects a nonempty list, even for a single record.
It loads the explicit path and returns a list in input order.
There is no service process, network socket, or hidden background job.
For a high-throughput service, loading once and reusing the model would be a sensible later refactor.
That optimization must preserve the same validation and preprocessing semantics.

## Prediction path, operation by operation

1. Validate the request container and batch length.
2. Load the trusted local checkpoint with `weights_only=True` on CPU.
3. Check its format, schema, preprocessing, feature order, candidate, and finite weights.
4. Reconstruct the known model and load its state dictionary strictly.
5. Enter evaluation mode.
6. Validate every raw record.
7. Impute/scale numeric fields using saved training statistics.
8. Build saved-order missing indicators and category one-hot columns.
9. Predict logits with gradient recording disabled.
10. Apply sigmoid and compare probabilities with the saved 0.5 threshold.

Nothing in this path calls `make_split`, `fit_preprocessor`, or `train_candidate`.
There is no opportunity for a request to become extra training data.
Saving a fitted mapping is what makes new-request inference reproducible.

The transformed tensor has shape `[B,10]`.
The chosen logistic release outputs `[B,1]`.
`.squeeze(-1)` preserves `[B]` even for one record.
Sigmoid maps each logit to a number between zero and one.
`.tolist()` converts the CPU tensor into JSON-serializable Python values.

## Output contract

Each result contains:

```text
probability:              model probability for the synthetic review label
decision:                 "review" at probability >= 0.5, otherwise "routine"
unknown_machine_type:     whether the type was outside the fitted vocabulary
missing_numeric_fields:   number of explicitly null numeric measurements
```

The two flags are important context.
An unknown type uses the saved unknown column without changing feature width.
Its initially neutral coefficient does not certify transfer to a new real machine type.
Missing measurements are mean-imputed with separate indicators; no real values are recovered.

The default demonstration has a low-risk-looking fan, a high-risk-looking pump,
and an input with null temperature and the unknown type `prototype`.
These are illustrative new records, not domain-approved operating cases.
Do not treat `"routine"` as permission to ignore an actual equipment concern.
This remains a synthetic course exercise, not a safety decision system.

## Explicit rejection behavior

The schema requires exactly the four declared input keys.
Omitted keys and unexpected extra keys raise `ValueError`.
Null numeric values are allowed, but numeric strings and booleans raise `TypeError`.
Python bool is an int subclass, so the validator rejects it before the numeric type check.

NaN, positive/negative infinity, and values outside declared ranges raise `ValueError`.
An empty, whitespace-padded, overlong, or reserved category spelling is rejected.
Unknown but well-formed category strings are accepted and flagged.
An empty or oversized batch is rejected.

The default command exercises invalid numeric, boolean, range, extra-field,
missing-field, empty-category, and empty-batch examples.
Each demonstration catches only the expected validation exception classes.
An unexpected loading or tensor error is not hidden by those handlers.
Invalid inputs do not get a probability that looks successful.

## Determinism and focused tests

The default demonstration loads and predicts the same records twice.
The ordinary Python result lists must compare equal.
Session 53 already checked exact live-model versus saved-model validation logits.
Together these cover persistence fidelity and repeat prediction behavior in this CPU environment.

Run the focused standard-library tests after session 53:

```bash
.venv/bin/python -m unittest lessons.54-capstone-inference.test_contracts
```

The tests cover accepted missing/unknown inputs, invalid request rejection,
safe-load schema mismatch rejection, explicit missing-release behavior,
and the no-retraining/no-test path of a cached selection rerun.
There is no pytest dependency.
The tests never generate or rescore the final test.

The test class derives from `unittest.TestCase`.
Methods named `test_...` are discovered and run by the shown command.
`setUpClass` checks the shared release prerequisite once.
`setUp` supplies a fresh valid record before each test.
`assertRaises` checks an explicit rejection; `subTest` identifies the particular invalid case.
`mock.patch` temporarily replaces a function or value only inside its `with` block.
For example, patched training/evaluation functions raise if a cached selection calls them.
That tests forbidden side effects without creating another dataset or modifying the release.
The missing-release test substitutes a nonexistent path without writing or deleting files.

## Controlled experiments

### A. Change only machine type to `prototype`

Predict: feature width stays ten and `unknown_machine_type` becomes true.
The probability can change because known-category coefficients are no longer active.
Do not silently append a new one-hot column.

### B. Compare null temperature with the saved numeric mean

Predict: both produce standardized temperature zero.
Only the null case sets the missingness indicator.
They can therefore have different predictions without a schema inconsistency.

### C. Submit the string `"82"` for temperature

Predict: it is rejected instead of coerced.
Silent coercion can hide an upstream schema/version error.
Convert deliberately in a separate documented ingestion layer if your application requires it.

### D. Run session 53 again

Predict: the existing release is reported as cached.
No retraining or final-test scoring occurs.
Inference remains tied to the same frozen artifact.

## Checkpoint questions

<details>
<summary>Why does inference load saved preprocessing instead of calling fit_preprocessor?</summary>

The weights were trained against a particular feature mapping.
Refitting on a request batch changes that mapping and can make predictions depend on neighboring requests.
Inference transforms inputs; it does not learn from them.
</details>

<details>
<summary>Does probability 0.9 mean this real machine has a 90% chance of failure?</summary>

No. The target is a synthetic review event and the model is not a real-machine estimator.
Even within the synthetic distribution, probability calibration needs its own evaluation.
The interface must not overstate what the dataset supports.
</details>

<details>
<summary>Why not auto-run session 53 when the file is missing?</summary>

That would merge training, model selection, final evaluation, and inference into an opaque side effect.
A missing release is an explicit prerequisite failure.
The user or runner chooses when the training/release workflow is allowed to run.
</details>

## Completed capstone judgment

The reference chose the baseline, met its synthetic educational goal, and preserved a reproducible API.
It did not establish real-world reliability, calibration, fairness, or equipment safety.
The completed outcome is an auditable small pipeline with honest limits.
That is more valuable than forcing the most complex model to be the winner.
