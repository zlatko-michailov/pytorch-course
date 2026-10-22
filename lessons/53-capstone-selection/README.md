# Project 53 / Session 53 — Freeze, evaluate once, and persist the release

## Prerequisites

Complete sessions 50–52.
Review validation selection, state dictionaries, `torch.save`, and `weights_only=True`.
You should be able to explain why the smaller baseline can legitimately win.
You should also know the exact preprocessing feature order from session 51.

**Review:** train/validation/test separation, deterministic fits, checkpoint reconstruction.
**New integration:** a persisted selection receipt, one final evaluation, auditable release metadata.
The default command completes the full release workflow.
It does not ask you to fill in a selection function.

## Attempt independently first

Write down the order of operations before opening the source.
At what exact point is the candidate choice frozen?
What prevents a second invocation from silently scoring the final test again?
What must be saved besides tensors?
How would you prove a loaded model behaves exactly like the saved model on this CPU?

Then compare with the completed implementation.
You can run it immediately; sessions 51 and 52 need not have left intermediate files.
Their fixed procedures are reconstructed deterministically here.

## Run from the course root

```bash
.venv/bin/python -m lessons.53-capstone-selection.train
```

Session source: [train.py](train.py).
Shared persistence and evaluation source:
[`course/capstone.py`](../../course/capstone.py).
Focus on `checkpoint`, `load_checkpoint`, and the order of operations in `main`.

The first successful run writes:

```text
lessons/53-capstone-selection/artifacts/selection.pt
lessons/53-capstone-selection/artifacts/release.pt
lessons/53-capstone-selection/artifacts/experiment.json
```

Paths derive from the helper's `__file__`, resolved relative to the course root.
They do not depend on the shell's accidental current output directory.
The directory is created if needed, with no destructive cleanup of existing artifacts.
Session 54 explicitly consumes `release.pt`.

## First-run order of operations

1. Check whether a completed release already exists.
2. Refuse an ambiguous interrupted state with a receipt but no release.
3. Generate training and validation, and fit training-only preprocessing.
4. Train exactly the two candidates declared in session 50.
5. Select the smallest validation BCE.
6. Persist the selected weights and decision metadata to `selection.pt`.
7. Load that receipt safely and verify exact validation logits.
8. Only now construct the final-test partition.
9. Evaluate only the selected model on that test.
10. Save final metrics in `release.pt` and the readable experiment record.

The code ordering is intentional.
The test tensor is not passed into training or into the candidate-selection function.
There is no loop that chooses a model or threshold based on final-test metrics.
The same fixed threshold 0.5 is used throughout.

## Selection logic

The script stores one record and one trained model for each candidate.
`min(range(len(experiments)), key=...)` returns the index of the smallest validation BCE.
Python's first-minimum behavior breaks an exact tie in declared candidate order.
This is stated in the checkpoint's selection rule.

The verified validation BCEs were approximately:

```text
logistic: 0.5812607
mlp16:    0.6032354
```

Therefore the release selects **logistic**, not the neural candidate.
The larger model's better training fit does not enter the choice.
The final test does not break a close validation tie.

The candidate pool is deliberately small and declared before evaluation.
A large adaptive validation search could overfit that partition too.
This reference demonstrates a bounded protocol, not immunity to all selection bias.

## What the checkpoint contains

The payload is a dictionary of tensors and simple Python values:

- Format version and input schema.
- Training-only means, scales, category vocabulary, and feature names.
- Candidate architecture configuration.
- The selected model's state dictionary.
- Threshold, training seed, update count, and learning rate.
- Selection rule and development experiment records.
- In the final release only, the single final-test result.

No live `nn.Module` object is pickled as the application artifact.
No lambda, custom dataset instance, or executable preprocessing object is required.
Inference reconstructs a known architecture and loads its weights strictly.

State tensors are detached, moved to CPU, and cloned.
That creates an independent snapshot without gradient history.
Preprocessing stays paired with the weights that were fitted against its exact column meanings.
Saving weights alone would not preserve the raw-input prediction function.

## Safe reconstruction walkthrough

`torch.load(path, map_location="cpu", weights_only=True)` uses the restricted loading path.
The format and schema must match the reference.
The saved threshold must match the fixed policy.
Means must be finite, and scales must be finite and positive.
The saved vocabulary must have unique strings and the correct unknown sentinel.

The feature-name sequence is reconstructed from the schema and vocabulary and compared exactly.
The candidate must be one of the two known configurations.
`make_model` builds that architecture.
`load_state_dict(..., strict=True)` rejects missing or unexpected parameter keys.
All loaded parameters must be finite.
Finally, the model enters evaluation mode.

`weights_only=True` reduces arbitrary-object deserialization risk.
It is not a claim that any untrusted file is safe against corruption or resource exhaustion.
Use this locally produced trusted artifact and do not disable the restricted loader to bypass errors.
Schema validation protects compatibility; it is not a substitute for artifact provenance.

## Exact roundtrip check

Before scoring test, the live selected model predicts validation logits.
The safely reconstructed receipt model predicts the same tensor.
`torch.equal` requires bit-for-bit equality in this same CPU environment.
The preprocessing dictionary must also compare equal.

This is stronger than merely getting the same class labels.
Small probability changes can matter even when thresholded decisions match.
Cross-device, cross-version, or different-kernel bitwise reproducibility is not promised.
The recorded seed and configuration support repetition within the stated environment.

## One final result, reported honestly

The verified selected logistic model produced:

```text
test BCE:               0.4138303
test accuracy:          0.8250
test positive recall:   0.6338
test balanced accuracy: 0.7696
test positives:         71 of 240
confusion:              TP=45, FN=26, TN=153, FP=16
```

It met the predeclared educational BCE/balanced-accuracy goal.
This does not turn the synthetic problem into a real maintenance validation.
The test happened to be easier than validation for this fixed generated sample.
That is an observation, not a reason to change the split or selection rule.

Six readings per unit are correlated.
Two hundred forty readings are not 240 independent units of evidence.
A serious real deployment would need representative data, group-aware uncertainty,
calibration analysis, and a domain-specific risk review.
This exercise does not provide those.

## Repeated and interrupted runs

If `release.pt` exists, the script loads it and prints **CACHED RELEASE**.
It does not retrain, reselect, or construct test data.
Showing the saved result again is not a new evaluation.

If only `selection.pt` exists, the script raises an explicit interrupted-run error.
It cannot know whether a prior process already observed final metrics before failing to save.
It refuses to silently spend the same evaluation again.
Investigate the interruption rather than deleting receipts to obtain a more appealing result.

The small reference is not a transactional artifact registry or a concurrency-safe release service.
Do not run simultaneous first releases into the same artifact folder.
For a genuinely new experiment, define a new evaluation plan and artifact identity.
Do not repurpose this already observed test as an untouched final test.

## Controlled experiments

### A. Rerun the unchanged command

Predict: it prints cached metadata without training or another final-test pass.
This is the safe repeatability path used by the integration runner after a release exists.

### B. Compare validation rank with training rank

Predict: the MLP ranks better on training BCE and worse on validation BCE.
The frozen rule chooses logistic.
Do not replace that rule after reading the final result.

### C. Inspect the saved metadata with the restricted loader

Use the loader shown in the source rather than unrestricted pickle loading.
Predict: feature names, candidate, and experiment records are available as ordinary types.
They are part of the inference contract, not optional commentary.

## Checkpoint questions

<details>
<summary>Why persist the selection before creating the test tensor?</summary>

It makes the order auditable and prevents the evaluation result from entering the choice.
The selected procedure is already fixed when the final outcome becomes available.
Code structure reinforces the experimental claim.
</details>

<details>
<summary>Why not retrain the chosen model on train plus validation now?</summary>

That would be a different fitted model and potentially different preprocessing.
It can be a legitimate predeclared procedure, but it is not this reference's procedure.
Here we evaluate and deploy exactly the selected training-only fit.
</details>

<details>
<summary>Does meeting the educational target mean the inference system is safe for real machines?</summary>

No. The entire dataset and event mechanism are synthetic.
The score tests the course pipeline under authored assumptions only.
It supplies no evidence of real equipment reliability or operational safety.
</details>

## Design judgment

A release is a frozen decision procedure plus its evidence and preprocessing.
The best architecture name is less important than preserving that complete contract.
Next, use the release without giving inference permission to train.
