# Project 5, Session 21: save the image model's meaning, not only its tensors

[Course curriculum](../../CURRICULUM.md) · [Complete train.py](train.py)

## Prerequisites and new ideas

Review Session 9's `state_dict`, safe loading mode, and model reconstruction.
Review Session 18's image normalization and class order.
Review Sessions 19–20's validation-selected model and error inspection.

**New:** an image-specific inference contract, exact CPU prediction roundtrip,
and a deliberate final evaluation boundary for Project 5.
We do not introduce another model or optimize against the official test set.

## The failure we want to prevent

Suppose the weights reload correctly, but a caller feeds `[0,255]` pixels instead
of normalized `[-1,1]` values. Or class 0 is displayed as Ankle boot.
The tensor dimensions can all match while every displayed result is wrong.
A usable artifact couples weights to architecture, preprocessing, and label meaning.

Our payload stores:

| Field | Why it matters |
|---|---|
| `schema`, `task` | Identify this artifact format and intended task |
| `architecture` | Reconstruct the exact 784–64–10 network |
| `state` | Learned parameter tensors |
| `preprocess` | One 28×28 channel, byte scaling, mean, standard deviation |
| `classes` | Ordered index-to-name mapping |
| `split`, `split_seed` | Development split protocol |
| `training` | Epoch budget and explicit subset limits |
| `initialization_seed` | Reproduce this example's initialization |
| `smoke` | Keep synthetic pipeline checks separate from real models |

These are necessary metadata, not a complete production model registry.
They do not encode a license, approval, or guarantee of model quality.

## Exact commands from the course root

```sh
.venv/bin/python -m lessons.21-image-checkpoint.train --smoke --epochs 1
```

This trains a synthetic fixture, writes it, reconstructs it, and compares logits.
To exercise the separate final-test branch with **synthetic** test examples:

```sh
.venv/bin/python -m lessons.21-image-checkpoint.train --smoke --epochs 1 --final-test
```

For the real path, first complete Session 19 and analyze validation in Session 20:

```sh
.venv/bin/python -m lessons.21-image-checkpoint.train --data data/fashion
```

It requires `.artifacts/vision/fashion-mlp.pt`; absence is an error, not a fallback.
The default command still does not read official test data.
After fixing architecture, hyperparameters, preprocessing, and selection procedure:

```sh
.venv/bin/python -m lessons.21-image-checkpoint.train --data data/fashion --final-test
```

This is the one planned Project 5 official test evaluation.
Do not repeatedly tune and rerun until its accuracy looks good.
Use the same development subset limits as before if you used any.
Official data paths and provenance are in [Session 18](../18-images-flat/README.md).
There is no network access or new dependency.

## Read all the code

[train.py](train.py) is the complete executable.
The serialization functions are in the full
[image_training.py](../../course/image_training.py), introduced in Session 19.

1. The parser adds `--final-test`, a boolean flag requiring an explicit decision.
2. We reconstruct the development split for repeatable validation inputs.
3. In explicit smoke mode only, `mlp()` is trained for the requested synthetic epochs
   and saved. The real branch never trains a missing replacement.
4. `artifact(args,"mlp")` chooses `smoke-mlp.pt` or `fashion-mlp.pt`.
5. `load_model(path,mlp,args.smoke)` loads onto CPU with `weights_only=True`.
   It compares stored metadata with the supported contract before prediction.
6. The loader calls `mlp()` to allocate a matching model and
   `load_state_dict(...,strict=True)` to check every expected state key and shape.
7. The entry point checks the architecture tag, not just compatible tensor sizes.
8. `images = val.tensors[0][:8]` obtains a fixed `[8,1,28,28]` probe batch.
9. `with torch.no_grad(): before = model(images)` obtains `[8,10]` logits without
   constructing a gradient graph. The loader already selected evaluation mode.
10. `torch.save(payload,copy_path)` writes a second artifact named `roundtrip`.
    It includes the same metadata and state, not a pickled model instance.
11. `load_model(copy_path,...)` constructs another independent network.
12. `torch.testing.assert_close(before,after,rtol=0,atol=0)` demands exact matching
    values on this CPU computation. Merely matching argmax classes is weaker.
13. `after.argmax(dim=1).tolist()` gives eight ordinary integer predictions.
14. The printed saved `classes` determines the meaning of those integers.
15. Only `if args.final_test:` calls `test_partition`.
    That function opens the official `t10k` files in real mode, or a separate
    seeded synthetic fixture in smoke mode.
16. `evaluate` reports loss and accuracy for the already fixed state.
    No optimizer exists in the real evaluation branch.

Python's named keyword arguments `rtol=...`, `atol=...`, and `weights_only=True`
make API choices explicit at the call site. They are not assignments to global
settings. A `with` block scopes gradient disabling, restoring the previous setting
on exit even if an exception occurs.

## Why reconstruct rather than deserialize a whole object?

A state dictionary is numerical state, separate from executable model code.
We control the builder and validate the metadata; strict loading catches drift.
`weights_only=True` narrows accepted serialized objects.
It is not a promise that any untrusted file is harmless: hostile tensors can
consume resources, and metadata can be falsified. Use trusted local artifacts.
Never “fix” a loading error by disabling the safety option.

`eval()` selects inference behavior for mode-dependent modules.
`no_grad()` disables recording a differentiation graph.
The current MLP has neither dropout nor BatchNorm, but maintaining both controls
is important when we replace its architecture.
Neither operation changes the learned parameter values.

## Controlled edits

1. Change one stored class name in a copy of the payload.
   Predict contract rejection before inference, even though the weight shapes fit.
2. Change the builder hidden width to 32.
   Predict strict state loading fails on tensor size mismatches.
3. Replace exact-logit comparison with an argmax-only assertion.
   Predict a small parameter perturbation might pass the weaker check while
   changing confidence scores. Restore the stronger roundtrip test.
4. Feed unnormalized bytes converted to floats to the restored MLP.
   Predict different logits; matching dimensions does not establish matching units.
   Do this only as a controlled local probe, not as your real inference path.

## Checkpoint questions

**Does a successful roundtrip prove the classifier is useful?**
<details>
<summary>Solution</summary>
No. It proves this saved artifact reconstructs the same computation for the probe
inputs. A consistently wrong model can roundtrip perfectly. Use held-out real
evaluation for predictive quality and deployment-specific evaluation for usefulness.
</details>

**Can test accuracy select between two hidden widths?**
<details>
<summary>Solution</summary>
That would use the test for development. Choose width with validation, then report
one final test result. After repeated test-informed decisions the test no longer
estimates a completely unseen final check.
</details>

**Why store normalization when it contains no learned statistics?**
<details>
<summary>Solution</summary>
Because the learned weights depend on that numerical convention. Fixed constants
still form part of the input contract and can be changed accidentally by a caller.
</details>

## Design judgment and limitations

This is inference packaging, not exact training resumption: optimizer state,
sampler progress, and all RNG states are not stored.
Exact CPU equality here does not promise bitwise agreement across devices or
library versions. Project 6 reuses the development split to study image structure.
If you already saw Project 5's test score, avoid using it to steer Project 6;
reserve an additional untouched deployment holdout for a stronger final claim.
