# Project 8, Session 33: reconstruct and use the adapted classifier

[Course curriculum](../../CURRICULUM.md) · [Complete train.py](train.py)

## Prerequisites and new ideas

Review Session 21's state-plus-contract packaging and roundtrip check.
Review Session 30's RGB conversion and V1 preprocessing.
Review Sessions 31–32's head/fine-tuning comparison and validation-selected stages.

**New:** reconstructing an adapted pretrained architecture without its original
download, rejecting fixture/real confusion, predicting class names from saved
mapping, and optionally reading genuinely new local IDX images.
The model and preprocessing are no longer choices to make during test evaluation.

## The problem

The trainer's in-memory model is not a deployment artifact.
An inference program must know exactly which architecture to construct, which
preprocessing to apply, and which class name belongs to each output column.
The ImageNet head and its mapping are gone; they must not reappear by accident.
Likewise, a random fixture checkpoint must never stand in for a pretrained model.

## Run from the course root

Self-contained procedure check:

```sh
.venv/bin/python -m lessons.33-transfer-inference.train --smoke --epochs 1
```

It explicitly trains the tiny fixture's two stages, saves, reconstructs, and
requires exact CPU logits on the probe inputs.
It verifies the transfer procedure, not ResNet-18 or Fashion-MNIST quality.
For the focused standard-library test suite, with no pytest dependency:

```sh
.venv/bin/python -m lessons.33-transfer-inference.check_contracts
```

It checks IDX gzip/header failures, repeatable splits, unchanged validation pixels,
metadata rejection, state roundtrip, trainable-group updates, and frozen
BatchNorm buffers. Fixtures are created under `.artifacts/` and cleaned afterward.
To check the separate synthetic final-evaluation branch:

```sh
.venv/bin/python -m lessons.33-transfer-inference.train --smoke --epochs 1 --final-test
```

For real inference after Session 32, with the same offline packages and data:

```sh
.venv/bin/python -m lessons.33-transfer-inference.train --data data/fashion --checkpoint .artifacts/transfer/fashion-finetuned.pt --max-train 512 --max-val 128
```

If Session 32's validation comparison favored head-only training, use it instead:

```sh
.venv/bin/python -m lessons.33-transfer-inference.train --data data/fashion --checkpoint .artifacts/transfer/fashion-head.pt --max-train 512 --max-val 128
```

Default probes are four development-validation images, not new independent
evaluation evidence. The command prints predictions and contract metadata.
Neither command reads the official test files.
The `--epochs` option is relevant only to the self-contained smoke training here;
real inference never retrains.

When the checkpoint choice and procedure are fixed, one explicit final evaluation:

```sh
.venv/bin/python -m lessons.33-transfer-inference.train --data data/fashion --checkpoint .artifacts/transfer/fashion-head.pt --max-train 512 --max-val 128 --final-test
```

Use the checkpoint that won your declared validation comparison; the head path
above is an example of a concrete valid choice, not a claim it always wins.
The final score must not choose between the two stages.

## Optional genuinely new local inputs

Supply a local 28×28 grayscale IDX image file, with the standard image header,
unsigned-byte pixels, and no labels required:

```sh
.venv/bin/python -m lessons.33-transfer-inference.train --data data/fashion --checkpoint .artifacts/transfer/fashion-head.pt --input-images data/new-clothes-idx3-ubyte
```

The file can contain one or several images and may be gzip-compressed.
It must have the same foreground/background and framing convention as the training
data. Do not relabel arbitrary resized photos as equivalent Fashion-MNIST inputs.
Use a small file: this readable inference path transforms its whole input at once.
For bulk serving, iterate in bounded batches as the evaluation helper already does.
This entry point still reconstructs the development split for its default probe
and smoke path; supplying new images does not remove the local dataset requirement.

Data provenance and optional-dependency requirements remain in
[Session 30](../30-transfer-contract/README.md).
No model, data, package, or image is downloaded.

## Complete source and walkthrough

Read [train.py](train.py), [transfer_contract.py](../../course/transfer_contract.py),
and [transfer_training.py](../../course/transfer_training.py).
The helpers were fully explained at their first appearances in Sessions 30–31.

1. `Path` converts `--input-images` to a filesystem path; absence is `None`.
2. The parser separately adds `--final-test`, making final evaluation explicit.
3. `partitions` reconstructs the stable development data without opening test.
4. In default smoke mode, create an explicit random fixture, train both stages,
   and save its complete fine-tuned state and honest provenance. An explicitly
   supplied smoke checkpoint is loaded instead, without training a replacement.
5. `args.checkpoint or transfer_path(...)` chooses an explicit checkpoint if given,
   otherwise the stage's conventional filename.
   Python `or` returns an operand, not necessarily a boolean.
6. `load_transfer` reads trusted local tensor/data state with `weights_only=True`
   and `map_location="cpu"`.
7. Schema, class order, input transform, split, and smoke status must match.
8. The smoke branch requires `kind="tiny-random-fixture"` and `pretrained=False`.
   The real branch requires the declared ResNet V1 pretrained provenance.
9. For real reconstruction, `resnet18(weights=None)` allocates architecture only.
   Replacing `fc` gives the saved ten-class dimensions before strict loading.
   Loading the complete adapted state supplies both backbone and head weights.
10. Neither reconstruction branch calls the original weight downloader or needs
    the original V1 file. Provenance remains in the saved payload.
11. The entry point requires a recognized adapted stage, `head` or `finetuned`.
12. Normally `source = val.tensors[0][:4]` supplies a known `[B,1,28,28]` probe.
13. With `--input-images`, `read_idx(parent,name,True)` validates the local image
    header and payload; `normalize(...)` applies our source byte convention.
    Unlike the official-pair reader, this generic reader permits arbitrary counts.
14. `preprocess(source,args.smoke)` checks source values and constructs the exact
    supported pretrained input contract.
    The loader compared saved metadata with that known contract; it does not
    silently ignore an unsupported transform and keep running.
15. The model is in eval mode and `no_grad` scopes the forward pass.
    Logits have `[B,len(saved_classes)]`, normally `[B,10]`.
16. In default smoke mode, compare reconstructed logits with the original fixture
    model's logits using zero relative and absolute tolerance. A custom checkpoint
    has no in-memory training reference, so that extra comparison is not claimed.
17. `argmax(dim=1)` chooses output columns.
    `[payload["classes"][index] for index in predicted.tolist()]` is a list
    comprehension mapping ordinary integer indices to saved human-readable names.
18. Assertions confirm every output name belongs to the saved mapping.
19. We print preprocessing and original source so fixture or provenance mistakes
    cannot be hidden behind an accuracy number.
20. Only `--final-test` calls `test_partition` and evaluates the fixed loaded model.
    No optimizer is used for the real inference or final-test path.

## What is durable and what is not

Durable: complete adapted weights, architecture kind, original source/digest,
clothes class mapping, input contract, stage, split, seed, and data limits.
The supported architecture and transform implementations still need compatible
code and packages; metadata alone is not executable inference code.
This is not a generic “load any vision model” format.

`weights_only=True` reduces accepted serialization objects, but does not
authenticate a claimed source or make hostile tensor sizes harmless.
Treat adapted checkpoints as trusted artifacts and retain original acquisition
records. Do not replace strict validation with permissive loading to fix a mismatch.

## Controlled edits

1. Pass `smoke-finetuned.pt` without `--smoke`.
   Predict metadata rejection, not a random model pretending to be pretrained.
2. Change the class order in a copied artifact.
   Predict rejection even though strict tensor loading alone could have succeeded.
3. Reconstruct the original 1,000-class head.
   Predict strict shape mismatches loading the saved ten-class weights.
4. Skip center cropping or ImageNet normalization.
   Predict changed inputs and logits; the intended inference contract is broken
   even if the network can technically process another spatial shape.

## Checkpoint questions

**Why can inference work without the original ImageNet weight file?**
<details>
<summary>Solution</summary>
The adapted checkpoint stores the full current state, including the backbone.
We construct only the architecture with weights=None and load that complete state.
Original provenance remains important for governance, not for another download.
</details>

**Can matching output names prove the class mapping is semantically correct?**
<details>
<summary>Solution</summary>
No. The assertion proves internal consistency with the saved list.
Its semantic correctness depends on the label contract established during training.
That is why we preserve and validate the official ordered mapping.
</details>

**When should you prefer the smaller scratch CNN?**
<details>
<summary>Solution</summary>
When its validated quality, latency, memory, and operational simplicity better fit
the use case. Reusing a famous pretrained model is a strategy, not a requirement.
</details>

## Judgment and limitations

Real execution still requires trusted local official weights for the first stage,
compatible torchvision, and real local images. Smoke does not validate those paths.
Repeated viewing of the same official test across projects weakens its independence;
a fresh untouched deployment holdout is preferable for a consequential final claim.
Keep class-level error analysis and compute cost beside the aggregate test score.
