# Project 8, Session 30: reuse a representation without breaking its contract

[Course curriculum](../../CURRICULUM.md) · [Complete train.py](train.py)

## Prerequisites and new ideas

Review Sessions 18–25: image tensors, CNN features, validation, augmentation,
dropout modes, and image checkpoints.
Sessions 26–29 are not required to understand this vision pipeline.

**New:** pretrained representations, a replacement classification head,
the ResNet-18 architecture boundary, ImageNet preprocessing, local weight
provenance, and the difference between an actual pretrained model and a fixture.
We inspect the contract before training anything.

## The motivating question

Can features learned on many natural images help classify our much smaller
grayscale clothing images?
Transfer learning starts from an externally trained representation instead of
learning every feature from random initialization.
That representation has assumptions about input units, channels, and scale.
Ignoring them can invalidate the transfer before optimization even begins.

Here “new task” means Fashion-MNIST's ten clothes classes, different from the
original 1,000 ImageNet classes. We keep Fashion-MNIST so earlier baselines remain
relevant, not because it is an ideal deployment setting for an ImageNet model.

## Run from the course root: explicit fixture

```sh
.venv/bin/python -m lessons.30-transfer-contract.train --smoke --epochs 1
```

This uses **TinyFixture**, a small randomly initialized model with BatchNorm.
It is not ResNet-18, not pretrained, and not a simulation of ImageNet accuracy.
It exists to exercise `[N,3,224,224]` inputs, head replacement procedures,
freezing, staged optimization, and checkpoint reconstruction with only torch.
Its preprocessing is a torch-only square-image implementation with the same
declared sizes and normalization. The real path calls the official torchvision
transform; smoke does not validate that external implementation.

## Real prerequisites: supply them offline

The real path requires:

1. The local official Fashion-MNIST IDX files from
   [Session 18](../18-images-flat/README.md).
2. A torchvision binary build compatible with the installed **torch 2.14.1 CPU**,
   Python 3.12, and the platform. It is optional for this course, not installed by
   this lesson. Use the official [compatibility table](https://github.com/pytorch/vision#installation);
   do not guess a version or install an arbitrary latest wheel.
3. The matching torchvision dependencies, including Pillow and NumPy as required
   by that distribution, supplied through an approved offline process.
4. A trusted local copy of **ResNet18_Weights.IMAGENET1K_V1**:
   [`resnet18-f37072fd.pth`](https://download.pytorch.org/models/resnet18-f37072fd.pth).
   This is a source reference, not a command the lesson executes.
5. A full SHA-256 digest recorded when acquiring that trusted artifact.
   A checksum calculated from an already untrusted file does not establish trust.

Torchvision's code license does not automatically grant every permission for
ImageNet training data or pretrained weights. Review the official
[ResNet-18 weight documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html),
its training-data terms, your dataset rights, and intended use before deployment.
No weights, ImageNet images, or Fashion-MNIST images are redistributed here.

On this macOS course environment you can inspect a supplied artifact's digest:

```sh
shasum -a 256 assets/resnet18-f37072fd.pth
```

After independently establishing its trusted source, set the shell variable
`RESNET18_SHA256` to that complete 64-character digest and run:

```sh
.venv/bin/python -m lessons.30-transfer-contract.train --data data/fashion --weights assets/resnet18-f37072fd.pth --weights-sha256 "$RESNET18_SHA256"
```

The file may be elsewhere; supply its actual local path.
No program in this project calls `weights="DEFAULT"` or asks torchvision to
retrieve parameters. Missing files or optional packages are hard errors in real
mode, never a switch to the fixture.

## Architecture boundary: what we reuse

ResNet-18 is a convolutional network with residual blocks.
A block combines a learned transformation with a shortcut: conceptually `F(x)+x`.
When shapes differ, a projection makes the shortcut compatible.
You need not reimplement its eighteen-layer recipe to learn transfer mechanics.
Its feature extractor produces 512 pooled features; `fc` maps them to 1,000
ImageNet scores. We discard that `fc` and construct `Linear(512,10)`.
The old ImageNet class mapping is deliberately not reused.
The new head starts random, so initial clothes logits have no learned task meaning.

ResNet also uses BatchNorm: per-channel normalization with learned affine
parameters and running statistics. It behaves differently in train/eval mode.
That distinction becomes essential in Session 31.

## The input contract, including every shape

```text
Fashion normalized input        [N,1,28,28], floats in [-1,1]
undo Fashion normalization      [N,1,28,28], floats in [0,1]
repeat grayscale channel        [N,3,28,28]
bilinear resize shorter side    [N,3,256,256] for these square inputs
center crop 224                 [N,3,224,224]
ImageNet channel normalization  [N,3,224,224]
feature extractor + new fc      [N,10] raw clothes logits
```

The channel means are `[0.485,0.456,0.406]`, standard deviations
`[0.229,0.224,0.225]`.
`(rgb-mean)/std` is channelwise, not a single global mean.
Replicating grayscale does not recover color information.
Upsampling cannot invent details absent from the original 28×28 image.

## Complete sources and walkthrough

Read [train.py](train.py) and all of
[transfer_contract.py](../../course/transfer_contract.py), introduced here.
The unchanged [fashion_data.py](../../course/fashion_data.py) is fully taught in
Session 18. These are the full sources, not an unmentioned training framework.

### Entry point

`transfer_arguments(...).parse_args()` extends the offline-data parser.
`partitions(args)` creates the same train/validation split as earlier projects.
`build_initial(args)` returns `(model,provenance)`.
We preprocess two training images, run the model in eval mode inside no-grad,
assert the input/output shapes, and print the replacement head and class mapping.
No optimizer runs and no checkpoint is saved.

### Every component in `transfer_contract.py`

`TRANSFER_PREPROCESS` is the durable dictionary of source scaling, RGB conversion,
resize/crop, interpolation, antialiasing, and channel normalization.
`dict(PREPROCESS)` copies the source dictionary rather than renaming it.
These constants identify our accepted pipeline; they are not fitted on holdouts.

`transfer_arguments` calls the earlier parser factory, then adds local weights,
their digest, and an optional adapted checkpoint path for later sessions.
`Path` is a filesystem path value. No URL is accepted as a loading instruction.

`torchvision_parts` performs a **lazy import** inside the function.
Importing our module in smoke mode never imports torchvision.
The `try/except` turns a missing/incompatible external binary into an actionable
error, preserving the original exception with `raise ... from error`.
It returns the `resnet18` factory and the specific V1 weights enum.
Reading enum metadata and constructing `weights.transforms()` do not download
weights. Only `resnet18(weights=None)` constructs a real model here.

`class TinyFixture(nn.Module)` introduces a concrete module subclass.
`__init__` is the constructor; `self` is the explicit instance reference,
analogous to C++/C# `this`.
`super().__init__()` initializes Module's child/parameter bookkeeping first.
Assigning `self.stem`, `self.layer4`, and `self.fc` registers child modules.
The two convolution blocks use strides 4 then 2 to stay inexpensive at 224×224.
`BatchNorm2d` has learned scale/bias and non-parameter running buffers.
`AdaptiveAvgPool2d(1)` averages each channel's spatial map to `[N,8,1,1]`.
`forward` composes stem, last block, flatten, and the eight-feature head.
Calling the instance invokes this forward method through Module's call machinery.
The name `layer4` matches the real model's final-block boundary for later code;
it does not imply architectural equivalence.

`build_initial` fixes initialization seed 3000.
The explicit smoke branch constructs only the fixture and records
`pretrained=False`, not a fictitious ImageNet source.
Real mode requires both weight path and full hexadecimal digest.
`hashlib.sha256(path.read_bytes()).hexdigest()` hashes actual local bytes.
We compare the full supplied digest and official V1 filename prefix `f37072fd`.
The prefix is an identity sanity check, not independent authenticity proof.
`torch.load(...,map_location="cpu",weights_only=True)` reads a trusted state dictionary.
`load_state_dict(...,strict=True)` loads the original 1,000-class architecture
*before* its head is replaced, catching incompatible files.
`nn.Linear(model.fc.in_features,len(CLASSES))` creates the new ten-class head.
The model is returned in evaluation mode with explicit source provenance.

`preprocess` checks rank, source shape, finite values, and allowed range.
`((images+1)/2).repeat(1,3,1,1)` reverses our source scaling and repeats channels.
The real branch calls `weights.transforms()(rgb)`, the official V1 tensor pipeline.
The smoke branch calls `F.interpolate` with bilinear antialiasing, crops
`[:, :, 16:240, 16:240]`, and normalizes using `[1,3,1,1]` mean/std tensors.
Broadcasting applies the three values to matching channels and all positions.
Both branches assert `[N,3,224,224]`, but only real execution validates torchvision.

`transfer_path` produces separate `smoke-...pt` and `fashion-...pt` paths.
`save_transfer` stores state, classes, preprocessing, stage, data limits, split,
seed, and pretrained/fixture provenance. It saves data, not a whole model object.
`load_transfer` uses safe loading mode and rejects mismatched metadata or smoke
status. It reconstructs the declared architecture, replaces `fc`, strictly loads
the **complete adapted state**, then returns an eval-mode model and metadata.
It never needs to download or reload the original ImageNet file for inference.
These serialization routines will be exercised in Sessions 31–33.
Only load trusted files; safe loading mode does not solve resource exhaustion or
authenticate declarations inside an artifact.

## Controlled edits

1. Skip reversal of Fashion normalization. Predict negative values enter a
   transform expecting [0,1] inputs and yield a different feature distribution.
2. Use only one input channel. Predict the real model's first convolution rejects
   the shape, because its kernels expect three channels.
3. Supply a wrong digest. Predict failure before parameter loading, not a warning.
4. Change the class order while keeping output width ten. Predict plausible shapes
   with wrong names; the checkpoint contract rejects this mismatch later.

## Checkpoint questions

**Are the replacement head's ten rows inherited from ImageNet?**
<details>
<summary>Solution</summary>
No. The ImageNet head has 1,000 different semantic labels and is discarded.
Our ten-row head is newly initialized and must learn the clothes mapping.
</details>

**Does a successful fixture run verify real pretrained transfer quality?**
<details>
<summary>Solution</summary>
No. It verifies procedure and shapes using random features. The actual ResNet
implementation, local official weights, transform, and real data remain separate
prerequisites and must be executed before any real quality claim.
</details>

## Design judgment and limitations

Transfer is most attractive when representation, data scale, and compute align.
This upsampled grayscale benchmark has a substantial domain gap from ImageNet.
Compare against the small CNN, including CPU cost; “pretrained” is not a guarantee
that the larger model is the better engineering choice.
