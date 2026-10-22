# Project 45 / Session 45 — Build an explicit encoder block

## Prerequisites

Complete sessions 42–44 and review MLPs, gradients, and training/evaluation modes.
You should be comfortable tracing `[B,H,T,d]` attention tensors.
You should also explain why padding keys need a mask.

**Review:** multi-head attention, ReLU MLPs, binary-logit training.
**New:** residual paths, LayerNorm, pre-normalization block structure.
Residual addition and normalization solve different problems.
Neither is an unexplained “Transformer ingredient” to copy without a purpose.

## Problem

One attention operation mixes token content.
A usable block also needs a way to preserve existing information and transform features locally.
Composing many transformations can make optimization difficult if every layer must reconstruct useful input.
Residual paths make a correction to an existing state rather than an obligatory replacement.

Feature scales can drift as states pass through these paths.
LayerNorm gives each token's feature vector a more controlled input scale.
It does not carry information across tokens.
Attention and the feed-forward subnetwork still perform the learned transformations.

## Run

From the course root:

```bash
.venv/bin/python -m lessons.45-transformer-block.train
```

Full source: [train.py](train.py).
The explicit `EncoderBlock` is the implementation; no `TransformerEncoder` wrapper is used.
Four synthetic feature sequences are used only for a wiring/overfitting check.
There is intentionally no held-out generalization claim.
The script is standalone and requires no saved artifact.

## Block equation

This is a **pre-normalization** encoder block:

```text
u = x + attention(LayerNorm_1(x))
y = u + feedforward(LayerNorm_2(u))
```

The attention output projection is included inside the attention branch.
Padding outputs are zeroed before returning y.
Each sum combines tensors with identical `[B,T,D]` shape.
Our D is 8, with 2 heads of width 4.

Pre-normalization places normalization inside each correction branch.
The residual path itself remains a direct addition path.
Post-normalization is another design, but moving LayerNorm changes optimization behavior.
It is not a cosmetic source-code rearrangement.

## What LayerNorm does

`nn.LayerNorm(width)` computes a mean and variance over the final D features of each token.
It subtracts that mean and divides by a stabilized standard deviation.
Learned per-feature scale and offset then adjust the normalized result.
The small epsilon inside the operation prevents division by a zero variance.

Normalization is per token, not across the batch.
It has no running mean/variance like BatchNorm.
Its basic statistics do not switch to stored population estimates in evaluation mode.
It does not remove the need for `train()`/`eval()` if other modules later require them.

A residual path preserves an identity route.
Normalization controls a branch's feature distribution.
These motives are independent; one cannot replace the other.

## Source walkthrough: modules

The constructor rejects widths not divisible by head count.
It registers two independent LayerNorm instances.
They have separate learned parameters because the two branches see different states.

`qkv = nn.Linear(D,3*D)` combines three projections into one matrix operation.
`.chunk(3,dim=-1)` later divides its output into Q, K, and V.
This is equivalent in shape to three separate projections from session 43.
It is not splitting the original input into query-only/key-only/value-only thirds.

The attention output layer maps the concatenated D head features back to D.
The feed-forward subnetwork is `D -> 2D -> D` with ReLU between.
`nn.Sequential` applies its listed modules in order.
The same feed-forward weights are applied independently at every token position.

## Source walkthrough: forward path

Input x is `[B,T,8]`, and valid is boolean `[B,T]`.
Each sequence must contain at least one real token.
For this **bidirectional** block, that guarantees every score row has an allowed key.
We do not add a causal triangle here.

The normalized input passes through the combined QKV projection.
Each projection is reshaped to `[B,2,T,4]`.
Scores are `[B,2,T,T]`, scaled by `sqrt(4)`.
`valid[:,None,None,:]` broadcasts the key-validity mask across heads and queries.

The weighted value states are merged back to `[B,T,8]`.
The output projection is added to x.
The second normalized branch applies the token-local MLP and adds another correction.

Finally, multiplication by the query-valid mask zeros padding outputs.
Biases in Linear and LayerNorm can create nonzero padding states inside the branch.
Real queries still cannot read padded keys.
Zeroing returned padded states makes subsequent pooling unambiguous.

## What this block does not include

Position addition is a caller responsibility.
The random input states in this wiring check are already distinct per position.
They are not a learned text embedding demonstration.
Session 46 supplies explicit sinusoidal positions to actual token embeddings.

There is no dropout in this small block.
That keeps padding and gradient checks deterministic.
A production architecture may add dropout according to its data and regularization needs.
Do not add mechanisms solely because a familiar diagram contains them.

## Training probe and assertions

A linear head consumes the masked mean of block outputs.
The output is one logit per example.
The labels are four fixed binary values with no domain meaning.

The loop performs 60 Adam updates.
After the first backward pass, every block parameter must have a finite nonzero gradient tensor.
The final BCE must be below the initial BCE.
This is a useful smoke test that the parameters participate in a learnable path.
It is also deliberate overfitting to four examples.

Replacing padded input states by 999 must not change the returned states.
All returned padded states must be exactly zero.
These are behavioral checks inherited from the attention-mask contract.
They matter even when the tiny optimization succeeds.

## Controlled experiments

### A. Remove the first residual addition

Predict: shape checks can still pass, but the input no longer has an identity path.
Training may still fit four examples.
A tiny successful fit is not evidence that residuals never matter in deeper models.
Compare gradient behavior before making any claim about deeper optimization.

### B. Remove LayerNorm while keeping residuals

Predict: the input path still exists, but branch scales are no longer normalized.
Try multiplying input features by 100 and inspect loss and gradient magnitudes.
This isolates normalization's motive from the identity path's motive.

### C. Increase depth of the feed-forward expansion

Change 2D to 4D in both feed-forward layers.
Predict: outer `[B,T,D]` shape remains unchanged while parameter count rises.
Four-example training loss is not a fair model-selection criterion for that added capacity.

## Checkpoint questions

<details>
<summary>Does LayerNorm normalize each feature across the training dataset?</summary>

No. It normalizes the feature vector of each individual token at runtime.
Train-only feature scaling is a separate fitted preprocessing step.
LayerNorm has learned affine parameters, but not a dataset-wide fitted vocabulary or scaler.
</details>

<details>
<summary>Why does the feed-forward network not mix neighboring tokens?</summary>

Linear layers act on the last dimension and share their weights across preceding dimensions.
They transform each token independently.
Attention is the branch that mixes information between token positions.
</details>

<details>
<summary>Why is this not yet evidence of good text classification?</summary>

The inputs are four synthetic random tensors and the model can memorize them.
There is no held-out dataset, position-generalization test, or natural-language task.
The demonstrated claim is correct wiring and mask behavior only.
</details>

## Design judgment

An explicit small block makes the information paths inspectable.
Keep that understanding when moving to higher-level libraries.
Next, test whether the additional order-aware machinery helps on a controlled held-out task.
