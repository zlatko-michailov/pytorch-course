# Project 43 / Session 43 — Attention with batches and heads

## Prerequisites

Complete session 42 and review tensor views, broadcasting, and batched matrix multiplication.
You should be able to explain `softmax(QK.T / sqrt(d_k)) @ V` for one sequence.
This session adds organization, not a new attention formula.

**Review:** Q/K/V roles, softmax over keys, gradient propagation.
**New:** head axes, split/merge operations, transposed strides, batched reference checks.
No padding or causal mask is used yet.
All rows contain exactly three real tokens.

## Motivation

A single learned projection can mix many feature types.
Multiple heads give separate projected feature groups their own attention-weight matrices.
They can learn different relationships, but are not assigned named linguistic jobs.
The final projection can mix information from these groups.

An incorrect transpose can still produce plausible tensor sizes.
We will check both dimensions and values against a slow explicit implementation.
Speed is secondary while establishing the contract.

## Run

From the course root:

```bash
.venv/bin/python -m lessons.43-attention-shapes.train
```

Full working source: [train.py](train.py).
The script compares manual batched attention with:

1. A Python loop over every batch/head pair.
2. PyTorch's `scaled_dot_product_attention` with dropout disabled.

It then backpropagates through the manually implemented path.
The built-in function is a reference, not the operation hidden inside our implementation.

## Name the axes first

The default dimensions are:

```text
B = 2 sequences
T = 3 tokens per sequence
D = 8 model features
H = 2 heads
d = D/H = 4 features per head
```

Input X has shape `[B,T,D] = [2,3,8]`.
Each of three `nn.Linear(D,D,bias=False)` layers preserves that shape.
Q, K, and V use different learned weight matrices.
Disabling biases makes this shape lesson slightly easier to inspect.

The layers live in an `nn.ModuleList`.
Unlike an ordinary unregistered list stored on a model, ModuleList participates in module tracking.
Here we also iterate it directly to apply each projection.
The separate output projection is another `nn.Linear(D,D)`.

## Split projected features into heads

`split_heads` first checks `D % H == 0`.
Without divisibility, equal-width heads cannot partition the projected feature axis.
A specific `ValueError` explains that configuration mistake.

The reshape:

```text
[B,T,D] -> [B,T,H,d]
```

splits the final feature axis.
It does not create more information or mix token positions.
Then `.transpose(1,2)` swaps token and head axes:

```text
[B,T,H,d] -> [B,H,T,d]
```

We place heads beside batch so the last two dimensions are a matrix per head.
Q, K, and V now have shape `[2,2,3,4]`.

## Batched scores

`k.transpose(-2,-1)` swaps only K's last two dimensions.
It produces logical shape `[B,H,d,T]`.
Using `.T` on a four-dimensional tensor would not express this intended operation.

PyTorch's `@` treats the leading dimensions as batch dimensions:

```text
[B,H,T,d] @ [B,H,d,T] -> [B,H,T,T]
```

The two T axes have different meanings.
The first indexes query positions; the second indexes key positions.
Each batch/head pair has its own score matrix.
The scale is `sqrt(d)`, not `sqrt(D)`, because each dot product uses d components.

`softmax(-1)` normalizes over the final key axis.
Multiplying by V gives:

```text
[B,H,T,T] @ [B,H,T,d] -> [B,H,T,d]
```

This is precisely session 42 repeated over batch and head axes.

## The slow reference

The nested loops select `q[b,h]`, `k[b,h]`, and `v[b,h]`.
Each selection has the two-dimensional shape used in session 42.
The reference computes one ordinary matrix formula at a time.
Writing into `torch.empty_like(attended)` fills a tensor with matching shape and dtype.
Every slot is assigned before comparison.

`assert_close` compares that reference with the batched calculation.
The second comparison checks against the built-in scaled-dot-product function.
Explicit `dropout_p=0.0` removes randomness from the reference contract.
These comparisons are useful for axis bugs that shape assertions alone cannot reveal.

## Merge the heads

The attended states first transpose back:

```text
[B,H,T,d] -> [B,T,H,d]
```

`transpose` often returns a view with non-contiguous strides.
The data is logically reordered without necessarily being physically copied.
`.contiguous()` requests ordinary contiguous storage in the new logical order.
Then `.reshape(B,T,D)` joins the adjacent H and d axes.

This is different from simply reshaping the original `[B,H,T,d]`.
Merging before swapping would interleave token/head contents incorrectly.
The final output projection mixes the concatenated head features.
Its result has the same `[B,T,D]` outer contract as X.

## Gradient check

A scalar `result.square().mean()` provides a simple differentiable probe.
It is not a language-learning objective.
After backward, every projection weight must have a finite, nonzero gradient tensor.
This includes Q, K, V, and the output projection.
The test checks connectivity, not optimization quality or generalization.

## Controlled experiments

### A. Change H from 2 to 4

Predict: d becomes 2, score shape becomes `[2,4,3,3]`.
The merged/output shape remains `[2,3,8]`.
Both references should still match.

### B. Choose H=3 with D=8

Predict: the divisibility guard rejects the configuration.
Do not hide the error by truncating features or changing the intended width silently.

### C. Remove the transpose before merging

Predict: a same-sized output may still be produced.
Compare its values with a correctly merged reference, not just its shape.
This is the central warning: shape compatibility is weaker than semantic axis correctness.

### D. Scale by sqrt(D) instead of sqrt(d)

Predict: the loop/built-in comparisons fail unless they are changed too.
The scores become smaller and softmax less sharp.
An apparently harmless constant changes the operation.

## Checkpoint questions

<details>
<summary>Does each head see a disjoint slice of the original input?</summary>

Not necessarily. Each full D-to-D projection can combine all input features.
We split its projected outputs into head groups afterward.
The head partition constrains the attention calculation, not direct access to raw input columns.
</details>

<details>
<summary>Why does adding heads not necessarily increase output width?</summary>

Here D is fixed and divided among H heads.
More heads means smaller per-head width d.
After concatenation, H*d is still D.
</details>

<details>
<summary>Is a successful gradient test enough to trust the model?</summary>

No. A transposed or semantically wrong calculation can still be differentiable.
That is why we also compare values with explicit references and check dimensions.
Later lessons add behavioral masking and evaluation contracts.
</details>

## Design judgment

Explicit attention costs O(B*H*T²) score storage in this implementation.
That is fine for tiny educational sequences, not a prescription for long-context production.
Next, constrain which keys may contribute and supply position information.
