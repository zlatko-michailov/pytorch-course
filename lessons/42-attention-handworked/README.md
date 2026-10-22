# Project 42 / Session 42 — Calculate attention by hand

## Prerequisites

Complete session 41 and review matrix multiplication, softmax, and autograd.
You should be able to multiply a `[2,2]` matrix by another `[2,2]` matrix.
No Transformer API knowledge is assumed.

**Review:** linear projections, dot products, differentiable tensor operations.
**New:** queries, keys, values, scaled scores, row-wise attention weights.
There are only two tokens and one attention head.
There is no optimizer, language model, mask, or pretrained network.

## Motivation

A mean gives every real token the same coefficient.
Attention lets one token select a content-dependent weighted mixture of other token values.
The weights are computed from the current representations, not stored as one fixed mean.
That is the operation to understand before introducing batches or blocks.

Attention by itself does not provide sequence positions.
Without positions, self-attention is permutation-equivariant:
reordering input tokens reorders their output states in the same way.
Mean-pooling those states would still lose the order distinction from session 41.
Positions will be introduced explicitly in session 44.

## Run

From the course root:

```bash
.venv/bin/python -m lessons.42-attention-handworked.train
```

Complete source: [train.py](train.py).
Everything is CPU-only and self-contained.
The script compares scalar arithmetic with tensor arithmetic and then checks gradients.

## The three projections

For token-state matrix X, define:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

A query is the requesting token's lookup representation.
A key is the candidate token's representation for matching that request.
A value is the content that gets mixed after matching.
These names describe roles in the calculation; they are not annotated linguistic concepts.

In a learned network, the W matrices are optimized through the task loss.
Here we choose them explicitly to make the calculation readable.
`torch.eye(2)` constructs the two-by-two identity matrix.
X, W_Q, and W_K are identities.
W_V is diagonal with entries 2 and 4.

Therefore:

```text
Q = K = [[1, 0],       V = [[2, 0],
         [0, 1]]            [0, 4]]
```

Rows represent tokens.
Columns represent features.
Using a different V from Q/K makes “where to read” distinct from “what to read.”

## Scores and scaling

The unnormalized compatibility scores are `Q @ K.T`.
The query row stays on the left; key rows become columns after transposition.
Score `[i,j]` asks how strongly query token i matches key token j.
The result has shape `[query_tokens, key_tokens]`, here `[2,2]`.

Divide by `sqrt(d_k)`, where `d_k=2` is the query/key feature width.
For roughly independent unit-variance features, a dot product's variance grows with width.
This scaling counteracts that growth.
It is a practical magnitude convention, not a proof that every learned feature is unit variance.

The scaled scores are:

```text
[[1/sqrt(2), 0],
 [0, 1/sqrt(2)]]
```

## Softmax over keys

`scores.softmax(dim=-1)` normalizes each query's row over candidate keys.
For the first row:

```text
a = exp(1/sqrt(2)) / (exp(1/sqrt(2)) + exp(0))
b = exp(0)         / (exp(1/sqrt(2)) + exp(0))
```

Thus `a ≈ 0.669762` and `b ≈ 0.330238`.
Both are positive, and `a+b=1`.
The second row has the coefficients in the opposite order.

The weights are:

```text
[[a, b],
 [b, a]]
```

Softmax is over **keys per query**, not over queries per key.
This orientation becomes especially important when batching multiple heads.

## Weighted values

`weights @ V` produces a new value vector for each query token.
The first output is `a*[2,0] + b*[0,4]`.
The second is `b*[2,0] + a*[0,4]`.

Expected outputs are approximately:

```text
[[1.3395, 1.3210],
 [0.6605, 2.6790]]
```

This is a convex combination of the current value vectors.
The later output projection and residual path can transform it further.
Nothing here has generated new words or interpreted a sentence.

## Autograd walkthrough

The chosen Q, K, and V tensors receive `.requires_grad_()`.
The trailing underscore marks an in-place setting change.
They are leaf tensors here because the chosen X and W tensors did not require gradients.
That keeps the gradient inspection focused on attention itself.

`output[0,0].backward()` differentiates one selected scalar output component.
Q and K receive gradients through score calculation and softmax.
V receives gradients through the weighted sum.
Each entire gradient tensor must be finite and have a nonzero total magnitude.
Not every individual component must be nonzero.

For example, the unused second output query need not have a query gradient for this scalar.
That sparsity is expected dependency structure, not a broken backward pass.
The script prints all three gradients so you can inspect the distinction.

## Controlled experiments

### A. Remove the square-root scale in both reference and tensor calculations

Predict: diagonal weights increase from about 0.67 toward about 0.73.
The matching preference becomes sharper.
With a larger feature width, uncontrolled score magnitudes can become much more extreme.
Do not change only one side unless you want the comparison assertion to fail.

### B. Make both keys identical

Predict: each query sees equal scores for the two candidates.
Each weight row becomes `[0.5,0.5]`, regardless of the two values.
The output is the same average value for both queries.

### C. Change V while holding Q and K fixed

Predict: attention weights remain unchanged but outputs and value gradients can change.
This separates matching from transferred content.
Restore the hand-calculable version before advancing.

## Checkpoint questions

<details>
<summary>Why use K.T but not V.T?</summary>

Q times K.T compares every query vector with every key vector.
The resulting key coefficients then multiply V's key-indexed rows.
The contracted dimension in the second multiplication is key count, not feature width.
</details>

<details>
<summary>Are attention weights probabilities that a word is important?</summary>

They are nonnegative normalized mixture coefficients for this query and this representation.
That does not make them calibrated semantic explanations or causal importance scores.
Their interpretation depends on the value, projection, residual, and downstream computations.
</details>

<details>
<summary>Why does this session not yet solve the order task?</summary>

There is no position signal.
Content-dependent mixing alone does not label first and last sequence slots.
We must supply the missing positional information before claiming order-sensitive behavior.
</details>

## Design judgment

Understand the weighted-sum contract before choosing a large attention API.
Next, preserve this same calculation across batch and head axes.
