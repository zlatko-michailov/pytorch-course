# Project 39 / Session 39 — Pool only real token positions

## Prerequisites and scope

Complete session 38 and review categorical embeddings and batched tensor reductions.
You should know that an embedding table is a trainable matrix indexed by integer IDs.
You should also recognize `[B, T, D]` as batch, tokens, and embedding width.

**Review:** `nn.Embedding`, autograd, reduction dimensions, PAD versus UNK.
**New:** ragged-to-rectangular collation, boolean masks, masked means, empty-input contracts.
We are not training a classifier yet.
We are testing the representation that the classifier will depend on.

## The motivating bug

Suppose a two-token message is batched beside a five-token message.
To form a rectangle, the first row receives three padding cells.
Taking `.mean(dim=1)` across all five cells dilutes the two-token representation.
The prediction could then change when an unrelated longer message joins the batch.
That is a batching artifact, not useful model behavior.

Setting a padding embedding to zero does not repair the denominator.
`(u + v + 0 + 0 + 0) / 5` is not `(u + v) / 2`.
We need a sum of real embeddings divided by the count of real embeddings.

## Run

From the course root:

```bash
.venv/bin/python -m lessons.39-masked-embeddings.train
```

Complete source: [train.py](train.py).
Vocabulary and fixture source, introduced in 38:
[`course/text_data.py`](../../course/text_data.py).
The script uses only CPU PyTorch and the Python standard library.
It performs assertions, including negative-path checks, before reporting success.

`torch.set_num_threads(1)` limits intra-operation CPU parallelism.
These tensors are tiny; managing many worker threads can cost more than the arithmetic.
This is a workload choice, not a requirement that neural networks use one CPU core.

## Step 1: collate a batch

`collate(texts, vocabulary)` begins with a list of variable-length encoded lists.
It rejects an empty batch and any sequence whose tokenizer produced zero tokens.
That rejection is deliberate; there is no meaningful mean over an empty message.
The caller must decide whether empty messages are invalid or deserve a separate policy.

`max(map(len, sequences))` computes the longest token count.
`map(len, ...)` applies `len` to every sequence.
The optional `extra_padding` argument adds right-padding columns for our invariant test.
Negative extra padding raises a specific `ValueError`.

`torch.full((B, T), PAD, dtype=torch.long)` creates the rectangular ID matrix.
The `long` dtype stores embedding indices, not floating-point measurements.
Each loop iteration writes the original IDs into a prefix of one row.
`ids[row, :len(sequence)]` is a basic slice view into that prefix.
Cells after the prefix remain PAD.

For the default messages, the ID tensor has shape `[3, 3]`.
The third message is unknown but nonempty.
Its unknown ID must be treated as a real token, not as padding.
No label values enter the collation operation in this session.

## Step 2: look up vectors

`nn.Embedding(vocabulary_size, 4, padding_idx=PAD)` creates a `[V, 4]` table.
Looking up an ID tensor `[3, 3]` produces embeddings `[3, 3, 4]`.
Each scalar ID selects one complete four-component row.

`padding_idx=PAD` initializes the designated row to zero and excludes it from lookup gradients.
It does not know which reduction you intend to apply afterward.
It is not a replacement for a mask.
Other update mechanisms, such as explicitly assigning weights, still need discipline.

## Step 3: construct the mask

`ids.ne(PAD)` means elementwise “not equal to PAD.”
It returns a boolean tensor `[B, T]`.
`.unsqueeze(-1)` changes the shape to `[B, T, 1]`.
The extra axis lets multiplication broadcast the same keep/drop decision over all D features.

The expression `embedded * mask` has shape `[B, T, D]`.
Real positions retain their vectors.
Padding positions contribute zeros.
UNK has ID 1, so `ne(PAD)` correctly keeps it.

`mask.sum(dim=1)` has shape `[B, 1]`.
Summing booleans counts `True` values.
This singleton feature axis is important for division against `[B, D]`.

## Step 4: compute the masked mean

The formula is:

```text
pooled[b,d] = sum_t(mask[b,t] * embedding[b,t,d]) / sum_t(mask[b,t])
```

The numerator reduces tokens and has shape `[B, D]`.
The denominator broadcasts from `[B, 1]`.
The resulting pooled representation has shape `[B, D]`, here `[3, 4]`.

The function checks for a zero count before dividing.
`.any().item()` converts the scalar tensor check into a Python boolean decision.
This tiny CPU demonstration prioritizes a readable error over silently clamping away bad input.
A clamp would avoid division by zero but could disguise an invalid example.

The second guard matters even though `collate` already rejects empty text.
The pooling function can be called with an independently constructed all-padding tensor.
Each public boundary protects its own contract.

## Step 5: test behavior, not only shapes

The script collates the same messages with four additional padding columns.
It recomputes the embeddings and pooled representations.
`torch.testing.assert_close` checks values, allowing ordinary floating-point rounding.
This tests a semantic invariant: padding must not influence the representation.

Two explicit invalid collation cases must raise `ValueError`.
A third test calls pooling directly on all-padding IDs.
Each `try` catches only the expected exception type.
The `else` branch raises if the expected failure never occurs.
Unexpected exceptions remain visible rather than being swallowed.

Finally, a scalar function of the pooled vectors is backpropagated.
The PAD row's gradient must remain zero.
The UNK row's gradient must be nonzero because a real unknown token was present.
This distinguishes masking semantics from mere tensor dimensions.

## Controlled experiments

### A. Replace the masked mean with an ordinary mean

Temporarily use `embedded.mean(dim=1)`.
Predict: appending padding changes pooled values and fails the invariance assertion.
Interpretation: zero-valued padding fixes the sum, not the count.

### B. Mask both PAD and UNK

Try `ids.gt(UNK)` instead of `ids.ne(PAD)`.
Predict: the one-token unknown message has count zero and is rejected.
Interpretation: unknown vocabulary is still observed input.

### C. Increase embedding width

Change 4 to 6 and update the single expected output-shape assertion.
Predict: masks remain `[B,T,1]`, counts remain `[B,1]`, output becomes `[B,6]`.
The masking algorithm should not need feature-specific logic.
Restore the completed source after the experiments.

## Checkpoint questions

<details>
<summary>Why is denominator shape [B,1], rather than [B]?</summary>

Broadcasting aligns dimensions from the right.
`[B,1]` divides every feature in one row by that row's count.
`[B]` can instead mismatch D, or accidentally align with features when B equals D.
</details>

<details>
<summary>Why reject all-padding input instead of returning a zero vector?</summary>

A zero vector would conflate invalid empty input with a legitimate learned mean.
Rejecting it makes the caller's missing-input policy explicit.
An intentional empty-document token would be a different, teachable design.
</details>

<details>
<summary>Does padding invariance prove the representation understands word order?</summary>

No. A mean is invariant to permutations of its real vectors too.
Padding invariance is desirable; order invariance may be an unwanted limitation.
Session 41 isolates that distinction.
</details>

## Design judgment

Masked means are inexpensive and useful baselines for variable-length inputs.
They remove one batching bug without solving language understanding.
The fixture and tokenizer remain synthetic and intentionally limited.
Next, put a binary logit head on this tested representation.
