# Project 44 / Session 44 — Positions and visibility are different contracts

## Prerequisites

Complete sessions 41–43.
Review the mean-model order collision and the query/key axes of attention scores.
You should know the difference between a real unknown token and a padding cell.

**Review:** scaled scores, row softmax, boolean masks, broadcasting.
**New:** sinusoidal positions, key-padding masks, causal masks, all-masked-row safety.
This session uses one head to make the visibility logic easy to inspect.
The same masks can broadcast across heads.

## Two separate problems

Without positions, the attention calculation has no explicit first/second/third-slot signal.
Without a padding mask, artificial batch cells can become keys and affect real outputs.
Neither problem is fixed merely by solving the other.

A third choice is task visibility.
A sentence classifier can normally use the entire input bidirectionally.
A next-token predictor must not see later tokens when predicting the current continuation.
Causal masking enforces that restriction.
It is not a universally “better” attention mode.

## Run

From the course root:

```bash
.venv/bin/python -m lessons.44-positions-masks.train
```

Complete implementation and checks: [train.py](train.py).
This script deliberately includes both right padding and a leading padding position.
The leading-padding case exposes a common all-masked causal-row bug.
No training or external data is required.

## Position construction

`positions(length, width)` returns a `[T,D]` fixed table.
The implementation requires even D so features can be organized in sine/cosine pairs.
`torch.arange(T).unsqueeze(1)` has shape `[T,1]`.
It stores position indices 0 through T-1.

The frequency vector has shape `[D/2]`.
Its entries decay exponentially according to the standard 10000-based scale.
Multiplying `[T,1]` by `[D/2]` broadcasts to `[T,D/2]`.
Even columns receive sine; odd columns receive cosine.

This gives every position a deterministic feature pattern at multiple frequencies.
Sine/cosine values are bounded and do not require a fitted maximum-length lookup table.
The network still has to learn how to use them.
Their existence does not guarantee generalization to longer sequences.

`z = x + positions(T,D)` broadcasts the table across the batch.
Both operands describe the same D feature coordinates.
Position is added **before** Q/K-like matching and value mixing.
Here, for inspection, z itself plays the Q, K, and V roles.
Learned projections return in session 45.

## Key-padding mask

`valid` is a boolean `[B,T]` tensor.
`True` means a real token.
`valid[:,None,:]` creates `[B,1,T]`.
Expanding it gives an allowed-key matrix `[B,query_T,key_T]`.

Every real query may read only keys whose `valid` entry is true.
The mask applies to the final **key** axis of scores.
Masking only query rows would leave padded keys available to real tokens.
Adding positions makes this especially dangerous: a padded cell need not be a zero vector.

Invalid score cells are filled with negative infinity before softmax.
For a row with at least one allowed key, those invalid cells receive exact zero weight.
The remaining allowed weights sum to one.

## Causal visibility

`torch.ones(T,T,dtype=torch.bool).tril()` keeps the lower triangle, including the diagonal.
Row i can read columns j where `j <= i`.
The diagonal allows a real token to read itself.
Combining this mask with key validity uses boolean `&`.

Bidirectional mode applies only key padding.
Causal mode applies key padding **and** the lower triangle.
The script verifies that causal weights above the diagonal are zero.
It also changes a later real token and checks earlier real outputs remain unchanged.
The corresponding bidirectional outputs are allowed to change and demonstrably do.

## Why an all-masked softmax is dangerous

A leading padding query in causal mode may have no valid earlier key.
Its score row would otherwise be all negative infinity.
Softmax subtracts a row maximum internally; `-inf - -inf` is not finite.
Multiplying the resulting NaN output by zero afterward does not repair it.

`has_key = allowed.any(-1, keepdim=True)` has shape `[B,T,1]`.
For a row with no allowed key, `torch.where` substitutes finite zero scores before softmax.
The resulting weights are then multiplied by `allowed`, making that row all zeros.
This preserves zero contribution while keeping the computation finite.

For real queries, the diagonal guarantees at least one key in causal mode.
For padded queries, outputs are explicitly zeroed with `valid.unsqueeze(-1)`.
The function separately rejects a sequence containing no real tokens at all.
Avoiding NaNs is not permission to accept an undefined empty example.

## Behavioral assertions

The script changes all padding vectors to the large value 1000.
Real outputs must remain unchanged in both attention modes.
It also appends three additional right-padding positions.
Existing outputs must remain close to their original values.

All outputs and weights must be finite.
Padding outputs must be zero.
Padding keys must receive zero weights.
The leading-padding causal row must not introduce NaNs.
An all-padding example must raise a specific `ValueError`.

Right-padding invariance does not imply left-padding invariance with these absolute positions.
Inserting tokens at the beginning shifts existing position indices.
Use a consistent padding/position policy or explicitly compute position IDs from valid tokens.
Our script intentionally tests right-padding invariance only.

## Controlled experiments

### A. Remove key-padding masking

Predict: replacing padded vectors with 1000 changes real-token outputs.
Extra padding can change the denominator of the attention softmax as well.
The invariance assertion should expose the bug.

### B. Remove the finite all-masked-row branch

Predict: the leading-padding causal query produces non-finite weights.
Zeroing its final output is too late.
Restore the branch rather than globally replacing all NaNs after the fact.

### C. Disable the causal triangle

Predict: earlier outputs can respond to changed future tokens.
That is valid for a bidirectional classifier and invalid for a causal prediction contract.
Explain which task you are implementing before judging the behavior.

### D. Remove the positional addition

Predict: the visibility checks can still pass.
Mask correctness and availability of order information are independent properties.
An order task can still fail despite perfect padding isolation.

## Checkpoint questions

<details>
<summary>Why not set padded token embeddings to zero and skip masking?</summary>

Zero keys still produce scores and occupy probability mass in softmax.
Position addition and projection biases can also make padded states nonzero.
Only explicit visibility masking excludes those candidates.
</details>

<details>
<summary>Does a causal mask exclude the current token?</summary>

Not in this implementation: the diagonal is included.
For next-token language training, the target is shifted relative to the input.
Visibility and target alignment must be designed together; changing one is not a substitute for the other.
</details>

<details>
<summary>Why can bidirectional attention be correct for this course's text classifier?</summary>

The whole message is available when assigning its label.
There is no promise to predict a token before later input arrives.
Restricting to the past would discard permitted information without a task-based reason.
</details>

## Design judgment

Masks encode allowed dependencies, not just implementation details.
Test those dependencies by changing forbidden inputs and observing the outputs.
Next, put the tested attention operation inside a trainable residual block.
