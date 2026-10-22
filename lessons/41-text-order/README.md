# Project 41 / Session 41 — When a representation makes success impossible

## Prerequisites

Complete session 40 and review masked means and binary-logit training.
You should know that changing an optimizer cannot recover information discarded by preprocessing.
You do not need any attention or Transformer knowledge yet.

**Review:** embedding lookup, mean pooling, BCE, optimizer updates.
**New:** a representational collision, permutation invariance, an order-sensitive counterexample.
This session is a deliberately constructed impossibility demonstration.
Its failure to separate the labels is the expected successful result.

## The problem

Consider these two synthetic strings:

```text
cat chases dog  -> 1: cat is the actor
dog chases cat  -> 0: dog is the actor
```

They have the same tokens with the same counts.
Only order distinguishes who performs the action.
This is a role-classification toy rule, not a claim that this grammar captures real language.
The labels are explicitly defined by us.

Let the embeddings be `e_cat`, `e_chases`, and `e_dog`.
Both means equal:

```text
(e_cat + e_chases + e_dog) / 3
```

Addition is mathematically commutative.
Any classifier applied only to that mean receives the same input for both labels.
No depth, optimizer, or extra epochs in the downstream classifier can undo this collision.

## Run

From the course root:

```bash
.venv/bin/python -m lessons.41-text-order.train
```

Complete source: [train.py](train.py).
The script is self-contained and needs no previous artifacts.
It trains the mean model and then checks the collision explicitly.
It exits successfully when the limitation is demonstrated.

## Source walkthrough

The vocabulary is written literally because only three ordinary tokens are needed.
Reserved IDs remain present for continuity with sessions 38–40.
There is no held-out vocabulary or statistical evaluation in this two-example proof.
No padding is needed because both sequences have length 3.

The list comprehension builds ID rows in string order.
`ids` has shape `[2,3]` and integer dtype.
`labels` has shape `[2]` and floating dtype.

`nn.Embedding(V,6)` maps the IDs to `[2,3,6]`.
`.mean(1)` removes the token axis, producing `[2,6]`.
`nn.Linear(6,1)` maps each mean to one logit.
`.squeeze(-1)` gives the `[2]` shape required by BCE.

Both the embedding table and output layer are optimized.
`list(embedding.parameters()) + list(classifier.parameters())`
creates one parameter list for Adam.
This matters: the demonstration does not freeze embeddings to manufacture failure.
Even trainable embeddings cannot make the same multiset yield two different exact means.

The loop performs 200 full-batch Adam updates.
It uses the familiar zero-gradient, forward, loss, backward, step sequence.
One CPU thread keeps the tiny loop inexpensive.
No random minibatch order is involved.

After training, a no-gradient block recomputes the means and logits.
`assert_close` checks that both rows collide numerically.
Floating-point addition may differ at the last few bits under a permutation.
Those rounding differences are not a reliable semantic order channel.

The script also computes probabilities with `sigmoid`.
These are printed for interpretation, not passed back into BCE.
The fixed threshold predicts the same class for both rows.
With one example of each class, accuracy is therefore 50%.

## The best possible shared probability

If the shared predicted probability is `p`, the average binary loss is:

```text
L(p) = -(log(p) + log(1-p)) / 2
```

It is minimized at `p = 0.5`.
The minimum is `log(2)`, approximately 0.693147.
The script prints that reference beside the trained loss.

This is not evidence that BCE is broken.
BCE correctly expresses uncertainty when an identical representation has conflicting labels.
The optimization problem is satisfiable only in the shared-probability sense.
The classification problem is not separable in this representation.

## A minimal order-sensitive counterexample

The last part uses `.flatten(1)` on the token embeddings.
Starting from `[2,3,6]`, it merges dimensions 1 onward into `[2,18]`.
The first six values represent position 0, the next six position 1, and so on.
Exchanging `cat` and `dog` now exchanges different feature slots.
The two flattened vectors differ.

This is a completed counterexample, not a proposed general text architecture.
It shows that preserving positions can remove this specific collision.
It also requires a fixed token count or an additional padding/truncation policy.
The output layer's size would grow with the maximum sequence length.
Attention will offer another way to mix token information without flattening all slots.

## Controlled experiments

### A. Increase training to 2,000 updates

Predict: the means still collide and the shared probability approaches or stays near 0.5.
Any tiny rounding-driven difference is not a meaningful learned role representation.
Interpret this as a representation limit rather than insufficient patience.

### B. Add more layers after the mean

Replace the linear classifier with a small ReLU MLP.
Predict: identical mean inputs still produce identical outputs in deterministic evaluation.
A more expressive function cannot distinguish two exactly identical arguments.

### C. Change one token, not its position

Change the second string to `dog chases mouse` and add `mouse` to the vocabulary.
Predict: the algebraic collision no longer follows because the multisets differ.
The model might now separate the examples for the wrong reason.
Distinguish “can fit this pair” from “understands actor/patient order.”

### D. Repeat an ambiguous pair unevenly

Conceptually repeat label 1 three times and label 0 once for the same mean.
Predict: the optimal shared probability becomes 0.75 rather than 0.5.
It still cannot assign different probabilities based on order.
This illustrates how class prevalence can hide a representation failure behind higher accuracy.

## Checkpoint questions

<details>
<summary>Would replacing the mean with a sum solve this pair?</summary>

No. Both strings contain exactly the same tokens with the same counts.
The sum is also permutation-invariant.
Changing normalization does not restore discarded order information.
</details>

<details>
<summary>Could an embedding learn a different cat vector at each position?</summary>

Not with the current lookup: it is indexed only by token ID.
A position-dependent representation needs position information as an additional input or operation.
We have not supplied that information to this mean model.
</details>

<details>
<summary>Does this prove attention will generalize well?</summary>

No. It proves one particular mean representation cannot separate one specific pair.
An order-aware architecture still needs suitable data, optimization, and evaluation.
Session 46 will compare alternatives without assuming a guaranteed gain.
</details>

## Design judgment

Inspect whether the target depends on information the representation discards.
Do that before spending a larger compute budget.
A mean can be appropriate when order is irrelevant or the task tolerates its loss.
Next, study attention first as an explicit weighted-sum calculation.
