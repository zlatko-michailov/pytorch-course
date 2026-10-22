# Project 46 / Session 46 — Compare inductive biases fairly

## Prerequisites

Complete sessions 38–45.
Review train-only vocabularies, validation checkpoint selection, positions, masks, and residual blocks.
You should be able to explain the exact order collision in session 41.

**Review:** the mean baseline and the explicit encoder computation.
**New integration:** a controlled architecture comparison with held-out text templates.
The hypothesis is that position-aware attention can represent actor/patient order.
The experiment must not assume that added capacity guarantees better held-out results.

## The task

Synthetic templates describe a cat or dog chasing the other animal.
Label 1 means the cat is the actor.
Label 0 means the dog is the actor.
Each template is instantiated with both assignments.

For example:

```text
today cat chases dog -> 1
today dog chases cat -> 0
```

Within each pair, the token multiset is identical.
A masked-mean embedding classifier must collide on the pair.
An order-aware model has access to a distinguishing signal but still has to learn it.

These are authored toy grammar strings, not human-language benchmark examples.
Perfect accuracy cannot establish broad compositional understanding.
The small number of templates also gives weak statistical evidence.

## Run

From the course root:

```bash
.venv/bin/python -m lessons.46-attention-comparison.train
```

Complete experiment: [train.py](train.py).
No previous artifact is required.
All data, models, and selection rules are inspectable in one file.
Execution uses CPU PyTorch only.

## Split by templates, not by rendered rows

`TEMPLATES` explicitly assigns string patterns to training, validation, and test.
`rows(split)` renders both labels for every pattern.
Training has six templates and twelve examples.
Validation has two templates and four examples.
Final test has two other templates and four examples.

Splitting the rendered rows randomly could put near-identical template forms in both partitions.
Here an entire template placement is held out.
This is still not an independent natural-language sampling process.
Some held-out strings are very close to training strings by design.

The vocabulary is fitted only from rendered training text.
All fixture words happen to occur there.
Held-out templates test their arrangement, not an unknown-word policy.
PAD and UNK remain distinct reserved IDs.

## Shared training contract

Both models receive exactly the same train and validation tensors.
Both use BCE with logits and Adam with learning rate 0.02.
Both receive 240 full-batch optimizer updates.
Both select their stored state by the lowest validation BCE.
Both use the same seed before construction.

The checkpoint dictionary clones detached tensors, as in session 40.
The evaluation helper enters evaluation mode and disables gradient recording.
Training re-enters training mode at every step.
The final-test tensor is constructed only after both fits are complete.

This matches data access and update budget.
It does **not** match parameter count or total floating-point work.
The script prints parameter counts to make that tradeoff visible.
Equal seeds also do not imply identical initial functions across different architectures.

## Mean model path

Embedding lookup produces `[B,T,16]`.
The key-independent mask excludes PAD cells from numerator and count.
The pooled representation is `[B,16]`.
A linear head returns logits `[B]`.

This baseline is not handicapped by a missing optimizer or frozen embeddings.
Its failure on minimal pairs follows from the representation.
Adding extra hidden layers after the mean would not remove that collision.
It remains a valuable low-cost baseline when order does not matter.

## Attention model path

The embedding lookup has the same width and vocabulary.
A fixed sinusoidal `[T,16]` table is added before attention.
The table's construction is the one taught in session 44.
It supports all default sequence lengths without an untrained learned-position row.

The block repeats session 45's explicit pre-normalization pattern.
QKV projects 16 features to 48, then splits into three 16-feature tensors.
Two heads each have width 8.
Scores have shape `[B,2,T,T]`.
The scale is `sqrt(8)`.

Padding keys are masked before the bidirectional softmax.
No causal mask is appropriate for this whole-message classification task.
The attention branch and token-local feed-forward branch each have a residual addition.
The feed-forward widths are 16 -> 32 -> 16.

The final masked mean pools **contextual, position-aware states**, not raw token embeddings.
That distinction is how a mean readout can retain an order-dependent result.
The output head is still a single binary logit.

## Assertions and interpretation

On the first attention backward pass, the QKV weights must receive nonzero gradients.
After training, appending four PAD columns must preserve each model's predictions.
Replacing the padding embedding vector by 1000 must also leave those predictions unchanged.
The saved padding vector is restored after that perturbation check.
The mean model's logits for both members of every test pair must remain close.
No assertion demands that attention beat the baseline.

In the verified CPU run:

```text
mean:                 129 parameters, final accuracy 0.50, BCE about 0.6931
attention+positions: 2353 parameters, final accuracy 1.00, very small BCE
```

Those are observations for this fixed toy fixture and environment.
The validation-selected steps were 126 and 38 respectively.
A step number after a near-flat mean loss can depend on floating-point rounding.
Do not attach significance to which tiny numerical minimum it selected.

## Controlled experiments

### A. Remove positions from attention

Predict: self-attention remains permutation-equivariant.
The masked mean then restores an invariant representation, so paired examples collide.
If a numerical difference appears, investigate it rather than calling it language learning.

### B. Reduce the update budget equally

Try 20 updates for **both** models.
Predict: the mean collision remains; attention may or may not optimize sufficiently.
Separate representational ability from the optimization budget needed to use it.

### C. Make order irrelevant

Design a new predeclared fixture where a signal token, regardless of position, determines the label.
Predict: mean pooling can become competitive with less compute and fewer parameters.
Do not reuse the original final test to decide which altered task/model looks best.

Treat all exploratory results as development.
Once you have inspected this test, it cannot support a fresh untouched-test claim.
Use a newly frozen evaluation design for a new final experiment.

## Checkpoint questions

<details>
<summary>Why can mean pooling after attention work when mean pooling before attention cannot?</summary>

Attention with positions changes each token state using ordered contextual information.
The states being averaged are no longer merely a reordered copy of fixed token embeddings.
The downstream mean can therefore differ for actor/patient reversals.
</details>

<details>
<summary>Is this a parameter-matched benchmark?</summary>

No. It matches splits, optimizer settings, update count, and checkpoint criterion.
Attention has more parameters and more work per update.
The printed counts and explicit limitation prevent claiming an equal-compute comparison.
</details>

<details>
<summary>What if attention performs worse on a modified held-out template set?</summary>

Report that result.
Representational capacity does not guarantee learning the intended invariant from twelve examples.
More development data or a better task-specific bias may be preferable to a larger model.
</details>

## Design judgment

Use a simple baseline, a target-dependent hypothesis, and a frozen comparison protocol.
Do not select an architecture because its diagram is more fashionable.
Next, turn common pipeline mistakes into explicit debugging contracts.
