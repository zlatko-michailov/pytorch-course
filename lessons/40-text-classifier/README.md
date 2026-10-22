# Project 40 / Session 40 — A complete tiny text classifier

## Prerequisites

Complete sessions 38–39 and review binary logits, BCE, validation selection, and `state_dict`.
You should be able to explain why PAD is excluded from a mean but UNK is included.
You should also distinguish a validation-selected checkpoint from a final test score.

**Review:** embeddings, masked means, Adam, binary cross-entropy with logits.
**New integration:** raw text through fitted IDs to a trained and selected classifier.
The individual components are familiar; their contracts must remain consistent together.

## Problem and honest scope

Classify the authored synthetic service messages introduced in session 38.
There are exactly 24 training examples, 8 validation examples, and 8 final-test examples.
Every example contains one deliberately obvious positive or negative signal word.
The fixture is balanced and has no realistic annotation disagreement.

**High accuracy here is not evidence of real NLP quality.**
The model has not learned robust negation, sarcasm, domain adaptation, or conversational context.
The held-out templates are few and closely related to training templates.
This is an integration exercise with inspectable data, not a product benchmark.

## Run from the course root

```bash
.venv/bin/python -m lessons.40-text-classifier.train
```

Complete working source: [train.py](train.py).
Fixture and vocabulary source:
[`course/text_data.py`](../../course/text_data.py).
No previous checkpoint is required.
The script fits its vocabulary and model from the fixed training partition each time.

## Data flow

```text
training strings -> fitted vocabulary
any split's strings -> IDs [B,T]
IDs -> embeddings [B,T,8]
masked mean -> features [B,8]
linear output -> logits [B]
logits + float labels [B] -> scalar BCE
```

The vocabulary is never extended during validation or test encoding.
Validation's `weekend` and test's `evening` therefore map to UNK.
PAD remains a separate ID used only for rectangular collation.

The collation function is repeated locally to keep the training script readable.
It is the session-39 operation plus label construction.
The labels are `float32`, because this is binary BCE rather than integer-index cross-entropy.
All examples in one split form one full batch; there is no DataLoader machinery to hide.

## Model walkthrough

`MeanClassifier` derives from `nn.Module`.
`super().__init__()` initializes PyTorch's parameter/module tracking.
Assigning the embedding and linear layer as attributes registers their parameters.

The embedding table has `[vocabulary_size, 8]` trainable entries.
The padding row is designated by `padding_idx`.
The UNK row is explicitly initialized to zero inside `torch.no_grad()`.
That context allows an initialization write without making it part of a gradient graph.

No UNK token occurs in training in this fixture.
Its embedding consequently receives no lookup gradient during training.
Starting it at zero gives unseen words a neutral numerator contribution rather than a random one.
It still counts in the masked-mean denominator.
This is a documented fallback, not a learned meaning for all unknown words.

`forward(ids)` builds the mask and checks that every row has a real token.
The mask broadcasts from `[B,T,1]` over `[B,T,8]`.
The mean has shape `[B,8]`.
`nn.Linear(8,1)` yields `[B,1]`.
`.squeeze(-1)` removes only that final singleton dimension, leaving `[B]` even for B=1.

The method returns **logits**, not sigmoid probabilities.
`binary_cross_entropy_with_logits` combines the stable sigmoid/log-loss calculation.
For accuracy, a logit threshold of zero equals a probability threshold of 0.5.

## Training and selection walkthrough

The seed is fixed before constructing the model.
One CPU thread avoids thread-management overhead for this tiny workload.
Adam uses learning rate 0.04 for 100 full-batch updates.
These are educational fixed settings, not a broad hyperparameter search.

Each epoch explicitly enters training mode.
`zero_grad(set_to_none=True)` clears gradients from the previous update.
The scalar training loss is differentiated.
`optimizer.step()` changes parameters using those gradients.

Validation is evaluated after each update.
`evaluate` has a `@torch.no_grad()` decorator and calls `model.eval()`.
The decorator is equivalent to wrapping the function body in a no-gradient context.
Evaluation mode and gradient recording are distinct mechanisms.
This model has no dropout, but establishing the correct pattern matters.

Only validation BCE chooses `best_epoch`.
When it improves, every state tensor is detached and cloned.
Detaching removes gradient history; cloning creates independent storage.
Without the clone, an alleged “best state” could continue changing with later optimizer steps.

The best state is restored after training.
Then train and validation summaries are printed.
Only afterward is the final-test batch constructed and evaluated.
The test loss never enters the epoch-selection condition.

## Read the output

The script prints the selected epoch and fitted vocabulary size.
Each score includes BCE and accuracy.
BCE uses the confidence of predictions, while accuracy uses only their side of a threshold.
A model can keep reducing BCE after accuracy reaches 100%.

The last assertion appends five PAD columns to test IDs.
Predictions must remain close to the unpadded predictions.
It also fills the padding embedding row with 1000 and checks the padded predictions again.
The original padding row is then restored.
This verifies exclusion of padding content, not just reliance on an initially zero vector.
This protects the connection between the representation test in 39 and the trained head.
The assertion does not establish robustness to new words or new distributions.

## Controlled experiments

### A. Make the embedding smaller

Try width 2 instead of 8 in both the embedding and output layer.
Predict: the script still runs with the same outer `[B]` logit contract.
The toy task may remain easy because one polarity direction is sufficient.
Use validation, not repeated final-test comparisons, for any design decision.

### B. Remove the unknown-row zero initialization

Predict: unknown held-out words contribute a random, untrained vector.
Training accuracy need not change because training contains no unknown tokens.
Validation behavior can change; that is a preprocessing/fallback issue, not new supervision.

### C. Omit `.clone()` in the saved state

Before running, predict what a storage-sharing checkpoint would mean.
Then inspect whether the supposedly saved tensor changes after another optimizer update.
A final good score does not prove the snapshotting implementation was correct.

For exploratory reruns, regard the original final test as already spent.
Do not keep adjusting the model until that same tiny test looks best.
Use a fresh predeclared evaluation for any new final claim.

## Checkpoint questions

<details>
<summary>Why select with BCE when the user may care about accuracy?</summary>

BCE supplies a graded probability-sensitive criterion rather than eight coarse decisions.
That makes this small demonstration easier to inspect, not statistically reliable.
In a real project, choose the selection metric from the intended decision costs.
</details>

<details>
<summary>Why do we not fit a separate validation vocabulary?</summary>

Embedding rows are shared model parameters with fixed meanings.
A separate ID assignment would feed the wrong word's embedding to the model.
Use the fitted training vocabulary and its documented unknown policy.
</details>

<details>
<summary>Does a zero UNK vector mean unknown words are ignored?</summary>

Only their numerator contribution starts at zero here.
They remain real tokens and increase the masked-mean count.
They can still alter feature magnitudes and therefore predictions.
</details>

## Design judgment and next step

This simple baseline is cheap, debuggable, and sometimes useful.
It represents the multiset of tokens, not the sequence of roles.
Before replacing it with a larger network, demonstrate a task it cannot represent.
That is the purpose of session 41.
