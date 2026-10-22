# Session 27 — An embedding is a learned lookup table

[Course map](../../CURRICULUM.md) · Project 7: structured records

## Problem and prior knowledge

In [Session 26](../26-tabular-preprocess/README.md), categories became one-hot
columns rather than fake numerical magnitudes.
Now imagine many equipment types and a small numerical measurement per record.
Can each type select a short trainable vector instead of a long sparse vector?

This **synthetic regression** exercise has eight known types.
Each type changes an offset and the slope of a continuous input.
The task is deliberately small enough to inspect a lookup and its gradients.
It is not evidence that embeddings beat other models on real tabular data.

You already know `nn.Module`, MLPs, MSE, scaling, dictionaries, category IDs,
and the train/validation/test separation.
The new ideas are embedding tables, repeated-index gradients, embedding dimension,
and the difference between an unknown category and padding.

## Run and inspect the full source

From the course root:

```bash
.venv/bin/python lessons/27-embeddings/train.py
```

The complete source is [train.py](train.py); no previous script must run first.
Everything is generated in memory with a local seeded `torch.Generator`.
Only Python and PyTorch are required; there are no downloads or saved artifacts.
A NumPy-not-installed warning from this PyTorch build does not affect this code.
One CPU thread avoids tiny-operation thread-pool overhead.
It is not a measured speed claim or a prescription for larger workloads.

## 1. The synthetic rule and honest input construction

There are 480 records.
The first 320 train, the next 80 validate, and the last 80 test.
The scalar measurement is uniformly drawn from `[-2,2)`.
The synthetic target is approximately:

```text
target = 0.7 * measurement + category_offset
         + 0.2 * measurement * category_offset + small noise
```

Offsets range from `-2.1` to `+2.1` across the eight known types.
The interaction term makes a type change more than an intercept.
The model receives only the measurement and the type ID, never the hidden offset.
The numeric mean and standard deviation come from the 320 training measurements.
The vocabulary comes only from training names.
Later `new-type` records map to ID zero.
Their synthetic offset is zero, but the model is not given that fact.

## 2. Read the table as a matrix

`nn.Embedding(vocabulary_size + 1, dimension)` owns a parameter matrix:

```text
table.weight: (V + 1, D)
```

`V=8` known types, one reserved UNK row, and default `D=3`.
`table(ids)` selects one row per ID.
With `ids.shape == (B,)`, the result has shape `(B,D)`.
This is a differentiable parameter selection, not a new kind of target or loss.
The input IDs must be `torch.long`, and every ID must be a valid row index.

The source verifies the identity:

```python
dense_lookup = F.one_hot(ids, V + 1).float() @ model.table.weight
assert torch.allclose(dense_lookup, model.table(ids))
```

One-hot multiplication explains what a lookup means.
The lookup implementation avoids materializing the dense one-hot matrix merely
to select rows.
It does not avoid storing the `(V+1)*D` trainable table entries.

Category ID `7` is not seven times category ID `1`.
It is an address into a table, like an array index in C++ or C#.
Reordering IDs and consistently reordering table rows leaves predictions unchanged.
Reordering only the vocabulary breaks the model's inference contract.

## 3. Where do the gradients go?

Suppose IDs are `[1,2,1]` and we sum every coordinate returned by the lookup.
Each coordinate of row 1 appears twice, and each coordinate of row 2 once.
Therefore:

```text
gradient row 0 = [0,0]
gradient row 1 = [2,2]
gradient row 2 = [1,1]
```

The small `probe` near the start of `main()` checks exactly this accumulation.
Autograd adds contributions for repeated IDs; it does not overwrite the first.
In the actual model, gradients flow from MSE through the MLP head into selected
embedding rows.
A row never selected during training receives no data gradient.
This is why rare and unknown categories deserve explicit attention.

## 4. Follow the two alternative models

`EmbeddingRegressor` concatenates a scaled numeric value with the selected vector:

```text
(B,1) plus (B,3) -> (B,4) -> Linear(4,16) -> ReLU -> Linear(16,1)
```

`OneHotRegressor` concatenates the numeric value with eight known-type columns:

```text
(B,1) plus (B,8) -> (B,9) -> Linear(9,16) -> ReLU -> Linear(16,1)
```

The unknown one-hot column is removed, so UNK is all zeros.
For the embedding model, row zero is explicitly initialized to a zero vector.
Training contains no zero IDs, and this optimizer has no weight decay.
Consequently that row remains zero; a post-training assertion checks it.
It is not evidence of a learned general-purpose representation for unknowns.
If training included zero IDs or the optimizer configuration changed, revisit
that contract rather than assuming the row stays fixed.

Both models use 300 full-batch Adam updates with the same learning rate.
They share raw data, splits, scaling, loss, and reported metric.
The embeddings change the parameterization and optimization path.
This is a transparent comparison, not a proof that their effective capacities
are identical.

### First use of Adam

Earlier SGD updates used `parameter -= learning_rate * gradient`.
`torch.optim.Adam` keeps moving averages of the gradient and its square for each
parameter coordinate:

```text
m = beta1*m + (1-beta1)*gradient
v = beta2*v + (1-beta2)*gradient^2
parameter -= learning_rate * bias_corrected_m / (sqrt(bias_corrected_v) + epsilon)
```

Bias correction compensates for starting both moving averages at zero.
PyTorch defaults are `beta1=0.9`, `beta2=0.999`, and `epsilon=1e-8`.
The running squared gradient changes the effective step size per coordinate;
the first average also smooths recent directions.
It is a practical optimizer alternative, not a guarantee of convergence or
better held-out predictions.
The familiar `zero_grad()`, `backward()`, `step()` sequence stays unchanged.
An SGD learning rate is not automatically appropriate for Adam.
This script uses the default zero weight decay, which also matters to the
unselected UNK row's behavior.
You do not need to implement Adam to understand the embedding's gradient lookup.

## 5. Parameter counts and the bottleneck

At default dimension three:

| Component | Parameters |
|---|---:|
| One-hot MLP | `9*16 + 16 + 16*1 + 1 = 177` |
| Embedding table | `9*3 = 27` |
| Embedding head | `4*16 + 16 + 16*1 + 1 = 97` |
| Embedding model total | `124` |

A small embedding can share downstream computations across categories.
Its dimension limits the representation passed to the head.
For a tiny vocabulary or a very large dimension, the embedding route can have
**more** parameters instead.
In a purely linear head, composing an embedding and another linear map also
introduces a factorization, not automatically a more expressive predictor.

The printed vectors do not have named coordinates such as “size” or “quality.”
Different initializations can rotate or otherwise change their representation
while retaining similar predictions.
Nearby vectors are not automatically a validated semantic taxonomy.

## 6. Shapes and output interpretation

| Object | Contract |
|---|---|
| `numeric`, `target` | `(480,1)`, float32 |
| `ids` | `(480,)`, int64; zero means unknown |
| table | `(9,D)`, trainable float32 parameters |
| embedding output | `(B,D)` |
| predictions | `(B,1)` |
| training criterion | MSE on training records only |
| report | MAE in the synthetic target's units |

A reference run reports about `0.083` validation MAE for one-hot and `0.084`
for embeddings.
That near tie is a valid result, not a failure to demonstrate deep learning.
For the first unknown record, one-hot predicts approximately `-1.76`, embeddings
approximately `-1.15`, and the actual value is `-1.39`.
The fallback is legal in both pipelines, but neither has learned this category.
Small backend/version differences may change the final decimals.

The script prints train, validation, and test diagnostics for fixed recipes.
Do not use the test reports to decide an embedding dimension.
The final-test gating introduced in later selection exercises makes that
separation more explicit.

## 7. Experiments with a prediction before execution

```bash
.venv/bin/python lessons/27-embeddings/train.py --embedding-dim 1
.venv/bin/python lessons/27-embeddings/train.py --embedding-dim 12
.venv/bin/python lessons/27-embeddings/train.py --steps 10
```

Predict that dimension one reduces representation size, but may or may not hurt
this particular scalar-target problem.
Dimension twelve increases parameter count; it does not create new evidence.
Ten updates should reveal incomplete fitting, not a different data distribution.
For each run, compare **validation** MAE, parameter count, and an unknown prediction.
Do not explain a small MAE difference as meaningful without considering noise and
the limited sample size.

## Checkpoint questions

1. Does an embedding require a special loss?
2. Why must two occurrences of ID 1 add their gradients?
3. Is ID zero a padded timestep in this example?
4. Why must a saved table retain its vocabulary and numeric scaler?

<details>
<summary>Explained solutions</summary>

1. No. The downstream task still determines the loss; this scalar regression uses
   MSE. Embeddings simply supply trainable inputs to later layers.
2. Both uses affect the same parameter row, so the chain rule sums their effects.
3. No. UNK means an actual record has an unsupported category. Padding means
   an artificial position inserted to equalize sequence lengths. This table has
   no padded sequences and deliberately does not set `padding_idx`.
   Later text lessons will introduce a distinct PAD ID and masking.
4. Table row addresses have no meaning without the exact category mapping.
   Numeric preprocessing must also reproduce the input distribution expected by
   the learned head.

</details>

## Design limitations

The example has independent records, few categories, and synthetic clean labels.
There is no evidence that a larger table, a deeper MLP, or embeddings themselves
are necessary.
Unknown and rare types need a validated cold-start policy.
Next, [Session 28](../28-group-splits/README.md) asks whether the records are
independent at all: repeated entities can make an impressive result misleading.
