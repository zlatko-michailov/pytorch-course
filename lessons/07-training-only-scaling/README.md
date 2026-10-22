# Session 7: scaling is a fitted part of the model

Project 2, lesson 4 of 6.

**Prerequisite:** [Session 6](../06-multiple-features/README.md).
**New:** column reductions, standardization, constant-feature handling, and a
preprocessing contract fitted using training rows only.

## Same information, very different numerical scale

The second input is now measured in units roughly 1,000 times larger. Its
generating weight is correspondingly smaller. Predictions can remain well scaled
while gradients in different weight directions have drastically different scales.
Session 2 showed why curvature affects stable learning rates.

We transform each feature separately:

```text
z_j = (x_j - training_mean_j) / training_scale_j
```

The model receives `z`, not raw `x`. This is not extra information or an
additional learned neural layer. It is a deterministic transformation whose
statistics must be saved and applied identically to later inputs.

We use population standard deviation (`correction=0`) for this transformation;
we are not trying to construct an unbiased sample-variance estimator.

## Run and read

[Complete source](train.py)

```sh
.venv/bin/python lessons/07-training-only-scaling/train.py
```

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 2 | Import the layer namespace. |
| 4 | Seed this controlled experiment. |
| 5 | Scale two random feature columns differently using broadcasting of `[2]` across `[120,2]`. |
| 6 | Generate the affine signal in original units. |
| 7 | Add noise to observations in place. |
| 8 | Separate independent feature rows into training and validation. |
| 9 | Apply the same split to targets. |
| 10 | `mean(dim=0)` reduces the row axis, leaving one mean per feature, shape `[2]`. |
| 11 | Calculate one training-only population standard deviation per feature. |
| 12 | `where` chooses 1 for constant-feature scales, avoiding division by zero; a centered constant feature then becomes zero. |
| 13 | Standardize training rows with training statistics. |
| 14 | Apply **the same** statistics to validation rows. |
| 15 | Create the affine layer over standardized inputs. |
| 16 | Configure SGD. |
| 18 | Repeat updates. |
| 19 | Clear gradients. |
| 20 | Compute MSE on standardized training features and unscaled targets. |
| 21 | Calculate gradients. |
| 22 | Update parameters. |
| 24 | Disable recording for diagnostics. |
| 25 | Calculate validation loss under the same transform. |
| 26 | Convert learned normalized coefficients to raw-unit coefficients by dividing by each feature scale. |
| 27 | Convert the bias: expanding `w*(x-mean)/scale+b` subtracts `sum(raw_weight*mean)`. |
| 29 | Print the fitted preprocessing statistics. |
| 30 | Show training column means are approximately zero after transformation. |
| 31 | Display recovered raw-unit coefficients and bias. |
| 32 | Display held-out validation error. |

The constant-feature rule is explicit, not a silent fallback for arbitrary bad
inputs. This synthetic dataset contains finite values. Handling missing values
and rejecting malformed inputs will be taught separately.

## Why validation gets the same transform

Fitting statistics on validation/test inputs would use information outside
training. Even unsupervised preprocessing is fitted information.
Normalizing each new batch independently would also change the function at
inference: an example's prediction could depend on its batch companions.

Do not expect validation column means to be exactly zero. Their difference from
training means is part of the data, not an error to normalize away.

## Experiments

1. Multiply the second feature's scale by another ten and divide its generating
   coefficient by ten. Predict whether standardized training stays well behaved.
2. Before calculating statistics, replace the second training column with a
   constant. Inspect its standardized values and discuss information loss.
3. For a deliberately incorrect comparison, normalize validation with its own
   means/scales. Explain why a nice score cannot justify the changed contract.
4. Remove scaling while retaining rate 0.1. Monitor `torch.isfinite(loss)`;
   large gradients can produce divergence. Stop the experiment if nonfinite
   values appear; do not hide them or replace them with zeros.

## Checkpoint

Why do normalized coefficients differ from raw coefficients? Must targets also
be standardized? What must accompany weights when saving this model?

<details>
<summary>Worked answers</summary>

Weights multiply transformed units, so scaling changes their numerical
interpretation. Target standardization is optional and would add an inverse
output transform; we do not use it here. Save the training means/scales, feature
order and units, architecture, and parameters. Weights alone no longer specify
raw-input predictions.

</details>

[Next: mini-batches](../08-mini-batches/README.md). [Course map](../../CURRICULUM.md).
