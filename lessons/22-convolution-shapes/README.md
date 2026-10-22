# Project 6, Session 22: locality, shared kernels, and shapes

[Course curriculum](../../CURRICULUM.md) · [Complete train.py](train.py)

## Prerequisites and new concepts

Review Sessions 18–21: image axes, the flattened MLP, multiclass logits, and
validation-first comparisons. Review Session 14's composition and ReLU.
The tabular project is not a prerequisite.

**New:** convolutional kernels, local receptive fields, parameter sharing,
feature channels, stride, padding, max pooling, and spatial shape arithmetic.
This is a deterministic shape demonstration, not a classifier training run.

## The problem with position-specific weights

The MLP's first layer treats pixel `(3,4)` and pixel `(4,4)` as unrelated inputs.
If the same edge moves one pixel, another set of weights must recognize it.
Images often contain local patterns that are useful in many positions.
A convolution applies the same small learned pattern detector at each position.
That sharing is a modeling assumption, not merely an optimization trick.

For one output channel, a 3×3 kernel multiplies a local 3×3 patch and sums.
With multiple input channels, it also sums across input channels.
Each output channel has a separate kernel and bias, producing a feature map.
PyTorch's operation is cross-correlation in mathematical terminology: it does not
reverse the kernel. Neural-network literature conventionally calls it convolution.

## Run from the course root

```sh
.venv/bin/python -m lessons.22-convolution-shapes.train
```

The course checker may equivalently run:

```sh
.venv/bin/python -m lessons.22-convolution-shapes.train --smoke
```

Both commands perform the same deterministic shape demonstration.
No data file, torchvision, or download is needed.
The constructed tensor is a numerical probe, not Fashion-MNIST.
The code checks a known local patch sum, traces every layer, and counts parameters.
This command makes no model-quality claim.

## Derive the shapes before reading the printout

For one spatial dimension with input size `H`, kernel size `K`, padding `P`,
stride `S`, and dilation `D`:

```text
H_out = floor((H + 2*P - D*(K-1) - 1) / S + 1)
```

All convolutions here use dilation 1 and stride 1.
Padding 1 with a 3×3 kernel preserves width and height.
Zero padding lets a kernel overlap an image boundary; those zeros are in the
normalized numerical domain, not necessarily raw black pixels.
Max pooling with kernel 2 and default stride 2 reduces each spatial size by half.

| Operation | Output shape for batch N | Learned parameters |
|---|---|---|
| Input | `[N,1,28,28]` | 0 |
| Conv 1→8, 3×3, padding 1 | `[N,8,28,28]` | `8*(1*3*3+1)=80` |
| ReLU | `[N,8,28,28]` | 0 |
| MaxPool 2 | `[N,8,14,14]` | 0 |
| Conv 8→16, 3×3, padding 1 | `[N,16,14,14]` | `16*(8*3*3+1)=1168` |
| ReLU | `[N,16,14,14]` | 0 |
| MaxPool 2 | `[N,16,7,7]` | 0 |
| Flatten from axis 1 | `[N,784]` | 0 |
| Dropout with p=0 | `[N,784]` | 0 |
| Linear 784→10 | `[N,10]` | `10*(784+1)=7850` |

Total: 9,098 parameters, compared with 50,890 for Session 19's MLP.
The spatial kernel has the same parameter count whether an image is 28×28 or
100×100. This particular final dense layer does not: its input shape is fixed.
Dropout is explicitly disabled with p=0 here; its behavior is taught in Session 25.

## Complete source and meaningful operations

Read [train.py](train.py) and the entire ten-line model factory
[image_models.py](../../course/image_models.py).
There are no omitted model blocks or pretrained weights.

`cnn(dropout=0.0)` returns an ordinary `nn.Sequential`, already familiar from the MLP.
The two `nn.Conv2d` calls specify input/output channels, kernel size, and padding.
Their learned weights have shapes `[8,1,3,3]` and `[16,8,3,3]`.
`nn.MaxPool2d(2)` takes the maximum in each non-overlapping 2×2 window by default.
It has no learned weights. It discards exact within-window position and values.
`nn.Flatten(start_dim=1)` preserves the batch dimension.
`nn.Dropout(dropout)` is an identity for the current default probability of zero.
`nn.Linear(16*7*7,10)` maps the final feature maps to ten raw logits.

In the entry point, the small `argparse` parser accepts the optional `--smoke`
flag so the course checker can use a uniform command. Neither mode loads data.
`arange(2*28*28,dtype=float32)` creates ascending probe values.
`.reshape(2,1,28,28)` forms a two-image batch; division keeps values moderate.
`current = images` initially binds a second name to the same tensor; it does not copy.
`for layer in model:` iterates through modules in Sequential order.
`current = layer(current)` computes the next intermediate tensor.
`type(layer).__name__` obtains a readable Python class name for the shape table.
`tuple(current.shape)` turns the shape object into a printable tuple.
The final assertion checks two examples and ten scores each.

`model.parameters()` yields parameter tensors, not outputs or buffers.
`p.numel()` returns the number of scalar elements in one tensor.
`sum(p.numel() for p in ...)` is a generator expression consumed by `sum`;
it is similar to a C# LINQ sum without allocating an intermediate list.
The assert protects the hand-derived 9,098 total.
We also instantiate the earlier MLP to compare counts, without training either.

The last probe isolates convolution from the rest of the architecture.
`nn.Conv2d(1,1,3,bias=False)` has a 3×3 kernel, no padding, and no bias.
Under `no_grad`, `kernel.weight.fill_(1.0)` makes it a local patch-sum operation.
The trailing underscore denotes an in-place tensor mutation.
Its output shape is `[1,1,26,26]`: `28-3+1=26`.
`images[:1,:,:3,:3].sum()` computes the top-left input patch sum directly.
`assert_close` compares that scalar with output `[0,0,0,0]`.
This verifies actual behavior, not just an API's name.

## Receptive fields and limits of the intuition

After the first 3×3 convolution, one feature depends on a local 3×3 input patch.
Pooling combines neighboring features, so subsequent layers see larger input
regions. Trace receptive fields separately from output dimensions.
Shared kernels encourage translation-equivariant intermediate features: shifts
can produce shifted features, away from boundaries and subject to stride effects.
They do not guarantee that the final classifier is exactly translation-invariant.
Padding, pooling, and the position-specific dense head all matter.

## Controlled edits

1. Change the first convolution to `padding=0`.
   Predict 28→26→13; the second convolution/pool then leads to 6×6 features.
   The final fixed 784-input layer will fail unless you recalculate its width.
2. Change the first convolution's stride to 2.
   Predict a roughly halved first map; calculate the exact result with the formula
   rather than assuming every later dimension is simply half.
3. Change 8 first-layer channels to 12.
   Predict both convolution parameter counts change, but spatial dimensions do not.
4. Change the isolated all-ones kernel to all zeros.
   Predict its entire output is zero because its bias is disabled.

## Checkpoint questions

**Why does the first convolution need only 80 parameters?**
<details>
<summary>Solution</summary>
It has eight filters, each with nine weights and one bias. Those same 80 values
are reused at all 28×28 locations; there is not one filter per location.
</details>

**Does max pooling learn which pixels to keep?**
<details>
<summary>Solution</summary>
No parameters are learned by this pooling layer. Its fixed rule takes each
window's maximum. Earlier learned features determine what values reach that rule.
</details>

**If both the MLP and CNN flatten to 784 values, are they equivalent?**
<details>
<summary>Solution</summary>
No. The MLP flattens raw pixels. The CNN flattens learned feature maps after shared
local operations and pooling. Equal vector widths do not imply equal computations.
</details>

## Design judgment

Locality is appropriate when neighboring pixels and recurring motifs matter.
It would be an unjustified assumption for arbitrary reordered spreadsheet columns.
Fewer parameters can help data efficiency but do not prove higher accuracy or
lower runtime. Session 23 tests the architectural hypothesis on the same split.
