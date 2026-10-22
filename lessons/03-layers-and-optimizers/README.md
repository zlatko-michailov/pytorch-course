# Session 3: layers, loss objects, and an optimizer

Project 1, lesson 3 of 3.

**Review:** [Session 2](../02-inspect-gradients/README.md)'s gradient formulas,
clearing gradients, and updates. **New:** `nn.Module`, `nn.Linear`, parameter
registration, `MSELoss`, SGD, and an explicit batch/feature shape.

## Same model, different packaging

We are not adding expressive power. This script must solve exactly the same line.
PyTorch now keeps the parameters inside a **layer object** and updates them through
an **optimizer object**. Think of `nn.Linear` as a class whose instance contains
learnable fields and implements a callable forward calculation.

`nn.Module` is the base class for layers and networks. It tracks registered
parameters; `model.parameters()` exposes those parameters to the optimizer.
Calling `model(inputs)` invokes the layer's forward calculation through the
module's call machinery. Prefer it to calling `model.forward(inputs)` directly.

```text
Old                         New
weight*x + bias              model(x)
mean((prediction-y)^2)       loss_function(prediction, y)
weight.grad = None, ...      optimizer.zero_grad(set_to_none=True)
loss.backward()              loss.backward()   [unchanged]
manual no_grad updates       optimizer.step()
```

## Shapes are part of the interface

`nn.Linear(in_features, out_features)` acts on the **last dimension**.
Here five rows are five examples; each row contains one feature.

```text
inputs       [5, 1]
weight       [1, 1]  (output features, input features)
bias         [1]
predictions  [5, 1]
targets      [5, 1]
```

For a matrix batch it computes `inputs @ weight.T + bias`. Unlike elementwise
`*`, `@` denotes matrix multiplication. The bias is broadcast across rows.
Keep predictions and targets identically shaped: `[5,1]` and `[5]` can broadcast
to a wrong `[5,5]` error calculation without the intended per-example alignment.

## Run and read every line

[Complete source](train.py)

```sh
.venv/bin/python lessons/03-layers-and-optimizers/train.py
```

| Line | Meaning |
|---|---|
| 1 | Import the PyTorch module. |
| 2 | Import its `nn` namespace, which contains layer and loss classes. |
| 4 | Construct a `[5,1]` floating-point tensor using nested Python lists. |
| 5 | Generate targets with that same shape. |
| 6 | Construct one affine layer, one input feature to one output feature. Its parameters are initially randomized by PyTorch. |
| 7 | Disable recording while replacing that initialization. |
| 8 | `.zero_()` modifies the existing weight tensor in place. A trailing underscore usually marks an in-place tensor method. |
| 9 | Zero the bias too, matching Session 1's initial conditions. |
| 11 | Create a loss object. Its default reduction averages all elementwise squared errors. |
| 12 | Construct plain SGD over the registered parameters, using rate 0.1 and no momentum or weight decay. |
| 14 | Repeat 60 training updates. |
| 15 | Set the optimizer's parameters' old gradient buffers to `None`. |
| 16 | Evaluate the layer on all five inputs. This records a differentiable graph. |
| 17 | Compare predictions and targets through the MSE loss object. |
| 18 | Populate parameter gradients, as before. |
| 19 | Apply SGD updates. The optimizer handles the update bookkeeping without recording ordinary parameter updates in autograd. |
| 21 | Disable recording for final measurements and inference. |
| 22 | Evaluate the final training MSE after the last update. |
| 23 | Predict a `[2,1]` output for a `[2,1]` input batch. |
| 25 | Convert the weight's shape to a tuple for display. |
| 26 | Display the bias's shape. |
| 27 | Extract scalar parameter values for display; `.item()` is valid because each contains one element. |
| 28 | Display the final scalar loss. |
| 29 | Display predictions as a nested list, preserving the row/feature layout. |

Zero initialization is used to isolate this refactoring. Do not copy it to all
hidden layers later: identical hidden units can fail to break symmetry.

## Experiments

1. Compare the final outputs with Session 1. They should agree to floating-point
   tolerance. The abstraction has not changed the mathematical task.
2. Remove the zeroing block and run twice. The initialization is now random.
   Then put `torch.manual_seed(7)` immediately before constructing the layer.
   It seeds PyTorch's random generator; reproducibility is distinct from quality.
3. Construct `nn.Linear(1, 1, bias=False)` and remove references to `model.bias`.
   Predict the residual error: no slope can make the output at input zero equal
   one. A parameter omitted from the architecture cannot be learned.
4. Change only the rate to `0.5`. The optimizer does not protect you from the
   oscillation derived in Session 2.

## Checkpoint

What does SGD replace? Does it replace backpropagation? Why is the input now
two-dimensional? Why must every learnable layer be registered with a module or
explicitly supplied to an optimizer?

<details>
<summary>Worked answers</summary>

SGD applies an update using existing gradients; `backward()` still calculates
them. The last input dimension is the feature dimension, with the preceding
dimension indexing examples. Registration makes parameters discoverable through
`parameters()` and, later, `state_dict()`. An optimizer cannot update a parameter
it was never given.

</details>

**Project 1 is complete.** We can fit a model, explain the updates, and express
the same computation through standard PyTorch objects. This does not yet prove
generalization. [Next: noisy measurements](../04-noise-and-baselines/README.md).
[Course map](../../CURRICULUM.md).
