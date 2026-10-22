# Session 15: hidden features can bend a decision boundary

Project 4, lesson 2 of 4.

**Prerequisite:** [Session 14](../14-nonlinear-regression/README.md).
**New:** nonlinear classification and a small custom `nn.Module` class.

## Explain why the previous classifier cannot work first

Label 1 means the two features have the same sign; label 0 means opposite signs.
The four representative corners have labels `[1,0,0,1]`.
No single line separates the two diagonal positive corners from both negatives.
This is XOR-like, although our label convention is equality of signs.

The change is a hidden ReLU layer, **not** a change to binary output semantics:
the model still emits one raw logit per example and uses `BCEWithLogitsLoss`.
Hidden features change the boundary's expressive power.

## Run and read

[Full source](train.py)

```sh
.venv/bin/python lessons/15-hidden-classifier/train.py
```

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 2 | Import layers and the module base class. |
| 5 | Declare a Python class deriving from `nn.Module`. |
| 6 | `__init__` is the instance initialization method; `self` explicitly names the instance, analogous to `this`. |
| 7 | Initialize the base class before assigning modules, so registration machinery exists. |
| 8 | Assign registered child layers: `[N,2] -> [N,16] -> [N,16] -> [N,1]`. |
| 10 | Define the forward computation method. |
| 11 | Return the child stack's output without an extra sigmoid. |
| 14 | Seed data and initialization. |
| 15 | Use one thread for these tiny operations. |
| 16 | Generate independent two-feature rows. |
| 17 | Construct equality-of-sign labels, converted to floating `[N,1]`. |
| 18 | Select training data. |
| 19 | Select validation data, reserving the final 80 rows. |
| 20 | Construct a model instance; this runs its initializer. |
| 21 | Register all nested parameters with SGD. |
| 22 | Construct binary logit loss. |
| 23 | Perform 1,000 small updates. |
| 24 | Clear gradients. |
| 25 | Call the module normally, which invokes its forward calculation. |
| 26 | Differentiate through hidden and output layers. |
| 27 | Update parameters. |
| 29 | Disable tracking for evaluation. |
| 30 | Calculate validation accuracy at the zero-logit threshold. |
| 31 | Define four representative corner examples. |
| 32 | Convert their logits to probability-like scores for inspection. |
| 33 | Report validation accuracy. |
| 34 | Show whether the corners follow the intended label pattern. |
| 35 | Show registered state names, including both layers' weights and biases. |

A Python class is not necessary for a plain stack; `Sequential` alone would
work. This minimal example teaches the interface before architectures need
branching, multiple inputs, or custom forward operations.

Ordinary tensors assigned as attributes are not automatically learnable
parameters; `nn.Linear` registers its `Parameter` objects, and assigning the
layer as `self.layers` registers the child module.

## Inspect errors before adding capacity

The boundary includes the two coordinate axes. Examples close to an axis may
have small logits and be harder to classify after finite training.
Inspect whether errors concentrate there rather than only reading accuracy.
A mathematically representable rule is not a guarantee that a particular seed,
width, optimizer, and budget will learn it perfectly.

## Experiments

1. Replace the hidden stack with `nn.Linear(2,1)`. Keep loss and data unchanged:
   this isolates the architectural limitation.
2. Compare widths 2, 8, and 16 using validation. Record budgets and parameter counts.
3. Zero **all** hidden weights and biases. Symmetric hidden units receive identical
   information and cannot spontaneously specialize; ReLU at zero also blocks
   hidden gradients initially. Explain why Session 1's zeroing is not a template.
4. Print predictions along `x0=0.01` as `x1` changes sign. Reason about boundary
   behavior and uncertain scores.

## Checkpoint

Which inherited mechanism makes `model(inputs)` call `forward`? Why does the
optimizer discover nested parameters? Why is no output sigmoid included?

<details>
<summary>Worked answers</summary>

`nn.Module`'s call machinery invokes `forward`. Assigning registered child modules
lets parameter/state traversal recurse through the hierarchy. The combined
binary loss requires raw logits; sigmoid is used only for interpreting scores.

</details>

[Next: capacity and overfitting](../16-capacity-and-overfitting/README.md).
[Course map](../../CURRICULUM.md).
