# Project 1, Session 1: teach a model to learn a line

## What you will learn

By the end of this session, you should be able to run a complete training loop and
identify its data, parameters, predictions, loss, gradients, updates, and inference.
You do not need to memorize the PyTorch API.

**Already assumed:** basic algebra, the meaning of a derivative, variables, and
loops. **Introduced here:** the required tensor operations, automatic
differentiation, the training loop, and all Python syntax used in the script.

This is intentionally a tiny first model, not an impressive application.

## 1. The problem

We have five examples:

| Input `x` | Desired output `y` |
|---|---|
| -2 | -3 |
| -1 | -1 |
| 0 | 1 |
| 1 | 3 |
| 2 | 5 |

We choose this family of possible predictions:

```text
prediction = weight * input + bias
```

The model is allowed to change `weight` and `bias`. It is not allowed to change the
examples or the form of this expression.

You can see the rule is `y = 2*x + 1`. The training code is not given `weight = 2`
or `bias = 1`. It starts both at zero and adjusts them from prediction errors.

This is linear regression: a single affine unit with one input and one output,
equivalent to the computation of one `nn.Linear(1, 1)` layer. There is no hidden
layer and no nonlinear activation. We use this simplest building block before
composing several units into a more expressive neural network.

For a real problem known to be linear, a least-squares solver would often be a
simpler choice. Here, iterative training is useful because the same mechanics
extend to neural networks that cannot be solved that way.

## 2. The few concepts we need

### A tensor is numerical data with a shape and a data type

A tensor can hold a scalar, a vector, a matrix, or a higher-dimensional array.
Think of array-like numerical storage with defined operations, not a Python
list of arbitrary objects.

In this script:

| Object | Shape | Role |
|---|---|---|
| `inputs`, `targets` | `(5,)` | Five scalar examples, stored as one-dimensional tensors |
| `weight`, `bias` | `()` | Scalar parameters; each is a zero-dimensional tensor |
| `predictions` | `(5,)` | One prediction per example |
| `loss` | `()` | One scalar summarizing the five errors |
| `new_inputs`, `new_predictions` | `(2,)` | Two inputs and predictions after training |

`(5,)` is Python's notation for a one-element tuple containing `5`; here it
describes one axis of length five. `()` describes no axes, not an empty tensor.
A scalar tensor still contains one number.

All these tensors use 32-bit floating-point values and CPU storage. Operations
between the scalar parameters and the input vector apply the scalars to every
element. This is a simple case of **broadcasting**.

### Parameters versus hyperparameters

`weight` and `bias` are **parameters**: training changes them.

`learning_rate` and `steps` are **hyperparameters**: we choose them to control
training. They are not learned by this loop.

### Loss gives us a quantity to minimize

For each example, subtract the target from the prediction, square the error, and
then average over all five examples:

```text
loss = mean((prediction - target)^2)
```

This is **mean squared error**, or MSE. The square prevents positive and negative
errors from canceling and penalizes large errors more strongly. The unit of MSE
is the square of the target's unit.

Zero loss would mean exact agreement with these training targets. It would not,
by itself, prove that the model is useful on other data.

### A gradient tells us how the loss changes locally

For this model, the gradient consists of two derivatives:

```text
d(loss)/d(weight)
d(loss)/d(bias)
```

Gradient descent updates each parameter in the opposite direction to its
derivative:

```text
parameter = parameter - learning_rate * parameter_gradient
```

A positive derivative means increasing that parameter locally increases the loss;
the update therefore decreases the parameter. A negative derivative does the
opposite. A learning rate that is too large can overshoot, so this is not a
guarantee that every possible update lowers the loss.

PyTorch's **automatic differentiation** computes these derivatives from the
operations we actually perform. It is not estimating them by perturbing the
parameters. We will derive and compare them explicitly in the next session.

## 3. Complete working code

The following is the complete contents of [train.py](train.py). It deliberately
uses no helper classes, dataset downloads, random initialization, or plotting.

```python
import torch

learning_rate = 0.1
steps = 60

inputs = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
targets = torch.tensor([-3.0, -1.0, 1.0, 3.0, 5.0], dtype=torch.float32)

weight = torch.tensor(0.0, requires_grad=True)
bias = torch.tensor(0.0, requires_grad=True)

for step in range(steps):
    weight.grad = None
    bias.grad = None

    predictions = weight * inputs + bias
    loss = torch.mean((predictions - targets) ** 2)
    loss.backward()

    if step == 0 or (step + 1) % 10 == 0:
        print(f"Before update {step + 1:02d}: MSE={loss.item():.6f}")

    with torch.no_grad():
        weight -= learning_rate * weight.grad
        bias -= learning_rate * bias.grad

with torch.no_grad():
    final_loss = torch.mean((weight * inputs + bias - targets) ** 2)
    new_inputs = torch.tensor([3.0, 4.0], dtype=torch.float32)
    new_predictions = weight * new_inputs + bias

print(f"\nLearned weight: {weight.item():.6f}")
print(f"Learned bias:   {bias.item():.6f}")
print(f"Final training MSE: {final_loss.item():.3e}")
print(f"New inputs: {new_inputs.tolist()}")
print(f"New predictions: {new_predictions.tolist()}")
```

Run it from the course folder:

```sh
cd ~/src/pytorch-course
.venv/bin/python lessons/01-learn-a-line/train.py
```

Use your actual course location if it differs from `~/src/pytorch-course`.
If the environment has not been prepared, follow the
[WSL/Linux setup instructions](../../SETUP.md), or the
[existing environment notes](../../README.md#environment), first.
No environment activation is required with this command.

In the current environment, PyTorch may first print a warning that NumPy is not
installed. This script does not use NumPy; its tensor operations and training
still run. The warning is separate from the training output below.

The initial MSE is `9.0`. After training, the weight should be approximately `2`,
the bias approximately `1`, and the new predictions approximately `[7, 9]`.
The final training MSE should be very small, not necessarily exactly zero.

There is no randomness in this example, so we do not need a random seed. Small
floating-point differences between environments are still possible.

## 4. Every line explained

Line numbers below refer to the supplied `train.py`, counting blank lines.
Blank lines only separate sections visually.

| Line | Explanation |
|---|---|
| 1 | `import torch` loads PyTorch and binds the module to the name `torch`. `torch.tensor`, for example, accesses a member of that module. |
| 3 | Set the learning rate to `0.1`. This is an ordinary Python floating-point value, not a trainable tensor. |
| 4 | Choose 60 parameter updates. `steps` is an ordinary Python integer. |
| 6 | Construct the five inputs from a Python list. `dtype=torch.float32` is a keyword argument requesting 32-bit floats, broadly analogous to C/C++ `float`. |
| 7 | Construct the corresponding desired outputs in the same order and with the same shape and type. Index `i` in each tensor describes the same example. |
| 9 | Create a scalar weight initially equal to zero. `requires_grad=True` asks PyTorch to track computations involving it so that derivatives with respect to it can be computed. It does not compute a gradient yet. Without an explicit `dtype`, this floating-point constructor uses PyTorch's default floating type, which is `float32` in this unmodified script. |
| 10 | Create a separate trainable scalar bias, also initially zero. Its derivative will be computed separately. |
| 12 | `range(steps)` produces integers from 0 through 59. The `for` loop binds each to `step`. A colon begins a block; indentation, not braces, determines its contents. Every iteration processes all five examples. |
| 13 | Set the weight's stored gradient to `None`. This does not change the weight's value. PyTorch accumulates gradients across `backward()` calls, so we clear the previous iteration's gradient before calculating a fresh one. |
| 14 | Clear the bias's stored gradient for the same reason. `None` is Python's singleton representing absence; after `backward()`, these attributes will hold scalar tensors. |
| 16 | Compute all five predictions. Tensor `*` is elementwise multiplication; it is not matrix multiplication. Scalar broadcasting makes this equivalent to computing `weight*x + bias` separately for each input. PyTorch also records the operations needed to differentiate the loss. |
| 17 | Subtract corresponding targets, square with Python's `**` exponentiation operator, then take the mean. The result is a scalar tensor. No tensor is converted to an ordinary Python number here, so the differentiation connection is preserved. |
| 18 | `loss.backward()` traverses the recorded computation backward using the chain rule. It populates `weight.grad` and `bias.grad`. It does **not** update either parameter. Because `loss` is scalar, no argument is required. |
| 20 | Log before the first update and then before every tenth update. `==` compares, `or` combines Boolean conditions, and `%` is the remainder operator. The loop is zero-based, so the human-readable update number is `step + 1`. |
| 21 | Print the current training loss. The `f` prefix creates an interpolated string; `{...}` contains an expression. `:02d` prints an integer in at least two digits with zero padding, and `:.6f` prints six digits after the decimal point. `loss.item()` extracts a Python number from the single-element tensor for display only. |
| 23 | Temporarily disable gradient tracking inside this indented block. `with` uses a context manager that restores the previous tracking state when the block exits. We want to calculate the next parameter values, not differentiate through the training update itself. |
| 24 | Update the weight in place by subtracting the learning rate times its gradient. `-=` changes the existing tensor's contents here. The gradient is available because line 18 just computed it. The `no_grad` block permits this update to a trainable tensor without recording it. |
| 25 | Apply the same update rule to the bias, using its own derivative. Both gradients were calculated at the parameter values from before either update. |
| 27 | After all loop iterations finish, disable tracking for the final measurements and inference. This block is outside the loop because it is not indented beneath `for`. |
| 28 | Recompute the training MSE using the final parameter values. The last `loss` from inside the loop was calculated **before** the last update, so we do not reuse it as the final measurement. |
| 29 | Create two inputs that were not among the five training examples. These are also 32-bit floating-point values. |
| 30 | Use the learned weight and bias to predict outputs for those inputs. This is **inference**: evaluating the trained function without changing its parameters. |
| 32 | Extract and print the learned weight. `\n` inserts a newline before this message. The six-decimal display is rounded, not the full stored precision. |
| 33 | Print the learned bias in the same way. The extra spaces only align the display. |
| 34 | Print the final MSE with scientific notation. `:.3e` means three digits after the decimal point in that notation, so a tiny nonzero value is not displayed as `0.000000`. |
| 35 | Convert the new-input tensor to a Python list with `.tolist()` for convenient printing. This is for presentation, not part of the model. |
| 36 | Convert and print the corresponding predictions. Their list positions match those of the inputs. |

Unlike a C++ fixed-width numeric declaration, assigning a value to a Python
variable does not specify a static type. The tensor object's `dtype`, however,
does specify its numerical element type.

## 5. Trace the first update

Initially, `weight = 0` and `bias = 0`, so all five predictions are zero.

The errors are:

```text
predictions:    0,  0,  0,  0,  0
targets:      -3, -1,  1,  3,  5
errors:        3,  1, -1, -3, -5
squared:       9,  1,  1,  9, 25
MSE: (9 + 1 + 1 + 9 + 25) / 5 = 9
```

At these initial parameter values, the derivatives are:

```text
weight.grad = -8
bias.grad   = -2
```

With learning rate `0.1`, the first update is:

```text
weight = 0 - 0.1 * (-8) = 0.8
bias   = 0 - 0.1 * (-2) = 0.2
```

Notice that subtracting a negative gradient increased both parameters.

The loss at the new values is `3.52`, lower than `9`. We calculate fresh
predictions and derivatives at those new values on the next iteration.

The output labels deliberately say **Before update**. A loss printed before
update 10 reflects nine completed updates. `Final training MSE` is measured after
all 60 updates.

The essential loop is:

```text
clear old gradients
    -> calculate predictions
    -> calculate loss
    -> calculate fresh gradients
    -> update parameters
    -> repeat
```

The five examples are processed together as one **batch**. Here, one update also
uses the entire dataset once. Later, an **epoch** will mean one pass through the
training dataset and can contain many batches and parameter updates. Update
count and epoch count are not generally interchangeable.

## 6. Controlled experiments

First run the original script unchanged. Then perform one experiment at a time.
Write down a prediction before running it: what should happen, and why?

Each script execution starts over from zero parameters. Restore the original
values before the next experiment so that you change only the stated factor.
There is no need to understand every outcome immediately; unexplained behavior
is useful material for Session 2.

### A. Stop training early

Change only:

```python
steps = 10
```

Does it still learn something? Are the weight and bias equally close to their
desired values? Is the final loss the same as the loss printed before update 10?

### B. Take smaller steps

Restore `steps = 60`, then change only:

```python
learning_rate = 0.01
```

Before running: do you expect a smaller learning rate to produce a better answer
after the **same number of updates**? A smaller step can be safer without being
faster.

### C. Take steps that are too large for convergence

Restore the original script, then change only:

```python
learning_rate = 0.5
```

Does the loss continue decreasing toward zero? Does more training necessarily
help? This particular value produces a revealing oscillation in this dataset,
not a universal "bad learning rate" for every model.

For this run only, replace the existing logging condition with `if True:` to
print every iteration. Run it again and inspect the losses. The weight
oscillation is easier to see if you temporarily add this line just before the
existing loss-printing line, at the same indentation level:

```python
print(weight.item(), bias.item())
```

`True` is Python's Boolean true value, so the logging block executes every time.
This extra change affects observation, not training.

### D. Learn a different line

Restore the original script, then replace only the target-construction line with:

```python
targets = torch.tensor([8.0, 5.0, 2.0, -1.0, -4.0], dtype=torch.float32)
```

These targets follow `y = -3*x + 2`. Predict the learned parameters and the outputs
for inputs 3 and 4 before running. Did the same training procedure adapt without
changing the prediction formula?

<details>
<summary>Expected qualitative outcomes: open after making your predictions</summary>

- **A:** Ten updates improve both parameters, but the bias still differs visibly
  from 1. The final MSE is lower than the value logged before the tenth update.
- **B:** Training progresses more slowly at this fixed update budget. A smaller
  learning rate does not automatically mean a more accurate final model.
- **C:** The weight alternates between 0 and 4 rather than settling at 2. The bias
  reaches 1, and the MSE settles at 8 instead of approaching zero. The chosen
  example has a precise oscillation; larger rates can diverge instead.
- **D:** The weight approaches -3, the bias approaches 2, and predictions approach
  -7 and -10.

We will derive the update behavior rather than treating these results as rules
to memorize.

</details>

After experimenting, restore `train.py` to its original contents so it remains
our known starting point.

## 7. What this example does not establish

The two new inputs lie outside the training range. Predicting them is
**extrapolation**. It happens to work because we deliberately chose an exact
linear rule and a model capable of representing it.

This example has no noise, no held-out validation/test sets, and no model
selection. Its purpose is to expose the mechanics, not demonstrate generalization
or recommend an evaluation procedure. We will introduce honest evaluation before
using realistic datasets or comparing architectures.

It also does not show that zero initialization is generally safe. It works here
because this model has just one affine unit; later we will explain why identical
initializations can prevent hidden units from learning different features.

We have not yet introduced layers as objects, optimizers, mini-batches, model
serialization, or GPU execution. They are not missing pieces you are expected to
already understand.

## 8. Checkpoint before the next session

Explain these in your own words, referring to the actual script:

1. What exactly is learned, and what exactly remains fixed?
2. Why is `loss` a scalar even though the model makes five predictions?
3. What does `loss.backward()` change, and what does it not change?
4. Why do we clear `.grad` but not reset the parameters each iteration?
5. Why are the parameter updates inside `torch.no_grad()`?
6. What makes lines 29-30 inference rather than training?
7. Why does near-zero training loss not establish real-world usefulness?

You are ready for Session 2 when you can trace the first update and distinguish
the role of each part of the loop. Exact API spelling can remain a reference.

**Next:** [Session 2](../02-inspect-gradients/README.md) derives the two gradients,
inspects them in the running program,
and explains the learning-rate experiments from the update equations. We will
stay with this same small model rather than adding complexity yet.

## Optional official references

These supplement the explanation; they are not required reading:

- [Tensor basics](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [`Tensor.backward()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)
- [`torch.no_grad()`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)
