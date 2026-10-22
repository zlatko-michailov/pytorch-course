# Project 48 / Session 48 — Debug the training state machine

## Prerequisites

Complete session 47.
Review backward accumulation, SGD/Adam updates, dropout, and train/eval modes.
You should be able to trace one complete optimizer step.

**Review:** gradient buffers, learning rate, stochastic layers, no-gradient evaluation.
**New:** BatchNorm's running-statistics state and controlled bug-isolation comparisons.
The default script demonstrates each bug and its repaired behavior.
It is not left broken for you to finish.

## The problem

Correct shapes do not guarantee a correct update schedule.
Gradients can contain stale work.
A learning rate can make even a convex scalar problem diverge.
A validation helper can leave a model in evaluation mode when training resumes.

These failures have different causes.
Changing the architecture before isolating them can hide rather than solve the defect.
We use three tiny independent experiments so each cause has a clear prediction.

## Run

From the course root:

```bash
.venv/bin/python -m lessons.48-debug-training.train
```

Complete worked solution: [train.py](train.py).
Only CPU PyTorch is needed.
The command finishes successfully after observing expected failures and asserting the repairs.
No broad exception handler turns an unknown failure into a success message.

## Experiment 1: stale gradients

The data is `x=[1,2,3,4]` with target `y=2*x`.
The scalar trainable parameter w starts at zero.
The loss is the mean squared error of `w*x` against y.
At w=0, its gradient is -30.

The first `.backward()` stores that gradient in `w.grad`.
The script saves a clone named `fresh`.
It then creates the same forward/loss expression again and calls backward again.
Without clearing the gradient buffer, `w.grad` becomes -60.

This is not a PyTorch bug.
Backward accumulates because multiple graph paths and intentional microbatches may contribute.
An optimizer step does not automatically clear that buffer.
Ordinary training must explicitly zero it before each independent update.

`w.grad = None` resets this one parameter for the repaired comparison.
In a real model, `optimizer.zero_grad(set_to_none=True)` handles all optimizer parameters.
Using `None` lets autograd allocate a fresh gradient rather than add into an old one.

## Intentional accumulation

The same data is divided into two equal-sized microbatches.
Each microbatch loss is a mean over two examples.
Dividing each by 2 makes their accumulated gradient equal the mean over all four examples.
The script asserts equality with the original full-batch gradient.

There is no optimizer step between those microbatches.
The desired sequence is:

```text
zero once -> backward microbatch 1 -> backward microbatch 2 -> step once
```

This is different from forgetting to zero between unrelated optimizer steps.
For unequal microbatch sizes, dividing by the number of microbatches is generally wrong.
Weight each mean loss by its example count divided by total accumulated example count.
The completed example uses equal sizes so that the arithmetic is transparent.

## Experiment 2: learning-rate instability

The scalar objective is:

```text
L(w) = (w-3)^2
gradient = 2(w-3)
```

Let the error be `e=w-3`.
One SGD update gives `e_next=(1-2*lr)*e`.
For `0 < lr < 1`, the error magnitude contracts.
For lr=1.1, the factor is -1.2, so errors alternate sign and grow.
For lr=0.1, the factor is 0.8, so errors shrink.

The script runs twelve updates for each learning rate.
Both paths clear gradients correctly.
The only changed variable is the step size.
It asserts increasing loss for the unstable path and decreasing loss for the stable path.

The verified run prints approximately:

```text
lr=1.1: 9.000 -> 496.856
lr=0.1: 9.000 ->   0.066
```

These are losses measured before the corresponding update.
Do not infer that lr=0.1 is universally safe from one quadratic with known curvature.
Real networks have more complicated local geometry.

## Experiment 3: modes and gradient recording

The small module contains BatchNorm1d, Dropout, and Linear.
Its input is `[16,4]`.
BatchNorm has enough examples to estimate training batch statistics.
Dropout randomly removes activations while in training mode.

### The new stateful layer: BatchNorm

`nn.BatchNorm1d(4)` treats the input as sixteen examples with four features each.
In training, it computes a mean and variance for each feature across the batch.
It subtracts each feature's batch mean and divides by a stabilized standard deviation.
Learned featurewise scales and offsets then adjust those normalized values.
This controls activation scale while preserving trainable affine flexibility.

Unlike session 45's per-token LayerNorm, BatchNorm also maintains running statistics.
`running_mean` and `running_var` are buffers, not gradient-trained parameters.
Training forwards update them with moving averages.
The default moving-average update mixes 10% new statistic with 90% old statistic.
Evaluation uses the stored running statistics instead of fitting the current request batch.
That is why mode affects inference even when no optimizer is active.

Tiny or unrepresentative training batches can give poor running estimates.
We use the layer here to expose state behavior, not recommend it for every small model.
The capstone deliberately avoids batch-dependent normalization in its inference interface.

First, `model.train()` is active inside `torch.no_grad()`.
Two outputs differ because dropout is still random.
BatchNorm running means change because training-mode buffer updates still occur.
Thus no-grad does not imply evaluation behavior.

Next, `model.eval()` is active without a no-grad context.
Two outputs match because dropout is disabled and BatchNorm uses stored statistics.
Running means no longer change.
The output still requires gradients because parameters require them.
Thus evaluation mode does not imply no-grad.

These are independent switches:

```text
train/eval: module behavior and relevant buffer updates
grad/no_grad: whether operations build an autograd graph
```

## A common state-machine mistake

A validation function calls `model.eval()`.
Training code resumes but forgets to call `model.train()`.
Dropout stays disabled and BatchNorm stays in inference behavior.
The optimizer can still run, so the mistake may not crash.

The demonstration intentionally checks `model.training` while eval mode is still active.
It catches that one expected `AssertionError`.
The repair explicitly restores `model.train()`.
The script checks the parent and relevant child modes, then performs a valid training update.

## Controlled experiments

### A. Accumulate without dividing the microbatch losses

Predict: the gradient becomes twice the full-batch mean gradient.
That effectively changes update magnitude, even if the configured learning rate is unchanged.
Do not compare accumulation schedules without agreeing on loss normalization.

### B. Try lr=1.0 on the scalar objective

Predict: error sign alternates but magnitude remains constant.
The loss does not converge.
This isolates the boundary between contraction and divergence.

### C. Remove BatchNorm and keep dropout

Predict: no-grad still does not disable randomness.
There will no longer be a running-mean buffer to inspect.
The two independent switches remain conceptually separate.

### D. Evaluate between every training update

Keep both explicit mode calls.
Predict: training remains correct.
Then omit the training-mode restoration and explain which modules change behavior.
Restore the working script before continuing.

## Checkpoint questions

<details>
<summary>Does zero_grad erase learned weights?</summary>

No. It clears gradient buffers, not parameter values or optimizer history.
Weights change during the optimizer step.
Momentum/Adam state has its own persistence and is not reset by zero_grad.
</details>

<details>
<summary>Why is the unstable example not repaired with gradient clipping?</summary>

Its known curvature makes the learning-rate cause explicit.
Clipping might limit symptoms without explaining the incorrect step-size regime.
Use the simplest diagnostic that isolates the actual failure.
</details>

<details>
<summary>Should validation update BatchNorm running statistics?</summary>

Not under the usual held-out evaluation contract.
Use eval mode so held-out inputs do not change trained running buffers.
Use no-grad as well to avoid unnecessary graph construction.
</details>

## Design judgment

Treat training as a state machine, not just a formula inside a loop.
Check gradient reset, update cadence, mode transitions, and data access separately.
Next, debug a score that looks good for the wrong reason.
