# Session 2: explain the gradients

Project 1, lesson 2 of 3. **Session 1 was a complete lesson**; this session
explains its mechanics more deeply rather than finishing omitted code.

**Prerequisite:** [Session 1](../01-learn-a-line/README.md). You can identify the
parameters, scalar loss, gradient calculation, and parameter update.

**New:** derive the derivatives, distinguish a fresh computation graph from a
stored gradient, and analyze a learning rate instead of guessing.

## Derive what `backward()` computes

For `N` examples define the error `e_i = w*x_i + b - y_i`. The loss is

```text
L(w,b) = (1/N) * sum_i e_i^2
```

The chain rule gives

```text
dL/dw = (1/N) * sum_i 2*e_i*x_i
dL/db = (1/N) * sum_i 2*e_i
```

The derivative of the square is `2*e_i`; the derivative of its argument with
respect to `w` is `x_i`, and with respect to `b` is `1`. The averaging factor is
part of both derivatives. Autograd follows those same operations.

At zero parameters, the five errors are `[3, 1, -1, -3, -5]`.
Their sum is `-5`; their input-weighted sum is `-20`.
Consequently `dL/db = -2` and `dL/dw = -8`.

## Run the complete experiment

[Full source: train.py](train.py)

```sh
.venv/bin/python lessons/02-inspect-gradients/train.py
```

Read the script beside this explanation. Nothing is downloaded or saved. It
first compares analytical derivatives with autograd, then deliberately omits a
gradient reset, and finally calculates learning-rate behavior in ordinary Python.
The formula in the final loop is valid for **this dataset**, not every model.

## Every nonblank line

| Line | Meaning |
|---|---|
| 1 | Import PyTorch. |
| 3 | Recreate the same five floating-point inputs. |
| 4 | Generate the exact targets with the known teaching rule; we are checking mathematics, not hiding the answer from ourselves. |
| 5 | Create a scalar trainable weight at zero. |
| 6 | Create a scalar trainable bias at zero. |
| 8 | Calculate the vector of signed errors; preserve its graph. |
| 9 | `.square()` squares elementwise; `.mean()` reduces all five values to one scalar. This equals Session 1's expression. |
| 10 | Compute and store the derivatives in the two leaf tensors' `.grad` attributes. |
| 12 | Disable recording for our independent derivative calculations. |
| 13 | Evaluate the derived weight-gradient formula over the five examples. |
| 14 | Evaluate the derived bias-gradient formula. |
| 16 | Extract and display the two autograd results. |
| 17 | Display the two formula results. |
| 18 | `torch.testing.assert_close` raises an error if tensors differ beyond its tolerances. It is an executable check, not a training operation. |
| 19 | Check the bias derivative too. |
| 21 | Build a **new** forward graph using unchanged parameters. |
| 22 | Backpropagate again; stored gradients are added to, not replaced. |
| 23 | Show doubled gradients: approximately `-16` and `-4`. |
| 25 | Remove the accumulated weight gradient. |
| 26 | Remove the accumulated bias gradient. |
| 27 | Build another fresh graph. |
| 28 | Compute gradients with no old values to accumulate into. |
| 29 | Show the original `-8` and `-2` again. No parameters have changed. |
| 31 | Iterate over four Python learning-rate values. |
| 32 | Reset an ordinary Python weight for this independent experiment. |
| 33 | Reset its bias. These are numbers, not tracked tensors. |
| 34 | Perform ten analytical updates. |
| 35 | Apply the weight derivative specific to this symmetric dataset. |
| 36 | Apply its bias derivative. |
| 37 | Calculate the exact dataset MSE at those final parameter values. |
| 38 | Print each experiment using fixed-point formatting. |

Calling `loss.backward()` twice on the **same** graph would normally fail after
its saved intermediates have been freed. We intentionally recompute the forward
expression. Recomputing a graph and clearing `.grad` are different requirements.
We are not using `retain_graph=True` to hide either requirement.

## Why the learning rate behaved that way

Here `mean(x)=0`, `mean(x^2)=2`, and `y=2*x+1`. Substitution gives

```text
L(w,b) = 2*(w-2)^2 + (b-1)^2
dL/dw = 4*(w-2)
dL/db = 2*(b-1)

(w_next-2) = (1-4*lr)*(w-2)
(b_next-1) = (1-2*lr)*(b-1)
```

For each error to shrink, its multiplier must have absolute value less than one.
For **both** errors here, `0 < lr < 0.5` suffices.
At `lr=0.1` the multipliers are `0.6` and `0.8`, explaining why the weight learns
faster. At `0.5`, the weight multiplier is `-1`: it oscillates without shrinking.
At `0.6` its magnitude exceeds one: the error grows.

This clean separation depends on centered inputs. Nonzero mean inputs couple
the two derivatives. Larger networks have more complicated curvature, but input
scale and curvature still affect usable learning rates.

## Experiments

1. Change the list to `[0.25, 0.49, 0.5]`. Predict oscillation versus convergence
   before running. Near the boundary, convergence can be slow even with large steps.
2. Increase the inner iteration count from 10 to 60. Which rates improve, and
   which get worse? A larger training budget cannot repair an unstable update.
3. Add `1` to every input. First rerun only the autograd/formula comparison:
   it still agrees. Then inspect the analytical loop: it is now the wrong model
   of this dataset because the simplification used zero-mean inputs.

## Checkpoint

Why does fresh forward computation not reset stored gradients? Why did neither
of the first three backward calls change the parameters? What happens if a
derivative is zero? Can learning rates be compared independently of input scale?

<details>
<summary>Worked answers</summary>

The graph records dependencies; `.grad` is separate accumulated storage on leaf
tensors. `backward()` fills that storage but is not a parameter update. A zero
derivative gives no update in plain gradient descent, which does not by itself
prove a global optimum. Scaling inputs changes the derivative and curvature, so
the same numerical rate can have very different effects.

</details>

**Next:** [Session 3](../03-layers-and-optimizers/README.md) replaces the manual
bookkeeping with PyTorch's named components. [Course map](../../CURRICULUM.md).
