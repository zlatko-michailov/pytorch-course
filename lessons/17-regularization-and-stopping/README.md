# Session 17: choose complexity with validation, then restore the chosen model

Project 4, lesson 4 of 4.

**Prerequisite:** [Session 16](../16-capacity-and-overfitting/README.md).
**New:** weight decay, validation-based stopping, patience, and independent
in-memory snapshots of learned state.

## Two distinct controls

Weight decay discourages parameter growth during fitting. In this script,
Adam's default coupled weight-decay option adds an L2-style contribution to
gradients; it is not AdamW's decoupled update. Its strength is a hyperparameter,
and it is not a guarantee that each learned function will be smooth or adequate.
We apply it to all parameters here, including biases, for a transparent example.

Early stopping chooses a parameter state based on validation performance.
It does not change the hypothesis family, but changes how far optimization follows
this particular training sample.

We keep both controls explicit. To isolate their effects, change only one at a
time in experiments instead of attributing every improvement to both together.

## Run and read

[Full source](train.py)

```sh
.venv/bin/python lessons/17-regularization-and-stopping/train.py
```

| Line | Meaning |
|---|---|
| 1 | Import `deepcopy` to make independent copies of nested tensor state. |
| 2 | Import PyTorch. |
| 3 | Import layers. |
| 5 | Seed the reference experiment. |
| 6 | Use one CPU thread for small operations. |
| 7 | Construct scalar input rows. |
| 8 | Generate noisy nonlinear observations. |
| 9 | Shuffle row indices. |
| 10 | Select 32 training examples. |
| 11 | Select 84 validation examples. |
| 12 | Reserve 84 final test examples. |
| 13 | Construct the wide network from the prior session. |
| 14 | Configure Adam with explicitly nonzero weight decay. |
| 15 | Start the best validation score at positive infinity. |
| 16 | Save an independent initial state, not references to live parameters. |
| 17 | Initialize the chosen update index. |
| 18 | Initialize the count since a meaningful improvement. |
| 19 | Set the allowed non-improving update count. |
| 21 | Train at most 800 full-batch updates, each also one epoch here. |
| 22 | Clear gradients. |
| 23 | Calculate training data MSE; the optimizer adds its decay contribution separately. |
| 24 | Differentiate. |
| 25 | Apply the regularized optimizer update. |
| 26 | Disable tracking for validation. |
| 27 | Measure validation MSE after this update. |
| 28 | Require improvement greater than a stated tolerance; tiny fluctuations do not reset patience. |
| 29 | Record the new validation score. |
| 30 | Copy the corresponding parameter tensors into independent storage. |
| 31 | Record the human-readable update index. |
| 32 | Reset the stale count. |
| 33 | Handle a non-improving update. |
| 34 | Increase the stale count. |
| 35 | Check the stopping policy. |
| 36 | Exit the loop if patience is exhausted. |
| 38 | Restore the selected snapshot, which may differ from the final loop state. |
| 39 | Disable tracking for final evaluation. |
| 40 | Evaluate the selected model on the untouched test partition once. |
| 41 | Evaluate a training-fitted constant on identical test rows. |
| 42 | Report the actual budget and selected checkpoint index. |
| 43 | Report the development score that made the selection. |
| 44 | Report final test error and baseline together. |

`state_dict()` alone contains references to the module's state tensors.
Assigning it to `best_state` without copying would let later updates mutate the
supposed best checkpoint. Saving and loading from a file, as in Session 9, is
another way to materialize a separate snapshot.

## Early stopping is a selection rule, not a guarantee

Validation curves are noisy and may improve again after a plateau. Patience and
minimum improvement are chosen policies, not proofs of optimal stopping.
Keep the test partition out of those decisions.

In models with dropout or batch normalization, selecting/evaluating also requires
the correct `train()`/`eval()` modes; those will be introduced with their
behavior-changing layers. `no_grad()` and module mode are separate controls.

## Experiments

1. Set weight decay to zero, leaving the split, seed, and stopping policy fixed.
2. Increase patience from 40 to 100. Report budget and selected validation state,
   not just whichever final iteration happens to execute.
3. As a deliberate bug, remove `deepcopy` from the assignment inside the
   improvement block. Compare predictions from the supposed best state with
   an independently saved snapshot; explain the aliasing.
4. Remove stopping but retain a fixed update budget. This isolates selection
   from regularization. Use validation, not repeated test observations, to judge.

## Checkpoint

Is the best checkpoint necessarily the last one? What information was used
to select it? Why restore it before testing? Does logged training MSE include the
weight penalty in this implementation?

<details>
<summary>Worked answers</summary>

The best validation state can occur earlier. Validation selected it; the test
set did not. Evaluating the final unselected state would test a different model.
The displayed data MSE omits the penalty, which the optimizer applies during
updates. Be explicit about which objective a reported number represents.

</details>

**Project 4 complete:** you can justify a nonlinear architecture and distinguish
representation, optimization, and generalization problems.
[Next: images and multiple classes](../18-images-flat/README.md).
[Course map](../../CURRICULUM.md).
