# Project 52 / Session 52 — Justify a nonlinear candidate without promising a win

## Prerequisites

Complete sessions 50–51.
Review ReLU MLPs, parameter counting, weight decay, and training/validation gaps.
You should be able to reproduce the ten-column feature contract.
You should also understand why the logistic baseline remains a release candidate.

**Review:** hidden layers, activation functions, BCE, deterministic training.
**New integration:** architecture rationale tied to a known task interaction and a resource budget.
There is no attention model in this capstone.
The target is small tabular synthetic data, not a reason to reuse the most recent architecture.

## Attempt independently first

The generator increases event probability when both vibration and load cross specified thresholds.
Ask whether an additive logit can represent that interaction exactly.
Propose the smallest nonlinear network you would try.
Count its parameters and define the evidence that would make you reject it.
Then compare with the completed reference.

Do not inspect a final-test score to justify the design.
The reference implementation is already complete and runnable.
No learner-written layer or unfinished callback is required.

## Run

From the course root:

```bash
.venv/bin/python -m lessons.52-capstone-network.train
```

Session source: [train.py](train.py).
Shared model and training source:
[`course/capstone.py`](../../course/capstone.py).
Focus on `CANDIDATES`, `make_model`, `train_candidate`, and `evaluate`.
Preprocessing is exactly the train-only pipeline introduced in session 51.

No previous artifact is needed.
The script regenerates development data and fits the single predeclared nonlinear candidate.
It does not request test rows.
It does not produce a final release merely because a network trained successfully.

## Why an additive logit may be limited

The baseline logit has the form:

```text
bias + sum_j(weight_j * feature_j)
```

It can make high vibration risky and high load risky separately.
It cannot directly switch on a special coefficient only when both conditions hold.
A manually engineered interaction feature could be another simple solution.
That would be a legitimate candidate if declared as part of a new development protocol.

Here the fixed candidate pool instead includes a small ReLU MLP.
Hidden units can form nonlinear combinations of numeric, missingness, and categorical features.
That provides the capacity to approximate interactions.
It does not force the model to discover the exact generator rule.

The labels remain noisy and measurements can be missing.
More representational power can therefore fit randomness rather than useful structure.
The hypothesis is “this capacity may help,” not “the larger model must win.”

## The completed architecture

```text
features [B,10]
  -> Linear(10,16) [B,16]
  -> ReLU          [B,16]
  -> Linear(16,1) [B,1]
  -> squeeze(-1)   [B]
```

The first layer has `10*16 + 16 = 176` parameters.
The second has `16*1 + 1 = 17`.
Total: 193, below the predeclared budget of 250.
The logistic baseline has 11.

`nn.Sequential` applies these layers in order.
ReLU keeps positive hidden activations and sets negative ones to zero.
Without the activation, two linear layers would collapse into one affine transformation.
The final layer returns logits, not probabilities.
BCE-with-logits handles the stable probability conversion during training.

No dropout, BatchNorm, embedding lookup, or Transformer block is needed.
Those are available tools, not mandatory course-completion decorations.
The small dataset and parameter budget favor a restrained architecture.
Inference is per record and should not depend on other records in the request.

## Initialization and unknown categories

`train_candidate` sets the same fixed seed before constructing each candidate.
The first layer's unknown-category column is explicitly initialized to zero.
That feature is absent from training examples, so no predictive signal can fit it.
A neutral unseen-category contribution is preferable to an arbitrary random one.

For the MLP, this means the unknown indicator adds zero to each hidden preactivation.
Other fields and the learned bias still affect the prediction.
This is not a guarantee that unknown machine types are safe or calibrated.
Session 54 exposes an explicit unknown-category flag alongside the output.

## The training budget

The candidate receives 160 full-batch Adam updates at lr 0.025.
Its weight decay is 0.002, declared in session 50.
There is no hidden width search or adaptive early-stopping rule in this reference.
The reported checkpoint is the final fixed-budget state.

The baseline and MLP use the same rows, feature mapping, update count, and learning rate.
They use different declared regularization strengths and have different parameter counts.
This is a comparison of two complete fixed procedures, not a pure parameter-matched ablation.
The experiment record preserves those details.

Every backward pass checks that all parameter gradients exist and are finite.
A reduction in training BCE checks basic learnability.
These are pipeline checks, not release acceptance criteria.
Validation performance remains necessary.

## Reference development result

The verified CPU run produced approximately:

| Procedure | Parameters | Train BCE | Validation BCE |
|---|---:|---:|---:|
| Logistic baseline | 11 | 0.4792 | 0.5813 |
| Width-16 ReLU MLP | 193 | 0.4445 | 0.6032 |

The MLP fits training better and validation worse.
Its validation accuracy was about 0.7208.
Its validation balanced accuracy was about 0.6696.
Its validation positive recall was about 0.4941.

The larger training/validation gap is consistent with extra variance.
It is not proof that every nonlinear model is inferior.
It also is not permission to expand the candidate pool after peeking at test.
The predeclared comparison in session 53 should allow the baseline to win.

## Batch-reordering contract

The script predicts all validation rows.
It reverses their order using `.flip(0)` and predicts again.
Reversing the second result must recover the original logits.

This checks that the network acts independently per record.
The model has no training-mode batch-statistics layer or cross-record attention.
The test is especially useful if a future refactor introduces a batch-dependent operation.
It says nothing about accuracy on a new machine type.

## Controlled experiments

### A. Remove ReLU

Predict: the composition is still affine despite its extra parameters.
It cannot create the intended nonlinear interaction.
Similar scores would not show that nonlinearity was unnecessary in every possible dataset.

### B. Increase width to 64 on a development-only copy

Predict: the parameter count rises to 769 and violates the frozen budget.
Training fit may improve while held-out behavior may not.
A forbidden candidate is not rescued by an attractive score.

### C. Add the explicit vibration/load interaction instead

Design the feature using only information available at inference.
Predict: a logistic model could now express an added interaction coefficient directly.
Treat this as a new declared development experiment, not an unrecorded tweak to the frozen release.

### D. Increase weight decay

Predict: coefficient magnitudes tend to shrink and training fit may become weaker.
Validation could improve or worsen.
Use only validation for exploration and record the changed procedure.
Do not overwrite the original final evaluation claim.

## Checkpoint questions

<details>
<summary>Why is a worse-validation MLP still a useful capstone result?</summary>

It tests a justified hypothesis and demonstrates why capacity is not the same as generalization.
The completed workflow can reject the network without being a failed exercise.
Choosing the baseline is a valid engineering outcome.
</details>

<details>
<summary>Could the MLP learn the generator's probabilities perfectly?</summary>

Not from these finite, partly missing observations with hidden unit effects.
Even exact probabilities would not predict every Bernoulli event correctly.
Distinguish probability estimation error from irreducible event randomness.
</details>

<details>
<summary>Why not use the training BCE to choose the release?</summary>

Extra capacity can lower training loss by fitting noise.
The deployment claim concerns new units.
The held-out validation units provide the predeclared selection evidence.
</details>

## Design judgment

Architecture should follow the information structure and resource budget.
The reference makes a nonlinear candidate available, not inevitable.
Next, freeze the validation choice and spend the final test exactly once.
