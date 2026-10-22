# Curriculum: from a learned line to defensible neural-network designs

## Destination

By the end, you should be able to take an unfamiliar, reasonably scoped problem
and explain:

- What the input and output representations should be.
- Which data and evaluation procedure would test the intended use.
- Whether a neural network is justified compared with simpler baselines.
- Which architecture, loss, and metrics fit the task, and why.
- Whether a failure comes from data, implementation, optimization, capacity, or
  a mismatch between training and deployment.
- How to package preprocessing and learned parameters for reliable inference.

There is no universal formula that determines an adequate architecture from a
problem statement. The skill is forming justified hypotheses, testing them
economically, and revising them using evidence.

## Teaching constraints

We start with tiny, transparent, synthetic problems whose correct behavior we can
inspect. We then add noise, unseen examples, more realistic data, and different
input structures. Toy success is a teaching aid, not evidence of real-world
usefulness.

We introduce PyTorch abstractions after seeing the work they replace. We introduce
mathematics at the point where it explains observed behavior. We use your existing
programming knowledge without assuming Python fluency.

Early projects use CPU computation. Later projects will be scoped to remain
manageable locally; if acceleration becomes helpful, device placement and the
available options will be taught explicitly before using them.

## Project sequence

The full reference course contains **54 sessions**, with no fixed session duration
and no penalty for splitting a session or adding prerequisites. A **project** is
a group of lessons: Session 1 was complete in itself, while Project 1 contains
Sessions 1-3. Read and run the lessons in numerical order; later lessons reuse
ideas taught earlier. The session count describes scope, not a deadline.

| Project | What you build | New ideas, in teaching order | Approx. sessions |
|---|---|---|---|
| 1. Learn a line | A model that discovers `y = w*x + b` from examples | Tensors, predictions, parameters, loss, gradients, manual updates; then `nn.Linear` and SGD | 3 |
| 2. Calibrate noisy sensors | A small, multifeature regression pipeline | Noise, unseen data, train/validation/test roles, baselines, scaling, batches, save/load | 6 |
| 3. Make a binary decision | A synthetic two-class classifier | Logits, sigmoid, binary cross-entropy, decision thresholds, imbalance, precision/recall | 4 |
| 4. Learn nonlinear relationships | A curve-fitting network, then a nonlinear binary classifier | Hidden layers, activations, composition, capacity, underfitting/overfitting, regularization | 4 |
| 5. Recognize several classes | A fully connected Fashion-MNIST classifier | Image tensors, multiclass logits, cross-entropy, class-index targets, confusion matrices | 4 |
| 6. Use image structure | A small convolutional classifier on the same task | Locality, shared kernels, channels, stride/padding, pooling, augmentation, ablations | 4 |
| 7. Model structured records | A small tabular prediction pipeline | Missing values, categorical representations/embeddings, group leakage, non-neural competitors | 4 |
| 8. Reuse a learned representation | A small pretrained image classifier adapted to a new dataset | Frozen backbones, preprocessing contracts, fine-tuning, learning-rate choices | 4 |
| 9. Predict a sequence | A short-horizon time-series forecaster | Windows, chronological splits, persistence/seasonal baselines, horizon errors, temporal convolution | 4 |
| 10. Classify text | A bag-of-embeddings text classifier | Tokenization, training-only vocabulary, unknown tokens, padding/masks, pooling, order limitations | 4 |
| 11. Model token interactions | A tiny attention-based sequence classifier | Attention through explicit tensor operations, positions, attention masks, residual connections, normalization | 5 |
| 12. Diagnose broken training | Repairs to deliberately flawed earlier projects | Systematic triage, tiny-subset overfit checks, gradients, controlled comparisons, error analysis | 3 |
| 13. Defend a design | A modest end-to-end capstone on an unfamiliar dataset | Independent framing, architecture hypotheses, experiment budgets, final evaluation, inference contract | 5 |

## Project contracts

### 1. Learn a line

**Prerequisites:** elementary algebra, derivatives, basic Python expressions and
loops. Required tensor and Python details are taught in the lesson.

**Artifact:** a short script that learns two numbers, then predicts outputs for
new inputs. It begins with raw tensors, not hidden layers or a training framework.

**Sessions:** first training loop; gradient and learning-rate experiments; express
the same computation using a layer, a loss object, and an optimizer.

**Ready to proceed when:** you can distinguish a parameter from a hyperparameter,
explain `backward()` versus an update, account for gradient clearing, and predict
the effect of changing a simple example.

### 2. Calibrate noisy sensors

**Builds on:** Project 1's affine model and training loop.

**Artifact:** a regression model with multiple measured inputs, a constant or
simple analytical baseline, honest evaluation, and a separate inference script.

**New content:** noise and irreducible error; training/validation/test roles;
feature/target shapes; input scale and training-only preprocessing; mini-batches
and data loaders; saving a state dictionary and reconstructing inference.

Introduce these in separate sessions. Use the validation set to make decisions;
reserve the test set for the final check. Derive splits from the intended use,
not automatically from random shuffling.

**Ready to proceed when:** you can train without touching test results, explain
why a scaling statistic must come only from training data, and reproduce
predictions after loading a saved model with its preprocessing.

### 3. Make a binary decision

**Builds on:** regression, held-out evaluation, and batched predictions.

**Artifact:** a classifier for an intentionally understandable synthetic problem.

**New content:** a score/logit versus a probability versus a class decision;
`BCEWithLogitsLoss` and its input/target contract; thresholds; confusion matrices;
precision/recall; why accuracy alone can hide failure on imbalanced data.

Explain why training uses logits directly rather than applying sigmoid before
this loss. Do not equate a sigmoid output with a well-calibrated probability.

**Ready to proceed when:** you can change a threshold without retraining and
explain which mistakes become more or less common.

### 4. Learn nonlinear relationships

**Builds on:** linear regression and linear classification.

**Artifact:** a small multilayer perceptron, first approximating a nonlinear
one-dimensional function and then classifying a nonlinear boundary.

**New content:** why stacked affine layers without an activation remain affine;
hidden features; ReLU and its limitations; shape tracing; width/depth; training
versus validation curves; weight decay and validation-based early stopping.

Start by demonstrating a failure of the previous model. Add the smallest
nonlinear change that addresses that failure, rather than presenting a larger
architecture as an unexplained recipe.

**Ready to proceed when:** you can justify an activation, sketch the data flow,
and distinguish optimization trouble from inadequate model capacity.

### 5. Recognize several classes

**Builds on:** hidden layers, binary classification, and data loaders.

**Artifact:** a fully connected Fashion-MNIST classifier with per-class analysis.

**New content:** channels and flattening; one logit per class; integer class-index
targets and `CrossEntropyLoss`; multiclass predictions; a validation split from
the training data; the untouched official test split.

Teach the loss's contract explicitly: feed raw logits, not softmax probabilities.
Investigate actual mistakes instead of reporting only one aggregate number.

**Ready to proceed when:** you can identify the output shape and loss for a
multiclass task and explain why this differs from binary classification.

### 6. Use image structure

**Builds on:** the same dataset, evaluation procedure, and task as Project 5.

**Artifact:** a small convolutional network compared with the previous model.

**New content:** receptive fields, weight sharing, channels, padding, stride, and
pooling; parameter counts; training-only augmentation; controlled architecture
comparisons. Introduce dropout and the distinction between `train()` and `eval()`
before relying on mode-dependent behavior.

**Ready to proceed when:** you can calculate intermediate shapes and explain the
assumption about images that makes a convolution useful.

### 7. Model structured records

**Builds on:** regression/classification and leakage-safe preprocessing.

**Artifact:** a model consuming numerical and categorical columns with a complete
preprocessing contract.

**New content:** missing-value handling fitted only on training data; categorical
encoding followed by an introduction to embeddings; group-aware evaluation;
comparison with an appropriate linear model and a tree-based baseline.

Dependencies beyond PyTorch are added only when needed and are explained.
Feature availability must reflect the moment a prediction would actually be made.

**Ready to proceed when:** you can explain why a neural network may lose to a
tree-based model, detect a plausible leakage path, and handle unknown categories.

### 8. Reuse a learned representation

**Builds on:** convolutional features and image preprocessing.

**Artifact:** an adapted pretrained classifier with a reproducible inference path.

**New content:** representation reuse; replacement heads; frozen versus trainable
parameters; pretrained normalization requirements; staged fine-tuning. Check
dataset permissions and pretrained-weight licensing before use.

**Ready to proceed when:** you can defend freezing versus fine-tuning given the
dataset size, domain difference, and available compute.

### 9. Predict a sequence

**Builds on:** regression, minibatches, and convolution.

**Artifact:** a window-based forecaster compared with persistence and, where
appropriate, seasonal baselines.

**New content:** what information exists at the forecast time; chronological
splits and overlapping windows; input/output horizons; a windowed MLP followed by
a small temporal convolution; errors by horizon.

An RNN/GRU session can be inserted here if it helps answer an observed limitation;
recurrent architectures are not unexplained prerequisites.

**Ready to proceed when:** you can build evaluation that does not let future
information leak backward and demonstrate improvement over a realistic baseline.

### 10. Classify text

**Builds on:** embeddings, multiclass/binary decisions, and batched inputs.

**Artifact:** a small text classifier starting with averaged token embeddings.

**New content:** tokenization and vocabulary; unknown tokens; variable-length
examples; padding and masked pooling; learned representations; text-specific data
leakage. Begin with inspectable examples before using a larger public dataset.

**Ready to proceed when:** you can trace a sentence into model inputs and explain
what information averaging discards.

### 11. Model token interactions

**Builds on:** token embeddings, masks, matrix multiplication, and nonlinear nets.

**Artifact:** a tiny attention-based classifier, compared with Project 10.

**New content:** a hand-worked attention example before a module; queries, keys,
and values; positional information; appropriate masks; residual connections and
normalization, each motivated separately; only then a small transformer block.

The goal is architectural understanding, not training a large language model.

**Ready to proceed when:** you can trace attention shapes, explain which
interactions are allowed, and justify whether attention improved this task enough
to warrant its added complexity.

### 12. Diagnose broken training

**Builds on:** all of the training and evaluation habits already practiced.

**Artifact:** short diagnoses and repairs for failures injected into earlier
projects. Debugging is practiced throughout; this is a synthesis, not its debut.

**Cases:** accidental broadcasting; wrong target types; unreset gradients; mode
errors; bad scaling; learning-rate failures; data leakage; constant predictions;
memorization; misleading aggregate metrics.

For each case, state a hypothesis, run the smallest discriminating check, and
explain why the repair addresses the cause.

**Ready to proceed when:** you investigate evidence rather than reflexively
adding layers or changing optimizers.

### 13. Defend a design

**Builds on:** the whole workflow, with a domain selected at that point.

**Artifact:** a reproducible baseline, a justified neural model, a small experiment
record, a final held-out evaluation, and a usable prediction interface.

Define the use case, data constraints, success criteria, and inference budget
before trying architectures. Compare at least a simple baseline and one justified
neural candidate. Use validation for development and inspect the test set only
after the design is settled.

**Completion criterion:** you can defend the representation, split, architecture,
loss, metrics, training decisions, and limits of the result. Bigger is not
automatically better; declining to use a neural network can be the right answer.

## All 54 lessons, in teaching order

Every linked folder contains teaching material and complete source, not just an
exercise specification. Early lessons explain every nonblank source line;
later lessons focus the walkthrough on new operations and refer back to familiar
training mechanics. Read code beside the walkthrough rather than memorizing APIs.

| Session | Lesson | Main addition |
|---|---|---|
| 1 | [Learn a line](lessons/01-learn-a-line/README.md) | First complete training loop |
| 2 | [Inspect gradients](lessons/02-inspect-gradients/README.md) | Derivatives, accumulation, learning-rate dynamics |
| 3 | [Layers and optimizers](lessons/03-layers-and-optimizers/README.md) | Same computation expressed with standard modules |
| 4 | [Noise and baselines](lessons/04-noise-and-baselines/README.md) | Observations versus underlying signal |
| 5 | [Held-out data](lessons/05-held-out-data/README.md) | Training, validation, test roles |
| 6 | [Multiple features](lessons/06-multiple-features/README.md) | Feature matrices and affine maps |
| 7 | [Training-only scaling](lessons/07-training-only-scaling/README.md) | Preprocessing fitted without leakage |
| 8 | [Mini-batches](lessons/08-mini-batches/README.md) | Dataset, loader, epoch, weighted loss totals |
| 9 | [Save and load](lessons/09-save-and-load/README.md) | Final evaluation and independent inference |
| 10 | [Logits and probabilities](lessons/10-logits-and-probabilities/README.md) | Binary output/loss contracts |
| 11 | [Binary classifier](lessons/11-binary-classifier/README.md) | Learn an affine boundary |
| 12 | [Decision thresholds](lessons/12-decision-thresholds/README.md) | Precision, recall, and error costs |
| 13 | [Imbalanced decisions](lessons/13-imbalanced-decisions/README.md) | Majority baselines and weighted objectives |
| 14 | [Nonlinear regression](lessons/14-nonlinear-regression/README.md) | Hidden features and ReLU |
| 15 | [Hidden classifier](lessons/15-hidden-classifier/README.md) | Nonlinear boundary and custom modules |
| 16 | [Capacity and overfitting](lessons/16-capacity-and-overfitting/README.md) | Train/validation gaps and Adam |
| 17 | [Regularization and stopping](lessons/17-regularization-and-stopping/README.md) | Validation-based checkpoint selection |
| 18 | [Images as tensors](lessons/18-images-flat/README.md) | Image dimensions and flattening |
| 19 | [Multiclass training](lessons/19-multiclass-training/README.md) | Raw class logits and class-index targets |
| 20 | [Confusion and errors](lessons/20-confusion-errors/README.md) | Per-class analysis |
| 21 | [Image checkpoint](lessons/21-image-checkpoint/README.md) | Persist an image prediction contract |
| 22 | [Convolution shapes](lessons/22-convolution-shapes/README.md) | Locality and shared filters |
| 23 | [CNN training](lessons/23-cnn-training/README.md) | Compare against the dense image baseline |
| 24 | [Augmentation](lessons/24-augmentation/README.md) | Training-only label-preserving transformations |
| 25 | [Dropout and ablation](lessons/25-dropout-ablation/README.md) | Modes and controlled comparisons |
| 26 | [Tabular preprocessing](lessons/26-tabular-preprocess/README.md) | Missing data and feature contracts |
| 27 | [Embeddings](lessons/27-embeddings/README.md) | Learned categorical representations |
| 28 | [Group splits](lessons/28-group-splits/README.md) | Repeated-entity leakage |
| 29 | [Tree baselines](lessons/29-tree-baselines/README.md) | When a non-neural competitor is appropriate |
| 30 | [Transfer contract](lessons/30-transfer-contract/README.md) | Pretrained representations and preprocessing |
| 31 | [Frozen head](lessons/31-frozen-head/README.md) | Train a new head without changing the backbone |
| 32 | [Fine-tuning](lessons/32-fine-tuning/README.md) | Unfreeze deliberately with separate update rates |
| 33 | [Transfer inference](lessons/33-transfer-inference/README.md) | Reconstruct the adapted predictor |
| 34 | [Forecast windows](lessons/34-forecast-windows/README.md) | Origins, horizons, and information availability |
| 35 | [Forecast baselines](lessons/35-forecast-baselines/README.md) | Persistence and seasonal naive comparison |
| 36 | [Temporal convolution](lessons/36-temporal-convolution/README.md) | Reuse convolution for historical sequences |
| 37 | [Forecast evaluation](lessons/37-forecast-evaluation/README.md) | Chronological selection and horizon errors |
| 38 | [Text vocabulary](lessons/38-text-vocabulary/README.md) | Train-only tokenization vocabulary |
| 39 | [Masked embeddings](lessons/39-masked-embeddings/README.md) | Variable-length batches and padding |
| 40 | [Text classifier](lessons/40-text-classifier/README.md) | End-to-end embedding-mean baseline |
| 41 | [Text order](lessons/41-text-order/README.md) | Demonstrate information discarded by pooling |
| 42 | [Hand-worked attention](lessons/42-attention-handworked/README.md) | Queries, keys, values, and weighted sums |
| 43 | [Attention shapes](lessons/43-attention-shapes/README.md) | Batched heads and matrix dimensions |
| 44 | [Positions and masks](lessons/44-positions-masks/README.md) | Order and allowed interactions |
| 45 | [Transformer block](lessons/45-transformer-block/README.md) | Residual paths, normalization, feed-forward updates |
| 46 | [Attention comparison](lessons/46-attention-comparison/README.md) | Test whether added complexity helps |
| 47 | [Debug contracts](lessons/47-debug-contracts/README.md) | Shape/type bugs and repaired behavior |
| 48 | [Debug training](lessons/48-debug-training/README.md) | Gradients, rates, modes, and fitting checks |
| 49 | [Debug evaluation](lessons/49-debug-evaluation/README.md) | Leakage and misleading scores |
| 50 | [Capstone framing](lessons/50-capstone-framing/README.md) | Define use, information, budget, and success |
| 51 | [Capstone baseline](lessons/51-capstone-baseline/README.md) | Establish a serious comparator |
| 52 | [Capstone network](lessons/52-capstone-network/README.md) | Justify and fit a neural candidate |
| 53 | [Capstone selection](lessons/53-capstone-selection/README.md) | Validation selection, then final test |
| 54 | [Capstone inference](lessons/54-capstone-inference/README.md) | Persistent, input-checked prediction interface |

Reference solutions are supplied throughout, including the capstone. For the
last project, attempt each decision yourself before reading the worked design.
The reference capstone demonstrates the procedure on a new synthetic problem;
it is not evidence that you can skip evaluating a real deployment dataset.

## Recurring design questions

From the first project onward, gradually build the habit of asking:

1. What would a useful prediction mean, and what information is available then?
2. What simple baseline must this model beat?
3. What input structure or invariance justifies this architecture?
4. Are output representation, loss, and evaluation metric compatible?
5. Can the model fit a tiny subset before we attempt a full training run?
6. Does validation failure suggest data trouble, optimization trouble, or capacity?
7. Is the improvement worth the extra compute, complexity, and inference cost?

The questions are the backbone of the course. The architectures are examples
through which we learn to answer them.
