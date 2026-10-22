# Learning neural networks with PyTorch

An incremental, project-based course for an experienced C/C++ and C# programmer
with some Python experience and a working knowledge of linear algebra, calculus,
and statistics.

**Goal:** learn to frame a problem, choose an appropriate baseline and neural
network, train it, diagnose its behavior, evaluate it honestly, and use it on new
inputs. Sometimes the appropriate conclusion will be not to use a neural network.

## Start here

1. Read the [curriculum](CURRICULUM.md) for the overall progression.
2. Start at [Session 1](lessons/01-learn-a-line/README.md), or continue with
   [Session 2](lessons/02-inspect-gradients/README.md) if you understand the first.
3. Run the supplied code, predict the effect of one change, then run that change.

From this folder, run the first lesson with:

```sh
.venv/bin/python lessons/01-learn-a-line/train.py
```

The environment must be created first; see the [WSL/Linux setup guide](SETUP.md)
or the environment notes below if `.venv/bin/python` does not exist.

## How the course works

The [curriculum](CURRICULUM.md#all-54-lessons-in-teaching-order) links all 54
sessions in prerequisite order, with complete reference programs, explanations,
controlled experiments, and checkpoints. A project is a group of sessions:
Lesson 1 is complete on its own; Project 1 comprises Lessons 1-3.

The complete course is available to read ahead, but you do not need to rush
through it. We can still adapt any session to your questions and experimental
results, or insert a prerequisite before continuing.

Sessions 1-17 explain every nonblank line of their training programs,
including unfamiliar Python syntax. There is no Python beginner course to sit
through, and no assumption that you already know Python-specific behavior.
Later sessions walk through the new operations and shapes in detail while
reusing already-explained training mechanics. All supplied programs are complete;
independent exercises have worked reference implementations, including the capstone.

The teaching cycle is:

**Problem -> minimal model -> prediction -> loss -> training -> inspection ->
controlled experiment -> explanation.**

We will distinguish "the code ran," "the training loss decreased," and "the model
is useful." Those are different achievements.

Each session separates concepts already introduced from new concepts. When a
mathematical or Python prerequisite is missing, we insert a focused session before
depending on it. Session estimates are not deadlines.

Exercises gradually change from editing one supplied line, to completing a small
part, to implementing from a specification, to defending a design. We do not remove
the scaffolding until you can explain the behavior.

## Environment

The course supports Python 3.12-3.13 and CPU execution on **WSL 2, native Linux,
and macOS**. A GPU, Apple Silicon, notebook server, or cloud account is not
required. Real-image lessons need local resources; the core examples do not.

**For a new WSL/Linux installation, follow [SETUP.md](SETUP.md).** It covers
Ubuntu prerequisites, source transfer without the Mac virtual environment,
a version-pinned CPU-only pip profile, offline downloads from Windows or Linux,
optional torchvision, and VS Code in WSL. The lesson commands remain the same.

Python 3.12 is a course environment choice, not a requirement of the first
lesson. Apple-provided Python 3.9 can also run that lesson with a compatible
PyTorch release such as 2.8.0.

`pyproject.toml` declares the supported Python/PyTorch range.
`requirements/linux-cpu.txt` selects the Linux CPU build, with matching optional
vision dependencies in `requirements/linux-vision-cpu.txt`. The existing
`uv.lock` records the original corporate-source environment, not this Linux CPU
profile. Use `.venv/bin/python` to run the lessons without learning an
additional package manager.

Do not replace or install packages into macOS's system Python.

**Existing Mac setup (preserved):** the project-local environment contains
Python 3.12.15 and PyTorch 2.14.1. On rechecking after the earlier installation
attempts were stopped, PyTorch was already installed, and `uv.lock` was present.
This working installation was preserved rather than replaced.

PyTorch may print a warning that NumPy is unavailable. The torch-only lessons
do not need NumPy, and their tensor calculations and training still work.
External image/pretrained workflows declare additional prerequisites in their
lessons; install them only through an approved method when needed. Do not
suppress the warning or modify system Python.

### Downloaded offline bundle

The transferred bundle is at:

```text
/Users/zlatkom/Downloads/pytorch-wheels-4b3b02e58e9443e68fe89657845a4f54
```

It contains PyTorch 2.8.0 and its nine dependencies as wheels for Python 3.12 on
macOS Apple Silicon. All ten wheel archives passed their ZIP checksum checks, and
an offline dependency-resolution dry run succeeded. No packages were changed by
those checks. These macOS wheels cannot be used on WSL/Linux; obtain the
[Linux CPU wheel set](SETUP.md#restricted-networks-and-offline-installation) instead.

Keep the whole folder. It provides an alternative compatible version for the
first lesson, not an exact restoration of the current PyTorch 2.14.1 environment
recorded in `uv.lock`.

### Future environment changes

Use only an IT-approved package source or approved offline wheels. `uv` does not
automatically inherit `pip`'s configuration. Changing Python versions does not
remove a network restriction. Do not bypass corporate network restrictions,
disable certificate verification, or copy credentials into course files.

Keep `uv.lock` and use `--locked` with future `uv sync` commands only when
reproducing the original Mac/corporate-source setup, not the Linux CPU profile.
The approved package source, compatible local artifacts, or an adequate existing
cache must also be available. If deliberately switching to the older offline
bundle, update the dependency record accordingly rather than assuming it matches
the current lockfile. On WSL/Linux, follow
the pip requirements in the setup guide and do not run `uv sync` over that
environment: the existing lock includes unneeded Linux CUDA dependencies.

Activating the virtual environment is optional. Using `.venv/bin/python`
explicitly selects the interpreter and dependencies for this course.

## Continuing in a later conversation

Use this folder as the durable course material. A useful continuation message is:

> Continue the course in `~/src/pytorch-course`. I am on Project 1,
> Session 2. Here are my predictions, results, and questions: ...

Your reading and experiments, not the fact that an example has been executed,
determine whether a session is complete.

## Offline examples versus external-data projects

The core examples use small, explicitly synthetic datasets so you can inspect
the intended behavior and run them locally without a download.

Image lessons use **local** dataset files, and transfer learning uses **local**
pretrained weights and its documented optional software. Their synthetic
`--smoke` paths exercise the pipeline without those resources; these are not
Fashion-MNIST benchmarks or evidence that pretrained transfer works on real data.
No lesson silently substitutes synthetic data for a missing real dataset, and
none downloads resources automatically.

Sessions [18](lessons/18-images-flat/README.md) and
[30](lessons/30-transfer-contract/README.md) explain the local image-file layout
and pretrained-weight requirements respectively. Their complete real-data
commands are separate from the synthetic smoke commands.

For any external dataset or weight artifact, follow its lesson's source,
format, permissions, and preprocessing instructions. Respect corporate network
restrictions. Generated checkpoints and reports go beneath the documented
`artifacts/` or `.artifacts/` folders, not beside the lesson sources, and are not
course content to upload.

## Course checks

These commands use the existing environment and install nothing:

```sh
.venv/bin/python scripts/check_course.py --static-only
.venv/bin/python scripts/check_course.py
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m lessons.33-transfer-inference.check_contracts
.venv/bin/python -m unittest lessons.54-capstone-inference.test_contracts
```

The full checker runs lessons in numerical order so saved-model prerequisites
are created before inference lessons. Image/transfer sessions use their
explicitly labeled synthetic smoke paths; the real-data paths are not validated
by those checks. To run a subset after the source/link checks:

```sh
.venv/bin/python scripts/check_course.py --lessons 2 3
```

Some later lessons use package execution (`python -m ...`) so their reusable
support code is imported correctly. Use the exact command in each lesson.

## References

The lessons are written for this course. Official documentation is provided as
reference, not as an assumed prerequisite:

- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [Tensor basics](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
