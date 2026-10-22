# Project 5, Session 18: from an image to a feature vector

[Course curriculum](../../CURRICULUM.md) · [Complete runnable source](train.py)

## Before you start

Review Sessions 7–9: fixed preprocessing, `TensorDataset`, `DataLoader`, and
training/validation/test roles. Review Sessions 14–17: an MLP is a composition of
affine layers and nonlinearities, and validation guides model choices.
Nothing from the tabular project in Sessions 26–29 is needed here.

**New:** image axes, integer pixel storage, flattening without losing the batch
axis, class-name mappings, and an offline dataset contract.
There is no training in this session. First make the input representation correct.

## The problem

Fashion-MNIST contains small grayscale clothing images with ten class labels.
We will build a recognizer, beginning with an MLP rather than a convolution.
“Recognize an image” must first become “map a numerical tensor to a class index.”
A file name or a picture displayed on screen is not yet that numerical contract.

Each example has one channel, 28 rows, and 28 columns.
An image batch is `[N, 1, 28, 28]`: batch, channel, height, width.
Our first MLP expects `[N, 784]`, because `1 * 28 * 28 = 784`.
The label tensor is `[N]` with `torch.long` values from 0 through 9.

## Run from the course root

```sh
.venv/bin/python -m lessons.18-images-flat.train --smoke
```

This explicitly generates synthetic stripe patterns. It checks tensor handling,
not the ability to recognize clothes. It uses only the installed PyTorch.

For real data, acquire the official files separately through an approved channel:

```text
data/fashion/train-images-idx3-ubyte.gz
data/fashion/train-labels-idx1-ubyte.gz
data/fashion/t10k-images-idx3-ubyte.gz
data/fashion/t10k-labels-idx1-ubyte.gz
```

Uncompressed versions without `.gz` also work.
Then run:

```sh
.venv/bin/python -m lessons.18-images-flat.train --data data/fashion
```

There is no downloader and no automatic synthetic fallback.
Missing files, malformed headers, wrong counts, and labels outside 0–9 fail.
The official source is
[Zalando Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist),
with files documented at
[`data/fashion`](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion).
The dataset repository identifies its MIT license; retain its notices and verify
your intended use. These lessons do not redistribute its images.
IDX's binary layout is documented on the
[MNIST format page](https://yann.lecun.com/exdb/mnist/); Fashion-MNIST uses that
format, not MNIST's handwritten-digit labels.

## Read the complete code before running experiments

There are two complete sources, not an omitted training framework:

1. [This session's train.py](train.py): inspect one batch and prove a reshape.
2. [Shared fashion_data.py](../../course/fashion_data.py): the entire offline
   reader, split, fixture, normalization, and loader. It is introduced here in full.

### Walk through `train.py`

1. `import torch` binds the module name, like referring to a library namespace.
2. `from course.fashion_data import ...` imports named objects from our helper.
   `course` and `lessons` are namespace packages; running with `-m` makes imports
   resolve from the course root. Do not execute this file by its path instead.
3. `def main():` defines a function; indentation delimits its body.
4. `arguments(...).parse_args()` builds the command-line parser and reads options.
   `args.smoke` is an attribute, analogous to a configuration object's property.
5. `torch.set_num_threads(1)` limits CPU parallelism for these small experiments.
   It is a runtime choice, not a modeling assumption.
6. `train, val = partitions(args)` destructures a returned pair, like C# tuples.
   Neither object contains official test examples.
7. `next(iter(loader(train)))` makes an iterator, then requests its first batch.
   In C++ terms, it obtains the first yielded value rather than indexing the loader.
   The batch is a pair, unpacked into `images` and `labels`.
8. `images.flatten(start_dim=1)` joins axes 1 onward. Axis 0 survives.
9. `assert images.shape[1:] == (1, 28, 28)` checks all axes after the batch.
   `1:` is a slice through the end; `(1, 28, 28)` is a tuple.
10. The next assertions check `[N,784]`, exact inverse reshaping, and target dtype.
    `reshape_as(images)` requests the other tensor's shape, not its values.
11. `torch.equal` checks exact shape and element equality, not approximate equality.
12. `.item()` converts a one-element tensor to a Python number.
    We use it for printing and indexing the ordinary Python `CLASSES` list.
13. The f-strings embed expressions; `:.3f` prints three decimal places.
14. `if __name__ == "__main__": main()` runs the entry point when invoked as a
    program. Importing the module does not start the demonstration.

### Walk through every part of `fashion_data.py`

`CLASSES` defines the official index-to-name mapping, in order:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
The order is part of the model contract, not an alphabetic presentation choice.

`PREPROCESS` is a dictionary: named values looked up with `["key"]`.
For every pixel we compute `(byte / 255 - 0.5) / 0.5`.
Thus byte 0 becomes -1, byte 255 becomes +1, and middle gray is near zero.
These are fixed constants, not statistics fitted on validation or test data.

`arguments` constructs an `argparse.ArgumentParser`.
`action="store_true"` makes `--smoke` a boolean flag.
`type=Path` converts a path string to a `pathlib.Path` object.
`type=int` parses integers; `default=...` applies when an option is absent.
`--max-train` and `--max-val` take prefixes of already separated partitions.
Zero means no limit, not an empty partition. `--epochs` is used in later lessons.

`read_idx(root, name, image)` joins paths using `/`, overloaded by `Path`.
It first looks for the uncompressed name, then that name plus `.gz`.
`is_file()` checks existence; failure raises an exception rather than continuing.
The conditional expression `gzip.open if ... else open` selects an opener.
`with opener(path, "rb") as handle:` closes the binary file even on an exception,
similar to C# `using` or RAII.

`handle.read()` reads bytes; `raw[:16]` selects the header.
`struct.unpack(">IIII", ...)` decodes four big-endian unsigned 32-bit integers:
magic number, example count, rows, columns.
The label header has two integers. We check magic numbers 2051 and 2049.
The payload length must match the header exactly.
`bytearray(...)` creates writable byte storage; `torch.frombuffer` views it as
unsigned bytes; `.clone()` gives the tensor its own storage.
The image result is reshaped to `[count,1,28,28]`.
Labels become `long()` integers rather than floating-point regression targets.

`raw_pair` reads matching images and labels and enforces the official sizes:
60,000 development examples and 10,000 official test examples.
`train=False` means the official test files; it does not mean “validation.”
It also verifies label range and count agreement.

`fixture` uses its own `torch.Generator`, seeded independently of model weights.
`arange(count) % 10` cycles labels 0–9.
`randint` creates dark random background pixels.
`enumerate(labels)` yields `(position, value)` pairs; a three-row bright stripe
is placed at a label-dependent position. This deliberately easy artificial rule
is not a clothing simulation.

`normalize` first converts bytes to `float32`, then applies the fixed formula.
Integer arithmetic would be the wrong numerical representation for a network.

`partitions` validates options, chooses real data or the explicit fixture, and
creates one seeded permutation with `torch.randperm`.
For real data, the first 6,000 permuted indices are validation and the remaining
54,000 are training. The seed is always 1801.
`torch.isin(...).any()` checks the partitions do not share indices.
Optional size limits are applied *after* this split.
`TensorDataset(images[indices], labels[indices])` pairs features and targets.
The official test files are never opened by this function.

`test_partition` is a separate operation reserved for explicit final evaluation.
Its smoke branch makes 40 different synthetic examples with seed 1802.
`loader` constructs a `DataLoader` with zero worker subprocesses, batch size 32,
and its own shuffle generator. `shuffle=False` preserves dataset order.
The last batch may be smaller; code must use `len(labels)`, not assume 32.

## Why flattening is a baseline, not an image model

Flattening does not discard numerical pixels: the inverse reshape proves that.
It discards the architectural knowledge that nearby pixels are related.
An MLP can learn spatial patterns, but every input position gets separate weights.
Its first `Linear(784,64)` has `784*64+64 = 50,240` parameters.
There is no automatic reuse of a pattern detector in another image location.
We will make that assumption explicit with convolutions in Session 22.

## Controlled edits: predict before running

1. Change `start_dim=1` to `start_dim=0`.
   Predict `[N*784]`, then observe the assertion fail: the batch axis was lost.
2. Set `--max-train 65`. Predict batches 32, 32, 1 if iterating the whole loader.
   The first batch in this script remains 32, and validation is unchanged.
3. Temporarily replace the normalization formula with `images.float()/255`.
   Predict a range near `[0,1]`; explain why this is a new preprocessing contract,
   not merely a harmless code cleanup. Restore it before Session 19.

## Check your understanding

**Why is the channel dimension still present for grayscale images?**
<details>
<summary>Solution</summary>
Convolutional APIs distinguish channels from height and width, even when channels
equals one. Consistent `[N,C,H,W]` axes also permit RGB images with C=3 later.
</details>

**Can we split the 10,000 official test images to get validation data?**
<details>
<summary>Solution</summary>
Not under this course's protocol. We derive validation from the official 60,000
training examples, and reserve all official test examples for a fixed final model.
Changing that protocol after seeing outcomes changes what the test means.
</details>

**Does a successful smoke run establish the Fashion-MNIST accuracy baseline?**
<details>
<summary>Solution</summary>
No. It verifies shapes, indexing, splitting, and normalization on generated data.
Only the real offline files can support statements about this clothing dataset.
</details>

## Design judgment and limitations

This in-memory reader favors inspectability over throughput or large-dataset scale.
The random split assumes independent, similarly distributed examples; it is not
a universal answer to grouped, chronological, or deployment-specific splits.
A low-resolution grayscale benchmark does not represent product photography.
Keep that distinction when we start reporting losses.
