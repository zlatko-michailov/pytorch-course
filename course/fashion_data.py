"""Offline Fashion-MNIST data contract, shared only by sessions 18-25 and 30-33."""

import argparse
import gzip
from pathlib import Path
import struct

import torch
from torch.utils.data import DataLoader, TensorDataset


CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
PREPROCESS = {"channels": 1, "height": 28, "width": 28, "scale": 255.0,
              "mean": 0.5, "std": 0.5}
SPLIT = "randperm(seed=1801); first 10% validation; remainder training"


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--smoke", action="store_true",
                        help="Synthetic pipeline check; NOT Fashion-MNIST evidence")
    parser.add_argument("--data", type=Path, default=Path("data/fashion"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=0,
                        help="0 means the entire training partition")
    parser.add_argument("--max-val", type=int, default=0)
    return parser


def read_idx(root, name, image):
    path = root / name
    if not path.is_file():
        path = root / (name + ".gz")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}[.gz] in {root}; no download attempted")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        raw = handle.read()
    header_size = 16 if image else 8
    if len(raw) < header_size:
        raise ValueError(f"Truncated IDX header: {path}")
    if image:
        magic, count, height, width = struct.unpack(">IIII", raw[:16])
        if magic != 2051 or (height, width) != (28, 28):
            raise ValueError(f"Expected 28x28 IDX images: {path}")
        expected = count * height * width
    else:
        magic, count = struct.unpack(">II", raw[:8])
        if magic != 2049:
            raise ValueError(f"Expected IDX labels: {path}")
        expected = count
    if count == 0 or len(raw) - header_size != expected:
        raise ValueError(f"IDX payload length does not match header: {path}")
    values = torch.frombuffer(bytearray(raw[header_size:]), dtype=torch.uint8).clone()
    return values.reshape(count, 1, 28, 28) if image else values.long()


def raw_pair(root, train):
    prefix = "train" if train else "t10k"
    images = read_idx(root, f"{prefix}-images-idx3-ubyte", True)
    labels = read_idx(root, f"{prefix}-labels-idx1-ubyte", False)
    expected = 60000 if train else 10000
    if len(images) != expected or len(labels) != expected:
        raise ValueError(f"Expected the official {expected}-example {prefix} split")
    if labels.min().item() < 0 or labels.max().item() >= len(CLASSES):
        raise ValueError("Labels must be class indices 0..9")
    return images, labels


def fixture(count, seed):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(count) % 10
    images = torch.randint(0, 50, (count, 1, 28, 28),
                           generator=generator, dtype=torch.uint8)
    for index, label in enumerate(labels):
        row = 2 * int(label) + 2
        images[index, :, row:row + 3, 3:25] = 220
    return images, labels


def normalize(images):
    return (images.float() / PREPROCESS["scale"] - PREPROCESS["mean"]) / PREPROCESS["std"]


def partitions(args):
    if args.epochs < 1 or args.max_train < 0 or args.max_val < 0:
        raise ValueError("epochs must be positive; subset limits must be nonnegative")
    if args.smoke:
        print("SYNTHETIC SMOKE: tensor/loop checks only; no clothing-quality claim")
        images, labels = fixture(160, 1800)
    else:
        images, labels = raw_pair(args.data, train=True)
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(1801))
    cut = len(order) // 10
    val_ids, train_ids = order[:cut], order[cut:]
    assert not torch.isin(train_ids, val_ids).any()
    if args.max_train:
        train_ids = train_ids[:args.max_train]
    if args.max_val:
        val_ids = val_ids[:args.max_val]
    images = normalize(images)
    train = TensorDataset(images[train_ids], labels[train_ids])
    val = TensorDataset(images[val_ids], labels[val_ids])
    print(f"train={len(train)} validation={len(val)}; official test NOT opened")
    return train, val


def test_partition(args):
    images, labels = fixture(40, 1802) if args.smoke else raw_pair(args.data, train=False)
    return TensorDataset(normalize(images), labels)


def loader(dataset, shuffle=False, seed=1900, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)
