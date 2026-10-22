"""Focused torch/stdlib checks; run with python -m lessons.33-transfer-inference.check_contracts."""

import gzip
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

import torch

from course.fashion_data import CLASSES, arguments, normalize, partitions, read_idx
from course.image_augmentation import make_augment
from course.image_training import load_model, mlp, save_model
from course.transfer_contract import (
    TinyFixture, build_initial, load_transfer, preprocess, save_transfer, transfer_arguments,
)
from course.transfer_training import assert_frozen, configure, snapshot, train_stage


class ContractChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3300)
        self.root = Path(".artifacts/owned-vision-contract-checks")
        self.root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self.root.rmdir)

    def path(self, name):
        path = self.root / name
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_idx_compression_endianness_and_corruption(self):
        pixels = bytes([0, 255] + [127] * (2 * 28 * 28 - 2))
        raw = struct.pack(">IIII", 2051, 2, 28, 28) + pixels
        image_path = self.path("images.gz")
        with gzip.open(image_path, "wb") as handle:
            handle.write(raw)
        images = read_idx(self.root, "images", True)
        self.assertEqual(images.shape, (2, 1, 28, 28))
        self.assertEqual(normalize(images)[0, 0, 0, 0].item(), -1.0)
        self.assertEqual(normalize(images)[0, 0, 0, 1].item(), 1.0)
        label_path = self.path("labels")
        label_path.write_bytes(struct.pack(">II", 2049, 2) + bytes([3, 9]))
        self.assertEqual(read_idx(self.root, "labels", False).tolist(), [3, 9])
        for payload in [b"x", raw[:-1], struct.pack(">IIII", 999, 2, 28, 28) + pixels]:
            bad = self.path("bad-idx")
            bad.write_bytes(payload)
            with self.assertRaises(ValueError):
                read_idx(self.root, "bad-idx", True)
        with self.assertRaises(FileNotFoundError):
            read_idx(self.root, "missing", True)

    def test_split_repeatability_and_train_only_augmentation(self):
        args = arguments("check").parse_args(["--smoke", "--epochs", "1"])
        with patch("course.fashion_data.raw_pair", side_effect=AssertionError("Real data accessed")):
            train, val = partitions(args)
            repeated_train, repeated_val = partitions(args)
        self.assertTrue(torch.equal(train.tensors[0], repeated_train.tensors[0]))
        self.assertTrue(torch.equal(val.tensors[0], repeated_val.tensors[0]))
        before = val.tensors[0].clone()
        augmenter = make_augment()
        first = augmenter(train.tensors[0][:8])
        repeated = make_augment()(train.tensors[0][:8])
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, train.tensors[0][:8]))
        self.assertTrue(torch.equal(before, val.tensors[0]))

    def test_image_roundtrip_and_metadata_rejection(self):
        args = arguments("check").parse_args(["--smoke", "--epochs", "1"])
        model = mlp().eval()
        images = torch.randn(3, 1, 28, 28)
        path = self.path("mlp.pt")
        save_model(path, model, "mlp-784-64-10", args)
        restored, payload = load_model(path, mlp, True)
        with torch.no_grad():
            torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            load_model(path, mlp, False)
        payload["classes"] = list(reversed(CLASSES))
        torch.save(payload, path)
        with self.assertRaises(ValueError):
            load_model(path, mlp, True)

    def test_frozen_buffers_finetune_and_transfer_roundtrip(self):
        args = transfer_arguments("check").parse_args(["--smoke", "--epochs", "1",
                                                       "--max-train", "24", "--max-val", "8"])
        train, val = partitions(args)
        model, provenance = build_initial(args)
        train_stage(model, train, val, args)
        train_stage(model, train, val, args, fine_tune=True)
        path = self.path("adapted.pt")
        save_transfer(path, model, provenance, args, "finetuned")
        restored, payload = load_transfer(path, True)
        images = preprocess(val.tensors[0][:2], True)
        with torch.no_grad():
            logits = restored(images)
            torch.testing.assert_close(model(images), logits, rtol=0, atol=0)
        names = [payload["classes"][index] for index in logits.argmax(dim=1).tolist()]
        self.assertEqual(len(names), 2)
        self.assertTrue(all(name in CLASSES for name in names))
        with self.assertRaises(ValueError):
            load_transfer(path, False)
        with self.assertRaises(ValueError):
            preprocess(torch.full((1, 1, 28, 28), 255.0), True)
        incorrect = TinyFixture()
        configure(incorrect)
        before = snapshot(incorrect)
        incorrect.train()
        with torch.no_grad():
            incorrect(images)
        with self.assertRaisesRegex(AssertionError, "running_mean|running_var|num_batches_tracked"):
            assert_frozen(before, incorrect)


if __name__ == "__main__":
    unittest.main()
