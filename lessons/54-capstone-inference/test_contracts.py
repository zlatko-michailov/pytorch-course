"""Focused inference/release contracts; run after session 53, with stdlib unittest."""
import copy
import importlib
import io
from pathlib import Path
import unittest
from unittest import mock

import torch

from course.capstone import RELEASE_PATH, load_checkpoint, predict_records


class ReleaseContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RELEASE_PATH.exists():
            raise RuntimeError("Run .venv/bin/python -m lessons.53-capstone-selection.train first.")

    def setUp(self):
        self.record = {"temperature_c": 82., "vibration_mm_s": 2.6,
                       "load_pct": 75., "machine_type": "pump"}

    def test_new_inputs_and_rejections(self):
        unknown_missing = {**self.record, "temperature_c": None, "machine_type": "prototype"}
        results = predict_records(RELEASE_PATH, [self.record, unknown_missing])
        self.assertTrue(results[1]["unknown_machine_type"])
        self.assertEqual(results[1]["missing_numeric_fields"], 1)
        for result in results:
            self.assertTrue(0 <= result["probability"] <= 1)
        invalid = [
            {**self.record, "load_pct": True},
            {**self.record, "temperature_c": float("inf")},
            {**self.record, "temperature_c": float("nan")},
            {**self.record, "temperature_c": 10 ** 1000},
            {**self.record, "vibration_mm_s": -0.1},
            {**self.record, "machine_type": " pump "},
            {**self.record, "extra": 1},
            {key: value for key, value in self.record.items() if key != "load_pct"},
        ]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises((TypeError, ValueError)):
                predict_records(RELEASE_PATH, [record])
        for batch in ([], [self.record] * 129, self.record):
            with self.subTest(batch_type=type(batch)), self.assertRaises(ValueError):
                predict_records(RELEASE_PATH, batch)

    def test_repeat_load_and_schema_rejection(self):
        self.assertEqual(predict_records(RELEASE_PATH, [self.record]),
                         predict_records(RELEASE_PATH, [self.record]))
        model, preprocessing, payload = load_checkpoint(RELEASE_PATH)
        other, other_preprocessing, _ = load_checkpoint(RELEASE_PATH)
        self.assertEqual(preprocessing, other_preprocessing)
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, other.state_dict()[name]))
        incompatible = copy.deepcopy(payload)
        incompatible["schema"]["numeric"] = list(reversed(incompatible["schema"]["numeric"]))
        with mock.patch("course.capstone.torch.load", return_value=incompatible):
            with self.assertRaisesRegex(ValueError, "schema"):
                load_checkpoint(RELEASE_PATH)

    def test_missing_release_does_not_train(self):
        inference = importlib.import_module("lessons.54-capstone-inference.train")
        missing = Path(__file__).resolve().parent / "artifacts/unavailable-release.pt"
        self.assertFalse(missing.exists())
        with mock.patch.object(inference, "RELEASE_PATH", missing):
            with mock.patch("sys.argv", ["train"]):
                with mock.patch("course.capstone.train_candidate", side_effect=AssertionError("trained")):
                    with self.assertRaisesRegex(SystemExit, "53-capstone-selection"):
                        inference.main()
        self.assertFalse(missing.exists())

    def test_cached_selection_never_generates_or_trains(self):
        selection = importlib.import_module("lessons.53-capstone-selection.train")
        with mock.patch.object(selection, "make_split", side_effect=AssertionError("generated data")):
            with mock.patch.object(selection, "train_candidate", side_effect=AssertionError("trained")):
                with mock.patch.object(selection, "evaluate", side_effect=AssertionError("evaluated")):
                    with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                        selection.main()
        self.assertIn("CACHED RELEASE", output.getvalue())


if __name__ == "__main__":
    unittest.main()
