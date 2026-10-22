import contextlib
import io
from pathlib import Path
import runpy
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def execute_lesson(name):
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(ROOT / "lessons" / name / "train.py"))


class FoundationTests(unittest.TestCase):
    def test_autograd_agrees_with_derived_gradients_after_reset(self):
        result = execute_lesson("02-inspect-gradients")
        torch.testing.assert_close(result["weight"].grad, result["manual_weight_gradient"])
        torch.testing.assert_close(result["bias"].grad, result["manual_bias_gradient"])

    def test_layer_refactor_preserves_the_learned_function(self):
        result = execute_lesson("03-layers-and-optimizers")
        torch.testing.assert_close(
            result["new_predictions"], torch.tensor([[7.0], [9.0]]),
            atol=1e-5, rtol=0,
        )

    def test_saved_preprocessing_and_weights_preserve_predictions(self):
        result = execute_lesson("09-save-and-load")
        torch.testing.assert_close(result["actual"], result["expected"], atol=0, rtol=0)

    def test_binary_loss_remains_finite_for_confident_wrong_scores(self):
        result = execute_lesson("10-logits-and-probabilities")
        torch.testing.assert_close(result["stable_losses"], torch.full((2, 1), 1000.0))


if __name__ == "__main__":
    unittest.main()
