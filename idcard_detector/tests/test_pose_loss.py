import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from ultralytics.utils.loss import KeypointLoss

from idcard_detector.training import (
    DEFAULT_CONFIG,
    TrainingError,
    _configure_pose_loss,
    load_training_config,
)


class PoseLossTest(unittest.TestCase):
    def test_large_k2_error_keeps_coordinate_gradient_with_training_sigmas(self):
        for sigma, has_gradient in ((0.025, False), (0.25, True)):
            with self.subTest(sigma=sigma):
                prediction = torch.tensor(
                    [[[0.01, 0.0], [0.01, 0.0], [1.49, 0.0], [0.01, 0.0]]],
                    requires_grad=True,
                )
                loss = KeypointLoss(torch.full((4,), sigma))(
                    prediction,
                    torch.zeros_like(prediction),
                    torch.ones((1, 4)),
                    torch.ones((1, 1)),
                )
                loss.backward()
                gradient = prediction.grad[0, 2, 0].item()
                self.assertEqual(gradient > 0, has_gradient)
                self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_training_and_ema_use_broad_sigmas_without_changing_evaluation(self):
        model = torch.nn.Module()
        model.kpt_oks_sigmas = [0.025] * 4
        model.criterion = object()
        ema = copy.deepcopy(model)
        data = {"kpt_oks_sigmas": [0.025] * 4}
        trainer = SimpleNamespace(model=model, ema=SimpleNamespace(ema=ema), data=data)

        _configure_pose_loss(trainer, [0.25] * 4)

        for active in (model, ema):
            self.assertEqual(active.kpt_oks_sigmas, [0.25] * 4)
            self.assertIsNone(active.criterion)
        self.assertEqual(data["kpt_oks_sigmas"], [0.025] * 4)

    def test_loss_sigmas_are_not_forwarded_as_ultralytics_arguments(self):
        configuration = load_training_config(DEFAULT_CONFIG)
        self.assertEqual(configuration["keypoint_loss_sigmas"], [0.25] * 4)
        self.assertNotIn("keypoint_loss_sigmas", configuration["arguments"])

    def test_invalid_loss_sigmas_are_rejected(self):
        for invalid in ("[0.25]", "[0, 0.25, 0.25, 0.25]", "[.nan, 0.25, 0.25, 0.25]"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "config.yaml"
                config.write_text(
                    DEFAULT_CONFIG.read_text().replace(
                        "keypoint_loss_sigmas: [0.25, 0.25, 0.25, 0.25]",
                        "keypoint_loss_sigmas: " + invalid,
                    )
                )
                with self.assertRaisesRegex(TrainingError, "keypoint_loss_sigmas"):
                    load_training_config(config)
