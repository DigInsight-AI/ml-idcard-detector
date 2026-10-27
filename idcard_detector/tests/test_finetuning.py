import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from idcard_detector.training import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    TrainingError,
    load_training_config,
    train_pose_from_config,
)


class FineTuningTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.weights = self.root / "best.pt"
        self.weights.write_bytes(b"checkpoint")
        self.document = yaml.safe_load(DEFAULT_CONFIG.read_text())
        self.document["model"].update(
            pretrained=True, initial_weights=str(self.weights)
        )
        self.document["output"].update(
            project=str(self.root / "runs"), name="fine-tuned", existing_run="error"
        )
        self.document["training"].update(
            epochs=50, learning_rate=0.0001, warmup_bias_lr=0.0
        )
        patcher = mock.patch(
            "idcard_detector.training._build_combined_dataset",
            return_value={"data_yaml": str(self.root / "data.yaml")},
        )
        self.dataset = patcher.start()
        self.addCleanup(patcher.stop)

    def load(self, document=None):
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump(document or self.document))
        return load_training_config(config)

    def test_initial_weights_resolve_relative_to_project_root(self):
        with mock.patch("idcard_detector.training.PROJECT_ROOT", self.root):
            self.document["model"]["initial_weights"] = "best.pt"
            config = self.load()
        self.assertEqual(config["model"], str(self.weights))
        self.assertEqual(config["initial_weights"], str(self.weights))
        self.assertTrue(config["arguments"]["pretrained"])
        self.assertNotIn("initial_weights", config["arguments"])
        self.assertEqual(config["arguments"]["warmup_bias_lr"], 0.0)
        self.assertIsNone(config["resume_from"])

    def test_initial_weights_require_pretrained_true(self):
        self.document["model"]["pretrained"] = False
        with self.assertRaisesRegex(TrainingError, "pretrained must be true"):
            self.load()
        self.dataset.assert_not_called()

    def test_missing_or_invalid_initial_checkpoint_is_rejected(self):
        for filename, error in [("missing.pt", "does not exist"), ("best.yaml", "must be a .pt")]:
            with self.subTest(filename=filename):
                self.document["model"]["initial_weights"] = str(self.root / filename)
                with self.assertRaisesRegex(TrainingError, error):
                    self.load()
        self.dataset.assert_not_called()

    def test_initial_weights_and_explicit_resume_are_mutually_exclusive(self):
        self.document["training"]["resume_from"] = str(self.weights)
        with self.assertRaisesRegex(TrainingError, "cannot both be set"):
            self.load()
        self.dataset.assert_not_called()

    def test_fresh_fine_tuning_preserves_weights_and_disables_resume(self):
        config = self.load()
        captured = {}

        class FakeYOLO:
            def __init__(self, model):
                captured["model"] = model

            def add_callback(self, event, callback):
                pass

            def train(self, **arguments):
                captured.update(arguments)
                return SimpleNamespace(save_dir=str(Path(arguments["project"]) / arguments["name"]))

        with mock.patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=FakeYOLO)}):
            with mock.patch("idcard_detector.training.load_training_config", return_value=config):
                result = train_pose_from_config(materialize=False)
        self.assertEqual(captured["model"], str(self.weights))
        self.assertTrue(captured["pretrained"])
        self.assertFalse(captured["resume"])
        self.assertEqual(captured["epochs"], 50)
        self.assertEqual(captured["lr0"], 0.0001)
        self.assertEqual(result["initial_weights"], str(self.weights))
        self.assertIsNone(result["resume_from"])

    def test_fine_tuning_will_not_overwrite_existing_output(self):
        config = self.load()
        output = Path(config["arguments"]["project"]) / config["arguments"]["name"]
        output.mkdir(parents=True)
        marker = output / "results.csv"
        marker.write_text("existing results")
        with mock.patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=mock.Mock())}):
            with mock.patch("idcard_detector.training.load_training_config", return_value=config):
                with self.assertRaisesRegex(TrainingError, "already exists"):
                    train_pose_from_config(materialize=False)
        self.assertEqual(marker.read_text(), "existing results")

    def test_bias_warmup_must_be_finite_and_nonnegative(self):
        for value in [-0.1, float("nan"), float("inf")]:
            with self.subTest(value=value):
                self.document["training"]["warmup_bias_lr"] = value
                with self.assertRaisesRegex(TrainingError, "warmup_bias_lr"):
                    self.load()

    def test_v2_recipe_has_consistent_initialization_and_schedule(self):
        document = yaml.safe_load((PROJECT_ROOT / "config/config_v2.yaml").read_text())
        document["model"]["initial_weights"] = str(self.weights)
        config = self.load(document)
        args = config["arguments"]
        self.assertEqual(args["name"], "idcard-pose-v2")
        self.assertEqual(config["existing_run"], "error")
        self.assertEqual(args["epochs"], 50)
        self.assertEqual(args["lr0"], 0.0001)
        self.assertEqual(args["lrf"], 0.1)
        self.assertEqual(args["mosaic"], 0.0)
        self.assertEqual(args["multi_scale"], 0.0)
        self.assertEqual(config["keypoint_loss_sigmas"], [0.25] * 4)
