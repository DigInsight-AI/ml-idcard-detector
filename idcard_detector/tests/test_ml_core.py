import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

from idcard_detector.cli import build_parser
from idcard_detector.geometry import GeometryError, validate_keypoints
from idcard_detector.training import (
    DEFAULT_CONFIG,
    TrainingError,
    _configure_device_runtime,
    _prepare_output_directory,
    load_training_config,
    train_pose_from_config,
)
from idcard_detector.training_logging import EpochMetricsLogger


class MLPipelineContractTest(unittest.TestCase):
    def test_reversed_keypoint_winding_is_rejected(self):
        with self.assertRaises(GeometryError):
            validate_keypoints([[0, 0], [0, 80], [100, 80], [100, 0]])

    def test_cli_contains_only_train_and_infer(self):
        parser = build_parser()
        commands = next(
            action for action in parser._actions if getattr(action, "choices", None)
        )
        self.assertEqual(set(commands.choices), {"train", "infer"})

    def test_training_defaults_to_config_v1(self):
        arguments = build_parser().parse_args(["train"])
        self.assertEqual(Path(arguments.config), DEFAULT_CONFIG)
        self.assertEqual(DEFAULT_CONFIG.name, "config_v1.yaml")

    def test_config_contains_full_scratch_training_arguments(self):
        configuration = load_training_config(DEFAULT_CONFIG)
        arguments = configuration["arguments"]
        self.assertEqual(configuration["model"], "yolo11n-pose.yaml")
        self.assertFalse(arguments["pretrained"])
        self.assertEqual(arguments["imgsz"], 640)
        self.assertEqual(arguments["optimizer"], "AdamW")
        self.assertIn("mosaic", arguments)
        self.assertIn("project", arguments)
        self.assertNotIn("resume", arguments)
        self.assertIsNone(configuration["resume_from"])
        self.assertEqual(configuration["keep_last_n"], 3)
        self.assertEqual(arguments["save_period"], 5)
        self.assertTrue(arguments["exist_ok"])
        self.assertEqual(configuration["existing_run"], "prompt")
        self.assertEqual(
            Path(arguments["project"]) / arguments["name"],
            DEFAULT_CONFIG.parent.parent / "runs" / "idcard-pose-v1",
        )
        dataset = configuration["dataset"]
        self.assertEqual(
            dataset["included_images"],
            dataset["train_images"] + dataset["validation_images"],
        )
        self.assertGreater(dataset["train_images"], 0)
        self.assertGreater(dataset["validation_images"], 0)
        self.assertGreaterEqual(dataset["duplicate_entries_replaced"], 0)
        self.assertGreaterEqual(len(dataset["source_datasets"]), 3)

    def test_config_rejects_pretrained_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
                "pretrained: false", "pretrained: true"
            )
            config.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(TrainingError, "pretrained must be false"):
                load_training_config(config)

    def test_config_resolves_resume_checkpoint_from_output_project(self):
        checkpoint = DEFAULT_CONFIG.parent.parent / "runs" / "resume-test" / "last.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        try:
            with tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "config.yaml"
                text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
                    "resume_from: null", "resume_from: resume-test/last.pt"
                )
                config.write_text(text, encoding="utf-8")
                configuration = load_training_config(config)
                self.assertEqual(configuration["resume_from"], str(checkpoint))
                self.assertNotIn("resume_from", configuration["arguments"])
        finally:
            checkpoint.unlink(missing_ok=True)
            checkpoint.parent.rmdir()

    def test_training_passes_config_values_to_ultralytics(self):
        captured = {}

        class FakeYOLO:
            def __init__(self, model):
                captured["model"] = model

            def add_callback(self, event, callback):
                captured.setdefault("callback_events", []).append(event)

            def train(self, **arguments):
                captured.update(arguments)
                return SimpleNamespace(save_dir="/tmp/run")

        fake_module = SimpleNamespace(YOLO=FakeYOLO)
        configuration = load_training_config(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            configuration["arguments"] = dict(configuration["arguments"])
            configuration["arguments"]["project"] = directory
            with mock.patch.dict("sys.modules", {"ultralytics": fake_module}):
                with mock.patch(
                    "idcard_detector.training.load_training_config",
                    return_value=configuration,
                ):
                    train_pose_from_config(DEFAULT_CONFIG, materialize=False)

        self.assertEqual(captured["model"], "yolo11n-pose.yaml")
        self.assertEqual(captured["epochs"], 150)
        self.assertEqual(captured["imgsz"], 640)
        self.assertFalse(captured["pretrained"])
        self.assertEqual(
            captured["callback_events"],
            [
                "on_train_start",
                "on_train_epoch_start",
                "on_fit_epoch_end",
                "on_model_save",
            ],
        )
        self.assertNotIn("resume", captured)

    def test_training_resumes_checkpoint_in_configured_output_directory(self):
        captured = {}

        class FakeYOLO:
            def __init__(self, model):
                captured["model"] = model
                self.trainer = None

            def add_callback(self, event, callback):
                captured.setdefault("callback_events", []).append(event)

            def train(self, **arguments):
                captured.update(arguments)
                self.trainer = SimpleNamespace(save_dir="/tmp/runs/resumed")

        class FakePoseTrainer:
            pass

        ultralytics = ModuleType("ultralytics")
        ultralytics.YOLO = FakeYOLO
        models = ModuleType("ultralytics.models")
        yolo = ModuleType("ultralytics.models.yolo")
        pose = ModuleType("ultralytics.models.yolo.pose")
        pose.PoseTrainer = FakePoseTrainer
        modules = {
            "ultralytics": ultralytics,
            "ultralytics.models": models,
            "ultralytics.models.yolo": yolo,
            "ultralytics.models.yolo.pose": pose,
        }
        configuration = {
            "config_path": "/tmp/config_v1.yaml",
            "model": "yolo11n-pose.yaml",
            "existing_run": "prompt",
            "resume_from": "/tmp/runs/resumed/weights/last.pt",
            "arguments": {
                "data": "/tmp/data.yaml",
                "project": "/tmp/runs",
                "name": "resumed",
            },
            "keep_last_n": 3,
            "dataset": {},
        }
        with mock.patch.dict("sys.modules", modules):
            with mock.patch(
                "idcard_detector.training.load_training_config",
                return_value=configuration,
            ):
                result = train_pose_from_config(DEFAULT_CONFIG, materialize=False)

        self.assertEqual(captured["model"], configuration["resume_from"])
        self.assertTrue(captured["resume"])
        self.assertEqual(captured["_resume_save_dir"], "/tmp/runs/resumed")
        self.assertIn("trainer", captured)
        self.assertEqual(result["resume_from"], configuration["resume_from"])

    def test_epoch_logger_writes_detailed_metrics(self):
        class FakeMetric:
            def __init__(self, result, map75):
                self._result = result
                self.map75 = map75
                self.all_ap = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, map75, 0.7]])

            def class_result(self, index):
                self.assert_index(index)
                return self._result

            @staticmethod
            def assert_index(index):
                if index != 0:
                    raise AssertionError("unexpected class result index")

        with tempfile.TemporaryDirectory() as directory:
            logger = EpochMetricsLogger()
            pose_metrics = SimpleNamespace(
                ap_class_index=np.array([0]),
                names={0: "idcard"},
                nt_per_image=np.array([7]),
                nt_per_class=np.array([8]),
                box=FakeMetric((0.9, 0.8, 0.85, 0.7), 0.75),
                pose=FakeMetric((0.88, 0.77, 0.82, 0.66), 0.72),
            )
            trainer = SimpleNamespace(
                save_dir=directory,
                args=SimpleNamespace(val=True),
                epoch=0,
                epochs=10,
                tloss={"box_loss": 1.2, "pose_loss": 2.3},
                metrics={
                    "metrics/precision(B)": 0.9,
                    "metrics/recall(B)": 0.8,
                    "metrics/mAP50(B)": 0.85,
                    "metrics/mAP50-95(B)": 0.7,
                    "metrics/precision(P)": 0.88,
                    "metrics/recall(P)": 0.77,
                    "metrics/mAP50(P)": 0.82,
                    "metrics/mAP50-95(P)": 0.66,
                    "val/box_loss": 1.0,
                    "val/pose_loss": 2.0,
                },
                lr={"lr/pg0": 0.001},
                fitness=0.75,
                best_fitness=0.75,
                epoch_time=62.0,
                validator=SimpleNamespace(metrics=pose_metrics),
                label_loss_items=lambda values: {
                    "train/%s" % key: value for key, value in values.items()
                },
            )

            logger.on_train_start(trainer)
            logger.on_train_epoch_start(trainer)
            logger.on_fit_epoch_end(trainer)
            logger.on_fit_epoch_end(trainer)

            train_log = (Path(directory) / "train.log").read_text(encoding="utf-8")
            self.assertIn("Epoch [1/10] completed", train_log)
            self.assertIn("mAP75=0.750000", train_log)
            self.assertIn("per-class validation metrics", train_log)
            metric_lines = (
                Path(directory) / "metrics.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(metric_lines), 1)
            record = json.loads(metric_lines[0])
            self.assertEqual(record["class_metrics"][0]["instances"], 8)
            self.assertAlmostEqual(record["class_metrics"][0]["pose_map75"], 0.72)
            class_csv = (Path(directory) / "class_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("pose_map50_95", class_csv)
            self.assertIn("idcard", class_csv)

    def test_mps_runtime_filters_known_pytorch_noise(self):
        _configure_device_runtime({"device": "mps"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn(
                "scatter_reduce_mps does not have a deterministic implementation, "
                "but deterministic algorithms are enabled.",
                UserWarning,
            )
        self.assertEqual(caught, [])

    def test_existing_run_prompt_can_resume_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "pose-v1"
            checkpoint = run_directory / "weights" / "last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            configuration = {
                "resume_from": None,
                "existing_run": "prompt",
                "arguments": {"project": directory, "name": "pose-v1"},
            }

            selected = _prepare_output_directory(configuration, lambda _: "yes")

            self.assertEqual(selected, str(checkpoint.resolve()))
            self.assertTrue(run_directory.is_dir())

    def test_existing_run_prompt_decline_removes_old_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "pose-v1"
            checkpoint = run_directory / "weights" / "last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            (run_directory / "results.csv").write_text("old", encoding="utf-8")
            configuration = {
                "resume_from": None,
                "existing_run": "prompt",
                "arguments": {"project": directory, "name": "pose-v1"},
            }

            selected = _prepare_output_directory(configuration, lambda _: "no")

            self.assertIsNone(selected)
            self.assertFalse(run_directory.exists())
