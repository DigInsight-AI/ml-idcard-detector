import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from idcard_detector.training import _save_training_checkpoints


class PoseCheckpointTest(unittest.TestCase):
    def test_pose_improvement_is_saved_despite_worse_combined_fitness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = root / "last.pt"
            trainer = SimpleNamespace(wdir=root, last=last, epoch=0,
                                      metrics={"metrics/mAP50-95(P)": 0.5}, fitness=1.5)
            last.write_bytes(b"first")
            _save_training_checkpoints(trainer, 1)
            trainer.epoch = 1
            trainer.metrics["metrics/mAP50-95(P)"] = 0.6
            trainer.fitness = 1.4
            last.write_bytes(b"better corners")
            (root / "epoch0.pt").write_bytes(b"old")
            (root / "epoch1.pt").write_bytes(b"new")
            _save_training_checkpoints(trainer, 1)
            self.assertEqual((root / "best_pose.pt").read_bytes(), b"better corners")
            self.assertEqual(json.loads((root / "best_pose.json").read_text()),
                             {"epoch": 2, "pose_map50_95": 0.6})
            self.assertFalse((root / "epoch0.pt").exists())
            self.assertTrue((root / "epoch1.pt").exists())
            trainer.metrics["metrics/mAP50-95(P)"] = 0.55
            last.write_bytes(b"worse corners")
            _save_training_checkpoints(trainer, 1)
            self.assertEqual((root / "best_pose.pt").read_bytes(), b"better corners")
            self.assertEqual(last.read_bytes(), b"worse corners")

    def test_missing_or_nonfinite_metric_does_not_save_a_pose_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = root / "last.pt"
            last.write_bytes(b"checkpoint")
            for score in (None, float("nan"), float("inf")):
                trainer = SimpleNamespace(wdir=root, last=last, epoch=0,
                                          metrics={"metrics/mAP50-95(P)": score})
                _save_training_checkpoints(trainer, 1)
                self.assertFalse((root / "best_pose.pt").exists())
