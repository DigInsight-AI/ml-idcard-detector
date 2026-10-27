import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from idcard_detector.augmentation import (
    AugmentationError,
    encode_label,
    generate_dataset,
    transform_sample,
)
from idcard_detector.cli import build_parser
from idcard_detector.training import _build_combined_dataset


class AugmentationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "source"
        self.points = np.float32([[90, 90], [450, 90], [450, 310], [90, 310]])
        self.image = np.full((400, 600, 3), 70, dtype=np.uint8)
        cv2.fillConvexPoly(self.image, self.points.astype(int), (170, 190, 210))
        for split in ("train", "val"):
            (self.dataset / "images" / split).mkdir(parents=True)
            (self.dataset / "labels" / split).mkdir(parents=True)
            for index in range(2):
                name = f"{split}_{index}"
                cv2.imwrite(str(self.dataset / "images" / split / (name + ".jpg")), self.image)
                sample = {"shape": (600, 400), "points": self.points,
                          "visibility": np.array([2, 2, 2, 2]), "bbox": np.array([90, 90, 450, 310])}
                (self.dataset / "labels" / split / (name + ".txt")).write_text(encode_label(index, sample))
        document = {"path": str(self.dataset), "train": "images/train", "val": "images/val",
                    "kpt_shape": [4, 3], "names": {0: "front", 1: "back"}, "flip_idx": [1, 0, 3, 2]}
        (self.dataset / "data.yaml").write_text(yaml.safe_dump(document))
        (self.dataset / "dataset_report.json").write_text(json.dumps({"keypoint_semantics": "idcard_canonical_v1"}))

    def test_tight_crop_fills_both_axes_without_stretching(self):
        sample = transform_sample(self.image, self.points, "large", np.random.default_rng(1),
                                  size=768, occupancy=(.999, .999))
        self.assertTrue(np.all(sample["visibility"] == 2))
        self.assertGreater(sample["metrics"]["bbox_width_fraction"], .997)
        self.assertGreater(sample["metrics"]["bbox_height_fraction"], .997)
        self.assertGreater(sample["metrics"]["card_area_fraction"], .995)
        self.assertAlmostEqual(sample["matrix"][0, 0], sample["matrix"][1, 1])
        self.assertLess(sample["shape"][1], sample["shape"][0])
        # The actual rendered card must fill the frame, independently of labels.
        pixels = sample["image"].mean(axis=2)
        self.assertGreater(np.mean(pixels > 140), .99)

    def test_crop_marks_missing_physical_corners_and_recomputes_bbox(self):
        sample = transform_sample(self.image, self.points, "clipped", np.random.default_rng(4),
                                  variant_index=0)
        self.assertEqual(sample["visibility"].tolist(), [0, 2, 2, 0])
        self.assertAlmostEqual(sample["bbox"][0], 0)
        self.assertGreater(sample["metrics"]["retained_card_area_fraction"], .65)
        triples = np.array(list(map(float, encode_label(0, sample).split())))[5:].reshape(4, 3)
        np.testing.assert_array_equal(triples[[0, 3]], [[0, 0, 0], [0, 0, 0]])
        # A corner is not replaced with a new crop-boundary intersection.
        self.assertLess(sample["points"][0, 0], 0)

    def test_occlusion_keeps_exact_coordinates_and_changes_visibility(self):
        plain = transform_sample(self.image, self.points, "large", np.random.default_rng(7))
        covered = transform_sample(self.image, self.points, "occluded", np.random.default_rng(7), variant_index=4)
        np.testing.assert_allclose(plain["points"], covered["points"])
        np.testing.assert_allclose(plain["bbox"], covered["bbox"])
        self.assertEqual(np.count_nonzero(covered["visibility"] == 1), 2)
        self.assertGreater(np.count_nonzero(plain["image"] != covered["image"]), 100)

    def test_boundary_profile_targets_all_four_corners_and_edges(self):
        variants, anchors = [], set()
        for index in range(10):
            sample = transform_sample(self.image, self.points, "occluded", np.random.default_rng(index),
                                      size=256, variant_index=index)
            variants.append(sample["variant"])
            self.assertGreaterEqual(sample["metrics"]["occluded_card_area_fraction"], .01)
            self.assertLessEqual(sample["metrics"]["occluded_card_area_fraction"], .35)
            self.assertTrue(np.all(sample["visibility"] > 0))
            if index < 4:
                self.assertEqual(np.where(sample["visibility"] == 1)[0].tolist(), [index])
                anchors.add(sample["occlusion"]["anchor"])
        self.assertEqual(anchors, {0, 1, 2, 3})
        self.assertEqual(variants, ["corner"] * 4 + ["two_corners"] * 2 +
                         ["opposite_corners", "edge", "edge", "corner_edge"])

    def test_mixed_profile_keeps_interior_occlusion_optional(self):
        sample = transform_sample(self.image, self.points, "occluded", np.random.default_rng(3),
                                  variant_index=3, occlusion_profile="mixed")
        self.assertEqual(sample["variant"], "center")
        self.assertEqual(sample["visibility"].tolist(), [2, 2, 2, 2])

    def test_exact_counts_reproducibility_and_training_only_shard(self):
        first = self.root / "first"
        second = self.root / "second"
        before = (self.dataset / "labels/train/train_0.txt").read_bytes()
        for output in (first, second):
            report = generate_dataset(self.dataset, output, large=3, clipped=5, occluded=4,
                                      preview_count=3, image_size=128)
            self.assertEqual(report["counts"], {"large": 3, "clipped": 5, "occluded": 4})
            self.assertEqual(report["included_images"], 12)
            self.assertEqual(report["occlusion_variants"], {"corner": 4})
            self.assertEqual(report["occluded_corner_counts"], {f"K{i}": 1 for i in range(4)})
            self.assertEqual(len(list((output / "images/train").glob("*.jpg"))), 12)
            self.assertEqual(len(list((output / "labels/train").glob("*.txt"))), 12)
            self.assertEqual(len(list((output / "previews").glob("*.jpg"))), 3)
            self.assertEqual(yaml.safe_load((output / "data.yaml").read_text())["val"], [])
            records = [json.loads(line) for line in (output / "lineage.jsonl").read_text().splitlines()]
            self.assertTrue(all("/images/train/" in r["source"] for r in records))
            self.assertTrue(all(r["source_split"] == "train" for r in records))
        for file in (first / "images/train").glob("*.jpg"):
            self.assertEqual(file.read_bytes(), (second / "images/train" / file.name).read_bytes())
        self.assertEqual(before, (self.dataset / "labels/train/train_0.txt").read_bytes())
        config = {"root": str(self.root), "pattern": "*/data.yaml", "deduplicate_by": "stem",
                  "selection_policy": "latest_export", "image_mode": "copy",
                  "manifest_directory": str(self.root / "combined"),
                  "required_keypoint_semantics": "idcard_canonical_v1"}
        combined = _build_combined_dataset(config, yaml, materialize=True)
        self.assertEqual(combined["train_images"], 14)
        self.assertEqual(combined["validation_images"], 2)

    def test_output_directory_is_never_overwritten(self):
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "marker"
        marker.write_text("keep")
        with self.assertRaisesRegex(AugmentationError, "already exists"):
            generate_dataset(self.dataset, existing, large=1)
        self.assertEqual(marker.read_text(), "keep")

    def test_invalid_arguments_and_empty_candidates_publish_nothing(self):
        output = self.root / "result"
        for options in ({}, {"large": -1}, {"large": 1, "occupancy_max": 1.0},
                        {"large": 1, "occupancy_min": float("nan")}, {"large": 1, "seed": -1},
                        {"occluded": 1, "occlusion_profile": "invalid"}):
            with self.subTest(options=options), self.assertRaises(AugmentationError):
                generate_dataset(self.dataset, output, **options)
            self.assertFalse(output.exists())
        for label in (self.dataset / "labels/train").glob("*.txt"):
            label.write_text(label.read_text() * 2)
        with self.assertRaisesRegex(AugmentationError, "no eligible"):
            generate_dataset(self.dataset, output, large=1)
        self.assertFalse(output.exists())

    def test_cli_exposes_counts_and_occupancy(self):
        args = build_parser().parse_args(["augment", "--dataset", "source", "--output", "out",
                                         "--large", "100", "--clipped", "20", "--occluded", "30",
                                         "--occupancy-min", ".995", "--occupancy-max", ".999"])
        self.assertEqual((args.large, args.clipped, args.occluded), (100, 20, 30))
        self.assertEqual(args.occupancy_min, .995)
        self.assertEqual(args.occlusion_profile, "boundary")


if __name__ == "__main__":
    unittest.main()
