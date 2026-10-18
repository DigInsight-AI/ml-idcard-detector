import tempfile
import unittest
from pathlib import Path

from PIL import Image

from idcard_detector.training import _materialize_samples


class TrainingImageTest(unittest.TestCase):
    def materialize(self, root, image_format, suffix, trailer=b""):
        source = root / ("sample" + suffix)
        Image.new("RGB", (32, 48), (123, 45, 67)).save(source, format=image_format)
        source.write_bytes(source.read_bytes() + trailer)
        label = root / "sample.txt"
        label.write_text("0 0.5 0.5 1 1\n", encoding="ascii")
        samples = {"sample": {"image": source, "label": label, "split": "train"}}
        original = source.read_bytes()
        count = _materialize_samples(samples, root / "snapshot")
        target = samples["sample"]["training_image"]
        self.assertEqual(source.read_bytes(), original)
        self.assertTrue((root / "snapshot/labels/train/sample.txt").is_file())
        self.assertEqual(count, 2)
        self.assertEqual(_materialize_samples(samples, root / "snapshot"), 0)
        return source, target

    def test_jpeg_trailer_is_removed_without_reencoding(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = self.materialize(Path(directory), "JPEG", ".jpg", b"extra metadata")
            self.assertEqual(target.read_bytes(), source.read_bytes()[:-14])
            with Image.open(source) as original, Image.open(target) as copied:
                self.assertEqual(original.tobytes(), copied.tobytes())

    def test_suffix_follows_actual_format(self):
        for image_format, suffix, expected in [("PNG", ".jpg", ".png"), ("JPEG", ".png", ".jpg")]:
            with self.subTest(image_format=image_format), tempfile.TemporaryDirectory() as directory:
                source, target = self.materialize(Path(directory), image_format, suffix)
                self.assertEqual(target.suffix, expected)
                self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_valid_jpeg_is_copied_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = self.materialize(Path(directory), "JPEG", ".jpeg")
            self.assertEqual(target.suffix, ".jpeg")
            self.assertEqual(target.read_bytes(), source.read_bytes())
