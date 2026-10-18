import tempfile
import unittest
from pathlib import Path

from PIL import Image

from idcard_detector.training import _copy_training_image, _materialize_samples


class TrainingImageTest(unittest.TestCase):
    def materialize(self, root, image_format, suffix, trailer=b"", **save_options):
        source = root / ("sample" + suffix)
        Image.new("RGB", (32, 48), (123, 45, 67)).save(
            source, format=image_format, **save_options
        )
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

    def test_missing_jpeg_end_marker_is_restored(self):
        for comment in (b"", b"embedded \xff\xd9 marker"):
            with self.subTest(comment=comment), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "sample.jpg"
                Image.new("RGB", (32, 48), (123, 45, 67)).save(source, comment=comment)
                complete = source.read_bytes()
                source.write_bytes(complete[:-2])
                destination = root / "snapshot"
                destination.mkdir()
                target, count = _copy_training_image(source, destination)
                self.assertEqual(count, 1)
                self.assertEqual(target.read_bytes(), complete)
                self.assertEqual(source.read_bytes(), complete[:-2])
                with Image.open(target) as repaired:
                    repaired.load()
                    self.assertEqual(repaired.size, (32, 48))
                self.assertEqual(_copy_training_image(source, destination), (target, 0))

    def test_mpo_primary_frame_is_saved_as_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            exif = Image.Exif()
            exif[274] = 6
            source, target = self.materialize(
                Path(directory), "MPO", ".jpg", save_all=True,
                append_images=[Image.new("RGB", (16, 24), (0, 255, 0))],
                exif=exif,
            )
            with Image.open(source) as original, Image.open(target) as copied:
                self.assertEqual(original.format, "MPO")
                self.assertEqual(original.n_frames, 2)
                self.assertEqual(copied.format, "JPEG")
                self.assertEqual(getattr(copied, "n_frames", 1), 1)
                self.assertEqual(copied.size, original.size)
                self.assertEqual(copied.getexif()[274], 6)
                for actual, expected in zip(copied.getpixel((0, 0)), original.getpixel((0, 0))):
                    self.assertLessEqual(abs(actual - expected), 2)
