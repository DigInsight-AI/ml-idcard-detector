import json
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .geometry import GeometryError, destination_size, validate_keypoints


class PipelineError(RuntimeError):
    pass


@dataclass
class QualityConfig:
    min_width: int = 800
    min_height: int = 500
    min_source_width: int = 320
    min_source_height: int = 200
    min_laplacian_variance: float = 80.0
    max_dark_fraction: float = 0.35
    max_glare_fraction: float = 0.12
    min_mean_luma: float = 45.0
    max_mean_luma: float = 220.0


def rectify_card(image, keypoints, target_width=1016, aspect_ratio=1.5857725):
    cv2, np = _cv_modules()
    validate_keypoints(keypoints, image.shape[1], image.shape[0])
    width, height = destination_size(keypoints, target_width, aspect_ratio)
    source = np.asarray(keypoints, dtype=np.float32)
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    if not np.isfinite(homography).all() or abs(float(np.linalg.det(homography))) < 1e-12:
        raise GeometryError("perspective homography is unstable")
    inverse = np.linalg.inv(homography)
    rectified = cv2.warpPerspective(
        image,
        homography,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return {
        "image": rectified,
        "homography": homography.tolist(),
        "inverse_homography": inverse.tolist(),
        "width": width,
        "height": height,
    }


def evaluate_quality(image, config=None, source_size=None):
    cv2, np = _cv_modules()
    config = config or QualityConfig()
    if image is None or getattr(image, "size", 0) == 0:
        raise PipelineError("quality evaluation received an empty image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    height, width = gray.shape[:2]
    mean_luma = float(np.mean(gray))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_fraction = float(np.mean(gray <= 30))
    glare_fraction = float(np.mean(gray >= 245))
    signals = {
        "width": width,
        "height": height,
        "mean_luma": round(mean_luma, 4),
        "laplacian_variance": round(laplacian_variance, 4),
        "dark_fraction": round(dark_fraction, 6),
        "glare_fraction": round(glare_fraction, 6),
    }
    failures = []
    if width < config.min_width or height < config.min_height:
        failures.append("low_resolution")
    if source_size is not None:
        signals["source_card_width"] = round(float(source_size[0]), 3)
        signals["source_card_height"] = round(float(source_size[1]), 3)
        if source_size[0] < config.min_source_width or source_size[1] < config.min_source_height:
            failures.append("low_source_resolution")
    if laplacian_variance < config.min_laplacian_variance:
        failures.append("blur")
    if mean_luma < config.min_mean_luma or dark_fraction > config.max_dark_fraction:
        failures.append("underexposure")
    if mean_luma > config.max_mean_luma:
        failures.append("overexposure")
    if glare_fraction > config.max_glare_fraction:
        failures.append("glare")
    return {
        "passed": not failures,
        "failures": failures,
        "signals": signals,
        "thresholds": asdict(config),
    }


class CommandOCREngine:
    def __init__(self, command, timeout=60, version="external-command"):
        self.command = command
        self.timeout = int(timeout)
        self.version = version

    def recognize(self, image):
        cv2, _ = _cv_modules()
        with tempfile.TemporaryDirectory(prefix="idcard-ocr-") as directory:
            image_path = Path(directory) / "card.png"
            if not cv2.imwrite(str(image_path), image):
                raise PipelineError("could not write the OCR input image")
            command = [
                part.replace("{image}", str(image_path))
                for part in shlex.split(self.command)
            ]
            try:
                completed = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise PipelineError("OCR command timed out")
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace").strip()
                raise PipelineError("OCR command failed%s" % ((": " + detail[-500:]) if detail else ""))
            try:
                result = json.loads(completed.stdout.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PipelineError("OCR command did not return JSON: %s" % error)
            if not isinstance(result, dict):
                raise PipelineError("OCR command must return one JSON object")
            return result


class YOLOPoseDetector:
    def __init__(self, model, confidence=0.25, device=None):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise PipelineError("ultralytics is required for YOLO Pose inference; install the pinned project dependency")
        self.model_path = str(model)
        self.model = YOLO(self.model_path)
        self.confidence = float(confidence)
        self.device = device

    def predict(self, image):
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        predictions = []
        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue
            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            points = result.keypoints.xy.cpu().tolist()
            keypoint_scores = result.keypoints.conf
            keypoint_scores = keypoint_scores.cpu().tolist() if keypoint_scores is not None else None
            for index, box in enumerate(boxes):
                if len(points[index]) != 4:
                    raise PipelineError("pose model must output exactly four keypoints")
                predictions.append({
                    "class_id": int(classes[index]),
                    "score": float(scores[index]),
                    "bbox": [float(value) for value in box],
                    "keypoints": [[float(value) for value in point] for point in points[index]],
                    "keypoint_scores": (
                        [float(value) for value in keypoint_scores[index]] if keypoint_scores is not None else [1.0] * 4
                    ),
                    "model_version": self.model_path,
                })
        return predictions


class IDCardPipeline:
    def __init__(self, detector, ocr_engine=None, quality_config=None, keypoint_confidence=0.5,
                 target_width=1016, aspect_ratio=1.5857725):
        self.detector = detector
        self.ocr_engine = ocr_engine
        self.quality_config = quality_config or QualityConfig()
        self.keypoint_confidence = float(keypoint_confidence)
        self.target_width = int(target_width)
        self.aspect_ratio = float(aspect_ratio)

    def process(self, image):
        results = []
        predictions = self.detector.predict(image)
        if not predictions:
            return [{"status": "no_detection"}]
        for prediction in predictions:
            item = dict(prediction)
            keypoint_scores = item.get("keypoint_scores") or [1.0] * 4
            if len(keypoint_scores) != 4 or min(keypoint_scores) < self.keypoint_confidence:
                item.update({"status": "review", "reason": "low_keypoint_confidence"})
                results.append(item)
                continue
            try:
                geometry = validate_keypoints(
                    item["keypoints"], image.shape[1], image.shape[0], bbox=item.get("bbox")
                )
                rectified = rectify_card(
                    image, item["keypoints"], self.target_width, self.aspect_ratio
                )
            except GeometryError as error:
                item.update({"status": "invalid_geometry", "reason": str(error)})
                results.append(item)
                continue
            edge_lengths = geometry["edge_lengths"]
            source_size = (max(edge_lengths[0], edge_lengths[2]), max(edge_lengths[1], edge_lengths[3]))
            quality = evaluate_quality(rectified["image"], self.quality_config, source_size=source_size)
            if _touches_image_edge(item["keypoints"], image.shape[1], image.shape[0]):
                quality["failures"].append("clipping")
                quality["passed"] = False
            item.update({
                "geometry": geometry,
                "rectification": {
                    "width": rectified["width"],
                    "height": rectified["height"],
                    "homography": rectified["homography"],
                    "inverse_homography": rectified["inverse_homography"],
                },
                "quality": quality,
                "quality_policy_version": "idcard-quality-v1",
                "_rectified_image": rectified["image"],
            })
            if not quality["passed"]:
                item.update({"status": "quality_failed", "reason": quality["failures"]})
                results.append(item)
                continue
            if self.ocr_engine is None:
                item.update({"status": "ready_for_ocr"})
            else:
                item.update({
                    "status": "ok",
                    "ocr_version": getattr(self.ocr_engine, "version", "unknown"),
                    "ocr": self.ocr_engine.recognize(rectified["image"]),
                })
            results.append(item)
        return results


def load_image(path):
    cv2, _ = _cv_modules()
    image = cv2.imread(str(Path(path).expanduser()))
    if image is None:
        raise PipelineError("could not decode image: %s" % path)
    return image


def save_pipeline_outputs(results, destination):
    cv2, _ = _cv_modules()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    serializable = []
    for index, result in enumerate(results):
        value = dict(result)
        image = value.pop("_rectified_image", None)
        if image is not None:
            name = "card-%03d.png" % index
            if not cv2.imwrite(str(destination / name), image):
                raise PipelineError("could not write rectified card %d" % index)
            value["rectified_image"] = name
        serializable.append(value)
    (destination / "result.json").write_text(
        json.dumps({"results": serializable}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return serializable


def _cv_modules():
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise PipelineError("opencv-python and numpy are required for rectification and quality checks")
    return cv2, np


def _touches_image_edge(points, width, height, tolerance=2.0):
    return any(
        float(point[0]) <= tolerance
        or float(point[1]) <= tolerance
        or float(point[0]) >= float(width) - tolerance
        or float(point[1]) >= float(height) - tolerance
        for point in points
    )
