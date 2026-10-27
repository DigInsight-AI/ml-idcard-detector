"""Construct independent YOLO Pose training shards with transformed labels."""

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from .geometry import GeometryError, KEYPOINT_SEMANTICS, validate_keypoints
from .training import TrainingError, _dataset_root, _image_paths, _label_path, _normalized_names


class AugmentationError(ValueError):
    pass


OCCLUSION_PROFILES = {
    "boundary": ("corner", "corner", "corner", "corner", "two_corners", "two_corners",
                 "opposite_corners", "edge", "edge", "corner_edge"),
    "mixed": ("corner", "two_corners", "edge", "center"),
}


def add_arguments(parser):
    parser.add_argument("--dataset", type=Path, required=True, help="Source data.yaml or its directory; reads train only.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for generated samples; must not exist.")
    parser.add_argument("--large", type=int, default=0, help="Number of complete, tightly cropped cards.")
    parser.add_argument("--clipped", type=int, default=0, help="Number of cards with corners outside the image.")
    parser.add_argument("--occluded", type=int, default=0, help="Number of cards covered by synthetic textured objects.")
    parser.add_argument("--occlusion-profile", choices=OCCLUSION_PROFILES, default="boundary",
                        help="boundary focuses on corners and edges (default); mixed also covers the interior.")
    parser.add_argument("--occupancy-min", type=float, default=0.99, help="Minimum target bbox occupancy for tight crops.")
    parser.add_argument("--occupancy-max", type=float, default=0.999, help="Maximum target bbox occupancy; must be below 1.")
    parser.add_argument("--image-size", type=int, default=768, help="Longest output dimension; preserves image proportions.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=24, help="Maximum annotated previews, distributed across types.")


def _project(points, matrix):
    return cv2.perspectiveTransform(np.float32(points).reshape(1, -1, 2), matrix)[0]


def _square_window(x0, y0, side, size):
    scale = size / side
    return np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale], [0, 0, 1]], dtype=float)


def _tight_matrix(points, occupancy, size):
    lo, hi = points.min(axis=0), points.max(axis=0)
    extent = hi - lo
    width, height = np.maximum(16, np.rint(size * extent / max(extent))).astype(int)
    scale = occupancy * min(width / extent[0], height / extent[1])
    offset = (np.array([width, height]) - extent * scale) / 2 - lo * scale
    matrix = np.array([[scale, 0, offset[0]], [0, scale, offset[1]], [0, 0, 1]], dtype=float)
    return matrix, (int(width), int(height))


def _clipping_matrix(points, variant, fraction, size, corner):
    rotation = np.eye(3)
    if variant == "corner":
        direction = points.mean(axis=0) - points[corner]
        direction /= np.linalg.norm(direction)
        ux, uy = direction
        rotation[:2, :2] = [[ux, uy], [-uy, ux]]
        points = _project(points, rotation)
    lo, hi = points.min(axis=0), points.max(axis=0)
    width, height = hi - lo
    margin = 0.03 * max(width, height)
    if variant in ("left", "corner"):
        if variant == "corner":
            ordered = np.sort(points[:, 0])
            cut = ordered[0] + fraction * (ordered[1] - ordered[0])
        else:
            cut = lo[0] + fraction * width
        side = max(height + 2 * margin, hi[0] - cut + margin)
        x0, y0 = cut, (lo[1] + hi[1] - side) / 2
    elif variant == "right":
        cut = hi[0] - fraction * width
        side = max(height + 2 * margin, cut - lo[0] + margin)
        x0, y0 = cut - side, (lo[1] + hi[1] - side) / 2
    elif variant == "top":
        cut = lo[1] + fraction * height
        side = max(width + 2 * margin, hi[1] - cut + margin)
        x0, y0 = (lo[0] + hi[0] - side) / 2, cut
    else:
        cut = hi[1] - fraction * height
        side = max(width + 2 * margin, cut - lo[1] + margin)
        x0, y0 = (lo[0] + hi[0] - side) / 2, cut - side
    return _square_window(x0, y0, side, size) @ rotation


def _occluder(points, shape, variant, rng, anchor):
    width, height = shape
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), np.float32(points))
    corner = np.float32([[-.14, -.15], [.10, -.14], [.22, .01], [.20, .18], [.08, .26], [-.12, .22]])
    corner *= rng.uniform(.75, 1.35)

    def turn(polygon, turns):
        for _ in range(turns % 4):
            polygon = np.column_stack((1 - polygon[:, 1], polygon[:, 0]))
        return polygon

    def edge(turns):
        x, span, depth = rng.uniform(.22, .36), rng.uniform(.24, .40), rng.uniform(.10, .30)
        polygon = np.float32([[x, 1.15], [x, 1 - depth * .5], [x + span * .25, 1 - depth],
                              [x + span * .70, 1 - depth], [x + span, 1 - depth * .5], [x + span, 1.15]])
        return turn(polygon, turns)

    if variant in ("corner", "two_corners", "opposite_corners", "corner_edge"):
        polygons = [turn(corner, anchor)]
        if variant in ("two_corners", "opposite_corners"):
            offset = 1 if variant == "two_corners" else 2
            polygons.append(turn(corner * rng.uniform(.8, 1.1), anchor + offset))
        elif variant == "corner_edge":
            polygons.append(edge(anchor + 1))
    elif variant == "edge":
        polygons = [edge(anchor)]
    elif variant == "center":
        x, y = rng.uniform(.12, .32), rng.uniform(.15, .35)
        polygons = [np.float32([[x, y], [x + .37, y - .02], [x + .50, y + .11],
                                 [x + .46, y + .32], [x + .08, y + .35]])]
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.rint(_project(polygon, matrix)).astype(np.int32)], 255)
    return mask


def _paint_occluder(image, mask, rng):
    height, width = mask.shape
    color = rng.integers(45, 220, size=3).astype(np.float32)
    noise = rng.normal(0, 4, (height, width, 1)).astype(np.float32)
    y, x = np.mgrid[:height, :width]
    grain = (2.5 * np.sin(x * .47) + 2 * np.cos(y * .37))[..., None]
    texture = np.clip(color + noise + grain, 0, 255)
    # A translated shadow must not wrap around the opposite image edge.
    shifted = cv2.warpAffine(mask, np.float32([[1, 0, 4], [0, 1, 5]]), (width, height))
    shadow = cv2.GaussianBlur(shifted, (15, 15), 0)[..., None] / 255
    alpha = cv2.GaussianBlur(mask, (3, 3), 0)[..., None] / 255
    shaded = image.astype(np.float32) * (1 - .18 * shadow)
    return np.clip(shaded * (1 - alpha) + texture * alpha, 0, 255).astype(np.uint8)


def transform_sample(image, points, kind, rng, size=768, occupancy=(.99, .999), variant_index=0,
                     occlusion_profile="boundary"):
    """Return pixels, physical corners, visibility and bbox in output coordinates."""
    points = np.float32(points)
    matrix, shape = _tight_matrix(points, float(rng.uniform(*occupancy)), size)
    variant = "tight"
    if kind == "clipped":
        variant = ("left", "right", "top", "bottom", "corner")[variant_index % 5]
        fraction = rng.uniform(.3, .65) if variant == "corner" else rng.uniform(.06, .24)
        matrix = _clipping_matrix(points, variant, fraction, size, int(rng.integers(4)))
        shape = (size, size)
    transformed = _project(points, matrix)
    width, height = shape
    canvas = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    retained_area, polygon = cv2.intersectConvexConvex(transformed, canvas)
    if polygon is None or retained_area < 16:
        raise AugmentationError("insufficient visible card area")
    polygon = polygon.reshape(-1, 2)
    bbox = np.r_[polygon.min(axis=0), polygon.max(axis=0)]
    bbox = np.clip(bbox, 0, [width, height, width, height])
    visibility = np.where(((transformed >= 0) & (transformed < shape)).all(axis=1), 2, 0)
    retained = float(retained_area / cv2.contourArea(transformed))
    if kind == "clipped" and not (1 <= np.count_nonzero(visibility == 0) <= 2 and .65 <= retained < .9999):
        raise AugmentationError("crop does not retain two corners and 65% of card area")
    output = cv2.warpPerspective(image, matrix, shape, flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    mask = np.zeros((height, width), dtype=np.uint8)
    occlusion = None
    if kind == "occluded":
        if occlusion_profile not in OCCLUSION_PROFILES:
            raise AugmentationError("unknown occlusion profile")
        schedule = OCCLUSION_PROFILES[occlusion_profile]
        variant = schedule[variant_index % len(schedule)]
        anchor = (variant_index + variant_index // len(schedule)) % 4
        mask = _occluder(transformed, shape, variant, rng, anchor)
        output = _paint_occluder(output, mask, rng)
        xy = np.clip(np.rint(transformed).astype(int), 0, [width - 1, height - 1])
        visibility[mask[xy[:, 1], xy[:, 0]] > 127] = 1
        expected = {"corner": {anchor}, "two_corners": {anchor, (anchor + 1) % 4},
                    "opposite_corners": {anchor, (anchor + 2) % 4}, "edge": set(),
                    "center": set(), "corner_edge": {anchor}}[variant]
        if set(np.flatnonzero(visibility == 1)) != expected:
            raise AugmentationError("occluder did not cover the intended corners")
        occlusion = {"profile": occlusion_profile, "anchor": anchor,
                     "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()}
    card_mask = np.zeros_like(mask)
    cv2.fillPoly(card_mask, [np.rint(transformed).astype(np.int32)], 255)
    covered = np.count_nonzero((card_mask > 0) & (mask > 127)) / max(np.count_nonzero(card_mask), 1)
    if kind == "occluded" and not .01 <= covered <= .35:
        raise AugmentationError("occluder must cover between 1% and 35% of the visible card")
    extent = (bbox[2:] - bbox[:2]) / shape
    metrics = {
        "retained_card_area_fraction": retained,
        "occluded_card_area_fraction": float(covered),
        "bbox_width_fraction": float(extent[0]), "bbox_height_fraction": float(extent[1]),
        "bbox_area_fraction": float(np.prod(extent)),
        "card_area_fraction": float(retained_area / (width * height)),
    }
    return {"image": output, "points": transformed, "visibility": visibility, "bbox": bbox,
            "matrix": matrix, "shape": shape, "variant": variant, "metrics": metrics, "occlusion": occlusion}


def encode_label(class_id, sample):
    width, height = sample["shape"]
    bbox = sample["bbox"] / [width, height, width, height]
    center, extent = (bbox[:2] + bbox[2:]) / 2, bbox[2:] - bbox[:2]
    parts = [str(class_id)] + [f"{v:.9f}" for v in np.r_[center, extent]]
    for point, visible in zip(sample["points"], sample["visibility"]):
        x, y = point / [width, height] if visible else (0, 0)
        parts.extend((f"{x:.9f}", f"{y:.9f}", str(int(visible))))
    label = " ".join(parts) + "\n"
    values = np.array(list(map(float, label.split())))
    triples = values[5:].reshape(4, 3)
    if not np.isfinite(values).all() or np.any(values[1:5] < 0) or np.any(values[1:5] > 1):
        raise AugmentationError("invalid output bbox")
    if np.any(triples[:, :2] < 0) or np.any(triples[:, :2] > 1):
        raise AugmentationError("invalid output keypoints")
    visible = sample["visibility"] > 0
    if not np.allclose(triples[visible, :2] * [width, height], sample["points"][visible], atol=1e-4):
        raise AugmentationError("label coordinate round trip failed")
    return label


def _sources(dataset):
    dataset = Path(dataset).expanduser().resolve()
    data_yaml = dataset / "data.yaml" if dataset.is_dir() else dataset
    try:
        document = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("kpt_shape") != [4, 3]:
            raise AugmentationError("source must use kpt_shape: [4, 3]")
        names = _normalized_names(document.get("names"))
        root = _dataset_root(data_yaml, document.get("path"))
        train = _image_paths(root, document.get("train"))
        val = _image_paths(root, document["val"]) if document.get("val") else []
        val_stems = {p.stem.lower() for p in val}
        source_report = data_yaml.parent / "dataset_report.json"
        report = json.loads(source_report.read_text()) if source_report.exists() else {}
        if report.get("synthetic"):
            raise AugmentationError("source is already synthetic; select the original dataset")
        if report.get("keypoint_semantics", KEYPOINT_SEMANTICS) != KEYPOINT_SEMANTICS:
            raise AugmentationError("source has incompatible keypoint semantics")
        candidates, skipped = [], Counter()
        for image in sorted(set(train)):
            if image.stem.lower() in val_stems:
                skipped["also_in_validation"] += 1
                continue
            try:
                lines = _label_path(image).read_text().strip().splitlines()
                if len(lines) != 1:
                    skipped["not_single_target"] += 1
                    continue
                values = np.array(list(map(float, lines[0].split())))
                if len(values) != 17 or not np.isfinite(values).all():
                    raise ValueError("invalid pose row")
                if not np.all(values[7::3] == 2):
                    skipped["not_four_visible_corners"] += 1
                    continue
                points = values[5:].reshape(4, 3)[:, :2]
                if np.any(points < 0) or np.any(points > 1) or values[0] != int(values[0]) or int(values[0]) not in names:
                    raise ValueError("invalid class or corner coordinates")
                validate_keypoints((points * 1024).tolist(), 1024, 1024, min_area=1)
                candidates.append({"image": image, "class_id": int(values[0]), "points": points})
            except (OSError, ValueError, GeometryError):
                skipped["invalid_label"] += 1
    except (OSError, ValueError, yaml.YAMLError, TrainingError) as error:
        raise AugmentationError(f"could not read source dataset: {error}") from error
    if not candidates:
        raise AugmentationError("no eligible single-target training images with four visible corners")
    return data_yaml, root, document, candidates, dict(skipped)


def _annotate(sample):
    image = sample["image"].copy()
    height, width = image.shape[:2]
    points = np.rint(sample["points"]).astype(np.int32)
    cv2.polylines(image, [points], True, (255, 209, 78), 2, cv2.LINE_AA)
    box = np.rint(sample["bbox"]).astype(int)
    cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 199, 255), 2)
    for index, (point, visible) in enumerate(zip(points, sample["visibility"])):
        if not visible:
            continue
        color = (59, 218, 95) if visible == 2 else (199, 75, 244)
        cv2.circle(image, tuple(point), 6, color, -1, cv2.LINE_AA)
        position = (int(np.clip(point[0] + 8, 3, width - 80)), int(np.clip(point[1] + 20, 20, height - 4)))
        cv2.putText(image, f"K{index}:{visible}", position, cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, f"K{index}:{visible}", position, cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
    return image


def _write_preview(output, previews):
    if not previews:
        return
    columns, cell_width, cell_height = 4, 310, 350
    rows = math.ceil(len(previews) / columns)
    sheet = Image.new("RGB", (columns * cell_width + 20, rows * cell_height + 55), "#f3f5f6")
    draw = ImageDraw.Draw(sheet)
    path = Path("/System/Library/Fonts/Helvetica.ttc")
    font = ImageFont.truetype(str(path), 15) if path.exists() else ImageFont.load_default()
    draw.text((12, 12), "GENERATED TRAINING DATA / TRANSFORMED LABELS", fill="#18212a", font=font)
    for index, record in enumerate(previews):
        x, y = 10 + index % columns * cell_width, 45 + index // columns * cell_height
        with Image.open(output / record["preview"]) as image:
            image.thumbnail((290, 280), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + (290 - image.width) // 2, y + (280 - image.height) // 2))
        metrics = record["metrics"]
        draw.text((x, y + 285), f"{record['kind']} / {record['variant']}", fill="#18212a", font=font)
        draw.text((x, y + 307), f"bbox {metrics['bbox_width_fraction']:.1%} x {metrics['bbox_height_fraction']:.1%}", fill="#38454c", font=font)
        draw.text((x, y + 329), " ".join(f"K{i}:{v}" for i, v in enumerate(record["visibility"])), fill="#38454c", font=font)
    sheet.save(output / "preview.jpg", quality=93)


def generate_dataset(dataset, output, large=0, clipped=0, occluded=0,
                     occupancy_min=.99, occupancy_max=.999, image_size=768,
                     seed=42, preview_count=24, occlusion_profile="boundary", progress=None):
    counts = {"large": large, "clipped": clipped, "occluded": occluded}
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts.values()) or not sum(counts.values()):
        raise AugmentationError("request at least one sample; counts must be non-negative integers")
    if not (0 < occupancy_min <= occupancy_max < 1):
        raise AugmentationError("occupancy must satisfy 0 < min <= max < 1")
    if occlusion_profile not in OCCLUSION_PROFILES:
        raise AugmentationError("unknown occlusion profile")
    if not isinstance(image_size, int) or image_size < 64 or preview_count < 0 or seed < 0:
        raise AugmentationError("image-size must be >=64; preview-count and seed must be non-negative")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise AugmentationError(f"output already exists; choose a new directory: {output}")
    data_yaml, source_root, document, candidates, skipped = _sources(dataset)
    if output.is_relative_to(source_root):
        raise AugmentationError("output must be outside the source dataset directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    total, previews, lineage_count, source_hashes = sum(counts.values()), [], 0, {}
    active = [kind for kind, count in counts.items() if count]
    preview_limits = {kind: min(counts[kind], preview_count // len(active) + (i < preview_count % len(active)))
                      for i, kind in enumerate(active)}
    rng = np.random.default_rng(seed)
    occlusion_variants, occluded_corners = Counter(), Counter()
    try:
        for folder in ("images/train", "labels/train", "previews"):
            (stage / folder).mkdir(parents=True)
        with (stage / "lineage.jsonl").open("w", encoding="ascii") as lineage:
            for kind in active:
                order = rng.permutation(len(candidates))
                for index in range(counts[kind]):
                    sample_seed = int(rng.integers(0, 2**32))
                    local_rng = np.random.default_rng(sample_seed)
                    for attempt in range(50):
                        candidate = candidates[int(order[(index + attempt) % len(order)])]
                        image = cv2.imread(str(candidate["image"]))
                        if image is None:
                            continue
                        height, width = image.shape[:2]
                        points = candidate["points"] * [width, height]
                        try:
                            sample = transform_sample(image, points, kind, local_rng, image_size,
                                                      (occupancy_min, occupancy_max), index, occlusion_profile)
                            reconstructed = _project(sample["points"], np.linalg.inv(sample["matrix"]))
                            if not np.allclose(reconstructed, points, atol=.005):
                                raise AugmentationError("inverse transform check failed")
                            label = encode_label(candidate["class_id"], sample)
                        except (AugmentationError, cv2.error, np.linalg.LinAlgError):
                            continue
                        break
                    else:
                        raise AugmentationError(f"could not construct {kind} sample {index + 1} after 50 attempts")
                    source = candidate["image"]
                    if source not in source_hashes:
                        source_hashes[source] = hashlib.sha256(source.read_bytes()).hexdigest()
                    identity = json.dumps([source_hashes[source], kind, index, sample_seed,
                                           points.tolist(), sample["matrix"].tolist(), list(sample["shape"]),
                                           sample["occlusion"]])
                    name = "aug_" + kind + "_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
                    image_path, label_path = f"images/train/{name}.jpg", f"labels/train/{name}.txt"
                    if not cv2.imwrite(str(stage / image_path), sample["image"], [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise AugmentationError("could not save generated image")
                    (stage / label_path).write_text(label, encoding="ascii")
                    record = {"image": image_path, "label": label_path, "kind": kind, "variant": sample["variant"],
                              "source": str(source), "source_split": "train", "source_sha256": source_hashes[source],
                              "source_keypoints": points.tolist(), "class_id": candidate["class_id"], "seed": sample_seed,
                              "matrix_source_to_output": sample["matrix"].tolist(), "output_size": list(sample["shape"]),
                              "keypoints": sample["points"].tolist(), "visibility": sample["visibility"].tolist(),
                              "bbox": sample["bbox"].tolist(), "metrics": sample["metrics"],
                              "occlusion": sample["occlusion"]}
                    if kind == "occluded":
                        occlusion_variants[sample["variant"]] += 1
                        occluded_corners.update(f"K{i}" for i, v in enumerate(sample["visibility"]) if v == 1)
                    if index < preview_limits[kind]:
                        record["preview"] = f"previews/{name}.jpg"
                        if not cv2.imwrite(str(stage / record["preview"]), _annotate(sample)):
                            raise AugmentationError("could not save annotation preview")
                        previews.append(record)
                    lineage.write(json.dumps(record) + "\n")
                    lineage_count += 1
                    if progress and (lineage_count % 100 == 0 or lineage_count == total):
                        progress(f"Generated {lineage_count}/{total}")
        exported = {"path": str(output), "train": "images/train", "val": [],
                    "names": document["names"], "kpt_shape": [4, 3]}
        for key in ("flip_idx", "kpt_oks_sigmas"):
            if key in document:
                exported[key] = document[key]
        (stage / "data.yaml").write_text(yaml.safe_dump(exported, sort_keys=False), encoding="utf-8")
        report = {"synthetic": True, "exported_at": datetime.now(timezone.utc).isoformat(),
                  "annotation_format": "yolo_pose", "keypoint_semantics": KEYPOINT_SEMANTICS,
                  "included_images": total, "train_images": total, "validation_images": 0,
                  "source_data_yaml": str(data_yaml), "eligible_source_images": len(candidates),
                  "used_source_images": len(source_hashes), "skipped_source_images": skipped,
                  "counts": counts, "seed": seed, "occupancy_range": [occupancy_min, occupancy_max],
                  "occlusion_profile": occlusion_profile, "occlusion_variants": dict(occlusion_variants),
                  "occluded_corner_counts": {f"K{i}": occluded_corners[f"K{i}"] for i in range(4)},
                  "image_size": image_size, "preview_images": len(previews), "checks": "passed",
                  "notes": ["Training-only shard; combine with original datasets for validation.",
                            "Side occupancy refers to the card bbox, not quadrilateral area.",
                            "Occluders are synthetic textured polygons."]}
        (stage / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
        _write_preview(stage, previews)
        if output.exists():
            raise AugmentationError(f"output was created by another process: {output}")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"output": str(output), **report}
