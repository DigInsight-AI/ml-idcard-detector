import io
import json
import re
import shutil
import tempfile
import warnings
from pathlib import Path

from .training_logging import EpochMetricsLogger


class TrainingError(RuntimeError):
    pass


class ResumeInPlaceTrainerMixin:
    """Keep a resumed run in the output directory selected by the config."""

    def __init__(self, *args, **kwargs):
        overrides = dict(kwargs.get("overrides") or {})
        save_dir = overrides.pop("_resume_save_dir", None)
        self._resume_save_dir = Path(save_dir).resolve() if save_dir else None
        kwargs["overrides"] = overrides
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self._resume_save_dir is None:
            return
        self.args.project = str(self._resume_save_dir.parent)
        self.args.name = self._resume_save_dir.name
        self.args.save_dir = str(self._resume_save_dir)
        self.args.exist_ok = True


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config_v1.yaml"
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_EPOCH_CHECKPOINT = re.compile(r"epoch(\d+)\.pt")

_SECTION_KEYS = {
    "data": {
        "root", "pattern", "manifest_directory", "deduplicate_by",
        "selection_policy", "required_keypoint_semantics", "keypoint_oks_sigmas",
        "image_mode",
    },
    "model": {"architecture", "pretrained"},
    "output": {
        "project", "name", "existing_run", "save", "save_period", "plots",
        "save_json", "save_txt", "save_conf",
    },
    "training": {
        "epochs", "image_size", "batch_size", "device", "seed", "patience",
        "workers", "optimizer", "learning_rate", "final_learning_rate_factor",
        "momentum", "weight_decay", "warmup_epochs", "amp", "deterministic",
        "cache", "validate", "keep_last_n", "cosine_lr", "nominal_batch_size",
        "fraction", "rect", "dropout", "verbose", "resume_from",
    },
    "augmentation": {
        "degrees", "scale", "translate", "shear", "perspective", "fliplr",
        "flipud", "mosaic", "mixup", "cutmix", "copy_paste",
        "copy_paste_mode", "close_mosaic", "multi_scale", "bgr", "hsv_h",
        "hsv_s", "hsv_v",
    },
}

_ARGUMENT_NAMES = {
    "image_size": "imgsz",
    "batch_size": "batch",
    "learning_rate": "lr0",
    "final_learning_rate_factor": "lrf",
    "validate": "val",
    "cosine_lr": "cos_lr",
    "nominal_batch_size": "nbs",
}


def load_training_config(path=DEFAULT_CONFIG, materialize=False):
    try:
        import yaml
    except ImportError:
        raise TrainingError("PyYAML is required; install requirements-ml.txt")

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise TrainingError("training configuration does not exist: %s" % config_path)
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise TrainingError("could not read training configuration: %s" % error)
    if not isinstance(document, dict):
        raise TrainingError("training configuration root must be a mapping")
    _validate_sections(document)

    dataset = _build_combined_dataset(document["data"], yaml, materialize)

    model = document["model"]
    architecture = str(model.get("architecture") or "").strip()
    if not architecture.endswith(".yaml"):
        raise TrainingError("model.architecture must be a Pose architecture .yaml")
    if model.get("pretrained") is not False:
        raise TrainingError("model.pretrained must be false for from-scratch training")

    output = document["output"]
    project = _project_path(output.get("project"))
    name = str(output.get("name") or "").strip()
    if not name:
        raise TrainingError("output.name is required")
    existing_run = str(output.get("existing_run") or "").strip().lower()
    if existing_run not in ("prompt", "resume", "overwrite", "error"):
        raise TrainingError(
            "output.existing_run must be prompt, resume, overwrite, or error"
        )
    resume_from = _resume_checkpoint(document["training"].get("resume_from"), project)

    arguments = {
        "data": dataset["data_yaml"],
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "pretrained": model["pretrained"],
    }
    for section in ("output", "training", "augmentation"):
        for key, value in document[section].items():
            if section == "output" and key in ("project", "name", "existing_run"):
                continue
            if key in ("keep_last_n", "resume_from"):
                continue
            arguments[_ARGUMENT_NAMES.get(key, key)] = value
    _validate_arguments(arguments, document["training"]["keep_last_n"])
    return {
        "config_path": str(config_path),
        "model": architecture,
        "existing_run": existing_run,
        "resume_from": str(resume_from) if resume_from else None,
        "arguments": arguments,
        "keep_last_n": int(document["training"]["keep_last_n"]),
        "dataset": dataset,
    }


def train_pose_from_config(path=DEFAULT_CONFIG, materialize=True, prompt=None):
    configuration = load_training_config(path, materialize=materialize)
    try:
        from ultralytics import YOLO
    except ImportError:
        raise TrainingError(
            "ultralytics is required for training; install requirements-ml.txt"
        )

    resume_from = _prepare_output_directory(configuration, prompt)
    _configure_device_runtime(configuration["arguments"])
    trainer = YOLO(resume_from or configuration["model"])
    metrics_logger = EpochMetricsLogger()
    trainer.add_callback("on_train_start", metrics_logger.on_train_start)
    trainer.add_callback("on_train_epoch_start", metrics_logger.on_train_epoch_start)
    trainer.add_callback("on_fit_epoch_end", metrics_logger.on_fit_epoch_end)
    trainer.add_callback(
        "on_model_save",
        lambda active: prune_epoch_checkpoints(
            active.wdir, configuration["keep_last_n"]
        ),
    )
    arguments = dict(configuration["arguments"])
    if resume_from is None:
        result = trainer.train(**arguments)
    else:
        try:
            from .ultralytics_resume import ResumeInPlacePoseTrainer
        except ImportError as error:
            raise TrainingError("could not load the resume trainer: %s" % error)
        arguments["resume"] = True
        resume_save_dir = Path(arguments["project"]) / arguments["name"]
        trainer.train(
            trainer=ResumeInPlacePoseTrainer,
            _resume_save_dir=str(resume_save_dir),
            **arguments,
        )
        result = getattr(trainer, "trainer", None)
    save_dir = getattr(result, "save_dir", None)
    if save_dir is None:
        save_dir = getattr(getattr(trainer, "trainer", None), "save_dir", None)
    return {
        "save_dir": str(save_dir) if save_dir else None,
        "config_path": configuration["config_path"],
        "resume_from": resume_from,
        "dataset": configuration["dataset"],
    }


def _configure_device_runtime(arguments):
    device = str(arguments.get("device") or "").lower()
    if device != "mps":
        return
    # Some PyTorch versions emit this even when the loader later disables
    # pinned memory for MPS. It is expected and has no effect on training.
    warnings.filterwarnings(
        "ignore",
        message=r"'pin_memory' argument is set as true but not supported on MPS now.*",
        category=UserWarning,
        module=r"torch\.utils\.data\.dataloader",
    )
    for operation in ("index_put_with_accumulate_mps", "scatter_reduce_mps"):
        warnings.filterwarnings(
            "ignore",
            message=r"%s does not have a deterministic implementation.*" % operation,
            category=UserWarning,
        )


def prune_epoch_checkpoints(weights_directory, keep_last_n):
    numbered = []
    for path in Path(weights_directory).glob("epoch*.pt"):
        match = _EPOCH_CHECKPOINT.fullmatch(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda value: value[0])
    for _, path in numbered[:-int(keep_last_n)]:
        path.unlink(missing_ok=True)


def _resume_checkpoint(value, output_project):
    if value is None or not str(value).strip():
        return None
    checkpoint = Path(str(value)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = output_project / checkpoint
    checkpoint = checkpoint.resolve()
    if checkpoint.suffix.lower() != ".pt":
        raise TrainingError("training.resume_from must be a .pt checkpoint: %s" % checkpoint)
    if not checkpoint.is_file():
        raise TrainingError("training.resume_from does not exist: %s" % checkpoint)
    return checkpoint


def _prepare_output_directory(configuration, prompt):
    explicit_resume = configuration["resume_from"]
    if explicit_resume:
        return explicit_resume

    arguments = configuration["arguments"]
    run_directory = Path(arguments["project"]) / arguments["name"]
    if not run_directory.exists():
        return None
    if run_directory.is_symlink():
        raise TrainingError("training output must not be a symlink: %s" % run_directory)
    if not run_directory.is_dir():
        raise TrainingError("training output path is not a directory: %s" % run_directory)

    action = configuration["existing_run"]
    checkpoint = run_directory / "weights" / "last.pt"
    if action == "prompt":
        if checkpoint.is_file():
            ask = prompt or input
            try:
                answer = ask(
                    "Training output %s already exists. Resume from last.pt? [y/N]: "
                    % run_directory
                )
            except EOFError:
                raise TrainingError(
                    "cannot prompt for an existing run; set output.existing_run "
                    "to resume, overwrite, or error"
                )
            action = "resume" if str(answer).strip().lower() in ("y", "yes") else "overwrite"
        else:
            action = "overwrite"

    if action == "error":
        raise TrainingError("training output already exists: %s" % run_directory)
    if action == "resume":
        if not checkpoint.is_file():
            raise TrainingError("resume checkpoint does not exist: %s" % checkpoint)
        return str(checkpoint.resolve())
    if action == "overwrite":
        shutil.rmtree(run_directory)
        return None
    raise TrainingError("unsupported existing run action: %s" % action)


def _build_combined_dataset(config, yaml, materialize):
    root = _project_path(config.get("root"))
    if not root.is_dir():
        raise TrainingError("data.root does not exist: %s" % root)
    pattern = str(config.get("pattern") or "").strip()
    if not pattern:
        raise TrainingError("data.pattern is required")
    if config.get("deduplicate_by") != "stem":
        raise TrainingError("data.deduplicate_by must be stem")
    if config.get("selection_policy") != "latest_export":
        raise TrainingError("data.selection_policy must be latest_export")
    if config.get("image_mode") != "copy":
        raise TrainingError("data.image_mode must be copy")

    source_yamls = sorted(path for path in root.glob(pattern) if path.is_file())
    if not source_yamls:
        raise TrainingError("no child dataset configurations matched %s" % (root / pattern))

    expected = None
    sources = []
    required_semantics = str(config.get("required_keypoint_semantics") or "").strip()
    for data_yaml in source_yamls:
        try:
            document = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            raise TrainingError("could not read child dataset %s: %s" % (data_yaml, error))
        dataset_root = _dataset_root(data_yaml, document.get("path"))
        contract = {
            "names": _normalized_names(document.get("names")),
            "kpt_shape": document.get("kpt_shape"),
            "flip_idx": document.get("flip_idx"),
        }
        if contract["kpt_shape"] != [4, 3]:
            raise TrainingError("child dataset must use kpt_shape: [4, 3]: %s" % data_yaml)
        if expected is None:
            expected = contract
        elif contract != expected:
            raise TrainingError("child dataset contract differs: %s" % data_yaml)

        report_path = data_yaml.parent / "dataset_report.json"
        report = _read_json(report_path)
        if required_semantics and report.get("keypoint_semantics") != required_semantics:
            raise TrainingError("child dataset has incompatible keypoint semantics: %s" % data_yaml)
        sources.append({
            "data_yaml": data_yaml,
            "dataset_root": dataset_root,
            "document": document,
            "exported_at": str(report.get("exported_at") or ""),
        })

    sources.sort(key=lambda value: (value["exported_at"], str(value["data_yaml"])))
    samples = {}
    duplicate_entries = 0
    split_changes = 0
    for source in sources:
        for split in ("train", "val"):
            for image_path in _image_paths(
                source["dataset_root"], source["document"].get(split)
            ):
                label_path = _label_path(image_path)
                if not label_path.is_file():
                    raise TrainingError("image is missing its pose label: %s" % image_path)
                key = image_path.stem.lower()
                previous = samples.get(key)
                if previous is not None:
                    duplicate_entries += 1
                    if previous["split"] != split:
                        split_changes += 1
                samples[key] = {
                    "image": image_path,
                    "label": label_path,
                    "split": split,
                    "source": source["data_yaml"].parent.name,
                }

    manifest_directory = _project_path(config.get("manifest_directory"))
    manifest_directory.mkdir(parents=True, exist_ok=True)
    copied_files = 0
    if materialize:
        print(
            "Preparing copied dataset snapshot: %d images -> %s"
            % (len(samples), manifest_directory),
            flush=True,
        )
        copied_files = _materialize_samples(samples, manifest_directory)
        print(
            "Dataset snapshot ready: %d files copied, %d files reused"
            % (copied_files, len(samples) * 2 - copied_files),
            flush=True,
        )
    train_images = sorted(
        str(value["training_image"] if materialize else value["image"])
        for value in samples.values() if value["split"] == "train"
    )
    val_images = sorted(
        str(value["training_image"] if materialize else value["image"])
        for value in samples.values() if value["split"] == "val"
    )
    if not train_images or not val_images:
        raise TrainingError("combined dataset requires non-empty train and val splits")

    train_list = manifest_directory / "train.txt"
    val_list = manifest_directory / "val.txt"
    data_yaml = manifest_directory / "data.yaml"
    report_path = manifest_directory / "dataset_report.json"
    _write_text(train_list, "\n".join(train_images) + "\n")
    _write_text(val_list, "\n".join(val_images) + "\n")

    combined = {
        "train": str(train_list),
        "val": str(val_list),
        "kpt_shape": expected["kpt_shape"],
        "flip_idx": expected["flip_idx"],
        "names": expected["names"],
        "kpt_oks_sigmas": config.get("keypoint_oks_sigmas"),
    }
    _write_text(data_yaml, yaml.safe_dump(combined, sort_keys=False, allow_unicode=False))
    report = {
        "source_root": str(root),
        "source_datasets": [str(value["data_yaml"]) for value in sources],
        "selection_policy": "latest_export",
        "deduplicate_by": "stem",
        "image_mode": "copy",
        "materialized": bool(materialize),
        "files_copied": copied_files,
        "files_reused": len(samples) * 2 - copied_files if materialize else 0,
        "duplicate_entries_replaced": duplicate_entries,
        "split_assignments_replaced": split_changes,
        "included_images": len(samples),
        "train_images": len(train_images),
        "validation_images": len(val_images),
    }
    _write_text(report_path, json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    return {"data_yaml": str(data_yaml), **report}


def _materialize_samples(samples, destination):
    copied = 0
    for value in samples.values():
        split = value["split"]
        image_directory = destination / "images" / split
        label_directory = destination / "labels" / split
        image_directory.mkdir(parents=True, exist_ok=True)
        label_directory.mkdir(parents=True, exist_ok=True)
        image_target, image_copied = _copy_training_image(value["image"], image_directory)
        label_target = label_directory / value["label"].name
        copied += image_copied
        copied += _copy_if_changed(value["label"], label_target)
        value["training_image"] = image_target
    return copied


def _copy_training_image(source, directory):
    from PIL import Image

    suffixes = {
        "JPEG": (".jpg", ".jpeg"), "MPO": (".jpg", ".jpeg"),
        "PNG": (".png",), "BMP": (".bmp",),
        "TIFF": (".tif", ".tiff"), "WEBP": (".webp",),
    }
    normalized = None
    try:
        with Image.open(source) as image:
            extensions = suffixes.get(image.format)
            if extensions is None:
                raise TrainingError("unsupported image format %s: %s" % (image.format, source))
            suffix = source.suffix if source.suffix.lower() in extensions else extensions[0]
            destination = directory / (source.stem + suffix)
            if image.format == "MPO":
                # Training uses the primary frame; retain orientation for label alignment.
                image.seek(0)
                buffer = io.BytesIO()
                image.convert("RGB").save(
                    buffer, format="JPEG", quality=95, subsampling=0,
                    exif=image.info.get("exif", b""),
                    icc_profile=image.info.get("icc_profile"),
                )
                normalized = buffer.getvalue()
            elif image.format == "JPEG":
                with source.open("rb") as stream:
                    stream.seek(-2, 2)
                    if stream.read() != b"\xff\xd9":
                        stream.seek(0)
                        data = stream.read()
                        end = data.rfind(b"\xff\xd9")
                        if end >= 0:
                            candidate = data[:end + 2]
                            # Strip trailer bytes only if decoding preserves every pixel.
                            try:
                                with Image.open(io.BytesIO(candidate)) as trimmed:
                                    image.load()
                                    trimmed.load()
                                    if (image.mode, image.size, image.tobytes()) == (
                                        trimmed.mode, trimmed.size, trimmed.tobytes()
                                    ):
                                        normalized = candidate
                            except (OSError, ValueError):
                                # An EOI inside metadata may not terminate the main image.
                                pass
                        if normalized is None:
                            candidate = data + b"\xff\xd9"
                            with Image.open(io.BytesIO(candidate)) as repaired:
                                repaired.load()
                            normalized = candidate
    except (OSError, ValueError) as error:
        raise TrainingError("could not prepare image %s: %s" % (source, error)) from error

    if normalized is None:
        return destination, _copy_if_changed(source, destination)
    if destination.is_file() and destination.read_bytes() == normalized:
        return destination, 0
    destination.write_bytes(normalized)
    shutil.copystat(source, destination)
    return destination, 1


def _copy_if_changed(source, destination):
    if destination.is_file():
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (
            source_stat.st_size == destination_stat.st_size
            and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
        ):
            return 0
    shutil.copy2(source, destination)
    return 1


def _validate_sections(document):
    unknown_sections = sorted(set(document) - set(_SECTION_KEYS))
    if unknown_sections:
        raise TrainingError("unknown configuration sections: %s" % ", ".join(unknown_sections))
    for section, allowed_keys in _SECTION_KEYS.items():
        values = document.get(section)
        if not isinstance(values, dict):
            raise TrainingError("configuration section %s must be a mapping" % section)
        unknown_keys = sorted(set(values) - allowed_keys)
        if unknown_keys:
            raise TrainingError(
                "unknown %s fields: %s" % (section, ", ".join(unknown_keys))
            )


def _project_path(value):
    text = str(value or "").strip()
    if not text:
        raise TrainingError("configured path must not be empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _dataset_root(data_yaml, value):
    path = Path(str(value or data_yaml.parent)).expanduser()
    if not path.is_absolute():
        path = data_yaml.parent / path
    path = path.resolve()
    if not path.is_dir():
        raise TrainingError("child dataset root does not exist: %s" % path)
    return path


def _normalized_names(value):
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        try:
            return {
                int(index): str(name)
                for index, name in sorted(value.items(), key=lambda item: int(item[0]))
            }
        except (TypeError, ValueError):
            pass
    raise TrainingError("child dataset names must be an indexed list or mapping")


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingError("could not read dataset report %s: %s" % (path, error))
    if not isinstance(value, dict):
        raise TrainingError("dataset report must contain one object: %s" % path)
    return value


def _image_paths(dataset_root, value):
    values = value if isinstance(value, list) else [value]
    images = []
    for item in values:
        path = Path(str(item or "")).expanduser()
        if not path.is_absolute():
            path = dataset_root / path
        path = path.resolve()
        if path.is_dir():
            images.extend(
                child for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in _IMAGE_SUFFIXES
            )
        elif path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                image = Path(line.strip()).expanduser()
                if not image.is_absolute():
                    image = path.parent / image
                images.append(image.resolve())
        else:
            raise TrainingError("child dataset split does not exist: %s" % path)
    return sorted(images)


def _label_path(image_path):
    parts = list(image_path.parts)
    indexes = [index for index, part in enumerate(parts) if part == "images"]
    if not indexes:
        raise TrainingError("image path does not contain an images directory: %s" % image_path)
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _write_text(path, content):
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _validate_arguments(arguments, keep_last_n):
    for name in ("epochs", "imgsz", "batch", "patience"):
        if int(arguments[name]) <= 0:
            raise TrainingError("%s must be greater than zero" % name)
    if int(arguments["workers"]) < 0:
        raise TrainingError("workers must be non-negative")
    if int(keep_last_n) <= 0:
        raise TrainingError("keep_last_n must be greater than zero")
    if not 0 < float(arguments["fraction"]) <= 1:
        raise TrainingError("fraction must be greater than zero and at most one")
    if not 0 <= float(arguments["dropout"]) <= 1:
        raise TrainingError("dropout must be between zero and one")
    for name in (
        "fliplr", "flipud", "mosaic", "mixup", "cutmix", "copy_paste",
        "bgr", "hsv_h", "hsv_s", "hsv_v", "multi_scale",
    ):
        if not 0 <= float(arguments[name]) <= 1:
            raise TrainingError("%s must be between 0 and 1" % name)
    if not 0 <= float(arguments["perspective"]) <= 0.001:
        raise TrainingError("perspective must be between 0 and 0.001")
    if int(arguments["close_mosaic"]) < 0:
        raise TrainingError("close_mosaic must be non-negative")
