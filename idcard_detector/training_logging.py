import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path


_CLASS_METRIC_FIELDS = (
    "epoch",
    "class_id",
    "class_name",
    "images",
    "instances",
    "box_precision",
    "box_recall",
    "box_f1",
    "box_map50",
    "box_map75",
    "box_map50_95",
    "pose_precision",
    "pose_recall",
    "pose_f1",
    "pose_map50",
    "pose_map75",
    "pose_map50_95",
)


class EpochMetricsLogger:
    """Write readable and machine-readable metrics for every training epoch."""

    def __init__(self):
        self.save_dir = None
        self.log_path = None
        self.metrics_path = None
        self.class_metrics_path = None
        self.epoch_started_at = None
        self.logged_epochs = set()

    def on_train_start(self, trainer):
        if not _is_primary_process():
            return
        self.save_dir = Path(trainer.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.save_dir / "train.log"
        self.metrics_path = self.save_dir / "metrics.jsonl"
        self.class_metrics_path = self.save_dir / "class_metrics.csv"
        self._log("Training started")
        self._log("Output directory: %s" % self.save_dir)
        validation = "enabled" if getattr(trainer.args, "val", False) else "disabled"
        self._log("Epoch validation: %s" % validation)
        self._log("Metrics history: %s" % self.metrics_path)
        self._log("Per-class metrics: %s" % self.class_metrics_path)

    def on_train_epoch_start(self, trainer):
        if not _is_primary_process():
            return
        self.epoch_started_at = time.monotonic()
        self._log("Epoch [%d/%d] started" % (trainer.epoch + 1, trainer.epochs))

    def on_fit_epoch_end(self, trainer):
        if not _is_primary_process():
            return
        epoch = trainer.epoch + 1
        if epoch > trainer.epochs or epoch in self.logged_epochs:
            return
        metrics = self._epoch_metrics(trainer)
        class_metrics = self._class_metrics(trainer)
        duration = self._duration(trainer)
        metrics["time/epoch_seconds"] = duration

        record = {
            "event": "epoch_end",
            "epoch": epoch,
            "total_epochs": trainer.epochs,
            "metrics": metrics,
            "class_metrics": class_metrics,
        }
        self._append_json(record)
        self._append_class_metrics(class_metrics)
        self._print_epoch_summary(trainer, metrics, class_metrics, duration)
        self.logged_epochs.add(epoch)

    def _epoch_metrics(self, trainer):
        values = {}
        tloss = getattr(trainer, "tloss", None)
        if tloss is not None:
            values.update(trainer.label_loss_items(tloss))
        values.update(getattr(trainer, "metrics", {}) or {})
        values.update(getattr(trainer, "lr", {}) or {})
        return {key: _number(value) for key, value in values.items()}

    def _class_metrics(self, trainer):
        validator = getattr(trainer, "validator", None)
        metrics = getattr(validator, "metrics", None)
        if metrics is None or not hasattr(metrics, "pose"):
            return []

        class_ids = list(getattr(metrics, "ap_class_index", []))
        names = getattr(metrics, "names", {}) or {}
        rows = []
        for result_index, class_id in enumerate(class_ids):
            class_id = int(class_id)
            box = metrics.box.class_result(result_index)
            pose = metrics.pose.class_result(result_index)
            rows.append({
                "epoch": trainer.epoch + 1,
                "class_id": class_id,
                "class_name": str(names.get(class_id, class_id)),
                "images": _count(metrics.nt_per_image, class_id),
                "instances": _count(metrics.nt_per_class, class_id),
                "box_precision": _number(box[0]),
                "box_recall": _number(box[1]),
                "box_f1": _f1(box[0], box[1]),
                "box_map50": _number(box[2]),
                "box_map75": _map75(metrics.box, result_index),
                "box_map50_95": _number(box[3]),
                "pose_precision": _number(pose[0]),
                "pose_recall": _number(pose[1]),
                "pose_f1": _f1(pose[0], pose[1]),
                "pose_map50": _number(pose[2]),
                "pose_map75": _map75(metrics.pose, result_index),
                "pose_map50_95": _number(pose[3]),
            })
        return rows

    def _duration(self, trainer):
        duration = getattr(trainer, "epoch_time", None)
        if duration is not None:
            return _number(duration)
        if self.epoch_started_at is not None:
            return time.monotonic() - self.epoch_started_at
        return 0.0

    def _print_epoch_summary(self, trainer, metrics, class_metrics, duration):
        validation_metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
        self._log("Epoch [%d/%d] completed:" % (trainer.epoch + 1, trainer.epochs))
        self._log("  train loss: %s" % _metric_group(metrics, "train/", "loss"))
        self._log("  val loss: %s" % _metric_group(metrics, "val/", "loss"))
        self._log(
            "  box: %s"
            % _task_metrics(metrics, class_metrics, "box", "B", validation_metrics)
        )
        self._log(
            "  pose: %s"
            % _task_metrics(metrics, class_metrics, "pose", "P", validation_metrics)
        )
        fitness = _number(getattr(trainer, "fitness", 0.0))
        best_fitness = _number(getattr(trainer, "best_fitness", fitness))
        self._log("  fitness: current=%.6f, best=%.6f" % (fitness, best_fitness))
        self._log("  learning rate: %s" % _metric_group(metrics, "lr/", None))
        self._log("  duration: %s" % _format_duration(duration))
        if class_metrics:
            self._log("  per-class validation metrics:")
            for row in class_metrics:
                self._log(
                    "    %d:%s images=%d instances=%d "
                    "box[P=%.6f, R=%.6f, F1=%.6f, mAP50=%.6f, mAP75=%.6f, mAP50-95=%.6f] "
                    "pose[P=%.6f, R=%.6f, F1=%.6f, mAP50=%.6f, mAP75=%.6f, mAP50-95=%.6f]"
                    % (
                        row["class_id"], row["class_name"], row["images"], row["instances"],
                        row["box_precision"], row["box_recall"], row["box_f1"],
                        row["box_map50"], row["box_map75"], row["box_map50_95"],
                        row["pose_precision"], row["pose_recall"], row["pose_f1"],
                        row["pose_map50"], row["pose_map75"], row["pose_map50_95"],
                    )
                )

    def _append_json(self, record):
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")

    def _append_class_metrics(self, rows):
        if not rows:
            return
        write_header = not self.class_metrics_path.exists()
        with self.class_metrics_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CLASS_METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _log(self, message):
        line = "[%s] %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)
        print(line, flush=True)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _number(value):
    if hasattr(value, "item"):
        value = value.item()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _count(values, index):
    try:
        return int(values[index])
    except (IndexError, TypeError):
        return 0


def _f1(precision, recall):
    precision = _number(precision)
    recall = _number(recall)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _map75(metric, result_index):
    try:
        return _number(metric.all_ap[result_index, 5])
    except (AttributeError, IndexError, TypeError):
        return 0.0


def _metric_group(metrics, prefix, suffix):
    parts = []
    for key, value in metrics.items():
        if not key.startswith(prefix) or (suffix and not key.endswith(suffix)):
            continue
        name = key[len(prefix):]
        parts.append("%s=%.6f" % (name, value))
    return ", ".join(parts) if parts else "not available"


def _task_metrics(metrics, class_metrics, task, abbreviation, validation_metrics=None):
    precision = metrics.get("metrics/precision(%s)" % abbreviation, 0.0)
    recall = metrics.get("metrics/recall(%s)" % abbreviation, 0.0)
    map50 = metrics.get("metrics/mAP50(%s)" % abbreviation, 0.0)
    map50_95 = metrics.get("metrics/mAP50-95(%s)" % abbreviation, 0.0)
    aggregate = getattr(validation_metrics, task, None)
    map75 = _number(getattr(aggregate, "map75", 0.0))
    if not map75 and class_metrics:
        map75_values = [row["%s_map75" % task] for row in class_metrics]
        map75 = sum(map75_values) / len(map75_values)
    return "P=%.6f, R=%.6f, F1=%.6f, mAP50=%.6f, mAP75=%.6f, mAP50-95=%.6f" % (
        precision, recall, _f1(precision, recall), map50, map75, map50_95
    )


def _format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "%dh %02dm %02ds" % (hours, minutes, seconds)
    return "%dm %02ds" % (minutes, seconds)


def _is_primary_process():
    return int(os.getenv("RANK", "-1")) in (-1, 0)
