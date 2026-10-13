import argparse
from pathlib import Path

from .pipeline import (
    CommandOCREngine,
    IDCardPipeline,
    PipelineError,
    YOLOPoseDetector,
    load_image,
    save_pipeline_outputs,
)
from .training import DEFAULT_CONFIG, TrainingError, train_pose_from_config


def build_parser():
    parser = argparse.ArgumentParser(description="Train and run the ID-card pose model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train", help="Train or resume a pose model from configuration."
    )
    train.add_argument("--config", default=str(DEFAULT_CONFIG))

    infer = subparsers.add_parser("infer", help="Run pose, rectification, quality checks, and optional OCR.")
    infer.add_argument("--model", required=True)
    infer.add_argument("--image", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--confidence", type=float, default=0.25)
    infer.add_argument("--keypoint-confidence", type=float, default=0.5)
    infer.add_argument("--device")
    infer.add_argument("--ocr-command", help="Command returning JSON; use {image} for the rectified crop path.")
    infer.add_argument("--ocr-timeout", type=int, default=60)
    infer.add_argument("--ocr-version", default="external-command")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        try:
            result = train_pose_from_config(args.config)
        except TrainingError as error:
            raise SystemExit(str(error))
        print("Training config: %s" % result["config_path"])
        print(
            "Combined dataset: %d train, %d val from %d exports"
            % (
                result["dataset"]["train_images"],
                result["dataset"]["validation_images"],
                len(result["dataset"]["source_datasets"]),
            )
        )
        if result["resume_from"]:
            print("Resumed checkpoint: %s" % result["resume_from"])
        print("Training output: %s" % (result["save_dir"] or "completed"))
        return

    try:
        detector = YOLOPoseDetector(args.model, args.confidence, args.device)
        ocr = (
            CommandOCREngine(args.ocr_command, args.ocr_timeout, args.ocr_version)
            if args.ocr_command else None
        )
        pipeline = IDCardPipeline(
            detector,
            ocr_engine=ocr,
            keypoint_confidence=args.keypoint_confidence,
        )
        results = pipeline.process(load_image(args.image))
        serialized = save_pipeline_outputs(results, args.output)
    except (PipelineError, OSError) as error:
        raise SystemExit(str(error))
    print(
        "Processed %d card candidates into %s"
        % (len(serialized), Path(args.output).expanduser().resolve())
    )
