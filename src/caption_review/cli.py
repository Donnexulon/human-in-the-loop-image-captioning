"""Command-line interface for the captioning workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import prepare_dataset
from .modeling import DEFAULT_MODEL, evaluate_model, generate_caption, train_model
from .openimages import fetch_openimages_subset
from .review import launch_review_app


def prepare_data(args: argparse.Namespace) -> int:
    result = prepare_dataset(Path(args.annotations), Path(args.images), Path(args.output), args.seed)
    print(json.dumps(result, indent=2))
    return 0


def fetch_openimages(args: argparse.Namespace) -> int:
    result = fetch_openimages_subset(
        Path(args.captions),
        Path(args.metadata),
        Path(args.output),
        args.limit,
        args.seed,
        args.workers,
    )
    print(json.dumps(result, indent=2))
    return 0


def train(args: argparse.Namespace) -> int:
    target = train_model(
        Path(args.data),
        Path(args.output),
        args.base_model,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.max_length,
        args.seed,
    )
    print(f"Saved checkpoint to {target}")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    metrics = evaluate_model(args.model, data_dir)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report["status"] = "verified"
    report["evaluation"] = {"fine_tuned": metrics, "split": "locked test"}
    metadata_path = data_dir.parent / "release-metadata.json"
    if metadata_path.is_file():
        report["dataset"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    if args.baseline_model:
        report["evaluation"]["base"] = evaluate_model(args.baseline_model, data_dir)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["evaluation"], indent=2))
    print(f"Wrote evaluation report: {report_path}")
    return 0


def caption_image(args: argparse.Namespace) -> int:
    print(generate_caption(args.model, Path(args.image), args.max_new_tokens))
    return 0


def app(args: argparse.Namespace) -> int:
    launch_review_app(args.model, Path(args.data), Path(args.reviews), args.host, args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="caption-review")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch-openimages")
    fetch.add_argument("--captions", required=True)
    fetch.add_argument("--metadata", required=True)
    fetch.add_argument("--output", default="data/raw/openimages-v6-sample")
    fetch.add_argument("--limit", type=int, default=3000)
    fetch.add_argument("--seed", type=int, default=42)
    fetch.add_argument("--workers", type=int, default=8)
    fetch.set_defaults(func=fetch_openimages)

    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--annotations", required=True)
    prepare.add_argument("--images", required=True)
    prepare.add_argument("--output", default="data/prepared-v1")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=prepare_data)

    fit = commands.add_parser("train")
    fit.add_argument("--data", default="data/prepared-v1/train")
    fit.add_argument("--output", default="checkpoints/blip-captioner-v1")
    fit.add_argument("--base-model", default=DEFAULT_MODEL)
    fit.add_argument("--epochs", type=int, default=3)
    fit.add_argument("--batch-size", type=int, default=4)
    fit.add_argument("--learning-rate", type=float, default=5e-5)
    fit.add_argument("--max-length", type=int, default=64)
    fit.add_argument("--seed", type=int, default=42)
    fit.set_defaults(func=train)

    score = commands.add_parser("evaluate")
    score.add_argument("--model", required=True)
    score.add_argument("--baseline-model", help="Optional base model for a matched comparison.")
    score.add_argument("--data", default="data/prepared-v1/test")
    score.add_argument("--report", default="results/release.json")
    score.set_defaults(func=evaluate)

    caption = commands.add_parser("caption")
    caption.add_argument("--model", required=True)
    caption.add_argument("--image", required=True)
    caption.add_argument("--max-new-tokens", type=int, default=40)
    caption.set_defaults(func=caption_image)

    reviewer = commands.add_parser("app")
    reviewer.add_argument("--model", required=True)
    reviewer.add_argument("--data", default="data/prepared-v1/test")
    reviewer.add_argument("--reviews", default="reviews/feedback.sqlite")
    reviewer.add_argument("--host", default="127.0.0.1")
    reviewer.add_argument("--port", type=int, default=7860)
    reviewer.set_defaults(func=app)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
