"""Validated, deterministic image-caption dataset preparation."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class CaptionSample:
    identifier: str
    image: str
    captions: tuple[str, ...]


def deterministic_split(identifier: str, seed: int) -> str:
    """Return a stable 80/10/10 split without depending on input-file ordering."""
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def load_annotations(path: Path, image_root: Path) -> list[CaptionSample]:
    samples: list[CaptionSample] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error
        identifier = str(row.get("id", "")).strip()
        image = str(row.get("image", "")).strip()
        captions = tuple(str(item).strip() for item in row.get("captions", []) if str(item).strip())
        if not identifier or not image or not captions:
            raise ValueError(f"Line {line_number} requires non-empty id, image, and captions.")
        if identifier in seen_ids:
            raise ValueError(f"Duplicate id '{identifier}' on line {line_number}.")
        image_path = image_root / image
        if not image_path.is_file():
            raise FileNotFoundError(f"Line {line_number} references missing image: {image_path}")
        seen_ids.add(identifier)
        samples.append(CaptionSample(identifier, image, captions))
    if not samples:
        raise ValueError("The annotation file has no valid samples.")
    return samples


def _safe_filename(sample: CaptionSample) -> str:
    suffix = Path(sample.image).suffix.lower() or ".jpg"
    safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in sample.identifier)
    return f"{safe_id}{suffix}"


def _write_jsonl(rows: list[dict[str, object]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    target.write_text(content, encoding="utf-8")


def prepare_dataset(annotations: Path, image_root: Path, output: Path, seed: int) -> dict[str, int]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    samples = load_annotations(annotations, image_root)
    grouped: dict[str, list[CaptionSample]] = {split: [] for split in SPLIT_NAMES}
    for sample in samples:
        grouped[deterministic_split(sample.identifier, seed)].append(sample)

    for split, split_samples in grouped.items():
        metadata: list[dict[str, object]] = []
        evaluation: list[dict[str, object]] = []
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for sample in sorted(split_samples, key=lambda item: item.identifier):
            file_name = _safe_filename(sample)
            shutil.copy2(image_root / sample.image, split_dir / file_name)
            metadata.append({"file_name": file_name, "text": sample.captions[0]})
            evaluation.append({"id": sample.identifier, "file_name": file_name, "references": list(sample.captions)})
        _write_jsonl(metadata, split_dir / "metadata.jsonl")
        _write_jsonl(evaluation, split_dir / "references.jsonl")

    digest = hashlib.sha256(annotations.read_bytes()).hexdigest()
    counts = Counter({split: len(split_samples) for split, split_samples in grouped.items()})
    metadata = {
        "annotation_sha256": digest,
        "seed": seed,
        "split_counts": dict(counts),
        "source_annotations": str(annotations),
    }
    (output / "release-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {split: counts[split] for split in SPLIT_NAMES}
