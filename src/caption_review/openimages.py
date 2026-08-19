"""Create a reproducible, provenance-tracked Open Images captioning subset."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


def _captions_by_image(path: Path) -> dict[str, list[str]]:
    captions: dict[str, list[str]] = defaultdict(list)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        image_id = str(row.get("image_id", "")).strip()
        caption = str(row.get("caption", "")).strip()
        if image_id and caption and caption not in captions[image_id]:
            captions[image_id].append(caption)
    if not captions:
        raise ValueError(f"No image_id/caption entries found in {path}")
    return dict(captions)


def _metadata_by_image(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)
        metadata = {
            row["ImageID"]: row
            for row in rows
            if row.get("ImageID") and (row.get("Thumbnail300KURL") or row.get("OriginalURL"))
        }
    if not metadata:
        raise ValueError(f"No ImageID/download URL entries found in {path}")
    return metadata


def _download_one(url: str, target: Path, retries: int = 3) -> bool:
    if target.is_file():
        return True
    temporary = target.with_suffix(f"{target.suffix}.part")
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "captioning-research/1.0"})
            with urlopen(request, timeout=30) as response:
                temporary.write_bytes(response.read())
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(target)
            return True
        except (HTTPError, URLError, OSError, ValueError):
            temporary.unlink(missing_ok=True)
            time.sleep(1 + attempt)
    return False


def fetch_openimages_subset(
    captions_file: Path,
    metadata_file: Path,
    output: Path,
    limit: int,
    seed: int,
    workers: int,
) -> dict[str, int]:
    """Download a fixed number of valid images and write annotations/provenance."""
    if limit < 1:
        raise ValueError("limit must be positive")
    metadata_path = output / "subset-metadata.json"
    if metadata_path.is_file():
        previous_run = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(previous_run.get("downloaded_count", 0)) >= limit:
            raise FileExistsError(f"Refusing to overwrite completed subset directory: {output}")
    captions = _captions_by_image(captions_file)
    metadata = _metadata_by_image(metadata_file)
    candidates = [image_id for image_id in captions if image_id in metadata]
    random.Random(seed).shuffle(candidates)
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    images.mkdir(exist_ok=True)

    downloaded: list[str] = []
    remaining_candidates = iter(candidates)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(downloaded) < limit:
            futures = {}
            for _ in range(min(workers * 2, limit - len(downloaded))):
                try:
                    image_id = next(remaining_candidates)
                except StopIteration:
                    break
                download_url = metadata[image_id].get("Thumbnail300KURL") or metadata[image_id]["OriginalURL"]
                suffix = Path(download_url.split("?", 1)[0]).suffix.lower()
                suffix = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"
                futures[
                    pool.submit(_download_one, download_url, images / f"{image_id}{suffix}")
                ] = image_id
            if not futures:
                break
            for future in as_completed(futures):
                if future.result():
                    downloaded.append(futures[future])
                    if len(downloaded) % 50 == 0 or len(downloaded) == limit:
                        print(f"Downloaded or reused {len(downloaded)}/{limit} images", flush=True)

    if not downloaded:
        raise RuntimeError("No images downloaded successfully. Check the network and source files.")
    selected = sorted(downloaded)[:limit]
    annotations = []
    provenance = []
    for image_id in selected:
        row = metadata[image_id]
        image_file = next(images.glob(f"{image_id}.*"))
        annotations.append(
            {"id": image_id, "image": image_file.name, "captions": captions[image_id]}
        )
        provenance.append(
            {
                "image_id": image_id,
                "download_url": row.get("Thumbnail300KURL") or row.get("OriginalURL", ""),
                "original_url": row["OriginalURL"],
                "landing_url": row.get("LandingURL", ""),
                "license": row.get("License", ""),
                "author": row.get("Author", ""),
            }
        )
    (output / "annotations.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in annotations) + "\n", encoding="utf-8"
    )
    with (output / "provenance.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=provenance[0].keys())
        writer.writeheader()
        writer.writerows(provenance)
    source_hashes = {
        "captions_sha256": hashlib.sha256(captions_file.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_file.read_bytes()).hexdigest(),
        "seed": seed,
        "requested_count": limit,
        "downloaded_count": len(selected),
    }
    (output / "subset-metadata.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
    return {"downloaded": len(selected), "candidate_count": len(candidates)}
