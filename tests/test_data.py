import json

import pytest
from PIL import Image

from caption_review.data import (
    deterministic_split,
    load_annotations,
    prepare_dataset,
    training_caption,
)
from caption_review.openimages import _captions_by_image, _metadata_by_image


def test_split_is_deterministic_and_within_expected_distribution() -> None:
    first = [deterministic_split(str(index), 42) for index in range(10_000)]
    second = [deterministic_split(str(index), 42) for index in range(10_000)]
    assert first == second
    assert 0.77 < first.count("train") / len(first) < 0.83
    assert 0.08 < first.count("val") / len(first) < 0.12
    assert 0.08 < first.count("test") / len(first) < 0.12


def test_annotations_reject_missing_images(tmp_path) -> None:
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text('{"id":"1","image":"missing.jpg","captions":["caption"]}\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_annotations(annotations, tmp_path / "images")


def test_concise_caption_removes_leading_narration_and_background_sentences() -> None:
    caption = "In this image, we can see a blue bird on a rail. In the background, we can see trees."
    assert training_caption(caption, "concise") == "A blue bird on a rail"
    assert training_caption("In this picture I can observe a dog on the floor.", "concise") == "A dog on the floor"
    assert training_caption("This image consists of a painting.", "concise") == "A painting"
    assert training_caption(caption, "original") == caption
    with pytest.raises(ValueError, match="Unsupported caption mode"):
        training_caption(caption, "unknown")


def test_preparation_writes_disjoint_metadata_and_references(tmp_path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for index in range(20):
        name = f"sample-{index}.png"
        Image.new("RGB", (4, 4), "white").save(images / name)
        rows.append(
            {
                "id": str(index),
                "image": name,
                "captions": [f"In this image, we can see caption {index}.", "alternate"],
            }
        )
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "prepared"
    counts = prepare_dataset(annotations, images, output, seed=42)
    identifiers = []
    for split in ("train", "val", "test"):
        metadata = [json.loads(line) for line in (output / split / "metadata.jsonl").read_text().splitlines()]
        references = [json.loads(line) for line in (output / split / "references.jsonl").read_text().splitlines()]
        assert len(metadata) == len(references) == counts[split]
        identifiers.extend(item["id"] for item in references)
    assert sorted(identifiers, key=int) == [str(index) for index in range(20)]
    prepare_dataset(annotations, images, output, seed=42, caption_mode="concise", refresh_labels=True)
    refreshed = [json.loads(line) for line in (output / "train" / "metadata.jsonl").read_text().splitlines()]
    assert all(not item["text"].lower().startswith("in this image") for item in refreshed)


def test_openimages_parsers_keep_caption_references_and_metadata(tmp_path) -> None:
    captions = tmp_path / "captions.jsonl"
    captions.write_text(
        "\n".join(
            [
                json.dumps({"image_id": "one", "caption": "first caption"}),
                json.dumps({"image_id": "one", "caption": "alternate caption"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "ImageID,OriginalURL,LandingURL,License,Author\none,https://example.test/one.jpg,https://example.test,CC BY,Test\n",
        encoding="utf-8",
    )
    assert _captions_by_image(captions) == {"one": ["first caption", "alternate caption"]}
    assert _metadata_by_image(metadata)["one"]["OriginalURL"] == "https://example.test/one.jpg"
