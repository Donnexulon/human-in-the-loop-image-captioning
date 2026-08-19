"""Local Gradio reviewer interface with SQLite-backed corrections."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .modeling import generate_caption


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                generated_caption TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                hallucination INTEGER NOT NULL CHECK (hallucination IN (0, 1)),
                corrected_caption TEXT,
                reviewer TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def save_review(
    database: Path,
    image_id: str,
    image_path: Path,
    generated_caption: str,
    rating: int,
    hallucination: bool,
    corrected_caption: str,
    reviewer: str,
) -> None:
    initialize_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO reviews (
                image_id, image_path, generated_caption, rating, hallucination,
                corrected_caption, reviewer, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                str(image_path),
                generated_caption,
                rating,
                int(hallucination),
                corrected_caption.strip() or None,
                reviewer.strip() or None,
                datetime.now(UTC).isoformat(),
            ),
        )


def _load_samples(data_dir: Path) -> list[dict[str, object]]:
    path = data_dir / "references.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def launch_review_app(
    model: str,
    data_dir: Path,
    database: Path,
    host: str,
    port: int,
    max_new_tokens: int = 40,
    num_beams: int = 1,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
) -> None:
    import gradio as gr

    samples = _load_samples(data_dir)
    if not samples:
        raise ValueError(f"No review samples found in {data_dir}")
    initialize_database(database)

    def load(index: int):
        sample = samples[index % len(samples)]
        image_path = data_dir / str(sample["file_name"])
        caption = generate_caption(
            model,
            image_path,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )
        progress = f"Image {index % len(samples) + 1} of {len(samples)}"
        return image_path, caption, "", 3, False, progress, index % len(samples)

    def save_and_next(index, caption, rating, hallucination, correction, reviewer):
        sample = samples[int(index)]
        image_path = data_dir / str(sample["file_name"])
        save_review(
            database,
            str(sample["id"]),
            image_path,
            caption,
            int(rating),
            bool(hallucination),
            correction,
            reviewer,
        )
        next_index = (int(index) + 1) % len(samples)
        return load(next_index)

    first_image, first_caption, first_correction, first_rating, first_flag, first_progress, first_index = load(0)
    with gr.Blocks(title="Caption Review") as demo:
        gr.Markdown("# Caption Review\nReview generated captions and save corrections locally.")
        index = gr.State(first_index)
        with gr.Row():
            image = gr.Image(value=str(first_image), label="Image", type="filepath")
            with gr.Column():
                caption = gr.Textbox(value=first_caption, label="Generated caption", interactive=False)
                correction = gr.Textbox(value=first_correction, label="Corrected caption (optional)")
                rating = gr.Slider(1, 5, value=first_rating, step=1, label="Caption quality")
                hallucination = gr.Checkbox(value=first_flag, label="Contains a factual hallucination")
                reviewer = gr.Textbox(label="Reviewer name (optional)")
                progress = gr.Markdown(first_progress)
                button = gr.Button("Save review and next image")
        button.click(
            save_and_next,
            [index, caption, rating, hallucination, correction, reviewer],
            [image, caption, correction, rating, hallucination, progress, index],
        )
    demo.launch(server_name=host, server_port=port)
