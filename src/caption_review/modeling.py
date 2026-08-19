"""BLIP training and locked-test evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path

DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_split(data_dir: Path):
    from datasets import load_dataset

    return load_dataset("imagefolder", data_dir=str(data_dir), split="train")


def _collate(processor, examples: list[dict[str, object]], max_length: int):

    images = [example["image"].convert("RGB") for example in examples]
    captions = [str(example["text"]) for example in examples]
    batch = processor(
        images=images,
        text=captions,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = batch.input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {
        "pixel_values": batch.pixel_values,
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "labels": labels,
    }


def train_model(
    data_dir: Path,
    output: Path,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 64,
    seed: int = 42,
) -> Path:
    import torch
    from torch.utils.data import DataLoader
    from transformers import BlipForConditionalGeneration, BlipProcessor

    random.seed(seed)
    torch.manual_seed(seed)
    device = _device()
    dataset = _load_split(data_dir)
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda examples: _collate(processor, examples, max_length),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(loader, start=1):
            batch = {name: value.to(device) for name, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0 or step == len(loader):
                print(f"epoch {epoch + 1}/{epochs}: step {step}/{len(loader)} loss={loss.item():.4f}")
        print(f"epoch {epoch + 1}/{epochs}: loss={loss.item():.4f}")
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    (output / "training-metadata.json").write_text(
        json.dumps(
            {
                "base_model": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "train_data": str(data_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def generate_caption(model_path: str | Path, image_path: Path, max_new_tokens: int = 40) -> str:
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = _device()
    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output[0], skip_special_tokens=True).strip()


def _read_references(data_dir: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (data_dir / "references.jsonl").read_text(encoding="utf-8").splitlines() if line]


def evaluate_model(model_path: str | Path, data_dir: Path) -> dict[str, object]:
    from nltk.translate.meteor_score import meteor_score
    from pycocoevalcap.cider.cider import Cider
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU

    references = _read_references(data_dir)
    predictions: list[str] = []
    reference_texts: list[list[str]] = []
    cider_refs: dict[str, list[str]] = {}
    cider_hypotheses: dict[str, list[str]] = {}
    for index, item in enumerate(references, start=1):
        file_name = str(item["file_name"])
        prediction = generate_caption(model_path, data_dir / file_name)
        item_references = [str(reference) for reference in item["references"]]
        predictions.append(prediction)
        reference_texts.append(item_references)
        cider_refs[str(item["id"])] = item_references
        cider_hypotheses[str(item["id"])] = [prediction]
        if index % 50 == 0 or index == len(references):
            print(f"Evaluating {model_path}: {index}/{len(references)} images")

    reference_count = max(len(references) for references in reference_texts)
    bleu_references = [
        [references[index] if index < len(references) else references[-1] for references in reference_texts]
        for index in range(reference_count)
    ]
    bleu = BLEU(effective_order=True).corpus_score(predictions, bleu_references).score / 100
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(max(scorer.score(reference, prediction)["rougeL"].fmeasure for reference in refs) for prediction, refs in zip(predictions, reference_texts, strict=True)) / len(predictions)
    meteor = sum(max(meteor_score([reference.split()], prediction.split()) for reference in refs) for prediction, refs in zip(predictions, reference_texts, strict=True)) / len(predictions)
    cider, _ = Cider().compute_score(cider_refs, cider_hypotheses)
    return {"image_count": len(predictions), "bleu_4": bleu, "rouge_l": rouge_l, "meteor": meteor, "cider": cider}
