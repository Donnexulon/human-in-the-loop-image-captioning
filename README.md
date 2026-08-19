# Human-in-the-Loop Image Captioning with BLIP

Fine-tune BLIP Base for image captioning, evaluate it against human-written references, and collect structured human corrections in a local review interface. The project keeps training, evaluation, and reviewer feedback separate so reported results are reproducible.

## Features

- Deterministic preparation from an image folder plus JSONL annotations.
- BLIP Base fine-tuning with automatic CPU/GPU selection.
- Locked-test evaluation with BLEU-4, ROUGE-L, METEOR, and CIDEr.
- Local Gradio reviewer workflow that stores ratings, hallucination flags, and corrected captions in SQLite.
- Dataset/version metadata and a machine-readable release report.

## V1 released results

The V1 model was fine-tuned from `Salesforce/blip-image-captioning-base` on a deterministic 3,000-image Open Images Localized Narratives subset (seed 42). Evaluation used the locked 306-image test split.

| Metric | Fine-tuned BLIP | Base BLIP |
| --- | ---: | ---: |
| BLEU-4 | 0.103 | 0.003 |
| ROUGE-L | 0.426 | 0.157 |
| METEOR | 0.339 | 0.078 |
| CIDEr | 0.359 | 0.097 |

The accompanying single-reviewer, 40-image convenience audit reported a mean quality score of **2.83/5** and a **27.5% factual-hallucination rate**. It also found repeated narration templates, under-specific descriptions, and occasional truncated outputs. These human-review findings qualify the automatic metrics; they are not a universal quality claim. Full measurements and audit metadata are in [`results/release.json`](results/release.json).

## Dataset format

Prepare a JSONL file where each line has an image path relative to `--images`, an identifier, and one or more human captions:

```json
{"id":"000001","image":"train/000001.jpg","captions":["A dog running through grass.","A brown dog runs outside."]}
```

The recommended source is Open Images Localized Narratives. Download only the subset you plan to use, retain source/license information, then convert it to the schema above. See [the dataset card](docs/DATASET_CARD.md).

## Setup

On Windows, use a short virtual-environment path if package installation runs into long-path limits:

```powershell
python -m venv C:\venvs\blip-captioning
C:\venvs\blip-captioning\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Workflow

```powershell
# 1. Download the official Open Images caption annotations and image metadata.
Invoke-WebRequest https://storage.googleapis.com/localized-narratives/annotations/open_images_validation_captions.jsonl -OutFile data\raw\open_images_validation_captions.jsonl
Invoke-WebRequest https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv -OutFile data\raw\validation-images-with-rotation.csv

# 2. Download a deterministic 3,000-image subset and write annotations.jsonl + provenance.csv.
caption-review fetch-openimages --captions data\raw\open_images_validation_captions.jsonl --metadata data\raw\validation-images-with-rotation.csv --output data\raw\openimages-v6-sample --limit 3000

# 3. Create deterministic train/validation/test folders.
caption-review prepare-data --annotations data\raw\openimages-v6-sample\annotations.jsonl --images data\raw\openimages-v6-sample\images --output data\prepared-v1

# 4. Fine-tune BLIP Base. The model uses an available GPU automatically, otherwise CPU.
caption-review train --data data\prepared-v1\train --output checkpoints\blip-captioner-v1 --epochs 3 --batch-size 4

# 5. Evaluate only against the locked test split, with base BLIP as a matched reference.
caption-review evaluate --model checkpoints\blip-captioner-v1 --baseline-model Salesforce/blip-image-captioning-base --data data\prepared-v1\test --report results\release.json

# 6. Start the local reviewer interface.
caption-review app --model checkpoints\blip-captioner-v1 --data data\prepared-v1\test --reviews reviews\feedback.sqlite
```

## Evaluation protocol

Compare base BLIP and the fine-tuned checkpoint using the same locked test images. Use automatic metrics as supporting evidence and record a human-review audit for factuality and visible hallucinations. Do not claim that reviewer corrections improve the model until a later, separately evaluated retraining run demonstrates that result.

See [the evaluation protocol](docs/EVALUATION_PROTOCOL.md) and [`results/release.json`](results/release.json).

## Releasing V1

The repository contains code, tests, configuration, and the checked-in release report. Raw images, annotations, checkpoints, and reviewer databases are intentionally excluded. Attach `checkpoints/blip-captioner-v1` as a release asset if you distribute trained weights, subject to the [BLIP model terms](https://huggingface.co/Salesforce/blip-image-captioning-base).

See [`docs/V1_RELEASE.md`](docs/V1_RELEASE.md) for the exact V1 configuration and known limitations. Source code is available under the [MIT License](LICENSE); the source data and model weights have their own licenses.

## V2 experiment

V2 is intentionally separate from the V1 release. It creates concise training labels from the same source manifest, starts again from base BLIP, freezes the vision encoder, and uses repetition-aware decoding. It does **not** train on V1 reviewer corrections because those originate from locked test images.

```powershell
# Same image split; concise labels are used only as training targets.
caption-review prepare-data --annotations data\raw\openimages-v6-sample\annotations.jsonl --images data\raw\openimages-v6-sample\images --output data\prepared-v2 --caption-mode concise

# V2 starts from base BLIP with a conservative fine-tuning configuration.
caption-review train --data data\prepared-v2\train --output checkpoints\blip-captioner-v2 --epochs 2 --batch-size 4 --learning-rate 1e-5 --freeze-vision-encoder

# Keep V2 results separate and evaluate both models with the same V2 decoding settings.
caption-review evaluate --model checkpoints\blip-captioner-v2 --baseline-model Salesforce/blip-image-captioning-base --data data\prepared-v2\test --report results\release-v2.json --max-new-tokens 30 --num-beams 3 --no-repeat-ngram-size 3 --repetition-penalty 1.1
```

The full hypothesis and human-review comparison protocol are in [`docs/V2_EXPERIMENT.md`](docs/V2_EXPERIMENT.md).
