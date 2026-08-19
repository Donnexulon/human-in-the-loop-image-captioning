# Dataset card

## Intended source

Use an explicitly documented Open Images V6 validation subset. The `fetch-openimages` command joins the official Localized Narratives textual-caption file to the official Open Images image metadata, retaining original identifiers, URLs, authors, and image-license fields in `provenance.csv`. The narrative annotations are CC BY 4.0; image licenses can differ by source and must be retained. Raw images and annotations are not committed to this repository.

## Preparation contract

Input JSONL entries must contain `id`, `image`, and a non-empty `captions` list. Image paths are relative to the `--images` folder. Preparation validates every referenced file, preserves all references for evaluation, and copies images into a deterministic 80/10/10 train/validation/test split.

## Versioning

Record the annotation-file SHA-256, source subset description, seed, split counts, and preparation command in a release report. Never merge reviewer corrections into the original training set without creating a new data version.
