# V1 release record

## Scope

- Base model: `Salesforce/blip-image-captioning-base`
- Dataset: deterministic 3,000-image Open Images V6 / Localized Narratives validation subset
- Seed: 42
- Split counts: 2,392 train, 302 validation, 306 locked test
- Training: 3 epochs, batch size 4, learning rate `5e-5`, maximum caption length 64

## Locked-test results

| Metric | Fine-tuned BLIP | Base BLIP |
| --- | ---: | ---: |
| BLEU-4 | 0.1026 | 0.0029 |
| ROUGE-L | 0.4262 | 0.1570 |
| METEOR | 0.3393 | 0.0783 |
| CIDEr | 0.3588 | 0.0970 |

## Human-review audit

One reviewer assessed the first 40 outputs shown by the local review app, using a 1–5 usefulness scale and a factual-hallucination flag. This convenience sample had a mean score of 2.825/5; 11 of 40 outputs were flagged for factual hallucinations. Corrections were supplied for 38 outputs.

Observed limitations:

- Every reviewed output used the training-set narration prefix “In this image…”;
- captions were frequently generic rather than salient (for example, “food items” instead of the relevant item);
- counts, spatial relations, and uncommon objects were error-prone;
- four outputs were truncated after repeated phrases.

The audit supports a V2 experiment, but does not measure retraining improvement. See [`../results/release.json`](../results/release.json) for machine-readable values.

## Distribution notes

- Do not commit raw data, the reviewer database, or checkpoints.
- Retain the Open Images provenance manifest and image-specific license information when working with the data.
- The source code is MIT licensed. BLIP weights and the Open Images data remain subject to their respective terms.
