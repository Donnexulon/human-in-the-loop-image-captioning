# V2 concise-caption experiment

V2 is a separate experiment on branch `codex/v2-caption-style`. It does not alter the `v1.0.0` tag, the locked test split, or the V1 reviewer corrections.

## Hypothesis

V1 learned long Localized Narratives phrasing, including repeated “In this image…” prefixes, generic object inventories, and background details. V2 tests whether concise targets plus conservative fine-tuning and repetition-aware decoding produce more useful captions.

## Data contract

- Use the same raw 3,000-image manifest and seed 42.
- Create `data/prepared-v2` with `--caption-mode concise`.
- The concise mode keeps the first sentence of each source caption and removes only leading narration boilerplate.
- Test references remain untouched for locked-test evaluation.
- Never train on `reviews/feedback.sqlite`; it contains corrections from locked test images.

## Training and decoding plan

- Start from `Salesforce/blip-image-captioning-base`, not the V1 checkpoint.
- Train for 2 epochs at learning rate `1e-5` with the vision encoder frozen.
- Generate with 3 beams, no repeated 3-grams, repetition penalty 1.1, and a 30-token maximum.

## Evaluation plan

1. Evaluate V2 and base BLIP on the same locked 306-image test split using the V2 decoding configuration.
2. Save results to `results/release-v2.json`; do not overwrite V1 results.
3. Review the same first 40 test images with the V2 app settings and save to `reviews/feedback-v2.sqlite`.
4. Compare automatic metrics, mean quality score, hallucination rate, template-prefix frequency, and truncated-output count against V1.

Automatic metrics may move differently from human usefulness because concise captions intentionally differ from verbose narrative references. Report both.

## Completed V2 review

The 40-image single-reviewer audit is complete. V2's mean caption-quality rating was **3.525 / 5** with **2 / 40 (5%)** factual-hallucination flags. The comparable V1 audit recorded **2.825 / 5** and **11 / 40 (27.5%)** respectively. V2 also reduced known narration-template prefixes from 40 to 3 in this sample.

This indicates an improvement in human-rated usefulness despite lower overlap metrics against verbose source references. The review remains a convenience sample rather than a multi-rater study.
