# Evaluation protocol

1. Prepare one deterministic dataset version and reserve its test split before training.
2. Generate captions from base BLIP and the fine-tuned checkpoint using identical test images and generation settings.
3. Report corpus BLEU-4, ROUGE-L, METEOR, and CIDEr against all human references available for each image.
4. Review a fixed sample of outputs for factuality, object hallucinations, and usefulness. Store reviewer edits separately in SQLite.
5. State the reviewer count and protocol in every release report. A single-reviewer audit is useful evidence, not a universal quality claim.
