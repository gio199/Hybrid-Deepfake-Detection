# Example 5: AIGVDBench HunyuanVideo T2V

This directory holds the HunyuanVideo text-to-video result corresponding to
the prompt-aligned AIGVDBench kitchen test item used by example 4.

Prepare the pinned benchmark member and then run all three matched examples:

```bash
python benchmarks/run_aigvd_examples.py
```

The input and annotated MP4s are excluded from Git. `source_metadata.json`
records the exact dataset revision, archive member, checksum, prompt, license,
generator, and provenance.

Source: [AIGVDBench dataset](https://huggingface.co/datasets/AIGVDBench/AIGVDBench),
licensed CC BY 4.0.
