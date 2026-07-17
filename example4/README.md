# Example 4: AIGVDBench real reference

This directory holds the real, camera-captured version of AIGVDBench test item
`IjW3jibCCmw_16_574to774.mp4`. The scene shows an adult and child preparing a
cake in a kitchen.

Prepare the pinned benchmark member and then run all three matched examples:

```bash
python benchmarks/run_aigvd_examples.py
```

The input and annotated MP4s are excluded from Git. `source_metadata.json`
records the exact dataset revision, archive member, checksum, prompt, license,
and provenance.

Source: [AIGVDBench dataset](https://huggingface.co/datasets/AIGVDBench/AIGVDBench),
licensed CC BY 4.0.
