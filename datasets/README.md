# Dataset protocol

The project intentionally does not download datasets automatically. Dataset hosts,
licenses, terms of use, image provenance, and citation requirements can change; record
the exact version and access date in each experiment before processing data.

## Candidate sources

- **CIFAKE**: real and synthetic image benchmark. Use its original dataset card or
  publication for the authoritative download, license, and citation. Its real/fake
  labels do not by themselves provide a validated `ai_edited` class.
- **GenImage**: multi-generator image classification benchmark. Preserve the generator
  identity labels so that generator-held-out tests can be constructed.
- **Synthbuster**: synthetic-image benchmark; confirm the current release's permitted
  research uses and provenance labels before inclusion.
- **Fake Image Detection Benchmark** and relevant Hugging Face dataset cards: inspect
  label definitions and source overlap before combining with other benchmarks.

Do not treat these labels as insurance-fraud labels. Synthetic/edited-image detection
and fraudulent intent are different targets. Do not mix test images from the same
source image, near-duplicate group, or generator family into training data.
Many popular benchmarks are binary real-versus-generated datasets. They are useful for
that binary task but do not support the requested three-class experiment by themselves.
Do not relabel generated images as `ai_edited`; obtain a separately documented edited
image set with defensible labels before making three-class claims.

## Manifest schema

Copy [`manifest.example.csv`](./manifest.example.csv) and provide one row per image:

| Column | Meaning |
|---|---|
| `image_path` | Path to the local image; relative paths resolve from the manifest. |
| `label` | Exactly `real`, `ai_generated`, or `ai_edited`. |
| `dataset` | Dataset name and release identifier. |
| `generator` | Generation model/family; use `real` for ungenerated real photographs. |
| `split` | Explicit `train`, `validation`, or `test` split. |
| `group_id` | Source-image/near-duplicate group; must not cross data splits. |

For cross-generator evaluation, select one generator family for the test split and keep
it out of training. Include real and manipulated images in each split. The detector
training command requires `--heldout-generator` and enforces the held-out-generator
constraint for non-real test examples. Store dataset version, source URLs, licenses,
citations, transformation settings, and generator prompts/parameters (when available)
in a separate experiment protocol record.

Validate a manifest before using it:

```powershell
claimguard validate-manifest datasets\manifest.csv
```

Original images should remain read-only. Place generated transformations and analysis
outputs outside the original dataset directory.
