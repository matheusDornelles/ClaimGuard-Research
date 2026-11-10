# ClaimGuard Research Prototype

Research scaffolding for evaluating image provenance, forensic cues, and AI-generated /
AI-edited image classifiers on claim-like property-damage evidence. The target is image
origin/manipulation, **not fraudulent intent**. Public synthetic-image labels do not say
whether a claim is fraudulent.

## What is implemented

- **Metadata agent:** EXIF capture-time and camera fields, GPS-presence parsing, and
  descriptive metadata completeness. Missing EXIF is not treated as fraud evidence;
  exact GPS coordinates are not retained.
- **Forensics agent:** JPEG recompression difference, JPEG block-boundary contrast, and
  Laplacian residual summaries. These are exploratory raw cues, not calibrated likelihoods.
- **AI detector agent:** optional three-class Hugging Face model (`real`, `ai_generated`,
  `ai_edited`). It abstains unless a compatible trained model is configured.
- **Visual-consistency agent:** interpretable tile-level luminance, edge, and texture
  summaries. It does not claim generic shadow/reflection or semantic anomaly detection.
- **Decision agent:** logistic-regression fusion calibrated on a separate validation
  split. It abstains if no trained/calibrated artifact exists; it never averages
  unrelated heuristic values into a probability.
- **LangGraph workflow**, FastAPI endpoints, Streamlit dashboard, and SQLite experiment
  and report storage.
- Dataset manifest validation, detector training, agent-feature extraction, agent
  combination experiments, and metric reporting.
- The original browser-based C2PA provenance viewer remains available through Vite.

See [the architecture](docs/architecture.md), [dataset protocol](datasets/README.md),
and [paper outline](docs/paper_outline.md).

## Setup (Windows PowerShell)

Use Python 3.11 or 3.12. Install the research stack:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For PyTorch/Transformers training and inference, install the optional ML dependencies.
Choose the PyTorch build appropriate to your operating system and accelerator if the
default pip build is unsuitable:

```powershell
python -m pip install -e ".[dev,ml]"
```

Environment settings:

| Variable | Default | Purpose |
|---|---|---|
| `CLAIMGUARD_DB` | `data/claimguard.sqlite3` | SQLite report/experiment database. |
| `CLAIMGUARD_IMAGE_MODEL` | unset | Local path or Hugging Face ID for a trained three-class model. |
| `CLAIMGUARD_FUSION_MODEL` | `artifacts/fusion.joblib` | Trained fusion artifact. |
| `CLAIMGUARD_MAX_UPLOAD_BYTES` | 25 MiB | API upload limit. |

## Run the application

Start the REST API:

```powershell
claimguard api
```

API docs: <http://127.0.0.1:8000/docs>. Key routes: `GET /api/health`,
`POST /api/analyze` (multipart field `file`), `GET /api/analyses`, and
`GET /api/experiments`.

Start the dashboard in another terminal:

```powershell
claimguard dashboard
```

The original C2PA viewer can also be started in the project directory with `npm install`
and `npm run dev`. It reads provenance credentials in the browser; it is separate from
the Python agent pipeline.

The API and dashboard process images locally; SQLite stores report JSON and image hashes,
not image bytes. Reports may contain filenames and EXIF metadata. Do not upload personal
claim photos—use appropriately licensed public benchmark data.

## Dataset and experiment workflow

Prepare a CSV from [`datasets/manifest.example.csv`](datasets/manifest.example.csv) and
follow the [dataset protocol](datasets/README.md). Every dataset must use the documented
labels, explicit train/validation/test splits, and non-overlapping `group_id` values.
Test manipulated-generator families must be absent from training for cross-generator
evaluation.

1. Validate paths, labels, splits, and groups:

   ```powershell
   claimguard validate-manifest datasets\manifest.csv
   ```

2. Fine-tune a three-class classifier on train/validation data and report test results
   for a held-out generator:

   ```powershell
   claimguard train-detector `
     --manifest datasets\manifest.csv `
     --output-dir artifacts\image-detector `
     --heldout-generator flux-family
   ```

   Configure the resulting model for inference:

   ```powershell
   $env:CLAIMGUARD_IMAGE_MODEL = (Resolve-Path artifacts\image-detector).Path
   ```

3. Collect per-image metadata, forensic, visual, and configured AI-detector features:

   ```powershell
   claimguard collect `
     --manifest datasets\manifest.csv `
     --output data\processed\agent-predictions.csv
   ```

4. Evaluate every available single-agent/agent-combination feature set. Base fusion
   models use the train split, Platt calibration uses validation, and final metrics use
   the held-out test split:

   ```powershell
   claimguard evaluate-fusion `
     --predictions data\processed\agent-predictions.csv `
     --output reports\fusion-evaluation.json
   ```

   If a full fusion model was fitted, the artifact is saved to `artifacts\fusion.joblib`
   and is used by subsequent API/dashboard analyses. Evaluation reports accuracy,
   precision, recall, F1, ROC-AUC, FPR, FNR, counts, and per-test-generator results.
   The operating threshold is fixed at 0.5 in this prototype; do not tune it on test data.

Dataset downloads, licensing decisions, and citations are deliberately manual. Record
the exact dataset version, source, license, model, split protocol, seed, and transformations
for every paper result. The project does not fabricate baseline metrics or auto-download
datasets.

## Tests and checks

```powershell
python -m pytest
python -m ruff check src tests streamlit_app.py
npm run build
```

The PyTorch/Transformers training path is optional because it can require large model
downloads and accelerator resources. The rest of the application and research pipeline
can be tested without downloading model weights.

## Research and deployment limitations

- C2PA credentials represent provenance claims; absence does not mean an image is
  synthetic. Credential validity alone does not establish truth of the depicted damage.
- Public datasets classify images, not insurance or rental-claim intent. This project
  estimates only the explicitly trained image-label target and does **not** return a
  probability that a person or claim is fraudulent.
- Handcrafted features require empirical validation across codecs, devices, and content.
- A trained model's confidence is not automatically calibrated. This prototype calibrates
  fusion on a dedicated validation split, but generalization still requires measurement.
- False positives can harm claimants. The system is for research and human-reviewed
  investigation only—not automated denial, accusation, or legal judgment.
