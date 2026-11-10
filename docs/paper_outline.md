# Paper outline

**Working title:** Multi-Agent Framework for Detecting AI-Manipulated Property-Damage
Evidence in Insurance and Rental Claims

> This is a structure, not a report of completed experiments. Fill every result with
> reproducible measurements; do not infer fraud from image-origin labels.

## Abstract

State the narrowly defined image-classification target, datasets and split protocol,
agents compared, metrics, and measured results. Explicitly distinguish manipulation
detection from fraud determination.

## 1. Introduction

- Motivation: synthetic and edited media may be submitted as evidence.
- Research gap: robustness/generalization and the value of combining heterogeneous
  signals need measurement, not assumption.
- Research question and testable hypotheses.
- Contributions: benchmark protocol, modular agent implementation, and empirical
  comparison (only if supported by completed results).

## 2. Related Work

- AI-generated image detection and cross-generator evaluation.
- Image manipulation forensics and compression/noise analysis.
- Content provenance standards and credential validation.
- Evidence review in insurance/rental workflows; distinguish operational fraud from
  synthetic-image classification.
- Cite original papers, dataset versions, and model cards.

## 3. Methodology

- Define three-class labels: real, AI-generated, AI-edited. Specify annotation policy,
  ambiguous/benign edits, exclusions, and annotator agreement.
- Describe the metadata, forensic, visual-cue, AI-classifier, and fusion agents.
- Describe model training, calibration, thresholds, missing-feature handling, and
  explanation generation.
- Specify leakage controls: source/near-duplicate groups and generator-held-out tests.
- Document agentic coding environment and human review/reproducibility safeguards.

## 4. Experiments

1. Single-agent baselines.
2. Agent combinations (all available subsets).
3. Full system vs. standalone learned detector.
4. Held-out generator and dataset/domain generalization.
5. Common transformations (JPEG quality, resizing, crops) as a separately labeled,
   paired robustness benchmark.
6. Ablations, missing credentials/features, runtime, and confidence calibration.

Report accuracy, precision, recall, F1, ROC-AUC, FPR, and FNR, with sample counts,
confidence intervals, and per-generator results. Pre-register threshold selection and
repeat each stochastic training run with declared seeds.

## 5. Results

Insert measured results only. Include class balance, group counts, generator coverage,
confidence intervals, calibration, and failure cases. Do not fill the tables with
placeholder values presented as findings.

## 6. Discussion

Interpret whether fusion improves over the strongest standalone detector; identify
where gains/losses occur and whether uncertainty is calibrated. Discuss distribution
shift, evidence workflows, and how false positives affect innocent submitters.

## 7. Limitations and Ethics

- Public image datasets are not representative of actual claims or fraudulent intent.
- Missing/invalid C2PA credentials do not establish that an image is synthetic.
- EXIF can be absent, edited, copied, or misleading; GPS is sensitive.
- Forensic and visual cues are content/codec-dependent.
- Open-set generators, post-processing, prevalence shift, and adversarial adaptation.
- Human review is required; no automated denial, accusation, or legal conclusion.
- Licensing, privacy, consent, and harms of false suspicion.

## 8. Future Work

Prospective, consent-based claim-like data; richer manipulation localization; provenance
and detection fusion; calibrated abstention; human-factors study; broader generator
coverage.

## 9. Conclusion

Answer the stated question using the measured test results and restate the boundary
between image-manipulation classification and fraud determination.
