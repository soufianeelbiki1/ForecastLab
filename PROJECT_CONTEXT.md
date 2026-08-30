# ForecastLab operating brief

ForecastLab is the applied ML flagship. Its next direction is passport-photo compliance computer vision with explainable per-rule scores, reproducible datasets, and evaluation—not a notebook gallery.

## Guardrails

- Establish dataset provenance, consent/licensing, and a versioned split before training.
- Keep inference deterministic and return per-rule evidence plus an overall decision.
- Report precision/recall and slice metrics; do not claim production accuracy without held-out evaluation.
- Keep image handling privacy-conscious and avoid storing raw photos in tests or CI.
- Add a small, synthetic fixture set and CI quality gates before model complexity.

## Current state

- Python package and GitHub Actions quality gates.
- Deterministic policy/evaluator boundary over pre-computed photo observations.
- Explainable per-rule results for image dimensions, face count, head pose, background uniformity, and occlusion.
- Versionable threshold policy with input-range validation and explicit evidence strings.
- Versioned synthetic observation dataset with fixed train/validation/test-style splits and provenance declaring that no photographs or personal identity data are included.
- Rule-level confusion counts plus precision, recall, and accuracy helpers for evaluator regression.
- Regression tests prove the reference policy matches the synthetic labels and detect stricter-threshold false positives.
- Narrow estimator interfaces for pose/face count, background uniformity, occlusion, and image quality signals.
- Observation pipeline composes estimator outputs into the existing compliance policy input while keeping blur/compression/illumination/shadow scores separate until validated rules exist.
- Deterministic estimator doubles exercise the full estimator-to-policy boundary in CI without raw images or heavyweight CV dependencies.
- No computer-vision accuracy claim is made yet because production pixel estimators and licensed held-out real-world evaluation are not implemented.

## Next slice

Add explicit quality policy rules for blur/compression and illumination/shadow only after defining labeled synthetic quality examples and threshold-regression metrics. Then add adapter contracts for licensed held-out image evaluation without storing raw images in the repository, followed by a small FastAPI inference surface that returns rule evidence and model/estimator version metadata.
