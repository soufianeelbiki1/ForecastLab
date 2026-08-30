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
- Observation pipeline composes estimator outputs into the existing compliance policy input while keeping blur/compression/illumination/shadow scores separate from geometric/photo rules.
- Versioned quality policy evaluates normalized blur, compression, illumination, and shadow scores with explainable thresholds, synthetic labels, and rule-level regression metrics.
- Deterministic estimator doubles exercise the full estimator-to-policy boundary in CI without raw images or heavyweight CV dependencies.
- Versioned signal-inference service combines geometric and quality policies, returns estimator/policy provenance, and is exposed through a FastAPI precomputed-signal endpoint with explicit rule evidence.
- Licensed held-out evaluation contracts keep raw image assets outside the repository, require explicit license provenance, run external assets through the estimator/policy pipeline, reject duplicate identities, and report overall plus named-slice precision/recall/accuracy with policy-version provenance.
- No computer-vision accuracy claim is made yet because production pixel estimators and a licensed held-out real-world evaluation run are not implemented.

## Next slice

Implement a raw-image adapter behind the privacy-conscious `ImageFrame`/estimator boundary, then run the held-out harness against a licensed versioned dataset. Keep the current FastAPI endpoint explicitly signal-only until real pixel estimators and held-out evaluation results exist.
