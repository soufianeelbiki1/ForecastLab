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
- Synthetic tests only; no raw passport or identity photos are stored in CI.
- No computer-vision accuracy claim is made yet because pixel-level estimators and held-out evaluation are not implemented.

## Next slice

Add synthetic observation datasets with versioned train/validation/test-style splits for evaluator regression, then introduce estimator interfaces for blur/illumination and pose/face/background signals. Add precision/recall-style evaluation helpers against labeled rule outcomes before integrating any heavyweight CV model.
