# ForecastLab operating brief

ForecastLab is the applied ML flagship. Its next direction is passport-photo compliance computer vision with explainable per-rule scores, reproducible datasets, and evaluation—not a notebook gallery.

## Guardrails

- Establish dataset provenance, consent/licensing, and a versioned split before training.
- Keep inference deterministic and return per-rule evidence plus an overall decision.
- Report precision/recall and slice metrics; do not claim production accuracy without held-out evaluation.
- Keep image handling privacy-conscious and avoid storing raw photos in tests or CI.
- Add a small, synthetic fixture set and CI quality gates before model complexity.

## Next slice

Write the rule schema and evaluator contract (image dimensions, face count, pose, background, and occlusion), with explainable scores and tests using generated fixtures.
