# ForecastLab

**Explainable passport-photo compliance evaluation system with versioned policies, estimator boundaries, held-out evaluation contracts, and FastAPI inference.**

ForecastLab is an applied ML / computer-vision engineering portfolio project. It focuses on the difficult parts around a vision model: measurable rules, explainable failures, versioned evaluation, privacy-conscious data handling, stable inference contracts, and honest accuracy boundaries.

> Current scope: the repository evaluates precomputed image signals and deterministic estimator doubles. It does **not** claim ICAO certification, production accuracy, or validated real-world pixel estimators yet.

## What is implemented

- Explainable compliance policy for image dimensions, face count, head pose, background uniformity, and occlusion.
- Separate image-quality policy for blur, compression, illumination, and shadow signals.
- Narrow estimator interfaces so real CV models can be introduced without changing policy/business logic.
- Deterministic estimator doubles that exercise the complete estimator-to-policy pipeline in CI without raw personal images or heavyweight model downloads.
- Versioned synthetic signal datasets with explicit provenance and fixed evaluation splits.
- Rule-level confusion counts, precision, recall, and accuracy regression helpers.
- Licensed held-out evaluation contract that keeps image assets outside the repository, requires provenance metadata, rejects duplicate identities, and reports overall plus named-slice metrics.
- Versioned inference result contract with estimator and policy provenance.
- FastAPI signal-evaluation endpoint.
- GitHub Actions quality gate with automated tests.

## Architecture

```text
Image / signal source
        |
        v
Estimator ports
  | pose / face count
  | background / occlusion
  | blur / compression
  | illumination / shadow
        |
        v
Versioned observations
        |
        +------------------+
        |                  |
        v                  v
Compliance policy     Quality policy
        |                  |
        +--------+---------+
                 v
      Explainable rule results
                 |
                 v
     Versioned inference contract
                 |
                 v
            FastAPI API
```

The separation between estimators and policy rules allows model components to evolve independently while keeping evaluation semantics stable and testable.

## Evaluation philosophy

ForecastLab deliberately separates three claims:

1. **Policy correctness** — are rule thresholds and decisions internally consistent?
2. **Estimator quality** — do pixel/model estimators produce accurate signals?
3. **End-to-end compliance quality** — does the full system generalize on licensed held-out images?

The repository currently has strong coverage for the first layer and the infrastructure required to evaluate the second and third. No real-world accuracy number is claimed before a licensed held-out run exists.

## Example API scope

The current API consumes validated precomputed signals rather than pretending a raw-image model exists.

```text
GET  /health
POST /v1/evaluate-signals
```

Responses expose individual rule outcomes, evidence, overall decision, and version provenance so failures are explainable rather than represented by a single opaque score.

## Testing and reproducibility

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
```

CI is designed to remain deterministic and credential-free. Test fixtures contain synthetic observations rather than passport photographs or identity data.

## Privacy and data boundaries

- No real passport-photo dataset is committed to this repository.
- Held-out assets are expected to remain external and explicitly licensed.
- Dataset provenance and split version must be recorded before results are reported.
- Duplicate identities across evaluation partitions are rejected by the held-out contract.
- This project does not claim government, ICAO, or biometric certification.

## Portfolio signal

ForecastLab demonstrates:

- applied ML system design beyond model training notebooks;
- explainable decision pipelines and rule-level diagnostics;
- evaluation design, slice metrics, and versioned evidence;
- privacy-conscious dataset boundaries;
- FastAPI inference contracts;
- testable abstraction of real CV estimators from product policy.

## Next engineering milestone

Implement a privacy-conscious raw-image adapter and real pixel-estimator implementations behind the existing interfaces, then execute the licensed held-out evaluation harness and report measured slice-level results without changing the existing claim boundaries.
