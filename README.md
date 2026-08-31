# ForecastLab

ForecastLab evaluates passport-photo compliance rules through versioned signal contracts. The current implementation focuses on explainable policy decisions, estimator interfaces and repeatable evaluation rather than claiming a finished raw-image model.

The API currently accepts precomputed image signals. It does not claim ICAO certification or measured real-world computer-vision accuracy.

## Rules

The compliance policy evaluates:

- image dimensions;
- face count;
- head pose;
- background uniformity;
- occlusion.

A separate quality policy handles normalized blur, compression, illumination and shadow signals. Each result includes the rule outcome and supporting evidence instead of returning only a single opaque score.

## Visual compliance demo

Generate a standalone HTML report from the versioned synthetic signal dataset:

```bash
python -m forecastlab.demo_report --output build/forecastlab-compliance.html
```

Open the generated file in a browser. It shows every synthetic case, expected failure labels, the overall policy decision, policy score and the evidence produced by each rule.

The report exercises the current signal-to-policy layer. It does not infer pose, faces, background or occlusion from image pixels, so it should not be read as a raw-image accuracy demonstration.

## Estimator boundary

Estimator interfaces isolate signal extraction from policy logic. Deterministic estimator doubles are used in tests so the complete estimator-to-policy path can run in CI without raw identity photos or heavyweight model downloads.

Real pixel estimators can later replace those adapters without changing the compliance rules or API contract.

## Evaluation

The repository includes:

- versioned synthetic signal datasets with fixed splits;
- rule-level confusion counts, precision, recall and accuracy helpers;
- a held-out evaluation contract for external licensed image data;
- duplicate-identity checks across evaluation partitions;
- overall and named-slice metrics with estimator/policy version provenance.

No real passport-photo dataset is committed to the repository.

## API

```text
GET  /health
POST /v1/evaluate-signals
```

The response contains per-rule results, evidence, an overall decision and estimator/policy versions.

## Run and test

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
python -m forecastlab.demo_report --output build/forecastlab-compliance.html
```

CI uses synthetic observations and does not require private image assets.

## Limitations

- raw-image inference is not implemented yet;
- the current estimator path in CI uses deterministic doubles;
- no production or certification claim is made;
- end-to-end accuracy must be measured on a licensed held-out dataset before publishing accuracy numbers.

## Roadmap

1. Add a privacy-conscious raw-image adapter.
2. Implement real pixel estimators behind the existing interfaces.
3. Run the held-out evaluation harness on a licensed, versioned dataset.
4. Extend the demo with real estimator output only after held-out measurements exist.
