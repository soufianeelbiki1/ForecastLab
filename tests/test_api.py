from fastapi.testclient import TestClient

from forecastlab.main import app

client = TestClient(app)


def valid_payload() -> dict[str, int | float]:
    return {
        "width_px": 800,
        "height_px": 800,
        "face_count": 1,
        "yaw_deg": 1.0,
        "pitch_deg": 1.0,
        "roll_deg": 1.0,
        "background_uniformity": 0.97,
        "occlusion_ratio": 0.01,
        "blur_score": 0.95,
        "compression_score": 0.94,
        "illumination_score": 0.93,
        "shadow_score": 0.92,
    }


def test_signal_evaluation_returns_versioned_rule_evidence() -> None:
    response = client.post("/v1/evaluate-signals", json=valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["acceptable"] is True
    assert body["compliance_policy_version"] == "photo-policy-v1"
    assert body["quality_policy_version"] == "quality-policy-v1"
    assert body["estimator_versions"]["pose"] == "external-signal-v1"
    assert len(body["geometric_rules"]) == 5
    assert len(body["quality_rules"]) == 4


def test_signal_evaluation_explains_quality_failure() -> None:
    payload = valid_payload()
    payload["blur_score"] = 0.4

    response = client.post("/v1/evaluate-signals", json=payload)

    assert response.status_code == 200
    body = response.json()
    blur = next(rule for rule in body["quality_rules"] if rule["code"] == "blur")
    assert body["acceptable"] is False
    assert blur["passed"] is False
    assert "minimum=0.800" in blur["evidence"]


def test_signal_endpoint_rejects_invalid_measurement_ranges() -> None:
    payload = valid_payload()
    payload["background_uniformity"] = 1.5

    response = client.post("/v1/evaluate-signals", json=payload)

    assert response.status_code == 422


def test_health_discloses_signal_only_inference_mode() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["inference_mode"] == "precomputed-signals"
