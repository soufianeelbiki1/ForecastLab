import pytest

from forecastlab.compliance import (
    PassportComplianceEvaluator,
    PhotoObservation,
    RuleCode,
)


def compliant_observation(**overrides: object) -> PhotoObservation:
    values: dict[str, object] = {
        "width_px": 800,
        "height_px": 1000,
        "face_count": 1,
        "yaw_deg": 1.5,
        "pitch_deg": -2.0,
        "roll_deg": 0.5,
        "background_uniformity": 0.97,
        "occlusion_ratio": 0.02,
    }
    values.update(overrides)
    return PhotoObservation(**values)  # type: ignore[arg-type]


def test_compliant_observation_passes_every_rule() -> None:
    evaluation = PassportComplianceEvaluator().evaluate(compliant_observation())

    assert evaluation.compliant is True
    assert evaluation.score == 1.0
    assert all(rule.passed for rule in evaluation.rules)


@pytest.mark.parametrize(
    ("overrides", "expected_rule"),
    [
        ({"width_px": 400}, RuleCode.IMAGE_DIMENSIONS),
        ({"face_count": 2}, RuleCode.FACE_COUNT),
        ({"yaw_deg": 13.0}, RuleCode.POSE),
        ({"background_uniformity": 0.70}, RuleCode.BACKGROUND),
        ({"occlusion_ratio": 0.22}, RuleCode.OCCLUSION),
    ],
)
def test_non_compliance_identifies_failed_rule(
    overrides: dict[str, object], expected_rule: RuleCode
) -> None:
    evaluation = PassportComplianceEvaluator().evaluate(compliant_observation(**overrides))

    assert evaluation.compliant is False
    failed = {rule.code for rule in evaluation.rules if not rule.passed}
    assert expected_rule in failed
    assert 0 <= evaluation.score < 1


def test_rule_evidence_contains_measured_values() -> None:
    evaluation = PassportComplianceEvaluator().evaluate(
        compliant_observation(yaw_deg=12.5, background_uniformity=0.81)
    )
    evidence = {rule.code: rule.evidence for rule in evaluation.rules}

    assert "yaw=12.5°" in evidence[RuleCode.POSE]
    assert "uniformity=0.810" in evidence[RuleCode.BACKGROUND]


def test_invalid_measurement_ranges_are_rejected() -> None:
    evaluator = PassportComplianceEvaluator()

    with pytest.raises(ValueError, match="background_uniformity"):
        evaluator.evaluate(compliant_observation(background_uniformity=1.2))
    with pytest.raises(ValueError, match="occlusion_ratio"):
        evaluator.evaluate(compliant_observation(occlusion_ratio=-0.1))
