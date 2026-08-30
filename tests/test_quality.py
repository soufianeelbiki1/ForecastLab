from forecastlab.estimators import QualityEstimate
from forecastlab.quality import QualityPolicy, QualityPolicyEvaluator, QualityRuleCode
from forecastlab.quality_evaluation import evaluate_quality_metrics
from forecastlab.synthetic_quality_dataset import (
    QUALITY_DATASET_PROVENANCE,
    QUALITY_DATASET_VERSION,
    synthetic_quality_dataset_v1,
)


def test_quality_policy_returns_explainable_per_rule_failures() -> None:
    evaluation = QualityPolicyEvaluator().evaluate(QualityEstimate(0.60, 0.90, 0.70, 0.95))

    failed = {rule.code for rule in evaluation.rules if not rule.passed}

    assert evaluation.acceptable is False
    assert failed == {QualityRuleCode.BLUR, QualityRuleCode.ILLUMINATION}
    assert 0 < evaluation.score < 1
    assert "score=0.600" in evaluation.rules[0].evidence


def test_synthetic_quality_dataset_has_versioned_non_image_provenance() -> None:
    dataset = synthetic_quality_dataset_v1()

    assert QUALITY_DATASET_VERSION == "synthetic-quality-v1"
    assert "no photographs" in QUALITY_DATASET_PROVENANCE
    assert {example.split for example in dataset} == {"train", "validation", "test"}
    assert len({example.id for example in dataset}) == len(dataset)


def test_reference_quality_policy_matches_synthetic_labels() -> None:
    metrics = evaluate_quality_metrics(synthetic_quality_dataset_v1(), QualityPolicyEvaluator())

    assert {metric.rule for metric in metrics} == set(QualityRuleCode)
    assert all(metric.false_positive == 0 for metric in metrics)
    assert all(metric.false_negative == 0 for metric in metrics)
    assert all(metric.accuracy == 1.0 for metric in metrics)


def test_stricter_quality_thresholds_create_measurable_false_positives() -> None:
    stricter = QualityPolicyEvaluator(
        QualityPolicy(
            version="quality-policy-v1-strict",
            min_blur_score=0.90,
            min_compression_score=0.90,
            min_illumination_score=0.90,
            min_shadow_score=0.90,
        )
    )
    metrics = {item.rule: item for item in evaluate_quality_metrics(synthetic_quality_dataset_v1(), stricter)}

    assert metrics[QualityRuleCode.BLUR].false_positive > 0
    assert metrics[QualityRuleCode.COMPRESSION].false_positive > 0
