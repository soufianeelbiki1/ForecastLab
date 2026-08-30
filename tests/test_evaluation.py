from forecastlab.compliance import PassportComplianceEvaluator, RuleCode
from forecastlab.evaluation import evaluate_rule_metrics
from forecastlab.synthetic_dataset import (
    DATASET_PROVENANCE,
    DATASET_VERSION,
    synthetic_dataset_v1,
)


def test_synthetic_dataset_has_fixed_version_provenance_and_splits() -> None:
    dataset = synthetic_dataset_v1()

    assert DATASET_VERSION == "synthetic-observations-v1"
    assert "no photographs" in DATASET_PROVENANCE
    assert {example.split for example in dataset} == {"train", "validation", "test"}
    assert len({example.id for example in dataset}) == len(dataset)


def test_reference_policy_matches_labeled_synthetic_failures() -> None:
    metrics = evaluate_rule_metrics(synthetic_dataset_v1(), PassportComplianceEvaluator())

    assert {metric.rule for metric in metrics} == set(RuleCode)
    assert all(metric.false_positive == 0 for metric in metrics)
    assert all(metric.false_negative == 0 for metric in metrics)
    assert all(metric.precision == 1.0 for metric in metrics)
    assert all(metric.recall == 1.0 for metric in metrics)
    assert all(metric.accuracy == 1.0 for metric in metrics)


def test_metrics_detect_policy_regression() -> None:
    from forecastlab.compliance import CompliancePolicy

    stricter = PassportComplianceEvaluator(
        CompliancePolicy(min_background_uniformity=0.99)
    )
    metrics = {metric.rule: metric for metric in evaluate_rule_metrics(synthetic_dataset_v1(), stricter)}

    assert metrics[RuleCode.BACKGROUND].false_positive > 0
    assert metrics[RuleCode.BACKGROUND].precision < 1.0
