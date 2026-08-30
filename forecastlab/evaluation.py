from dataclasses import dataclass
from typing import Iterable

from forecastlab.compliance import PassportComplianceEvaluator, PhotoObservation, RuleCode


@dataclass(frozen=True, slots=True)
class LabeledObservation:
    id: str
    split: str
    observation: PhotoObservation
    expected_failures: frozenset[RuleCode]

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if not self.id.strip():
            raise ValueError("observation id must not be empty")


@dataclass(frozen=True, slots=True)
class RuleMetrics:
    rule: RuleCode
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def accuracy(self) -> float:
        total = self.true_positive + self.false_positive + self.true_negative + self.false_negative
        correct = self.true_positive + self.true_negative
        return correct / total if total else 0.0


def evaluate_rule_metrics(
    examples: Iterable[LabeledObservation],
    evaluator: PassportComplianceEvaluator,
) -> tuple[RuleMetrics, ...]:
    materialized = tuple(examples)
    metrics: list[RuleMetrics] = []
    for rule in RuleCode:
        true_positive = false_positive = true_negative = false_negative = 0
        for example in materialized:
            result = evaluator.evaluate(example.observation)
            predicted_failure = any(item.code is rule and not item.passed for item in result.rules)
            expected_failure = rule in example.expected_failures
            if predicted_failure and expected_failure:
                true_positive += 1
            elif predicted_failure and not expected_failure:
                false_positive += 1
            elif not predicted_failure and expected_failure:
                false_negative += 1
            else:
                true_negative += 1
        metrics.append(
            RuleMetrics(
                rule=rule,
                true_positive=true_positive,
                false_positive=false_positive,
                true_negative=true_negative,
                false_negative=false_negative,
            )
        )
    return tuple(metrics)
