from dataclasses import dataclass
from enum import StrEnum

from forecastlab.estimators import QualityEstimate


class QualityRuleCode(StrEnum):
    BLUR = "blur"
    COMPRESSION = "compression"
    ILLUMINATION = "illumination"
    SHADOW = "shadow"


@dataclass(frozen=True, slots=True)
class QualityRuleResult:
    code: QualityRuleCode
    passed: bool
    score: float
    evidence: str


@dataclass(frozen=True, slots=True)
class QualityEvaluation:
    acceptable: bool
    score: float
    rules: tuple[QualityRuleResult, ...]


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    version: str = "quality-policy-v1"
    min_blur_score: float = 0.80
    min_compression_score: float = 0.80
    min_illumination_score: float = 0.80
    min_shadow_score: float = 0.80

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("quality policy version must not be empty")
        for name, value in (
            ("min_blur_score", self.min_blur_score),
            ("min_compression_score", self.min_compression_score),
            ("min_illumination_score", self.min_illumination_score),
            ("min_shadow_score", self.min_shadow_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


class QualityPolicyEvaluator:
    """Deterministic policy over normalized image-quality estimator scores.

    Scores are estimator outputs, not ICAO measurements by themselves. Thresholds
    are versioned and must be validated against labeled data before any accuracy
    claim is made.
    """

    def __init__(self, policy: QualityPolicy | None = None) -> None:
        self.policy = policy or QualityPolicy()

    def evaluate(self, estimate: QualityEstimate) -> QualityEvaluation:
        rules = (
            self._minimum(QualityRuleCode.BLUR, estimate.blur_score, self.policy.min_blur_score),
            self._minimum(
                QualityRuleCode.COMPRESSION,
                estimate.compression_score,
                self.policy.min_compression_score,
            ),
            self._minimum(
                QualityRuleCode.ILLUMINATION,
                estimate.illumination_score,
                self.policy.min_illumination_score,
            ),
            self._minimum(
                QualityRuleCode.SHADOW,
                estimate.shadow_score,
                self.policy.min_shadow_score,
            ),
        )
        return QualityEvaluation(
            acceptable=all(rule.passed for rule in rules),
            score=sum(rule.score for rule in rules) / len(rules),
            rules=rules,
        )

    @staticmethod
    def _minimum(code: QualityRuleCode, value: float, threshold: float) -> QualityRuleResult:
        passed = value >= threshold
        score = min(value / threshold, 1.0) if threshold > 0 else 1.0
        return QualityRuleResult(
            code=code,
            passed=passed,
            score=score,
            evidence=f"score={value:.3f}; minimum={threshold:.3f}",
        )
