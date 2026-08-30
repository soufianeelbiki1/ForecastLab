from dataclasses import dataclass

from forecastlab.compliance import PassportComplianceEvaluator, PhotoObservation
from forecastlab.estimators import QualityEstimate
from forecastlab.quality import QualityPolicyEvaluator


@dataclass(frozen=True, slots=True)
class EstimatorVersions:
    pose: str
    background: str
    occlusion: str
    quality: str

    def __post_init__(self) -> None:
        for name, value in (
            ("pose", self.pose),
            ("background", self.background),
            ("occlusion", self.occlusion),
            ("quality", self.quality),
        ):
            if not value.strip():
                raise ValueError(f"{name} estimator version must not be empty")


@dataclass(frozen=True, slots=True)
class RuleEvidence:
    code: str
    passed: bool
    score: float
    evidence: str


@dataclass(frozen=True, slots=True)
class InferenceResult:
    acceptable: bool
    score: float
    compliance_policy_version: str
    quality_policy_version: str
    estimator_versions: EstimatorVersions
    geometric_rules: tuple[RuleEvidence, ...]
    quality_rules: tuple[RuleEvidence, ...]


class SignalInferenceService:
    """Evaluate precomputed CV signals with explicit version provenance.

    This service does not decode pixels. Production image adapters can call it
    after versioned estimators produce observations and quality scores.
    """

    def __init__(
        self,
        *,
        compliance: PassportComplianceEvaluator | None = None,
        quality: QualityPolicyEvaluator | None = None,
        compliance_policy_version: str = "photo-policy-v1",
        estimator_versions: EstimatorVersions,
    ) -> None:
        if not compliance_policy_version.strip():
            raise ValueError("compliance_policy_version must not be empty")
        self._compliance = compliance or PassportComplianceEvaluator()
        self._quality = quality or QualityPolicyEvaluator()
        self._compliance_policy_version = compliance_policy_version
        self._estimator_versions = estimator_versions

    def evaluate(
        self,
        observation: PhotoObservation,
        quality: QualityEstimate,
    ) -> InferenceResult:
        geometric = self._compliance.evaluate(observation)
        quality_result = self._quality.evaluate(quality)
        geometric_rules = tuple(
            RuleEvidence(
                code=rule.code.value,
                passed=rule.passed,
                score=rule.score,
                evidence=rule.evidence,
            )
            for rule in geometric.rules
        )
        quality_rules = tuple(
            RuleEvidence(
                code=rule.code.value,
                passed=rule.passed,
                score=rule.score,
                evidence=rule.evidence,
            )
            for rule in quality_result.rules
        )
        combined_rules = geometric_rules + quality_rules
        return InferenceResult(
            acceptable=geometric.compliant and quality_result.acceptable,
            score=sum(rule.score for rule in combined_rules) / len(combined_rules),
            compliance_policy_version=self._compliance_policy_version,
            quality_policy_version=self._quality.policy.version,
            estimator_versions=self._estimator_versions,
            geometric_rules=geometric_rules,
            quality_rules=quality_rules,
        )
