from dataclasses import dataclass
from enum import StrEnum


class RuleCode(StrEnum):
    IMAGE_DIMENSIONS = "image_dimensions"
    FACE_COUNT = "face_count"
    POSE = "pose"
    BACKGROUND = "background"
    OCCLUSION = "occlusion"


@dataclass(frozen=True, slots=True)
class PhotoObservation:
    width_px: int
    height_px: int
    face_count: int
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    background_uniformity: float
    occlusion_ratio: float


@dataclass(frozen=True, slots=True)
class RuleResult:
    code: RuleCode
    passed: bool
    score: float
    evidence: str


@dataclass(frozen=True, slots=True)
class ComplianceEvaluation:
    compliant: bool
    score: float
    rules: tuple[RuleResult, ...]


@dataclass(frozen=True, slots=True)
class CompliancePolicy:
    min_width_px: int = 600
    min_height_px: int = 600
    max_abs_yaw_deg: float = 8.0
    max_abs_pitch_deg: float = 8.0
    max_abs_roll_deg: float = 5.0
    min_background_uniformity: float = 0.92
    max_occlusion_ratio: float = 0.08

    def __post_init__(self) -> None:
        if self.min_width_px <= 0 or self.min_height_px <= 0:
            raise ValueError("minimum image dimensions must be positive")
        if not 0 <= self.min_background_uniformity <= 1:
            raise ValueError("background uniformity threshold must be within [0, 1]")
        if not 0 <= self.max_occlusion_ratio <= 1:
            raise ValueError("occlusion threshold must be within [0, 1]")


class PassportComplianceEvaluator:
    """Deterministic evaluator over pre-computed computer-vision observations.

    The evaluator does not infer observations from pixels. Pose, segmentation, face
    detection, expression, and quality models can evolve independently while this
    policy layer remains explainable and versionable.
    """

    def __init__(self, policy: CompliancePolicy | None = None) -> None:
        self.policy = policy or CompliancePolicy()

    def evaluate(self, observation: PhotoObservation) -> ComplianceEvaluation:
        self._validate_observation(observation)
        rules = (
            self._dimensions(observation),
            self._face_count(observation),
            self._pose(observation),
            self._background(observation),
            self._occlusion(observation),
        )
        score = sum(rule.score for rule in rules) / len(rules)
        return ComplianceEvaluation(
            compliant=all(rule.passed for rule in rules),
            score=score,
            rules=rules,
        )

    @staticmethod
    def _validate_observation(observation: PhotoObservation) -> None:
        if observation.width_px <= 0 or observation.height_px <= 0:
            raise ValueError("image dimensions must be positive")
        if observation.face_count < 0:
            raise ValueError("face_count must be non-negative")
        if not 0 <= observation.background_uniformity <= 1:
            raise ValueError("background_uniformity must be within [0, 1]")
        if not 0 <= observation.occlusion_ratio <= 1:
            raise ValueError("occlusion_ratio must be within [0, 1]")

    def _dimensions(self, observation: PhotoObservation) -> RuleResult:
        width_score = min(observation.width_px / self.policy.min_width_px, 1.0)
        height_score = min(observation.height_px / self.policy.min_height_px, 1.0)
        score = min(width_score, height_score)
        passed = score >= 1.0
        return RuleResult(
            RuleCode.IMAGE_DIMENSIONS,
            passed,
            score,
            f"{observation.width_px}x{observation.height_px}px; minimum "
            f"{self.policy.min_width_px}x{self.policy.min_height_px}px",
        )

    @staticmethod
    def _face_count(observation: PhotoObservation) -> RuleResult:
        passed = observation.face_count == 1
        return RuleResult(
            RuleCode.FACE_COUNT,
            passed,
            1.0 if passed else 0.0,
            f"detected {observation.face_count} face(s); exactly 1 required",
        )

    def _pose(self, observation: PhotoObservation) -> RuleResult:
        ratios = (
            abs(observation.yaw_deg) / self.policy.max_abs_yaw_deg,
            abs(observation.pitch_deg) / self.policy.max_abs_pitch_deg,
            abs(observation.roll_deg) / self.policy.max_abs_roll_deg,
        )
        worst_ratio = max(ratios)
        score = max(0.0, 1.0 - max(0.0, worst_ratio - 1.0)) if worst_ratio > 1 else 1.0
        passed = worst_ratio <= 1.0
        return RuleResult(
            RuleCode.POSE,
            passed,
            score,
            f"yaw={observation.yaw_deg:.1f}°, pitch={observation.pitch_deg:.1f}°, "
            f"roll={observation.roll_deg:.1f}°",
        )

    def _background(self, observation: PhotoObservation) -> RuleResult:
        threshold = self.policy.min_background_uniformity
        score = min(observation.background_uniformity / threshold, 1.0)
        passed = observation.background_uniformity >= threshold
        return RuleResult(
            RuleCode.BACKGROUND,
            passed,
            score,
            f"uniformity={observation.background_uniformity:.3f}; minimum={threshold:.3f}",
        )

    def _occlusion(self, observation: PhotoObservation) -> RuleResult:
        threshold = self.policy.max_occlusion_ratio
        passed = observation.occlusion_ratio <= threshold
        if threshold == 0:
            score = 1.0 if observation.occlusion_ratio == 0 else 0.0
        else:
            score = max(0.0, 1.0 - observation.occlusion_ratio / threshold) if not passed else 1.0
        return RuleResult(
            RuleCode.OCCLUSION,
            passed,
            score,
            f"occlusion={observation.occlusion_ratio:.3f}; maximum={threshold:.3f}",
        )
