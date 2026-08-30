from forecastlab.estimators import QualityEstimate
from forecastlab.quality import QualityRuleCode
from forecastlab.quality_evaluation import LabeledQualityExample

QUALITY_DATASET_VERSION = "synthetic-quality-v1"
QUALITY_DATASET_PROVENANCE = (
    "Synthetic normalized quality scores only; no photographs, biometric templates, "
    "or personal identity data are included."
)


def synthetic_quality_dataset_v1() -> tuple[LabeledQualityExample, ...]:
    return (
        LabeledQualityExample(
            "q-train-pass",
            "train",
            QualityEstimate(0.92, 0.91, 0.94, 0.90),
            frozenset(),
        ),
        LabeledQualityExample(
            "q-train-blur",
            "train",
            QualityEstimate(0.55, 0.91, 0.94, 0.90),
            frozenset({QualityRuleCode.BLUR}),
        ),
        LabeledQualityExample(
            "q-validation-compression",
            "validation",
            QualityEstimate(0.90, 0.60, 0.92, 0.91),
            frozenset({QualityRuleCode.COMPRESSION}),
        ),
        LabeledQualityExample(
            "q-validation-lighting",
            "validation",
            QualityEstimate(0.91, 0.90, 0.62, 0.57),
            frozenset({QualityRuleCode.ILLUMINATION, QualityRuleCode.SHADOW}),
        ),
        LabeledQualityExample(
            "q-test-pass",
            "test",
            QualityEstimate(0.82, 0.83, 0.84, 0.85),
            frozenset(),
        ),
        LabeledQualityExample(
            "q-test-multi",
            "test",
            QualityEstimate(0.79, 0.78, 0.77, 0.76),
            frozenset(QualityRuleCode),
        ),
    )
