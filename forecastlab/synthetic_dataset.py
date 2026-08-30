from forecastlab.compliance import PhotoObservation, RuleCode
from forecastlab.evaluation import LabeledObservation

DATASET_VERSION = "synthetic-observations-v1"
DATASET_PROVENANCE = (
    "Programmatically authored observation vectors only; no photographs, biometric images, "
    "or personal identity data are included."
)


def synthetic_dataset_v1() -> tuple[LabeledObservation, ...]:
    base = dict(
        width_px=800,
        height_px=1000,
        face_count=1,
        yaw_deg=1.0,
        pitch_deg=1.0,
        roll_deg=0.5,
        background_uniformity=0.97,
        occlusion_ratio=0.02,
    )

    def observation(**overrides: object) -> PhotoObservation:
        values = {**base, **overrides}
        return PhotoObservation(**values)  # type: ignore[arg-type]

    return (
        LabeledObservation("train-clean", "train", observation(), frozenset()),
        LabeledObservation(
            "train-small",
            "train",
            observation(width_px=420),
            frozenset({RuleCode.IMAGE_DIMENSIONS}),
        ),
        LabeledObservation(
            "train-two-faces",
            "train",
            observation(face_count=2),
            frozenset({RuleCode.FACE_COUNT}),
        ),
        LabeledObservation(
            "validation-pose",
            "validation",
            observation(yaw_deg=12.0),
            frozenset({RuleCode.POSE}),
        ),
        LabeledObservation(
            "validation-background",
            "validation",
            observation(background_uniformity=0.74),
            frozenset({RuleCode.BACKGROUND}),
        ),
        LabeledObservation(
            "test-occlusion",
            "test",
            observation(occlusion_ratio=0.19),
            frozenset({RuleCode.OCCLUSION}),
        ),
        LabeledObservation(
            "test-multi-failure",
            "test",
            observation(height_px=430, roll_deg=9.0, background_uniformity=0.80),
            frozenset(
                {
                    RuleCode.IMAGE_DIMENSIONS,
                    RuleCode.POSE,
                    RuleCode.BACKGROUND,
                }
            ),
        ),
        LabeledObservation("test-clean", "test", observation(), frozenset()),
    )
