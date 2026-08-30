from dataclasses import dataclass

import pytest

from forecastlab.compliance import PassportComplianceEvaluator
from forecastlab.estimators import (
    BackgroundEstimate,
    ImageFrame,
    ObservationPipeline,
    OcclusionEstimate,
    PoseEstimate,
    QualityEstimate,
)


@dataclass
class StaticPoseEstimator:
    value: PoseEstimate

    def estimate(self, frame: ImageFrame) -> PoseEstimate:
        assert frame.frame_id
        return self.value


@dataclass
class StaticBackgroundEstimator:
    value: BackgroundEstimate

    def estimate(self, frame: ImageFrame) -> BackgroundEstimate:
        assert frame.width_px > 0
        return self.value


@dataclass
class StaticOcclusionEstimator:
    value: OcclusionEstimate

    def estimate(self, frame: ImageFrame) -> OcclusionEstimate:
        assert frame.height_px > 0
        return self.value


@dataclass
class StaticQualityEstimator:
    value: QualityEstimate

    def estimate(self, frame: ImageFrame) -> QualityEstimate:
        assert frame.frame_id
        return self.value


def pipeline(
    *,
    pose: PoseEstimate | None = None,
    background: BackgroundEstimate | None = None,
    occlusion: OcclusionEstimate | None = None,
) -> ObservationPipeline:
    return ObservationPipeline(
        pose=StaticPoseEstimator(pose or PoseEstimate(1, 1.0, -1.0, 0.5)),
        background=StaticBackgroundEstimator(background or BackgroundEstimate(0.97)),
        occlusion=StaticOcclusionEstimator(occlusion or OcclusionEstimate(0.02)),
        quality=StaticQualityEstimator(QualityEstimate(0.92, 0.95, 0.93, 0.91)),
    )


def test_pipeline_composes_estimators_into_policy_input() -> None:
    bundle = pipeline().observe(ImageFrame("synthetic-frame-1", 800, 1000))

    assert bundle.observation.width_px == 800
    assert bundle.observation.face_count == 1
    assert bundle.quality.blur_score == 0.92
    assert PassportComplianceEvaluator().evaluate(bundle.observation).compliant is True


def test_pipeline_keeps_quality_signals_separate_from_policy_until_rules_exist() -> None:
    bundle = pipeline().observe(ImageFrame("synthetic-frame-2", 800, 1000))

    assert bundle.quality.compression_score == 0.95
    assert not hasattr(bundle.observation, "blur_score")


def test_pipeline_rejects_invalid_estimator_ranges() -> None:
    with pytest.raises(ValueError, match="background uniformity"):
        pipeline(background=BackgroundEstimate(1.4)).observe(ImageFrame("bad-bg", 800, 1000))
    with pytest.raises(ValueError, match="occlusion ratio"):
        pipeline(occlusion=OcclusionEstimate(-0.1)).observe(ImageFrame("bad-occ", 800, 1000))
    with pytest.raises(ValueError, match="face_count"):
        pipeline(pose=PoseEstimate(-1, 0.0, 0.0, 0.0)).observe(ImageFrame("bad-face", 800, 1000))


def test_quality_estimate_rejects_out_of_range_scores() -> None:
    with pytest.raises(ValueError, match="blur_score"):
        QualityEstimate(1.1, 0.9, 0.9, 0.9)


def test_image_frame_rejects_invalid_identity_and_dimensions() -> None:
    with pytest.raises(ValueError, match="frame_id"):
        ImageFrame("", 800, 1000)
    with pytest.raises(ValueError, match="dimensions"):
        ImageFrame("bad", 0, 1000)
