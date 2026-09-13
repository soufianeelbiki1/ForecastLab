from dataclasses import dataclass

import pytest

from forecastlab.estimators import (
    BackgroundEstimate,
    ImageFrame,
    ObservationPipeline,
    OcclusionEstimate,
    PoseEstimate,
    QualityEstimate,
)
from forecastlab.heldout_evaluation import HeldOutAsset, HeldOutEvaluator
from forecastlab.inference import EstimatorVersions, SignalInferenceService


@dataclass
class ManifestFrameProvider:
    frames: dict[str, ImageFrame]

    def load(self, asset: HeldOutAsset) -> ImageFrame:
        assert asset.license_reference.startswith("license://")
        return self.frames[asset.locator]


class PoseEstimator:
    def estimate(self, frame: ImageFrame) -> PoseEstimate:
        del frame
        return PoseEstimate(1, 0.0, 0.0, 0.0)


class BackgroundEstimator:
    def estimate(self, frame: ImageFrame) -> BackgroundEstimate:
        del frame
        return BackgroundEstimate(0.98)


class OcclusionEstimator:
    def estimate(self, frame: ImageFrame) -> OcclusionEstimate:
        del frame
        return OcclusionEstimate(0.01)


class QualityEstimator:
    def estimate(self, frame: ImageFrame) -> QualityEstimate:
        del frame
        return QualityEstimate(0.95, 0.95, 0.95, 0.95)


def evaluator(provider: ManifestFrameProvider | None = None) -> HeldOutEvaluator:
    provider = provider or ManifestFrameProvider(
        {
            "vault://good": ImageFrame("asset-good", 800, 1000),
            "vault://small": ImageFrame("asset-small", 200, 250),
        }
    )
    pipeline = ObservationPipeline(
        pose=PoseEstimator(),
        background=BackgroundEstimator(),
        occlusion=OcclusionEstimator(),
        quality=QualityEstimator(),
    )
    inference = SignalInferenceService(
        estimator_versions=EstimatorVersions(
            pose="pose-test-v1",
            background="background-test-v1",
            occlusion="occlusion-test-v1",
            quality="quality-test-v1",
        )
    )
    return HeldOutEvaluator(provider=provider, pipeline=pipeline, inference=inference)


def test_heldout_evaluation_reports_overall_and_slice_metrics() -> None:
    report = evaluator().evaluate(
        (
            HeldOutAsset(
                asset_id="good",
                locator="vault://good",
                license_reference="license://dataset-v1/good",
                expected_acceptable=True,
                slices=("studio", "adult"),
            ),
            HeldOutAsset(
                asset_id="small",
                locator="vault://small",
                license_reference="license://dataset-v1/small",
                expected_acceptable=False,
                slices=("resolution-failure", "adult"),
            ),
        )
    )

    assert report.evaluated == 2
    assert report.overall.accuracy == pytest.approx(1.0)
    assert report.overall.precision == pytest.approx(1.0)
    assert report.overall.recall == pytest.approx(1.0)
    assert report.by_slice["adult"].accuracy == pytest.approx(1.0)
    assert report.by_slice["resolution-failure"].true_negative == 1
    assert report.policy_versions[0] == "photo-policy-v1"


def test_manifest_requires_external_license_provenance_and_unique_ids() -> None:
    with pytest.raises(ValueError, match="license_reference"):
        HeldOutAsset("asset", "vault://asset", "", True)

    duplicate = HeldOutAsset("same", "vault://good", "license://dataset-v1/a", True)
    with pytest.raises(ValueError, match="ids must be unique"):
        evaluator().evaluate((duplicate, duplicate))


@pytest.mark.parametrize(
    ("second", "message"),
    (
        (
            HeldOutAsset("other", "vault://good", "license://dataset-v1/b", True),
            "locators must be unique",
        ),
        (
            HeldOutAsset("other", "vault://small", "license://dataset-v1/a", False),
            "license references must be unique",
        ),
    ),
)
def test_manifest_rejects_duplicate_asset_references(
    second: HeldOutAsset,
    message: str,
) -> None:
    first = HeldOutAsset("first", "vault://good", "license://dataset-v1/a", True)

    with pytest.raises(ValueError, match=message):
        evaluator().evaluate((first, second))


def test_manifest_rejects_aliases_that_resolve_to_the_same_frame() -> None:
    duplicate_frame = ImageFrame("same-frame", 800, 1000)
    provider = ManifestFrameProvider(
        {
            "vault://alias-a": duplicate_frame,
            "vault://alias-b": duplicate_frame,
        }
    )
    heldout_evaluator = evaluator(provider)
    assets = (
        HeldOutAsset("a", "vault://alias-a", "license://dataset-v1/a", True),
        HeldOutAsset("b", "vault://alias-b", "license://dataset-v1/b", True),
    )

    with pytest.raises(ValueError, match="resolved frame ids must be unique"):
        heldout_evaluator.evaluate(assets)


def test_empty_evaluation_is_rejected_instead_of_reporting_fake_metrics() -> None:
    with pytest.raises(ValueError, match="at least one asset"):
        evaluator().evaluate(())
