"""Privacy-conscious held-out evaluation contracts for real image adapters.

The repository stores manifests and labels, not raw photos. Asset providers resolve
licensed images outside the repository and return only the core ImageFrame handle;
pixel ownership remains behind the adapter boundary.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from forecastlab.estimators import ImageFrame, ObservationPipeline
from forecastlab.inference import SignalInferenceService


@dataclass(frozen=True, slots=True)
class HeldOutAsset:
    asset_id: str
    locator: str
    license_reference: str
    expected_acceptable: bool
    slices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("asset_id must not be empty")
        if not self.locator.strip():
            raise ValueError("locator must not be empty")
        if not self.license_reference.strip():
            raise ValueError("license_reference must not be empty")
        if len(set(self.slices)) != len(self.slices):
            raise ValueError("slice labels must be unique per asset")
        if any(not value.strip() for value in self.slices):
            raise ValueError("slice labels must not be empty")


class HeldOutAssetProvider(Protocol):
    """Resolve a licensed external asset without exposing storage to the core."""

    def load(self, asset: HeldOutAsset) -> ImageFrame: ...


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0


@dataclass(frozen=True, slots=True)
class HeldOutEvaluationReport:
    evaluated: int
    overall: BinaryMetrics
    by_slice: dict[str, BinaryMetrics]
    policy_versions: tuple[str, str]


class HeldOutEvaluator:
    """Run licensed held-out assets through estimators and versioned policies."""

    def __init__(
        self,
        *,
        provider: HeldOutAssetProvider,
        pipeline: ObservationPipeline,
        inference: SignalInferenceService,
    ) -> None:
        self._provider = provider
        self._pipeline = pipeline
        self._inference = inference

    def evaluate(self, assets: tuple[HeldOutAsset, ...]) -> HeldOutEvaluationReport:
        if not assets:
            raise ValueError("held-out evaluation requires at least one asset")
        _require_unique((asset.asset_id for asset in assets), "asset ids")
        _require_unique((asset.locator for asset in assets), "asset locators")
        _require_unique(
            (asset.license_reference for asset in assets),
            "asset license references",
        )

        overall_pairs: list[tuple[bool, bool]] = []
        slice_pairs: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        policy_versions: tuple[str, str] | None = None
        resolved_frame_ids: set[str] = set()

        for asset in assets:
            frame = self._provider.load(asset)
            if frame.frame_id in resolved_frame_ids:
                raise ValueError("resolved frame ids must be unique")
            resolved_frame_ids.add(frame.frame_id)
            bundle = self._pipeline.observe(frame)
            result = self._inference.evaluate(bundle.observation, bundle.quality)
            pair = (result.acceptable, asset.expected_acceptable)
            overall_pairs.append(pair)
            for slice_name in asset.slices:
                slice_pairs[slice_name].append(pair)

            versions = (result.compliance_policy_version, result.quality_policy_version)
            if policy_versions is None:
                policy_versions = versions
            elif policy_versions != versions:
                raise ValueError("policy versions changed during held-out evaluation")

        assert policy_versions is not None
        return HeldOutEvaluationReport(
            evaluated=len(assets),
            overall=_metrics(overall_pairs),
            by_slice={name: _metrics(pairs) for name, pairs in sorted(slice_pairs.items())},
            policy_versions=policy_versions,
        )


def _metrics(pairs: list[tuple[bool, bool]]) -> BinaryMetrics:
    tp = sum(predicted and expected for predicted, expected in pairs)
    tn = sum(not predicted and not expected for predicted, expected in pairs)
    fp = sum(predicted and not expected for predicted, expected in pairs)
    fn = sum(not predicted and expected for predicted, expected in pairs)
    return BinaryMetrics(tp, tn, fp, fn)


def _require_unique(values: Iterable[str], label: str) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"held-out {label} must be unique")
