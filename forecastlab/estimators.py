from dataclasses import dataclass
from typing import Protocol

from forecastlab.compliance import PhotoObservation


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """Privacy-conscious image boundary.

    Production adapters can wrap decoded image bytes or tensors outside this package.
    The core pipeline only needs dimensions plus an opaque frame identifier for tracing.
    """

    frame_id: str
    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame_id must not be empty")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("frame dimensions must be positive")


@dataclass(frozen=True, slots=True)
class PoseEstimate:
    face_count: int
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass(frozen=True, slots=True)
class BackgroundEstimate:
    uniformity: float


@dataclass(frozen=True, slots=True)
class OcclusionEstimate:
    ratio: float


@dataclass(frozen=True, slots=True)
class QualityEstimate:
    blur_score: float
    compression_score: float
    illumination_score: float
    shadow_score: float

    def __post_init__(self) -> None:
        for name, value in (
            ("blur_score", self.blur_score),
            ("compression_score", self.compression_score),
            ("illumination_score", self.illumination_score),
            ("shadow_score", self.shadow_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


class PoseEstimator(Protocol):
    def estimate(self, frame: ImageFrame) -> PoseEstimate: ...


class BackgroundEstimator(Protocol):
    def estimate(self, frame: ImageFrame) -> BackgroundEstimate: ...


class OcclusionEstimator(Protocol):
    def estimate(self, frame: ImageFrame) -> OcclusionEstimate: ...


class QualityEstimator(Protocol):
    def estimate(self, frame: ImageFrame) -> QualityEstimate: ...


@dataclass(frozen=True, slots=True)
class ObservationBundle:
    observation: PhotoObservation
    quality: QualityEstimate


class ObservationPipeline:
    """Composes independent estimators into the deterministic policy input contract."""

    def __init__(
        self,
        *,
        pose: PoseEstimator,
        background: BackgroundEstimator,
        occlusion: OcclusionEstimator,
        quality: QualityEstimator,
    ) -> None:
        self._pose = pose
        self._background = background
        self._occlusion = occlusion
        self._quality = quality

    def observe(self, frame: ImageFrame) -> ObservationBundle:
        pose = self._pose.estimate(frame)
        background = self._background.estimate(frame)
        occlusion = self._occlusion.estimate(frame)
        quality = self._quality.estimate(frame)

        if pose.face_count < 0:
            raise ValueError("estimated face_count must be non-negative")
        if not 0.0 <= background.uniformity <= 1.0:
            raise ValueError("estimated background uniformity must be within [0, 1]")
        if not 0.0 <= occlusion.ratio <= 1.0:
            raise ValueError("estimated occlusion ratio must be within [0, 1]")

        return ObservationBundle(
            observation=PhotoObservation(
                width_px=frame.width_px,
                height_px=frame.height_px,
                face_count=pose.face_count,
                yaw_deg=pose.yaw_deg,
                pitch_deg=pose.pitch_deg,
                roll_deg=pose.roll_deg,
                background_uniformity=background.uniformity,
                occlusion_ratio=occlusion.ratio,
            ),
            quality=quality,
        )
