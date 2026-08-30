from fastapi import FastAPI
from pydantic import BaseModel, Field

from forecastlab.compliance import PhotoObservation
from forecastlab.estimators import QualityEstimate
from forecastlab.inference import EstimatorVersions, SignalInferenceService

app = FastAPI(title="ForecastLab passport-photo evaluation", version="0.1.0")

_service = SignalInferenceService(
    estimator_versions=EstimatorVersions(
        pose="external-signal-v1",
        background="external-signal-v1",
        occlusion="external-signal-v1",
        quality="external-signal-v1",
    )
)


class SignalEvaluationRequest(BaseModel):
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    face_count: int = Field(ge=0)
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    background_uniformity: float = Field(ge=0.0, le=1.0)
    occlusion_ratio: float = Field(ge=0.0, le=1.0)
    blur_score: float = Field(ge=0.0, le=1.0)
    compression_score: float = Field(ge=0.0, le=1.0)
    illumination_score: float = Field(ge=0.0, le=1.0)
    shadow_score: float = Field(ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "inference_mode": "precomputed-signals",
    }


@app.post("/v1/evaluate-signals")
def evaluate_signals(request: SignalEvaluationRequest) -> dict[str, object]:
    result = _service.evaluate(
        PhotoObservation(
            width_px=request.width_px,
            height_px=request.height_px,
            face_count=request.face_count,
            yaw_deg=request.yaw_deg,
            pitch_deg=request.pitch_deg,
            roll_deg=request.roll_deg,
            background_uniformity=request.background_uniformity,
            occlusion_ratio=request.occlusion_ratio,
        ),
        QualityEstimate(
            blur_score=request.blur_score,
            compression_score=request.compression_score,
            illumination_score=request.illumination_score,
            shadow_score=request.shadow_score,
        ),
    )
    return {
        "acceptable": result.acceptable,
        "score": result.score,
        "compliance_policy_version": result.compliance_policy_version,
        "quality_policy_version": result.quality_policy_version,
        "estimator_versions": {
            "pose": result.estimator_versions.pose,
            "background": result.estimator_versions.background,
            "occlusion": result.estimator_versions.occlusion,
            "quality": result.estimator_versions.quality,
        },
        "geometric_rules": [
            {
                "code": rule.code,
                "passed": rule.passed,
                "score": rule.score,
                "evidence": rule.evidence,
            }
            for rule in result.geometric_rules
        ],
        "quality_rules": [
            {
                "code": rule.code,
                "passed": rule.passed,
                "score": rule.score,
                "evidence": rule.evidence,
            }
            for rule in result.quality_rules
        ],
    }
