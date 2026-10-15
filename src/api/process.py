import asyncio

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = structlog.get_logger()
router = APIRouter(tags=["Inference"])


class ProcessRequest(BaseModel):
    features: dict[str, float | None] = Field(
        min_length=1,
        description="Признаки кандидата: имя колонки KOI и значение",
        examples=[
            {
                "koi_period": 9.488,
                "koi_duration": 2.96,
                "koi_depth": 615.8,
                "koi_prad": 2.26,
                "koi_teq": 793.0,
                "koi_steff": 5455.0,
                "koi_srad": 0.927,
            }
        ],
    )


class ProcessResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    missing_features: int
    model_name: str
    model_version: str


@router.post("/process", response_model=ProcessResponse)
async def process(payload: ProcessRequest, request: Request) -> ProcessResponse:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена из MLflow",
        )

    if not service.knows_any_feature(payload.features):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Ни один из переданных признаков не известен модели",
        )

    result = await asyncio.to_thread(service.predict, payload.features)
    logger.info(
        "process_called",
        prediction=result["prediction"],
        model_version=service.model_version,
        missing_features=result["missing_features"],
    )

    return ProcessResponse(
        **result,
        model_name=service.model_name,
        model_version=service.model_version,
    )
