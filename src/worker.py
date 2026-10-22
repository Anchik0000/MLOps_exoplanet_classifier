import asyncio
import os
import time
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from faststream import ContextRepo, FastStream
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue

from src.astrobiology import AstrobiologyEngine
from src.config import (
    INFERENCE_DELAY_SECONDS,
    MODEL_ALIAS,
    MODEL_NAME,
    PREFETCH_COUNT,
    QUEUE_MAX_LENGTH,
    QUEUE_OVERFLOW,
    RABBITMQ_URL,
    TASKS_QUEUE_NAME,
    WORKER_HEALTH_PORT,
)
from src.inference import ModelService, load_model_with_retries

logger = structlog.get_logger()

_model_service: ModelService | None = None
_health_server: uvicorn.Server | None = None

health_app = FastAPI(title="Worker Health Check API")


@health_app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model_service is not None,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }


def compute_sync(
    service: ModelService,
    features: dict[str, float | None],
    delay_sec: float = 0.0,
) -> dict:
    if delay_sec > 0:
        time.sleep(delay_sec)

    if not service.knows_any_feature(features):
        raise ValueError("None of the provided features are known by the model")

    prediction_result = service.predict(features)
    radius = features.get("koi_prad")
    teq = features.get("koi_teq")
    esi_metrics = AstrobiologyEngine.compute_all_metrics(radius, teq)

    return {
        "prediction": prediction_result["prediction"],
        "probabilities": prediction_result["probabilities"],
        "missing_features": prediction_result["missing_features"],
        **esi_metrics,
    }


worker_channel = Channel(prefetch_count=PREFETCH_COUNT)
broker = RabbitBroker(RABBITMQ_URL, default_channel=worker_channel)
tasks_queue = RabbitQueue(
    name=TASKS_QUEUE_NAME,
    durable=True,
    arguments={
        "x-max-length": QUEUE_MAX_LENGTH,
        "x-overflow": QUEUE_OVERFLOW,
    },
)


@broker.subscriber(tasks_queue, channel=worker_channel)
async def process_task(payload: dict) -> dict:
    global _model_service
    task_id = payload.get("task_id", "anonymous")
    features = (
        payload.get("features")
        if isinstance(payload.get("features"), dict)
        else payload
    )

    if _model_service is None:
        logger.error("worker_model_not_ready", task_id=task_id)
        raise RuntimeError("Model service is not loaded from MLflow")

    logger.info("worker_task_received", task_id=task_id)

    result = await asyncio.to_thread(
        compute_sync,
        _model_service,
        features,
        INFERENCE_DELAY_SECONDS,
    )

    logger.info(
        "worker_task_completed",
        task_id=task_id,
        prediction=result.get("prediction"),
        esi_percent=result.get("esi_percent"),
    )
    return result


@asynccontextmanager
async def lifespan(context: ContextRepo):
    global _model_service, _health_server
    logger.info(
        "worker_starting",
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        prefetch_count=PREFETCH_COUNT,
        inference_delay_seconds=INFERENCE_DELAY_SECONDS,
    )

    _model_service = await load_model_with_retries()
    if _model_service:
        logger.info(
            "worker_model_loaded",
            model_name=_model_service.model_name,
            version=_model_service.model_version,
            alias=MODEL_ALIAS,
        )
        context.set_global("model_service", _model_service)
    else:
        logger.warning(
            "worker_model_load_failed",
            detail="Proceeding without model; will fail on inference",
        )

    server_task = None
    if os.getenv("ENABLE_WORKER_HEALTH_SERVER", "true").lower() in ("true", "1"):
        try:
            cfg = uvicorn.Config(
                health_app,
                host="0.0.0.0",
                port=WORKER_HEALTH_PORT,
                log_level="warning",
            )
            _health_server = uvicorn.Server(cfg)
            server_task = asyncio.create_task(_health_server.serve())
            logger.info("worker_health_server_started", port=WORKER_HEALTH_PORT)
        except (OSError, RuntimeError) as exc:
            logger.warning("worker_health_server_start_failed", error=str(exc))

    try:
        yield
    finally:
        logger.info("worker_shutting_down")
        if _health_server:
            _health_server.should_exit = True
        if server_task:
            try:
                await server_task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("worker_shutdown_complete")


app = FastStream(broker, lifespan=lifespan)


if __name__ == "__main__":
    asyncio.run(app.run())
