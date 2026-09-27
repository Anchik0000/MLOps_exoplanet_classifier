import structlog
from fastapi import FastAPI

from src.api.v1.health import router as health_v1_router
from src.config import get_app_version

logger = structlog.get_logger()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MLOps Project API",
        version=get_app_version(),
        description="FastAPI сервис с мониторингом состояния и компонентов",
    )

    @app.get("/healthz", tags=["Probes"])
    async def healthz() -> dict[str, str]:
        logger.info("liveness_probe_called", endpoint="/healthz", status="ok")
        return {"status": "ok"}

    app.include_router(health_v1_router, prefix="/api/v1")

    return app


app = create_app()
