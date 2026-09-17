from fastapi import FastAPI

from src.api.v1.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="MLOps Project API",
        version="0.1.0",
        description="Учебный FastAPI-сервер для проекта MLOps",
    )
    app.include_router(health_router, prefix="/api/v1")
    return app


app = create_app()
