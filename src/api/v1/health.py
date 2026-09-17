import time

import asyncpg
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.config import DATABASE_URL, get_app_version

router = APIRouter(tags=["Health & Monitoring"])


@router.get("/version")
async def get_version() -> dict[str, str]:
    return {"version": get_app_version()}


@router.get("/health")
async def detailed_health_check() -> JSONResponse:
    start_time = time.perf_counter()
    db_status = "ok"
    db_version = "unknown"
    error_detail = None

    try:
        conn = await asyncpg.connect(DATABASE_URL, timeout=2.0)
        try:
            server_version = await conn.fetchval("SHOW server_version;")
            db_version = str(server_version)
        finally:
            await conn.close()
    except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
        db_status = "unavailable"
        error_detail = str(exc)

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    response_payload = {
        "status": "healthy" if db_status == "ok" else "unhealthy",
        "components": {
            "database": {
                "status": db_status,
                "version": db_version,
                "latency_ms": latency_ms,
            }
        },
    }

    if error_detail:
        response_payload["components"]["database"]["error"] = error_detail

    http_code = (
        status.HTTP_200_OK if db_status == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(status_code=http_code, content=response_payload)
