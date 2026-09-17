from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
