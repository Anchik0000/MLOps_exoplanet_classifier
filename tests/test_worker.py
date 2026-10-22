from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src import worker
from src.worker import (
    compute_sync,
    health_app,
    healthz,
    lifespan,
    process_task,
)


def make_dummy_service():
    service = MagicMock()
    service.model_name = "exoplanet_classifier"
    service.model_version = "1"
    service.knows_any_feature.return_value = True
    service.predict.return_value = {
        "prediction": "CONFIRMED",
        "probabilities": {
            "CANDIDATE": 0.05,
            "CONFIRMED": 0.90,
            "FALSE POSITIVE": 0.05,
        },
        "missing_features": 0,
    }
    return service


def test_healthz():
    client = TestClient(health_app)
    worker._model_service = None
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["model_loaded"] is False

    worker._model_service = make_dummy_service()
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["model_loaded"] is True


@pytest.mark.asyncio
async def test_healthz_direct():
    worker._model_service = None
    data = await healthz()
    assert data["model_loaded"] is False


def test_compute_sync_success():
    service = make_dummy_service()
    features = {"koi_period": 10.0, "koi_prad": 1.0, "koi_teq": 255.0}
    res = compute_sync(service, features, delay_sec=0.01)

    assert res["prediction"] == "CONFIRMED"
    assert res["esi"] == 1.0
    assert res["esi_percent"] == 100.0
    assert res["is_habitable_zone"] is True
    assert res["planet_category"] == "Землеподобный супер-кандидат"


def test_compute_sync_unknown_features():
    service = make_dummy_service()
    service.knows_any_feature.return_value = False

    with pytest.raises(ValueError, match="None of the provided features"):
        compute_sync(service, {"unknown_feature": 123.0})


@pytest.mark.asyncio
async def test_process_task_without_model():
    worker._model_service = None
    with pytest.raises(RuntimeError, match="Model service is not loaded"):
        await process_task({"task_id": "test-1", "features": {"koi_period": 10.0}})


@pytest.mark.asyncio
async def test_process_task_success():
    worker._model_service = make_dummy_service()
    payload = {
        "task_id": "test-1",
        "features": {"koi_period": 10.0, "koi_prad": 1.0, "koi_teq": 255.0},
    }
    res = await process_task(payload)
    assert res["prediction"] == "CONFIRMED"
    assert res["esi_percent"] == 100.0


@pytest.mark.asyncio
async def test_process_task_raw_dict():
    worker._model_service = make_dummy_service()
    payload = {"koi_period": 10.0, "koi_prad": 1.0, "koi_teq": 255.0}
    res = await process_task(payload)
    assert res["prediction"] == "CONFIRMED"


@pytest.mark.asyncio
async def test_lifespan_success():
    context = MagicMock()
    mock_service = make_dummy_service()

    with (
        patch(
            "src.worker.load_model_with_retries",
            new=AsyncMock(return_value=mock_service),
        ),
        patch("src.worker.uvicorn.Server") as mock_server_cls,
    ):
        mock_server = MagicMock()
        mock_server.serve = AsyncMock()
        mock_server_cls.return_value = mock_server

        async with lifespan(context):
            assert worker._model_service == mock_service
            context.set_global.assert_called_once_with("model_service", mock_service)

        assert mock_server.should_exit is True


@pytest.mark.asyncio
async def test_lifespan_model_load_failure():
    context = MagicMock()

    with (
        patch("src.worker.load_model_with_retries", new=AsyncMock(return_value=None)),
        patch("src.worker.uvicorn.Server") as mock_server_cls,
    ):
        mock_server_cls.side_effect = OSError("Port already bound")
        async with lifespan(context):
            assert worker._model_service is None
