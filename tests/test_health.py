import asyncio
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.app import app
from src.config import get_app_version

client = TestClient(app)


def test_healthz_success():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_success():
    response = client.get("/api/v1/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert isinstance(data["version"], str)


def test_detailed_health_db_success():
    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = "PostgreSQL 16.1"
    mock_conn.close = AsyncMock()

    with patch("asyncpg.connect", return_value=mock_conn):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["components"]["database"]["status"] == "ok"
        assert data["components"]["database"]["version"] == "PostgreSQL 16.1"
        assert data["components"]["database"]["latency_ms"] >= 0


def test_detailed_health_db_connection_error():
    with patch("asyncpg.connect", side_effect=ConnectionError("Database is down")):
        response = client.get("/api/v1/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["components"]["database"]["status"] == "unavailable"
        assert "Database is down" in data["components"]["database"]["error"]


def test_detailed_health_db_timeout():
    with patch("asyncpg.connect", side_effect=asyncio.TimeoutError("Query timed out")):
        response = client.get("/api/v1/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["components"]["database"]["status"] == "unavailable"


def test_not_found_route():
    response = client.get("/api/v1/non_existing_endpoint")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_method_not_allowed():
    response = client.post("/healthz")
    assert response.status_code == 405


def test_config_version_fallback_on_error():
    with patch("builtins.open", side_effect=OSError("File read error")):
        version = get_app_version()
        assert version == "0.1.0"
