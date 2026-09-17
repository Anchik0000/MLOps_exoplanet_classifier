from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.app import app

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


def test_detailed_health_db_failure():
    with patch("asyncpg.connect", side_effect=ConnectionError("Cannot connect to DB")):
        response = client.get("/api/v1/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["components"]["database"]["status"] == "unavailable"
        assert "Cannot connect to DB" in data["components"]["database"]["error"]


def test_detailed_health_db_success():
    mock_connection = AsyncMock()
    mock_connection.fetchval.return_value = "PostgreSQL 16.1"
    mock_connection.close = AsyncMock()

    with patch("asyncpg.connect", return_value=mock_connection):
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["components"]["database"]["status"] == "ok"
        assert data["components"]["database"]["version"] == "PostgreSQL 16.1"
        assert data["components"]["database"]["latency_ms"] >= 0
