"""Full-stack smoke tests: the real dashboard script against the real API."""

from pathlib import Path
from typing import Any

import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from tests.conftest import APPLE_HEALTH_XML

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
BACKEND_URL = "http://backend.test"


@pytest.fixture
def dashboard(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """The Streamlit app with its HTTP calls routed to the in-process API."""

    def route(self: requests.Session, method: str, url: str, **kwargs: Any) -> Any:
        kwargs.pop("timeout", None)
        return client.request(method, url.removeprefix(BACKEND_URL), **kwargs)

    monkeypatch.setattr(requests.Session, "request", route)
    monkeypatch.setenv("BACKEND_URL", BACKEND_URL)
    monkeypatch.syspath_prepend(str(FRONTEND))
    return AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=30)


def test_dashboard_renders_the_demo_data(dashboard: AppTest) -> None:
    dashboard.run()

    assert not dashboard.exception
    assert not dashboard.error
    labels = [metric.label for metric in dashboard.metric]
    assert "Burnout risk" in labels
    assert "ROC-AUC (model)" in labels
    assert any("synthetic demo data" in info.value for info in dashboard.info)
    assert any("not medical advice" in caption.value for caption in dashboard.caption)


def test_frontend_client_can_upload_and_delete(client: TestClient, dashboard: AppTest) -> None:
    from api_client import ApiError, BackendClient

    backend = BackendClient(BACKEND_URL)
    backend.wait_until_ready()
    slug = backend.create_dataset("Mine")

    result = backend.import_file(slug, "apple-health", "export.xml", APPLE_HEALTH_XML.read_bytes())
    assert result["days_added"] == 150

    with pytest.raises(ApiError) as excinfo:
        backend.import_file(slug, "csv", "bad.csv", b"date,sleep_hours\n2026-01-01,99\n")
    assert excinfo.value.status == 422
    assert excinfo.value.errors[0].startswith("Line 2: sleep_hours")

    backend.delete_dataset(slug)
    with pytest.raises(ApiError, match="does not exist"):
        backend.get(f"/datasets/{slug}")
