"""Full-stack smoke tests: the real dashboard script against the real API."""

import datetime as dt
from pathlib import Path
from typing import Any

import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from tests.conftest import APPLE_HEALTH_XML, DEMO_CSV

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


def dataset_choice(dashboard: AppTest) -> Any:
    """The sidebar's dataset selector."""
    return next(radio for radio in dashboard.radio if radio.label == "Dataset")


def test_dashboard_renders_the_demo_data(dashboard: AppTest) -> None:
    dashboard.run()

    assert not dashboard.exception
    assert not dashboard.error
    labels = [metric.label for metric in dashboard.metric]
    assert "Burnout risk" in labels
    assert "ROC-AUC (model)" in labels
    assert any("synthetic demo data" in info.value for info in dashboard.info)
    assert any("not medical advice" in caption.value for caption in dashboard.caption)


def test_a_private_link_reopens_the_uploaded_dataset(
    client: TestClient, dashboard: AppTest
) -> None:
    slug = client.post("/api/v1/datasets").json()["slug"]
    client.post(
        f"/api/v1/datasets/{slug}/import/csv",
        files={"file": ("data.csv", DEMO_CSV.read_bytes())},
    )
    dashboard.query_params["data"] = slug

    dashboard.run()

    assert not dashboard.exception
    assert dataset_choice(dashboard).options == ["Demo data (synthetic)", "My data"]
    assert dataset_choice(dashboard).value == "My data"
    assert not any("synthetic demo data" in info.value for info in dashboard.info)
    assert any("private link" in caption.value for caption in dashboard.caption)


def test_an_expired_private_link_falls_back_to_the_demo(dashboard: AppTest) -> None:
    dashboard.query_params["data"] = "no-longer-exists"

    dashboard.run()

    assert not dashboard.exception
    assert dataset_choice(dashboard).options == ["Demo data (synthetic)"]
    assert "data" not in dashboard.query_params
    assert any("has expired" in warning.value for warning in dashboard.warning)
    assert any("synthetic demo data" in info.value for info in dashboard.info)


def test_a_check_in_creates_a_private_dataset_and_saves_the_day(
    client: TestClient, dashboard: AppTest
) -> None:
    dashboard.run()
    today = dt.date.today()

    dashboard.number_input(key=f"checkin_sleep_hours_{today}").set_value(7.5)
    dashboard.number_input(key=f"checkin_study_hours_{today}").set_value(3.0)
    dashboard.button(key="checkin_save").click()
    dashboard.run()

    assert not dashboard.exception
    slug = dashboard.query_params["data"][0]
    stored = client.get(f"/api/v1/datasets/{slug}/metrics").json()
    assert len(stored) == 1
    assert (stored[0]["date"], stored[0]["sleep_hours"], stored[0]["study_hours"]) == (
        today.isoformat(),
        7.5,
        3.0,
    )
    assert stored[0]["mood_score"] is None
    assert any("Saved" in message.value for message in dashboard.success)
    assert dataset_choice(dashboard).value == "My data"
    # The form now shows what was saved.
    assert dashboard.number_input(key=f"checkin_sleep_hours_{today}").value == 7.5


def test_an_empty_check_in_is_not_saved(client: TestClient, dashboard: AppTest) -> None:
    dashboard.run()

    dashboard.button(key="checkin_save").click()
    dashboard.run()

    assert any("at least one field" in warning.value for warning in dashboard.warning)
    assert "data" not in dashboard.query_params


def test_frontend_client_fetches_several_paths_at_once(
    client: TestClient, dashboard: AppTest
) -> None:
    from api_client import ApiError, BackendClient

    backend = BackendClient(BACKEND_URL)

    loaded = backend.get_many({"a": "/datasets/demo", "b": "/datasets/demo/insights"})

    assert loaded["a"]["days"] == 180
    assert loaded["b"]["dataset"] == "demo"
    with pytest.raises(ApiError, match="does not exist"):
        backend.get_many({"a": "/datasets/demo", "b": "/datasets/missing"})


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
