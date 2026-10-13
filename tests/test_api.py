import datetime as dt

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.models import Dataset
from tests.conftest import APPLE_HEALTH_XML, DEMO_CSV

DEMO = "/api/v1/datasets/demo"


def csv_upload(content: bytes) -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("data.csv", content, "text/csv")}


# --- System and demo data ------------------------------------------------------


def test_health_check(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_demo_dataset_is_seeded_on_startup(client: TestClient) -> None:
    body = client.get(DEMO).json()

    assert body["days"] == 180
    assert body["source"] == "synthetic"
    assert (body["first_date"], body["last_date"]) == ("2026-04-01", "2026-09-27")


def test_metrics_can_be_filtered_by_date(client: TestClient) -> None:
    everything = client.get(f"{DEMO}/metrics").json()
    week = client.get(f"{DEMO}/metrics", params={"start": "2026-05-01", "end": "2026-05-07"})

    assert len(everything) == 180
    assert [row["date"] for row in week.json()] == [f"2026-05-0{day}" for day in range(1, 8)]
    assert set(week.json()[0]) >= {"sleep_hours", "training_minutes", "study_hours"}


def test_metrics_reject_an_inverted_date_range(client: TestClient) -> None:
    response = client.get(f"{DEMO}/metrics", params={"start": "2026-06-01", "end": "2026-05-01"})

    assert response.status_code == 422
    assert "must not be after" in response.json()["detail"]["message"]


def test_derived_metrics_cover_every_day(client: TestClient) -> None:
    days = client.get(f"{DEMO}/metrics/derived").json()

    assert len(days) == 180
    assert days[0]["acwr"] is None  # not enough history on day one
    assert days[-1]["acwr"] is not None
    assert 0 <= days[-1]["readiness"] <= 100
    assert 0 <= days[-1]["burnout_risk"] <= 100


def test_insights_for_the_demo_dataset(client: TestClient) -> None:
    body = client.get(f"{DEMO}/insights").json()

    assert body["days"] == 180
    assert {c["y"] for c in body["correlations"]} >= {"mood_score", "sugar_grams"}
    assert body["load"]["acwr_zone"] in {"undertraining", "optimal", "caution", "high"}
    assert body["recovery"]["direction"] in {"improving", "stable", "declining"}
    assert body["burnout"]["level"] in {"low", "moderate", "high"}
    assert len(body["burnout"]["components"]) == 5
    assert body["recommendations"]
    assert all(item["reason"] for item in body["recommendations"])
    assert "not medical advice" in body["disclaimer"]
    dates = [a["date"] for a in body["anomalies"]]
    assert dates == sorted(dates, reverse=True)


def test_prediction_for_the_demo_dataset(client: TestClient) -> None:
    body = client.get(f"{DEMO}/prediction").json()

    assert body["available"] is True
    assert body["prediction"]["predicts"] == "2026-09-28"
    assert 0 <= body["prediction"]["probability_low_readiness"] <= 1
    assert body["evaluation"]["test_days"] == 40
    assert body["feature_effects"][0]["label"]
    assert "not medical advice" in body["disclaimer"]


# --- Dataset lifecycle ---------------------------------------------------------


def test_unknown_dataset_returns_a_clear_404(client: TestClient) -> None:
    response = client.get("/api/v1/datasets/nope/insights")

    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]["message"]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("DELETE", DEMO),
        ("POST", f"{DEMO}/import/csv"),
        ("POST", f"{DEMO}/import/apple-health"),
    ],
)
def test_demo_dataset_is_read_only(client: TestClient, method: str, path: str) -> None:
    response = client.request(method, path, files=csv_upload(b"date,steps\n2026-01-01,1\n"))

    assert response.status_code == 403
    assert client.get(DEMO).json()["days"] == 180


def test_created_datasets_get_unguessable_slugs(client: TestClient) -> None:
    first = client.post("/api/v1/datasets", json={"name": "Mine"})
    second = client.post("/api/v1/datasets")

    assert first.status_code == 201
    assert first.json()["name"] == "Mine"
    assert first.json()["days"] == 0
    assert len(first.json()["slug"]) >= 20
    assert first.json()["slug"] != second.json()["slug"]


def test_empty_dataset_has_no_insights_yet(client: TestClient, upload_slug: str) -> None:
    response = client.get(f"/api/v1/datasets/{upload_slug}/insights")

    assert response.status_code == 409
    assert "no data yet" in response.json()["detail"]["message"]


def test_dataset_can_be_deleted(client: TestClient, upload_slug: str) -> None:
    assert client.delete(f"/api/v1/datasets/{upload_slug}").status_code == 204
    assert client.get(f"/api/v1/datasets/{upload_slug}").status_code == 404


def set_last_active(client: TestClient, slug: str, days_ago: int) -> None:
    with client.app.state.session_factory() as session:  # type: ignore[attr-defined]
        dataset = session.query(Dataset).filter_by(slug=slug).one()
        dataset.last_active_at = dt.datetime.now(dt.UTC) - dt.timedelta(days=days_ago)
        session.commit()


def days_since_active(client: TestClient, slug: str) -> float:
    with client.app.state.session_factory() as session:  # type: ignore[attr-defined]
        last_active = session.query(Dataset).filter_by(slug=slug).one().last_active_at
    if last_active.tzinfo is None:  # SQLite returns naive UTC datetimes
        last_active = last_active.replace(tzinfo=dt.UTC)
    return (dt.datetime.now(dt.UTC) - last_active).total_seconds() / 86400


def test_datasets_unused_for_the_retention_period_are_purged(
    client: TestClient, upload_slug: str
) -> None:
    set_last_active(client, upload_slug, days_ago=91)

    client.post("/api/v1/datasets")

    assert client.get(f"/api/v1/datasets/{upload_slug}").status_code == 404


def test_datasets_used_within_the_retention_period_are_kept(
    client: TestClient, upload_slug: str
) -> None:
    set_last_active(client, upload_slug, days_ago=89)

    client.post("/api/v1/datasets")

    assert client.get(f"/api/v1/datasets/{upload_slug}").status_code == 200


def test_opening_a_dataset_counts_as_activity(client: TestClient, upload_slug: str) -> None:
    set_last_active(client, upload_slug, days_ago=60)

    client.get(f"/api/v1/datasets/{upload_slug}")

    assert days_since_active(client, upload_slug) < 0.01


def test_activity_is_recorded_at_most_once_a_day(client: TestClient, upload_slug: str) -> None:
    set_last_active(client, upload_slug, days_ago=0)
    before = days_since_active(client, upload_slug)

    client.get(f"/api/v1/datasets/{upload_slug}")

    assert days_since_active(client, upload_slug) >= before


def test_a_check_in_creates_a_day_and_later_ones_merge_into_it(
    client: TestClient, upload_slug: str
) -> None:
    base = f"/api/v1/datasets/{upload_slug}"
    set_last_active(client, upload_slug, days_ago=30)

    morning = client.post(
        f"{base}/records", json={"records": [{"date": "2026-10-13", "sleep_hours": 7.5}]}
    )
    evening = client.post(
        f"{base}/records",
        json={"records": [{"date": "2026-10-13", "mood_score": 4, "study_hours": 3}]},
    )

    assert (morning.json()["days_added"], evening.json()["days_updated"]) == (1, 1)
    day = client.get(f"{base}/metrics").json()[0]
    assert (day["sleep_hours"], day["mood_score"], day["study_hours"]) == (7.5, 4, 3)
    assert days_since_active(client, upload_slug) < 0.01


def test_a_single_logged_day_already_has_insights(client: TestClient, upload_slug: str) -> None:
    base = f"/api/v1/datasets/{upload_slug}"
    client.post(
        f"{base}/records",
        json={"records": [{"date": "2026-10-13", "sleep_hours": 7, "mood_score": 4}]},
    )

    insights = client.get(f"{base}/insights").json()
    derived = client.get(f"{base}/metrics/derived").json()

    assert insights["days"] == 1
    assert insights["recovery"]["readiness_today"] is not None
    assert "Keep logging" in insights["recommendations"][0]["message"]
    assert derived[0]["acwr"] is None
    assert client.get(f"{base}/prediction").json()["available"] is False


def test_upload_capacity_is_enforced(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "max_upload_datasets", 1)

    assert client.post("/api/v1/datasets").status_code == 201
    assert client.post("/api/v1/datasets").status_code == 429


# --- Ingestion -----------------------------------------------------------------


def test_csv_import_then_insights(client: TestClient, upload_slug: str) -> None:
    base = f"/api/v1/datasets/{upload_slug}"

    result = client.post(f"{base}/import/csv", files=csv_upload(DEMO_CSV.read_bytes()))

    assert result.status_code == 200
    assert result.json() == {
        "dataset": upload_slug,
        "days_received": 180,
        "days_added": 180,
        "days_updated": 0,
        "warnings": [],
    }
    assert client.get(f"{base}/insights").json()["days"] == 180


def test_reimporting_updates_instead_of_duplicating(client: TestClient, upload_slug: str) -> None:
    base = f"/api/v1/datasets/{upload_slug}"
    client.post(f"{base}/import/csv", files=csv_upload(b"date,steps\n2026-01-01,5000\n"))

    again = client.post(
        f"{base}/import/csv", files=csv_upload(b"date,mood_score\n2026-01-01,4\n2026-01-02,3\n")
    )

    assert (again.json()["days_added"], again.json()["days_updated"]) == (1, 1)
    first_day = client.get(f"{base}/metrics").json()[0]
    assert first_day["steps"] == 5000  # kept from the first import
    assert first_day["mood_score"] == 4  # merged from the second


def test_invalid_csv_is_rejected_with_row_level_errors(
    client: TestClient, upload_slug: str
) -> None:
    base = f"/api/v1/datasets/{upload_slug}"
    content = b"date,sleep_hours\n2026-01-01,7\n2026-01-02,40\n"

    response = client.post(f"{base}/import/csv", files=csv_upload(content))

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "1 row(s) failed validation."
    assert detail["errors"][0].startswith("Line 3: sleep_hours")
    assert client.get(f"{base}").json()["days"] == 0  # nothing was stored


def test_apple_health_import(client: TestClient, upload_slug: str) -> None:
    base = f"/api/v1/datasets/{upload_slug}"

    result = client.post(
        f"{base}/import/apple-health",
        files={"file": ("export.xml", APPLE_HEALTH_XML.read_bytes(), "application/xml")},
    )

    assert result.status_code == 200
    assert result.json()["days_added"] == 150
    assert "no mood" in result.json()["warnings"][0]
    insights = client.get(f"{base}/insights").json()
    assert insights["load"]["acwr"] is not None
    assert insights["recovery"]["readiness_today"] is not None
    assert client.get(f"{base}/prediction").json()["available"] is True


def test_a_csv_sent_to_the_apple_health_endpoint_is_rejected(
    client: TestClient, upload_slug: str
) -> None:
    response = client.post(
        f"/api/v1/datasets/{upload_slug}/import/apple-health",
        files=csv_upload(DEMO_CSV.read_bytes()),
    )

    assert response.status_code == 422
    assert "not valid Apple Health XML" in response.json()["detail"]["message"]


def test_oversized_and_empty_uploads_are_rejected(
    client: TestClient, upload_slug: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"/api/v1/datasets/{upload_slug}/import/csv"
    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)

    assert client.post(url, files=csv_upload(b"date,steps\n2026-01-01,1\n")).status_code == 413
    assert client.post(url, files=csv_upload(b"")).status_code == 422


def test_json_records_are_validated_with_readable_messages(
    client: TestClient, upload_slug: str
) -> None:
    url = f"/api/v1/datasets/{upload_slug}/records"

    ok = client.post(url, json={"records": [{"date": "2026-01-01", "sleep_hours": 7.5}]})
    bad = client.post(url, json={"records": [{"date": "2026-01-02", "mood_score": 9}]})
    duplicate = client.post(
        url,
        json={"records": [{"date": "2026-01-03", "steps": 1}, {"date": "2026-01-03", "steps": 2}]},
    )

    assert ok.json()["days_added"] == 1
    assert bad.status_code == 422
    assert bad.json()["detail"]["errors"] == [
        "body.records.0.mood_score: Input should be less than or equal to 5"
    ]
    assert duplicate.status_code == 422


def test_short_datasets_get_insights_but_no_prediction(
    client: TestClient, upload_slug: str
) -> None:
    base = f"/api/v1/datasets/{upload_slug}"
    rows = "".join(f"2026-01-{day:02d},{6 + day % 3},{58 + day % 4}\n" for day in range(1, 31))
    client.post(
        f"{base}/import/csv",
        files=csv_upload(("date,sleep_hours,resting_heart_rate\n" + rows).encode()),
    )

    insights = client.get(f"{base}/insights")
    prediction = client.get(f"{base}/prediction").json()

    assert insights.status_code == 200
    assert prediction["available"] is False
    assert "about 81 days" in prediction["reason"]
