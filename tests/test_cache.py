import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.services import insights
from app.services.cache import ResultCache, frame_fingerprint
from tests.helpers import make_frame


def test_fingerprint_changes_with_values_and_dates() -> None:
    frame = make_frame(sleep_hours=[7.0, 8.0])
    edited = frame.copy()
    edited.iloc[1, 0] = 8.5
    shifted = make_frame(start="2026-02-01", sleep_hours=[7.0, 8.0])

    assert frame_fingerprint(frame) == frame_fingerprint(frame.copy())
    assert frame_fingerprint(frame) != frame_fingerprint(edited)
    assert frame_fingerprint(frame) != frame_fingerprint(shifted)


def test_cache_computes_once_per_key_and_evicts_the_oldest() -> None:
    cache: ResultCache[int] = ResultCache(max_entries=2)
    calls: list[str] = []

    def compute(key: str) -> int:
        calls.append(key)
        return len(calls)

    assert cache.get_or_compute("a", lambda: compute("a")) == 1
    assert cache.get_or_compute("a", lambda: compute("a")) == 1
    cache.get_or_compute("b", lambda: compute("b"))
    cache.get_or_compute("a", lambda: compute("a"))  # "a" is now the most recent
    cache.get_or_compute("c", lambda: compute("c"))  # evicts "b"
    cache.get_or_compute("b", lambda: compute("b"))

    assert calls == ["a", "b", "c", "b"]
    assert len(cache) == 2


@pytest.fixture
def training_runs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count how many times the model is actually trained."""
    insights.prediction_cache.clear()
    runs: list[int] = []
    original = insights.train_and_explain

    def counting(frame: pd.DataFrame) -> object:
        runs.append(len(frame))
        return original(frame)

    monkeypatch.setattr(insights, "train_and_explain", counting)
    return runs


def test_prediction_is_trained_once_for_unchanged_data(
    training_runs: list[int], client: TestClient
) -> None:
    first = client.get("/api/v1/datasets/demo/prediction").json()
    second = client.get("/api/v1/datasets/demo/prediction").json()

    assert first == second
    assert training_runs == [180]


def test_prediction_is_recomputed_after_the_data_changes(
    training_runs: list[int], client: TestClient, upload_slug: str
) -> None:
    base = f"/api/v1/datasets/{upload_slug}"
    rows = "".join(f"2026-01-{day:02d},{6 + day % 3}\n" for day in range(1, 31))
    client.post(f"{base}/import/csv", files={"file": ("a.csv", f"date,sleep_hours\n{rows}")})
    client.get(f"{base}/prediction")
    client.get(f"{base}/prediction")

    client.post(f"{base}/import/csv", files={"file": ("b.csv", "date,sleep_hours\n2026-01-31,7\n")})
    client.get(f"{base}/prediction")

    uploads = [days for days in training_runs if days != 180]
    assert uploads == [30, 31]
