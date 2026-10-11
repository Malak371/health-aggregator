from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app.db import Base, make_engine
from app.main import create_app

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
DEMO_CSV = SAMPLES / "demo_daily_metrics.csv"
APPLE_HEALTH_XML = SAMPLES / "apple_health_export_sample.xml"


@pytest.fixture(scope="session")
def demo_frame() -> pd.DataFrame:
    """The bundled synthetic dataset, indexed by date."""
    return pd.read_csv(DEMO_CSV, parse_dates=["date"]).set_index("date")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """An API client backed by a fresh SQLite database seeded with the demo data."""
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    app = create_app(sessionmaker(bind=engine, expire_on_commit=False))
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


@pytest.fixture
def upload_slug(client: TestClient) -> str:
    """An empty visitor dataset."""
    slug: str = client.post("/api/v1/datasets").json()["slug"]
    return slug
