import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app.config import Settings
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
    """An API client on a clean database seeded with the demo data.

    Uses a temporary SQLite file unless TEST_DATABASE_URL points elsewhere
    (CI also runs the suite against PostgreSQL).
    """
    url = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'test.db'}"
    engine = make_engine(Settings(database_url=url).database_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    app = create_app(sessionmaker(bind=engine, expire_on_commit=False))
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def upload_slug(client: TestClient) -> str:
    """An empty visitor dataset."""
    slug: str = client.post("/api/v1/datasets").json()["slug"]
    return slug
