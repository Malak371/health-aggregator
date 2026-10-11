from pathlib import Path

import pandas as pd
import pytest

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
DEMO_CSV = SAMPLES / "demo_daily_metrics.csv"
APPLE_HEALTH_XML = SAMPLES / "apple_health_export_sample.xml"


@pytest.fixture(scope="session")
def demo_frame() -> pd.DataFrame:
    """The bundled synthetic dataset, indexed by date."""
    return pd.read_csv(DEMO_CSV, parse_dates=["date"]).set_index("date")
