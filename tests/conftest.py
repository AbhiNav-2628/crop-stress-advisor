import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_advisory_rules, load_settings  # noqa: E402
from src.data.ingestion import RasterInput  # noqa: E402
from src.data.preprocessing import sample_from_raster  # noqa: E402
from src.data.synthetic import synthesize_plot  # noqa: E402

DEMO_MAPPING = {"blue": 1, "green": 2, "red": 3, "red_edge": 4, "nir": 5}


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def advisory_rules():
    return load_advisory_rules()


@pytest.fixture
def make_sample(settings):
    """Build a SpectralSample from a synthetic scenario (or a given cube)."""

    def _make(scenario="healthy", seed=0, cube=None, severity=0.8, typical=True):
        if cube is None:
            cube = synthesize_plot(scenario, np.random.default_rng(seed), severity=severity, typical=typical)
        raster = RasterInput(cube, -9999.0, f"{scenario}.tif", "test")
        return sample_from_raster(raster, DEMO_MAPPING, settings, 1.0, scenario)

    return _make
