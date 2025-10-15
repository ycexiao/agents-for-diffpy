import sys
from pathlib import Path

import pytest


# Import by path. Do need to re-install the package after every change.
SRC_ROOT = Path(__file__).parents[1].resolve() / "src"
if SRC_ROOT not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


environment_modules_path = SRC_ROOT / "environments"
if environment_modules_path not in sys.path:
    sys.path.insert(0, str(environment_modules_path))

TEST_ROOT = Path(__file__).parents[0].resolve()


@pytest.fixture()
def structure_and_profile_path():
    data_path = TEST_ROOT / "data"
    out = {}
    out["Ni"] = {
        "structure": str(data_path / "Ni.cif"),
        "profile": str(data_path / "Ni.gr"),
    }
    return out
