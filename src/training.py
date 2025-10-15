import gymnasium as gym
from environments import pdf_envs  # noqa: F401
from pathlib import Path

data_path = Path().cwd() / "data"
env_kwargs = {
    "structure": str(data_path / "Ni.cif"),
    "profile": str(data_path / "Ni.gr"),
    "configurations": {
        "spacegroup": "Fm-3m",
        "instrumental_params": {
            "qdamp": 0,
            "qbroad": 0,
        },
        "profile_calculation_params": {"xmin": 0, "xmax": 0, "dx": 0},
    },
}
env = gym.make(
    "pdf-single-phase",
    structure=str(data_path / "Ni.cif"),
    profile=str(data_path / "Ni.gr"),
    configurations={
        "spacegroup": "Fm-3m",
        "fixed_params": {
            "profile": {
                "xmin": 0.5,
                "xmax": 20.0,
                "dx": 0.01,
            },
            "generator": {"qdamp": 0.03, "qbroad": 0.01},
        },
    },
)
