import gymnasium as gym
from environments import pdf_envs
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
env = gym.make("pdf-single-phase", **env_kwargs)
