import gymnasium as gym
from environments import pdf_envs  # noqa: F401
from pathlib import Path

data_path = Path().cwd() / "data"
env = gym.make(
    "pdf-single-phase",
    structure=data_path / "Ni.cif",
    profile=data_path / "Ni.gr",
    configurations={
        "spacegroup": "Fm-3m",
    },
    fixed_params=["xmin", "xmax", "dx"],
    initial_values={
        "xmin": 1.5,
        "xmax": 50.0,
        "dx": 0.01,
        "qdamp": 0.03,
        "qbroad": 0.01,
        "scale": 0.4,
        "Uiso_0": 0.005,
        "delta2": 2,
    },
)
