import pytest
from PDFAdapter import PDFAdapter
from pathlib import Path
from scipy.optimize import least_squares
from diffpy.srfit.fitbase import FitResults
from diffpy.structure.parsers import getParser


def test_adapter_workflow():
    profile_path = Path().cwd() / "data" / "Ni.gr"
    structure_path = Path().cwd() / "data" / "Ni.cif"
    adapter = PDFAdapter(
        profile_path=str(profile_path),
        structure_path=str(structure_path),
        Qmax=25.0,
        Qmin=0.5,
    )
    # Set initial parameter values
    adapter.free_parameters(["all"])
    adapter.apply_pv_dict(
        {
            "scale": 0.4,
            "a": 3.52,
            "Uiso_0": 0.005,
            "delta2": 2.0,
            "qdamp": 0.04,
            "qbroad": 0.02,
        }
    )
    adapter.fix_parameters(["all"])
    # C3: perform fitting steps
    steps = ["a", "scale", "Uiso_0", "delta2", "qdamp", "qbroad"]
    # result = least_squares(
    #     adapter._recipe.residual,
    #     adapter._recipe.values,
    #     x_scale="jac",
    # )
    for step in steps:
        adapter.free_parameters([step])
        result = least_squares(
            adapter.residual,
            adapter.initial_values,
            x_scale="jac",
        )
        print(f"Fitting step: {step}, success: {result.success}")
