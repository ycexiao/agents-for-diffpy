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
        profile_path=str(profile_path), structure_path=str(structure_path)
    )
    # C1: Set initial parameter values
    adapter.update_parameters(
        {
            "scale": 0.4,
            "a": 3.52,
            "Uiso_0": 0.005,
            "delta2": 2.0,
            "qdamp": 0.04,
            "qbroad": 0.02,
        }
    )
    assert (
        adapter.residual_factory_components["recipe"].get("scale").value == 0.4
    )
    # C2: fix, free, and show parameters
    adapter.free_parameters(["scale"])
    assert adapter.residual_factory_components["recipe"].isFree(
        adapter.residual_factory_components["recipe"].get("scale")
    )
    adapter.fix_parameters(["all"])
    adapter.show_parameters()
    # C3: perform fitting steps
    steps = ["a", "scale", "Uiso_0", "delta2", "qdamp", "qbroad"]
    for step in steps:
        adapter.free_parameters([step])
        result = least_squares(
            adapter.residual_factory_components["recipe"].residual,
            adapter.residual_factory_components["recipe"].values,
            x_scale="jac",
        )
        print(f"Fitting step: {step}, success: {result.success}")
