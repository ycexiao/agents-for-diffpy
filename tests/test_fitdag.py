from types import SimpleNamespace
from FitDAG import FitDAG
from PDFAdapter import PDFAdapter
from pathlib import Path
import pytest


profile_path = Path().cwd() / "data" / "Ni.gr"
structure_path = Path().cwd() / "data" / "Ni.cif"
adapter = PDFAdapter(
    profile_path=str(profile_path),
    structure_path=str(structure_path),
    Qmax=25.0,
    Qmin=0.5,
)


def test_from_dict_valid():
    diag_data = {
        "nodes": [
            {"id": "1", "value": "scale"},
            {"id": "2", "value": "alpha"},
            {"id": "3", "value": "a"},
            {"id": "4", "value": "qdamp"},
            {"id": "5", "value": "scale"},
            {"id": "6", "value": "all"},
        ],
        "edges": [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6")],
    }
    fit_diag = FitDAG(adapter)
    with pytest.raises(ValueError) as excinfo:
        fit_diag.from_dict(diag_data)
    assert str(excinfo.value) == (
        f"Fitting diagram contains undefined operations: ['alpha']."
    )
    assert len(fit_diag.graph.nodes) == 6
    assert len(fit_diag.graph.edges) == 5


def test_from_str_valid():
    dag_str = "scale->a->qdamp->scale->all"
    fit_dag = FitDAG(adapter)
    fit_dag.from_str(dag_str)
    assert len(fit_dag.graph.nodes) == 5
    assert len(fit_dag.graph.edges) == 4
