from types import SimpleNamespace
from FitDiag import FitDiag

runner = SimpleNamespace(
    allowed_actions={
        "scale": None,
        "alpha": None,
        "a": None,
        "qdamp": None,
        "all": None,
    }
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
    fit_diag = FitDiag(runner)
    fit_diag.from_dict(diag_data)
    assert len(fit_diag.graph.nodes) == 6
    assert len(fit_diag.graph.edges) == 5


def test_from_str_valid():
    diag_str = "scale, alpha, a, qdamp, scale, all"
    fit_diag = FitDiag(runner)
    fit_diag.from_str(diag_str)
    assert len(fit_diag.graph.nodes) == 6
    assert len(fit_diag.graph.edges) == 5
