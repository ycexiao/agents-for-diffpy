from FitRunner import FitRunner
from PDFAdapter import PDFAdapter
from FitDAG import FitDAG
from pathlib import Path


def test_runner():
    profile_path = Path("data/Ni.gr")
    structure_path = Path("data/Ni.cif")
    inputs = {
        "profile_string": profile_path.read_text(),
        "structure_string": structure_path.read_text(),
        "xmin": 1.5,
        "xmax": 50,
        "dx": 0.01,
        "qmax": 25.0,
        "qmin": 0.1,
    }
    payload = {
        "scale": 0.4,
        "a": 3.52,
        "Uiso_0": 0.005,
        "delta2": 2.0,
        "qdamp": 0.04,
        "qbroad": 0.02,
    }
    dag = FitDAG()
    dag.from_str("a->scale->qdamp->Uiso_0->delta2->all")
    runner = FitRunner()
    runner.run_workflow(dag, PDFAdapter, inputs, payload)
