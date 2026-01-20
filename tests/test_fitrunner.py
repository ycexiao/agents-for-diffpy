from FitRunner import FitRunner
from PDFAdapter import PDFAdapter
from FitDAG import FitDAG
from pathlib import Path
import re


def test_run_dag():
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
    runner.watch(dag, "a", "all", "a", "append", "payload")
    runner.run_dag(dag, PDFAdapter, inputs, payload)
    assert list(runner.data_for_plot.values())[0]["ydata"].qsize() == 6


def test_run_sequential_dags():
    data_dir_path = Path().cwd() / "data" / "sequential_fit"
    files = [file for file in data_dir_path.glob("*.gr")]
    order = [
        int(re.search(r"(\d+)K\.gr", file.name).group(1)) for file in files
    ]
    files = [file for _, file in sorted(zip(order, files))]
    structure_path = Path().cwd() / "data" / "Ni.cif"

    # Configure the first file
    inputs = [
        {
            "profile_string": files[i].read_text(),
            "structure_string": structure_path.read_text(),
            "xmin": 1.5,
            "xmax": 50,
            "dx": 0.01,
            "qmax": 25.0,
            "qmin": 0.1,
        }
        for i in range(3)
    ]
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
    dags = [dag]
    for _ in range(2):
        new_dag = dag.clean_copy(
            with_payload=False, with_same_id=False, return_type="FitDAG"
        )
        dags.append(new_dag)

    runner = FitRunner()
    runner.watch(dags[0], "a", "all", "a", "append", "payload")
    runner.watch(dags[1], "a", "all", "a", "append", "payload")
    runner.run_sequential_dags(dags, PDFAdapter, inputs, payload)
