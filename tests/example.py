from pathlib import Path
from PDFAdapter import PDFAdapter
from FitDAG import FitDAG
from FitRunner import FitRunner
import time
import re
import pickle
import networkx as nx
import threading
from FitAnalyzer import FitAnalyzer

data_dir_path = Path().cwd() / "data" / "sequential_fit"
files = [file for file in data_dir_path.glob("*.gr")]
order = [int(re.search(r"(\d+)K\.gr", file.name).group(1)) for file in files]
files = [file for _, file in sorted(zip(order, files))]

structure_path = Path().cwd() / "data" / "Ni.cif"

# Configure the first file
inputs = {
    "profile_string": files[0].read_text(),
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


template_dag = FitDAG()
template_dag.from_str("a->scale->qdamp->Uiso_0->delta2->all")
dag = template_dag.clean_copy(
    with_payload=False,
    with_same_id=False,
    instance_type="FitDAG",
)

files = files[:2]
for i in range(1, len(files)):
    another_dag = template_dag.clean_copy(
        with_payload=False,
        with_same_id=False,
        instance_type="FitDAG",
    )
    if i > 1:
        parent_node_id, _ = dag.get_node_by_name(f"all_{i-1}")[0]
    else:
        parent_node_id, _ = dag.get_node_by_name("all")[0]
    child_node_id, _ = another_dag.get_node_by_name("start")[0]
    child_node_id = list(another_dag.successors(child_node_id))[0]
    dag.merge_dag(another_dag, parent_node_id, child_node_id)
    print(f"Merged DAG for file {i+1}")

all_inputs = [
    {
        "profile_string": files[i].read_text(),
        "structure_string": structure_path.read_text(),
        "xmin": 1.5,
        "xmax": 50,
        "dx": 0.01,
        "qmax": 25.0,
        "qmin": 0.1,
    }
    for i in range(len(files))
]
all_payloads = [None for _ in range(len(files))]
all_payloads[0] = payload

lock = threading.Lock()
runner = FitRunner(lock=lock)
analyzer = FitAnalyzer(lock=lock)
# analyzer.watch(
#     dag,
#     "all",
#     "a",
# )
analyzer.track(
    dag,
    dag.get_node_by_name("start")[0][0],
    dag.get_node_by_name(f"all_{len(files)-1}")[0][0],
    "ycalc",
    "replace",
)
runner.run_workflow(dag, PDFAdapter, all_inputs, all_payloads, lock)
analyzer.on(dag, life_time=300)  # must after runner.run_workflow
runner.thread.join()

# print(dag.watch("all", ["a"]))
# with open("example_dag.pkl", "wb") as f:
#     pickle.dump(
#         dag.clean_copy(
#             with_payload=True,
#             with_same_id=True,
#             instance_type="FitDAG",
#         ),
#         f,
#     )
