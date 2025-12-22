# DAG
from PDFAdapter import PDFAdapter
from FitRunner import FitRunner
from FitDAG import FitDAG
from pathlib import Path
from diffpy.srfit.fitbase import FitResults
import time
import pickle

# A simple test for PDFAdater
profile_path = Path().cwd() / "data" / "Ni.gr"
structure_path = Path().cwd() / "data" / "Ni.cif"
adapter = PDFAdapter()
inputs = {
    "profile_string": profile_path.read_text(),
    "structure_string": structure_path.read_text(),
    "xmin": 1.5,
    "xmax": 50,
    "dx": 0.01,
    "qmax": 25.0,
    "qmin": 0.1,
}
adapter.load_inputs(inputs)
# print(adapter.get_payload())
payload = {
    "scale": 0.4,
    "a": 3.52,
    "Uiso_0": 0.005,
    "delta2": 2.0,
    "qdamp": 0.04,
    "qbroad": 0.02,
}
adapter.apply_payload(payload)
# print(adapter.get_payload())

# A simple test for FitDAG
dag = FitDAG()
dag.from_str("a->scale->qdamp->Uiso_0->delta2->all")
dag.load_inputs([inputs])

# A simple test for FitRunner
start_time = time.time()
runner = FitRunner()
dag, adapters_dict = runner.run_workflow(dag, PDFAdapter, [inputs], [payload])
end_time = time.time()
print(f"FitRunner took {end_time - start_time} seconds.")
# have to separeate the visualization part to avoid multiprocessing issues
with open("debug_dag.pkl", "wb") as f:
    pickle.dump(dag.clean_copy(with_payload=True, with_besides_str=True), f)
