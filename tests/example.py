from pathlib import Path
from PDFAdapter import PDFAdapter
from FitDAG import FitDAG
from FitRunner import FitRunner
import time
import re

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

# show initial residual
adapter = PDFAdapter()
adapter.load_inputs(inputs)
print(adapter._residual_scalar())

dag = FitDAG()
dag.from_str("a->scale->qdamp->Uiso_0->delta2->all")
dag.load_inputs([inputs])

start_time = time.time()
runner = FitRunner()
dag, adapters_dict = runner.run_workflow(dag, PDFAdapter, [inputs], [payload])
end_time = time.time()
print(f"FitRunner took {end_time - start_time} seconds.")
