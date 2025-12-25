from FitDAG import FitDAG
from PDFAdapter import PDFAdapter
import matplotlib.pyplot as plt
import numpy
from pathlib import Path
import re

dag = FitDAG()
dag.from_workflow("example_dag.pkl")
results_node_name = ["all", "all_1", "all_2", "all_3", "all_4"]

lat_s = []
for name in results_node_name:
    node_id, node_contetn = dag.get_node_by_name(name)[0]
    lat_s.append(node_contetn["payload"]["a"])
    # input_source_node_id = dag.get_input_source(node_id)
    # adapter = PDFAdapter()
    # adapter.load_inputs(dag.nodes[input_source_node_id]["inputs"])
    # adapter.apply_payload(dag.nodes[node_id]["payload"])
    # # adapter.plot()

data_dir_path = Path().cwd() / "data" / "sequential_fit"
files = [file for file in data_dir_path.glob("*.gr")]
temperatures = [
    int(re.search(r"(\d+)K\.gr", file.name).group(1)) for file in files
]
sorted_items = [
    (temp, file) for temp, file in sorted(zip(temperatures, files))
]
files = [file for _, file in sorted_items]
temperatures = [temp for temp, _ in sorted_items]


plt.plot(range(len(lat_s)), lat_s, marker="o")
plt.xticks(
    range(len(lat_s)),
    [f"{temperatures[i]}K" for i in range(len(lat_s))],
    rotation=45,
)
plt.show()
