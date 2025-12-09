from pathlib import Path
from PDFAdapter import PDFAdapter
from FitRunner import FitRunner
from FitDAG import FitDAG

profile_path = Path().cwd() / "data" / "Ni.gr"
structure_path = Path().cwd() / "data" / "Ni.cif"
adapter = PDFAdapter(
    profile_path=str(profile_path),
    structure_path=str(structure_path),
    Qmax=25.0,
    Qmin=0.5,
)
# print(adapter.residual_scalar)

# initial values
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
# print(adapter.residual_scalar)

# Define fitting diagram
fit_dag = FitDAG(adapter)
dag_str = "scale->a->qdamp->scale->all"
fit_dag.from_str(dag_str)

# Run FitRunner
fit_runner = FitRunner()
fit_dag = fit_runner.run_workflow(fit_dag, adapter)
# leaf_nodes = [
#     n for n in fit_dag.graph.nodes() if fit_dag.graph.out_degree(n) == 0
# ]
# for node_id in leaf_nodes:
#     parameter_values = fit_dag.graph.nodes[node_id]["parameter_values"]
#     adapter.free_parameters(["all"])
#     adapter.apply_parameter_values_in_slot(parameter_values)

# print(adapter.residual_scalar)
