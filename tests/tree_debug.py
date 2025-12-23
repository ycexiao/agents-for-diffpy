# DAG
from PDFAdapter import PDFAdapter
from FitRunner import FitRunner
from FitDAG import FitDAG
from pathlib import Path
from diffpy.srfit.fitbase import FitResults
import time
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json

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
dag.from_str("a->scale->Uiso_0->delta2->all")
dag.load_inputs([inputs])
# print(dag.nodes(data=True))

# A simple test for FitRunner
start_time = time.time()
runner = FitRunner()
dag, adapters_dict, wait_list = runner.run_workflow(
    dag, PDFAdapter, [inputs], [payload]
)
end_time = time.time()
print(f"FitRunner took {end_time - start_time} seconds.")
print(list(adapters_dict.values())[0]._residual_scalar())
# have to separeate the visualization part to avoid multiprocessing issues
# with open("debug_dag.pkl", "wb") as f:
#     pickle.dump(dag.clean_copy(with_payload=True, with_besides_str=True), f)

copied_dag = dag.clean_copy(with_payload=True, with_besides_str=False)
result_nodes = []
for node_id, node_content in copied_dag.nodes(data=True):
    result_nodes.append((node_id, node_content))

with open("tree_debug.json", "w") as f:
    json.dump(dag.to_json(), f, indent=4)


# adapter = adapters_dict[list(adapters_dict.keys())[0]]
# # Get an array of the r-values we fitted over.
# r = adapter._recipe.pdfcontribution.profile.x
# # Get an array of the observed PDF.
# g = adapter._recipe.pdfcontribution.profile.y
# # Get an array of the calculated PDF.
# gcalc = adapter._recipe.pdfcontribution._eq()
# # Make an array of identical shape as g which is offset from g.
# diffzero = -0.65 * max(g) * np.ones_like(g)
# # Calculate the residual (difference) array and offset it vertically.
# diff = g - gcalc + diffzero
# # Change some style detials of the plot
# mpl.rcParams.update(mpl.rcParamsDefault)
# # Create a figure and an axis on which to plot
# fig, ax1 = plt.subplots(1, 1)
# # Plot the difference offset line
# ax1.plot(r, diffzero, lw=1.0, ls="--", c="black")
# # Plot the measured data
# ax1.plot(
#     r,
#     g,
#     ls="None",
#     marker="o",
#     ms=5,
#     mew=0.2,
#     mfc="None",
#     label="G(r) Data",
# )
# # Plot the calculated data
# ax1.plot(r, gcalc, lw=1.3, label="G(r) Fit")
# # Plot the difference
# ax1.plot(r, diff, lw=1.2, label="G(r) diff")
# # Let's label the axes!
# ax1.set_xlabel(r"r ($\mathrm{\AA}$)")
# ax1.set_ylabel(r"G ($\mathrm{\AA}$$^{-2}$)")
# # Tune the tick markers. We are picky!
# ax1.tick_params(axis="both", which="major", top=True, right=True)
# # Set the boundaries on the x-axis
# ax1.set_xlim(r[0], r[-1])
# # We definitely want a legend!
# ax1.legend()
# # Let's use a tight layout. Shun wasted space!
# plt.tight_layout()
# # This is going to make a figure pop up on screen for you to view.
# # The script will pause until you close the figure!
# plt.show()
