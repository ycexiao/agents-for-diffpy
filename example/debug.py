from FitDAG import FitDAG
import networkx as nx
from PDFAdapter import PDFAdapter
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from diffpy.srfit.fitbase import FitResults
from FitRunner import FitRunner

# dag = FitDAG()
# dag.from_json(
#     "example/results/fit_results_Ni_PDF_20250923-004037_8c9cd1_90K.json"
# )
# last_node_id = list(nx.topological_sort(dag))[-1]
# payload = dag.nodes[last_node_id]["payload"]

profile_path = Path(
    "example/data/sequential_fit/Ni_PDF_20250923-004037_8c9cd1_90K.gr"
)
structure_path = Path("example/data/Ni.cif")
inputs = {
    "profile_string": profile_path.read_text(),
    "structure_string": structure_path.read_text(),
    "xmin": 1.5,
    "xmax": 50,
    "dx": 0.01,
    "qmax": 25.0,
    "qmin": 0.1,
    "remove_vars": ["delta1"],
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
dag.from_str("a->scale->Uiso_0->delta2->all")
# dag.from_str("a->scale")
runner = FitRunner()
runner._run_dag(dag, PDFAdapter, inputs, payload=payload)
last_node_id = list(nx.topological_sort(dag))[-1]
# print(dag.nodes[last_node_id]["name"])
# for pname, pvalue in dag.nodes[last_node_id]["payload"].items():
#     print(f"{pname}: {pvalue}")

payload = dag.nodes[last_node_id]["payload"]
print(payload)
adapter = PDFAdapter()
adapter.load_inputs(inputs)
adapter.apply_payload(payload)
adapter._recipe.free("all")
res = FitResults(adapter._recipe)

r = adapter._recipe.pdfcontribution.profile.x
g = adapter._recipe.pdfcontribution.profile.y
gcalc = adapter._recipe.pdfcontribution.profile.ycalc
# Make an array of identical shape as g which is offset from g.
diffzero = -0.65 * max(g) * np.ones_like(g)

# Calculate the residual (difference) array and offset it vertically.
diff = g - gcalc + diffzero

fig, ax1 = plt.subplots(1, 1)

# Plot the difference offset line
ax1.plot(r, diffzero, lw=1.0, ls="--", c="black")

# Plot the measured data
ax1.plot(
    r,
    g,
    ls="None",
    marker="o",
    ms=5,
    mew=0.2,
    mfc="None",
    label="G(r) Data",
)

# Plot the calculated data
ax1.plot(r, gcalc, lw=1.3, label="G(r) Fit")

# Plot the difference
ax1.plot(r, diff, lw=1.2, label="G(r) diff")

# Let's label the axes!
ax1.set_xlabel(r"r ($\mathrm{\AA}$)")
ax1.set_ylabel(r"G ($\mathrm{\AA}$$^{-2}$)")

# Tune the tick markers. We are picky!
ax1.tick_params(axis="both", which="major", top=True, right=True)

# Set the boundaries on the x-axis
ax1.set_xlim(r[0], r[-1])

# We definitely want a legend!
ax1.legend()

# Let's use a tight layout. Shun wasted space!
plt.tight_layout()

# This is going to make a figure pop up on screen for you to view.
# The script will pause until you close the figure!
plt.show()
