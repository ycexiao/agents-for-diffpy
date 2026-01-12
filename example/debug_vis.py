from FitDAG import FitDAG
from PDFAdapter import PDFAdapter
from pathlib import Path
from diffpy.srfit.fitbase import FitResults
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

dag = FitDAG()
dag.from_workflow("forest_debug.pkl")
# dag.render("forest_debug.html")
order1 = ["a", "scale", "qdamp", "Uiso_0", "delta2", "all"]
order2 = [item + "_1" for item in order1]
order = order1 + order2

for name in order:
    node_id, node_content = dag.get_node_by_name(name)[0]
    payload = node_content["payload"]
    print(
        f"name: {name} ".ljust(20)
        + f"a: {payload['a']:.8f} ".ljust(15)
        + f"scale: {payload['scale']:.8f} ".ljust(15)
        + f"qdamp: {payload['qdamp']:.8f} ".ljust(15)
        + f"Uiso_0: {payload['Uiso_0']:.8f} ".ljust(15)
        + f"delta2: {payload['delta2']:.4f} ".ljust(15)
    )


adaper = PDFAdapter()
data_dir_path = Path().cwd() / "data" / "sequential_fit"
files = [file for file in data_dir_path.glob("*.gr")]
order = [int(re.search(r"(\d+)K\.gr", file.name).group(1)) for file in files]
files = [file for _, file in sorted(zip(order, files))]
structure_path = Path().cwd() / "data" / "Ni.cif"
input = {
    "profile_string": files[0].read_text(),
    "structure_string": structure_path.read_text(),
    "xmin": 1.5,
    "xmax": 50,
    "dx": 0.01,
    "qmax": 25.0,
    "qmin": 0.1,
}
adaper.load_inputs(input)
_, end_node_1 = dag.get_node_by_name("all_1")[0]
adaper.apply_payload(end_node_1["payload"])
print(end_node_1["payload"])
print(f"Residual scalar: {adaper._residual_scalar()}")

# Get an array of the r-values we fitted over.
r = adaper._recipe.pdfcontribution.profile.x
# Get an array of the observed PDF.
g = adaper._recipe.pdfcontribution.profile.y
# Get an array of the calculated PDF.
gcalc = adaper._recipe.pdfcontribution._eq()
# Make an array of identical shape as g which is offset from g.
diffzero = -0.65 * max(g) * np.ones_like(g)
# Calculate the residual (difference) array and offset it vertically.
diff = g - gcalc + diffzero
# Change some style detials of the plot
mpl.rcParams.update(mpl.rcParamsDefault)
# Create a figure and an axis on which to plot
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
