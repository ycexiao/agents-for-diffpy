from FitDAG import FitDAG
from pathlib import Path
from PDFAdapter import PDFAdapter
import time
import re
from FitRunner import FitRunner
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


dag = FitDAG()
dag.from_str("a->scale->qdamp->Uiso_0->delta2->all")

another_dag = dag.clean_copy(
    with_payload=False,
    with_same_id=False,
    instance_type="FitDAG",
)

parent_node_id, _ = dag.get_node_by_name("all")[0]
child_node_id, _ = another_dag.get_node_by_name("start")[0]
child_node_id = list(another_dag.successors(child_node_id))[0]

dag.merge_dag(another_dag, parent_node_id, child_node_id)
# dag.render("forest_debug.html")

for name in dag.all_names:
    node_id = dag.get_node_by_name(name)[0][0]
    input_source_id = dag.get_input_source_node_id(node_id)
    parent_id = dag.get_payload_source_node_id(node_id)
    print(
        f"name: {name} ".ljust(20)
        + f"source: {dag.nodes[input_source_id]['name']}".ljust(30)
        + f"parent: {dag.nodes[parent_id]['name']}".ljust(30)
    )


data_dir_path = Path().cwd() / "data" / "sequential_fit"
files = [file for file in data_dir_path.glob("*.gr")]
order = [int(re.search(r"(\d+)K\.gr", file.name).group(1)) for file in files]
files = [file for _, file in sorted(zip(order, files))]
structure_path = Path().cwd() / "data" / "Ni.cif"

inputs = [
    {
        "profile_string": files[0].read_text(),
        "structure_string": structure_path.read_text(),
        "xmin": 1.5,
        "xmax": 50,
        "dx": 0.01,
        "qmax": 25.0,
        "qmin": 0.1,
    },
    {
        "profile_string": files[1].read_text(),
        "structure_string": structure_path.read_text(),
        "xmin": 1.5,
        "xmax": 50,
        "dx": 0.01,
        "qmax": 25.0,
        "qmin": 0.1,
    },
]
payload = {
    "scale": 0.25,  # 0.26
    "a": 3.52,
    "Uiso_0": 0.0015,  # 0.0016
    "delta2": 0.65,  # 0.64
    "delta1": 0.3,  # 0.3
    "delta2": 0.65,  # 0.64
    "qdamp": 0.030,  # 0.028
    "qbroad": 0.02,  # 0.023
}

start_time = time.time()
runner = FitRunner()
dag = runner.run_workflow(
    dag,
    PDFAdapter,
    inputs,
    [payload, None],
)
end_time = time.time()
print(f"FitRunner took {end_time - start_time} seconds.")


def plot(adapter):
    # Get an array of the r-values we fitted over.
    r = adapter._recipe.pdfcontribution.profile.x
    # Get an array of the observed PDF.
    g = adapter._recipe.pdfcontribution.profile.y
    # Get an array of the calculated PDF.
    gcalc = adapter._recipe.pdfcontribution._eq()
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


names = ["all", "all_1"]
inputs_indx = [0, 1]
# for i, name in enumerate(names):
#     node_content = dag.get_node_by_name(name)[0][1]
#     print(node_content["name"])
#     print(node_content["payload"])
#     payload = node_content["payload"]
#     adapter = PDFAdapter()
#     adapter.load_inputs(inputs[inputs_indx[i]])
#     adapter.apply_payload(payload)
#     adapter._recipe._prepare()
#     print(adapter._residual_scalar())
#     plot(adapter)
