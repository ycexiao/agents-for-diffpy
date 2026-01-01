from PDFexperiment import PDFexperiment
from pathlib import Path
from matplotlib import pyplot as plt

experiment = PDFexperiment()
experiment.load_inputs(structure_txt=Path("data/Ni.cif").read_text())
for pname, param in experiment.parameters.items():
    print(f"{pname}: {param.value}")

r, ycalc = experiment.generate()
plt.plot(r, ycalc)
plt.show()
