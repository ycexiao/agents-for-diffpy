from pathlib import Path
from PDFexperiment import PDFExperiment
from diffpy.structure import loadStructure
from matplotlib import pyplot as plt


def test_fetch_mp_structure():
    # data_path = Path().cwd() / "mp_docs.pkl"
    # with open(data_path, "rb") as f:
    #     docs = pickle.load(f)
    # doc = docs[0]
    # experiment = PDFExperiment(doc["structure"])
    data_path = Path().cwd() / "data" / "Ni.cif"
    structure = loadStructure(str(data_path))
    experiment = PDFExperiment(structure)
    fig, ax = plt.subplots()
    r, G = experiment.generate()
    ax.plot(r, G, label="Original Structure")
    experiment.apply_lat_perturbation(magnitude=0.1, probability=1.0)
    experiment.apply_adp_perturbation(magnitude=0.01, probability=1.0)
    r, G = experiment.generate()
    ax.plot(
        r,
        G,
        label="Perturbed Structure",
        linestyle="--",
    )
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("G(r)")
    ax.legend()
    # plt.show()
