import pickle
from pathlib import Path
from experiment import PDFExperiment


def test_fetch_mp_structure():
    data_path = Path().cwd() / "mp_docs.pkl"
    with open(data_path, "rb") as f:
        docs = pickle.load(f)
    doc = docs[0]
    experiment = PDFExperiment(doc["structure"])
    experiment.apply_lat_perturbation(magnitude=0.1, probability=1.0)
    experiment.apply_adp_perturbation(magnitude=0.01, probability=1.0)
    r, G = experiment.generate()
