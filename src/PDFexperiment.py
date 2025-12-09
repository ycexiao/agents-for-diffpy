from pathlib import Path
from diffpy.structure.parsers import getParser
from diffpy.srreal.pdfcalculator import PDFCalculator
import random


class PDFexperiment:
    def __init__(self, structure_path):
        self.load_inputs(structure_path)

    def load_inputs(
        self, structure_path, qmin=0.5, qmax=25.0, rmin=0.0, rmax=20.0
    ):
        # Placeholder for loading structure
        stru_parser = getParser("cif")
        structure = stru_parser.parse(Path(structure_path).read_text())
        sg = getattr(stru_parser, "spacegroup", None)
        self.structure = structure
        self.spacegroup = sg.short_name if sg else "Unknown"
        self.pdfcalc = PDFCalculator(
            qmin=qmin, qmax=qmax, rmin=rmin, rmax=rmax
        )
        self.pname_to_perturb = {
            "qdamp": {"rng": (0, 0.1), "magnitude": None},
            "qbroad": {"rng": (0, 0.1), "magnitude": None},
            "scale": {"rng": (0.5, 1.5), "magnitude": None},
            "delta2": {"rng": (0, 0.2), "magnitude": None},
            "delta1": {"rng": (0, 0.2), "magnitude": None},
            "U": {"rng": None, "magnitude": (0.5, 1.5)},
            "length": {"rng": None, "magnitude": (0.5, 1.5)},
            "angle": {"rng": None, "magnitude": (0.5, 1.5)},
        }

    def get_parameter(self, pname):
        if pname in ["qdamp", "qbroad", "scale", "delta2", "delta1"]:
            parameter = getattr(self.pdfcalc, pname)
        if pname in ["a", "b", "c", "alpha", "beta", "gamma"]:
            lattice = self.structure.lattice
            parameter = getattr(lattice, pname)
        elif pname.startswith("U"):
            uij_name, atom_index = pname.split("_")
            atom_index = int(atom_index)
            atom = self.structure[atom_index]
            parameter = getattr(atom, uij_name)
        elif (
            pname.startswith("x")
            or pname.startswith("y")
            or pname.startswith("z")
        ):
            coord_name, atom_index = pname.split("_")
            atom_index = int(atom_index)
            atom = self.structure[atom_index]
            parameter = getattr(atom, coord_name)
        else:
            raise ValueError(f"Unknown parameter name: {pname}")
        return parameter

    def perturb_parameter(self, pname, probability, rng, magnitude):
        if random.random() > probability:
            return
        param = self.get_parameter(
            pname,
        )
        if rng is not None:
            param.setValue(
                random.random() * (rng[1] - rng[0]) + rng[0],
            )
        elif magnitude is not None:
            param.setValue(
                param.value
                * (
                    random.random() * (magnitude[1] - magnitude[0])
                    + magnitude[0]
                )
            )

    def perturb_parameter_default(self):
        for pname, settings in self.pname_to_perturb.items():
            self.perturb_parameter(
                pname,
                probability=0.8,
                rng=settings["rng"],
                magnitude=settings["magnitude"],
            )

    def generate(self):
        r, G = self.pdfcalc(self.structure)
        return r, G
