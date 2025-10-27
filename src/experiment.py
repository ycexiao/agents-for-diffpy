import numpy as np
import pickle
import random

from mp_api.client import MPRester
import pymatgen
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import diffpy
from diffpy.structure import loadStructure, Structure
from diffpy.structure.symmetryutilities import SymmetryConstraints
from diffpy.structure.parsers import getParser
from diffpy.structure.spacegroups import GetSpaceGroup
from diffpy.srfit.pdf import PDFGenerator
from diffpy.srfit.equation.builder import EquationFactory
from diffpy.srreal.pdfcalculator import PDFCalculator

from helper import adpmatrix_from_ij

# The output string from pymatgen is not periodic that can be used
# directly in PDFCalculator.
# Need to get rid of pymatgen


def fetch_mp_structure():
    with MPRester() as mpr:
        docs = mpr.materials.summary.search(
            elements=[
                "Li",
            ],
            energy_above_hull=(0, 0.05),  # stable or nearly stable
            num_elements=(1, 5),  # optional: 1-3 elements
            fields=[
                "nsites",
                "material_id",
                "structure",
                "formula_pretty",
            ],
            chunk_size=1000,
            num_chunks=10,
        )
    return docs


class PDFExperiment:
    def __init__(
        self,
        structure: diffpy.structure.Structure,
        rmin=0.0,
        rmax=20.0,
        rstep=0.01,
    ):
        self.structure = structure
        self.pdfcalc = PDFCalculator()
        self.pdfcalc.rmin = rmin
        self.pdfcalc.rmax = rmax
        self.pdfcalc.rstep = rstep
        self.remember_original_parameters()
        self.set_symmertry_constraints()

    def remember_original_parameters(self):
        self.latpar = {
            "a": self.structure.lattice.a,
            "b": self.structure.lattice.b,
            "c": self.structure.lattice.c,
            "alpha": self.structure.lattice.alpha,
            "beta": self.structure.lattice.beta,
            "gamma": self.structure.lattice.gamma,
        }

    def set_symmertry_constraints(self):
        dp_structure = self.structure
        # Get spacegroup number(How to do it without pymatgen?)
        sgn = SpacegroupAnalyzer(pmg_structure).get_space_group_number()
        sg = GetSpaceGroup(sgn)
        # Get variable ADP parameters
        pdfgenerator = PDFGenerator()
        pdfgenerator.setStructure(dp_structure, periodic=True)
        scatterers = pdfgenerator.phase.getScatterers()
        positions = []
        adpsymbols = ["Uiso", "U11", "U22", "U33", "U12", "U13", "U23"]
        for sidx, scatterer in enumerate(scatterers):
            pars = [scatterer.get(symbol) for symbol in adpsymbols]
            if None in pars:
                continue
            xyz = [scatterer.x, scatterer.y, scatterer.z]
            positions.append([p.value for p in xyz])
        g = SymmetryConstraints(sg, positions)
        free_adpnames = [name for name, val in g.Upars]
        fadp = g.UFormulas(free_adpnames)
        # Get variable lattice parameters
        system = sg.crystal_system
        if not system:
            system = "Triclinic"
        system = system.title()
        flat = self.get_constrained_lat(system)
        free_latnames = [value for value in flat.values()]
        free_latnames = list(
            {value for value in free_latnames if value is not None}
        )
        self.free_latnames = free_latnames
        self.free_adpnames = free_adpnames
        self.fadp = fadp
        self.flat = flat

    def apply_lat_perturbation(self, magnitude=0.02, probability=0.5):
        latpar_names = self.free_latnames
        lat_perturbed = self.latpar.copy()
        for i in range(len(latpar_names)):
            if random.random() > probability:
                continue
            value = self.latpar[latpar_names[i]] * (
                1 + magnitude * np.random.random()
            )
            lat_perturbed[latpar_names[i]] = value
        lat = self.structure.lattice
        lat.setLatPar(**lat_perturbed)

    def apply_adp_perturbation(self, magnitude=0.001, probability=0.5):
        adppar_names = self.free_adpnames
        eqf = EquationFactory()
        for i in range(len(adppar_names)):
            if random.random() > probability:
                continue
            value = (np.random.rand() + 0.001) * magnitude
            eqf.registerConstant(adppar_names[i], value)

        for i, atom in enumerate(self.structure):
            U_dict = self.fadp.copy()[i]
            for key, value in U_dict.items():
                if value == 0:
                    continue
                else:
                    eq = eqf.makeEquation(
                        value,
                    )
                    U_dict[key] = eq.value
            atom.U = adpmatrix_from_ij(U_dict)

    def apply_other_perturbation(
        self,
        kwargs={
            "qdamp": (0.005, 0.5),
            "qbroad": (0.01, 0.5),
            "scale": (0.2, 0.5),
            "delta2": (0.05, 0.5),
            "delta1": (0.05, 0.5),
        },
    ):
        for param, (magnitude, probability) in kwargs.items():
            if random.random() > probability:
                continue
            if param == "qdamp":
                self.pdfcalc.qdamp = abs(random.gauss(0, magnitude))
            elif param == "qbroad":
                self.pdfcalc.qbroad = abs(random.gauss(0, magnitude))
            elif param == "scale":
                self.pdfcalc.scale = random.uniform(
                    1 - magnitude, 1 + magnitude
                )
            elif param == "delta2":
                self.pdfcalc.delta2 = abs(random.gauss(0, magnitude))
            elif param == "delta1":
                self.pdfcalc.delta1 = abs(random.gauss(0, magnitude))

    def generate(self):
        pdfcalc = PDFCalculator(qmax=25.0, qmin=0.5, rmin=0.0, rmax=20.0)
        pdfcalc.qdamp = 0.04
        pdfcalc.qbroad = 0.02
        # r, G = self.pdfcalc(self.structure)
        r, G = pdfcalc(self.structure)
        return r, G

    def reset(self):
        lat = self.structure.lattice
        lat.setLatPar(
            self.latpar["a"],
            self.latpar["b"],
            self.latpar["c"],
            self.latpar["alpha"],
            self.latpar["beta"],
            self.latpar["gamma"],
        )
        for atom in self.structure:
            atom.U = None

    def get_constrained_lat(self, system):
        flat_name = ["a", "b", "c", "alpha", "beta", "gamma"]
        flats = {
            "Cubic": ["a", "a", "a", None, None, None],
            "Tetragonal": ["a", "a", "c", None, None, None],
            "Orthorhombic": ["a", "b", "c", None, None, None],
            "Hexagonal": ["a", "a", "c", None, None, None],
            "Trigonal": ["a", "a", "c", None, None, None],
            "Monoclinic": ["a", "b", "c", None, "beta", None],
            "Triclinic": ["a", "b", "c", "alpha", "beta", "gamma"],
        }
        flat = {name: value for name, value in zip(flat_name, flats[system])}
        return flat


if __name__ == "__main__":
    # docs = fetch_mp_structure()
    # docs = [doc.model_dump() for doc in docs]
    # data_path = "mp_docs.pkl"
    # with open(data_path, "wb") as f:
    #     pickle.dump(docs, f)
    pass
