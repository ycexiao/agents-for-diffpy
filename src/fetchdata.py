import diffpy
from mp_api.client import MPRester  # noqa: F401
import pickle  # noqa: F401
from diffpy.structure.parsers import getParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pymatgen
import random
from diffpy.srreal.pdfcalculator import PDFCalculator
import numpy as np
from diffpy.structure.symmetryutilities import SymmetryConstraints
from diffpy.srfit.structure.sgconstraints import _constraintMap
from diffpy.srfit.pdf import PDFGenerator
from diffpy.structure.spacegroups import GetSpaceGroup


def get_constraint_info(
    pmg_structure: pymatgen.core.Structure,
    dp_structure: diffpy.structure.Structure,
) -> dict:
    """Get symmetry constraint information from a pymatgen Structure."""
    # Get spacegroup number
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
    lattice = pdfgenerator.phase.getLattice()
    system = sg.crystal_system
    if not system:
        system = "Triclinic"
    system = system.title()
    f = _constraintMap[system]
    f(lattice)
    lat_pars = [
        lattice.a,
        lattice.b,
        lattice.c,
        lattice.alpha,
        lattice.beta,
        lattice.gamma,
    ]
    lat_pars = [p.name for p in lat_pars if not p.const and not p.constrained]
    return sgn, lat_pars, (free_adpnames, fadp)


def fetch_mp_structure():
    """Fetch structure from Materials Project by material ID."""
    with MPRester() as mpr:
        docs = mpr.materials.summary.search(
            elements=[
                "Li",
                "Na",
                "K",
                "Mg",
                "Ca",
                "Ti",
                "Ni",
                "Co",
                "Cu",
                "Zn",
                "Fe",
                "Mn",
                "Al",
                "Si",
                "P",
                "Sn",
                "Sb",
                "Cr",
                "Pb",
                "V",
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
            num_chunks=20,
        )
    return docs


def add_meta_data(docs):
    for doc in docs:
        pmg_struct = doc["structure"]
        dp_struct = getParser("cif").parse(pmg_struct.to(fmt="cif"))
        doc["diffpy_structure"] = dp_struct
        spacegroup_number, lat_names, adp_names = get_constraint_info(
            pmg_struct, dp_struct
        )
        doc["spacegroup_number"] = spacegroup_number
        doc["lat_names"] = lat_names
        doc["adp_names"] = adp_names
    return docs


def do_experiment(doc, lat_perturb_frac=0.02, adp_perturb_frac=0.001):
    pmg_struct = doc["structure"]
    dp_struct = getParser("cif").parse(pmg_struct.to(fmt="cif"))
    lat = dp_struct.lattice
    latpar_names = ["a", "b", "c", "alpha", "beta", "gamma"]
    latpar_value = [
        (
            getattr(lat, name)
            if name not in doc["lat_names"]
            else getattr(lat, name)
            * (1 + random.uniform(-lat_perturb_frac, lat_perturb_frac))
        )
        for name in latpar_names
    ]
    lat.setLatPar(*latpar_value)
    n_free_adp = len(doc["adp_names"][0])
    adppar_value = [
        (np.random.rand() + 0.001) * adp_perturb_frac
        for _ in range(n_free_adp)
    ]
    adppar_dict = dict(zip(doc["adp_names"][0], adppar_value))
    # adppar_names = ['U11','U22','U33','U12','U13','U23']
    for i, atom in enumerate(dp_struct):
        U = doc["adp_names"][1][i]
        for key, value in U:
            if value in adppar_dict:
                U[key] = adppar_dict[value]
        atom._U = [[U[0], U[3], U[4]], [U[3], U[1], U[5]], [U[4], U[5], U[2]]]
    pdfcalc = PDFCalculator()
    pdfcalc.rmin = 0.0
    pdfcalc.rmax = 20.0
    pdfcalc.rstep = 0.01
    pdfcalc.delta1 = abs(random.gauss(0.0, 0.05))  # correlated motion
    pdfcalc.delta2 = abs(random.gauss(0.0, 0.05))  # uncorrelated motion
    pdfcalc.qdamp = abs(random.gauss(0.03, 0.005))
    pdfcalc.qbroad = abs(random.gauss(0.05, 0.01))
    pdfcalc.scale = random.uniform(0.8, 1.2)
    r, G = pdfcalc(dp_struct)
    return r, G
