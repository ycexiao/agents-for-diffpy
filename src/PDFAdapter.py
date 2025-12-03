from typing import Optional, Union
from diffpy.structure import Structure, loadStructure
from diffpy.srfit.pdf import PDFParser
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.srfit.fitbase import (
    FitContribution,
    FitRecipe,
    Profile,
    FitResults,
)
from diffpy.srfit.pdf import PDFGenerator
from collections import OrderedDict
from scipy.optimize import least_squares
from pathlib import Path
import numpy


class PDFAdapter:
        """
        Adapter to expose PDF fitting interface for FitRunner.

        Parameters
        ----------
        profile_path : str
            File path to the PDF profile data.
        structure_path : str
            File path to the structure data.
        kwargs : dict
            Additional keyword arguments for configuration.
            e.g., xmin, xmax, dx for profile calculation range. 
                  qmin, qmax for PDF generator settings.
        
        Attributes
        ----------
        inputs : dict
            A dictionary containing loaded 'structure' and 'profile' objects.
        residual_factory : callable
            A callable that returns the current residual function.
        residual_factory_components : dict
            A dictionary containing components used to build the residual 
            function.
        pname_parameter_dict : dict
            A dictionary mapping parameter names to their corresponding 
            objects in the fitting model.
        parameter_names_group : dict
            A dictionary grouping parameter names by their categories.
        """
    def __init__(self, *inputs):
        self.load_inputs(*inputs)
        self.init_parameters_names()
        self.init_residual_factory()

    def load_inputs(self, profile_path: str, structure_path: str, **kwargs):
        """
        Load profile and structure from given file paths.

        Attributes
        ----------
        inputs: dict
            A dictionary containing loaded 'structure' and 'profile' objects.
        """
        self.inputs = {
            "structure": loadStructure(structure_path, fmt="cif"),
        }
        profile = Profile()
        parser = PDFParser()
        parser.parseFile(profile_path)
        profile.loadParsedData(parser)
        self.inputs["profile"] = profile

    def init_parameters_names(self):
        self.parameter_names_group = {
            "profile_parameters": ["xmin", "xmax", "dx"],
            "generator_instrument_parameters": ["Qmin", "Qmax"],
            "generator_parameters": [
                "qdamp",
                "qbroad",
                "scale",
                "delta1",
                "delta2",
            ],
            "structure_lattice_parameters": [
                "a",
                "b",
                "c",
                "alpha",
                "beta",
                "gamma",
            ],
            "structure_atom_xyz_parameters": (
                [f"x{i}" for i in range(len(self.inputs["structure"]))]
                + [f"y{i}" for i in range(len(self.inputs["structure"]))]
                + [f"z{i}" for i in range(len(self.inputs["structure"]))]
            ),
            "structure_atom_U_parameters": (
                [f"U_11{i}" for i in range(len(self.inputs["structure"]))]
                + [f"U_22{i}" for i in range(len(self.inputs["structure"]))]
                + [f"U_33{i}" for i in range(len(self.inputs["structure"]))]
                + [f"U_12{i}" for i in range(len(self.inputs["structure"]))]
                + [f"U_13{i}" for i in range(len(self.inputs["structure"]))]
                + [f"U_23{i}" for i in range(len(self.inputs["structure"]))]
            ),
        }

    def init_residual_factory(self, **kwargs):
        """
        Initialize the factory to generate residual function.

        Attributes
        ----------
        residual_factory: callable
            A callable that returns the current residual function.
        residual_factory_components: dict
            A dictionary containing components used to build the residual.
        """
        pdfgenerator = PDFGenerator("pdfgen")
        pdfgenerator.setStructure(self.inputs["structure"])
        contribution = FitContribution("pdfcontri")
        xmin = kwargs.get("xmin", numpy.min(self.inputs["profile"]._xobs))
        xmax = kwargs.get("xmax", numpy.max(self.inputs["profile"]._xobs))
        dx = kwargs.get(
            "dx", numpy.mean(numpy.diff(self.inputs["profile"]._xobs))
        )
        self.inputs["profile"].setCalculationRange(xmin=xmin, xmax=xmax, dx=dx)
        contribution.setProfile(self.inputs["profile"])
        contribution.addProfileGenerator(pdfgenerator)
        recipe = FitRecipe()
        recipe.addContribution(contribution)
        self.residual_factory_components = {
            "pdfgenerator": pdfgenerator,
            "contribution": contribution,
            "recipe": recipe,
        }
        self.residual_factory = lambda: self.residual_factory_components[
            "recipe"
        ].residual

        stru_parset = pdfgenerator.phase
        self.pname_parameter_dict = {
            "a": stru_parset.lattice.a,
            "b": stru_parset.lattice.b,
            "c": stru_parset.lattice.c,
            "alpha": stru_parset.lattice.alpha,
            "beta": stru_parset.lattice.beta,
            "gamma": stru_parset.lattice.gamma,
        }
        for i in range(len(stru_parset.atoms)):
            atom_parset = stru_parset.atoms[i]
            self.pname_parameter_dict[f"x_{i}"] = atom_parset._parameters["x"]
            self.pname_parameter_dict[f"y_{i}"] = atom_parset._parameters["y"]
            self.pname_parameter_dict[f"z_{i}"] = atom_parset._parameters["z"]
            self.pname_parameter_dict[f"U11_{i}"] = atom_parset._parameters[
                "U11"
            ]
            self.pname_parameter_dict[f"U22_{i}"] = atom_parset._parameters[
                "U22"
            ]
            self.pname_parameter_dict[f"U33_{i}"] = atom_parset._parameters[
                "U33"
            ]
            self.pname_parameter_dict[f"U12_{i}"] = atom_parset._parameters[
                "U12"
            ]
            self.pname_parameter_dict[f"U13_{i}"] = atom_parset._parameters[
                "U13"
            ]
            self.pname_parameter_dict[f"U23_{i}"] = atom_parset._parameters[
                "U23"
            ]
        self.pname_parameter_dict["qdamp"] = pdfgenerator.qdamp
        self.pname_parameter_dict["qbroad"] = pdfgenerator.qbroad
        self.pname_parameter_dict["scale"] = contribution.scale
        self.pname_parameter_dict["delta1"] = pdfgenerator.delta1
        self.pname_parameter_dict["delta2"] = pdfgenerator.delta2


    def apply_parameter_one(self, pname, pvalue):
        """
        Apply a single parameter value.
        Raise KeyError if the parameter is missing.
        """
        if pname == "qdamp":
            self.residual_factory_components["pdfgenerator"].qdamp = pvalue
        elif pname == "qbroad":
            self.residual_factory_components["pdfgenerator"].qbroad = pvalue
        elif pname == "scale":
            self.residual_factory_components["pdfgenerator"].scale = pvalue
        elif pname == "delta1":
            self.residual_factory_components["pdfgenerator"].delta1 = pvalue
        elif pname == "delta2":
            self.residual_factory_components["pdfgenerator"].delta2 = pvalue
        elif pname in ["a", "b", "c", "alpha", "beta", "gamma"]:
            self.residual_factory_components[
                "pdfgenerator"
            ].structure.lattice.__setattr__(pname, pvalue)
        elif pname.startswith("xyz_"):
            index = int(pname.split("_")[1])
            atom = self.residual_factory_components["pdfgenerator"].structure[
                index
            ]
            atom.xyz = pvalue
        elif pname.startswith("U_"):
            index = int(pname.split("_")[1])
            atom = self.residual_factory_components["pdfgenerator"].structure[
                index
            ]
            atom._U = pvalue

    def apply_parameter_values(self, pv_dict: dict):
        """
        Apply all parameter values from the provided dictionary.
        Raise KeyError if any parameter is missing.
        """
        pass

    def update_parameter_values(self, pv_dict: dict):
        """
        Only update given parameter values based on the provided dictionary to
        speed up the computing process.
        """
        pass

    def fix_parameters():
        """
        Fix parameters given their names.
        """
        pass

    def free_parameters():
        """
        Free parameters given their names.
        """
        pass

    def show_parameters():
        """
        Show current parameter values and their fix/free status.
        """
        pass

    def residual_factory():
        """
        Return the current residual function.
        """
        pass
