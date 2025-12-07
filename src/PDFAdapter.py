from typing import Optional, Union
from diffpy.structure import Structure
from diffpy.srfit.pdf import PDFParser
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.structure.parsers import getParser
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
import copy
import numpy
from BaseAdapter import BaseAdapter


class PDFAdapter(BaseAdapter):
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

    Properties
    ----------
    residual: callable
        The current residual function.
    initial_values: dict
        The dictionary of initial parameter values for residual function.
    parameters: dict
        The dictionary of parameter objects (both fixed and free) used in the
        fitting model.
    parameters_slots: list
        The sorted list of all parameter names used in ML/RL model. The
        parameter name might not appear in the sepcific structure model.

    Attributes
    ----------
    _inputs : dict
        A dictionary containing loaded 'structure' and 'profile' objects.
    _residual_factory_components : dict
        A dictionary containing components used to build the residual
        function.
    _pname_parameter_dict : dict
        A dictionary mapping parameter names to their corresponding
        objects in the fitting model.
    _parameter_names_group : dict
        A dictionary grouping parameter names by their categories.
    _parameters_ready : bool
        Flag indicating whether parameters have been initialized and bound.

    Methods
    -------
    fix_parameters(parameter_names: list)
        Fix parameters given their names.
    free_parameters(parameter_names: list)
        Free parameters given their names.
    show_parameters()
        Show current parameter values and their fix/free status.
    """

    def __init__(self, **inputs):
        self._parameters_ready = False
        self._load_inputs(
            profile_path=inputs.pop("profile_path"),
            structure_path=inputs.pop("structure_path"),
        )
        self._init_parameters()
        self._init_residual_factory(
            xmin=inputs.get("xmin", None),
            xmax=inputs.get("xmax", None),
            dx=inputs.get("dx", None),
        )
        self._impose_symmertry_constraints(inputs.get("spacegroup", None))
        self._bind_model_parameters()

    @property
    def residual(self):
        return self._residual_factory_components["recipe"].residual

    @property
    def initial_values(self):
        return self._residual_factory_components["recipe"].values

    @property
    def parameters(self):
        if not self._parameters_ready:
            raise RuntimeError(
                "Parameters are not ready. Ensure parameters are "
                "initialized and bound before accessing."
            )
        else:
            return self._pname_parameter_dict

    @property
    def pparameter_values_in_slots(self):
        static_parameter_groups = [
            "generator_parameters",
            "structure_lattice_parameters",
        ]
        parameter_names_slots = []
        for group in static_parameter_groups:
            parameter_names_slots.extend(
                self._possible_parameter_names_group[group]
            )
        max_atoms = 64
        for i in range(max_atoms):
            parameter_names_slots.append(f"x_{i}")
            parameter_names_slots.append(f"y_{i}")
            parameter_names_slots.append(f"z_{i}")
            parameter_names_slots.append(f"U11_{i}")
            parameter_names_slots.append(f"U22_{i}")
            parameter_names_slots.append(f"U33_{i}")
            parameter_names_slots.append(f"U12_{i}")
            parameter_names_slots.append(f"U13_{i}")
            parameter_names_slots.append(f"U23_{i}")
        parameter_values_in_slots = numpy.zeros(len(parameter_names_slots))
        for pname in self.parameters.keys():
            if pname in parameter_names_slots:
                index = parameter_names_slots.index(pname)
                parameter_values_in_slots[index] = self.parameters[pname].value
            elif pname.startswith("Uiso"):
                suffix = pname.split("Uiso")[-1]
                u11_name = f"U11{suffix}"
                index = parameter_names_slots.index(u11_name)
                parameter_values_in_slots[index] = self.parameters[pname].value
            else:
                raise KeyError(
                    f"Parameter {pname} not found in the parameter slots."
                )
        return parameter_values_in_slots

    def _load_inputs(self, profile_path: str, structure_path: str):
        """
        Load profile and structure from given file paths.

        Attributes
        ----------
        inputs: dict
            A dictionary containing loaded 'structure' and 'profile' objects.
        """
        stru_parser = getParser("cif")
        structure = stru_parser.parse(Path(structure_path).read_text())
        self._inputs = {
            "structure": structure,
        }
        sg = getattr(stru_parser, "spacegroup", None)
        self._inputs["spacegroup"] = sg.short_name if sg else "P1"
        profile = Profile()
        parser = PDFParser()
        parser.parseFile(str(profile_path))
        profile.loadParsedData(parser)
        self._inputs["profile"] = profile

    def _init_parameters(self):
        self._possible_parameter_names_group = {
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
            # "structure_atom_xyz_parameters": (
            #     [f"x_{i}" for i in range(len(self._inputs["structure"]))]
            #     + [f"y_{i}" for i in range(len(self._inputs["structure"]))]
            #     + [f"z_{i}" for i in range(len(self._inputs["structure"]))]
            # ),
            "structure_atom_U_parameters": (
                [f"U11_{i}" for i in range(len(self._inputs["structure"]))]
                + [f"U22_{i}" for i in range(len(self._inputs["structure"]))]
                + [f"U33_{i}" for i in range(len(self._inputs["structure"]))]
                + [f"U12_{i}" for i in range(len(self._inputs["structure"]))]
                + [f"U13_{i}" for i in range(len(self._inputs["structure"]))]
                + [f"U23_{i}" for i in range(len(self._inputs["structure"]))]
            ),
        }

    def _init_residual_factory(self, xmin=None, xmax=None, dx=None):
        """
        Initialize the factory to generate residual function.

        Attributes
        ----------
        residual_factory: callable
            A callable that returns the current residual function.
        residual_factory_components: dict
            A dictionary containing components used to build the residual.
        """
        # set up PDF generator, contribution, and recipe
        pdfgenerator = PDFGenerator("pdfgen")
        pdfgenerator.setStructure(self._inputs["structure"])
        contribution = FitContribution("pdfcontri")
        xmin = xmin if xmin else numpy.min(self._inputs["profile"]._xobs)
        xmax = xmax if xmax else numpy.max(self._inputs["profile"]._xobs)
        dx = (
            dx if dx else numpy.mean(numpy.diff(self._inputs["profile"]._xobs))
        )
        self._inputs["profile"].setCalculationRange(
            xmin=xmin, xmax=xmax, dx=dx
        )
        contribution.setProfile(self._inputs["profile"])
        contribution.addProfileGenerator(pdfgenerator)
        recipe = FitRecipe()
        recipe.fithooks[0].verbose = 0
        recipe.addContribution(contribution)
        RUN_PARALLEL = True
        if RUN_PARALLEL:
            try:
                import psutil
                import multiprocessing
                from multiprocessing import Pool
            except ImportError:
                print(
                    "\nYou don't appear to have the necessary packages for parallelization"
                )
            syst_cores = multiprocessing.cpu_count()
            cpu_percent = psutil.cpu_percent()
            avail_cores = numpy.floor(
                (100 - cpu_percent) / (100.0 / syst_cores)
            )
            ncpu = int(numpy.max([1, avail_cores]))
            pool = Pool(processes=ncpu)
            pdfgenerator.parallel(ncpu=ncpu, mapfunc=pool.map)
        self._residual_factory_components = {
            "pdfgenerator": pdfgenerator,
            "contribution": contribution,
            "recipe": recipe,
        }
        # find all parameters and map them to their names
        stru_parset = pdfgenerator.phase
        self._pname_parameter_dict = {
            "a": stru_parset.lattice.a,
            "b": stru_parset.lattice.b,
            "c": stru_parset.lattice.c,
            "alpha": stru_parset.lattice.alpha,
            "beta": stru_parset.lattice.beta,
            "gamma": stru_parset.lattice.gamma,
        }
        for i in range(len(stru_parset.atoms)):
            atom_parset = stru_parset.atoms[i]
            self._pname_parameter_dict[f"x_{i}"] = atom_parset._parameters["x"]
            self._pname_parameter_dict[f"y_{i}"] = atom_parset._parameters["y"]
            self._pname_parameter_dict[f"z_{i}"] = atom_parset._parameters["z"]
            self._pname_parameter_dict[f"U11_{i}"] = atom_parset._parameters[
                "U11"
            ]
            self._pname_parameter_dict[f"U22_{i}"] = atom_parset._parameters[
                "U22"
            ]
            self._pname_parameter_dict[f"U33_{i}"] = atom_parset._parameters[
                "U33"
            ]
            self._pname_parameter_dict[f"U12_{i}"] = atom_parset._parameters[
                "U12"
            ]
            self._pname_parameter_dict[f"U13_{i}"] = atom_parset._parameters[
                "U13"
            ]
            self._pname_parameter_dict[f"U23_{i}"] = atom_parset._parameters[
                "U23"
            ]
        self._pname_parameter_dict["qdamp"] = pdfgenerator.qdamp
        self._pname_parameter_dict["qbroad"] = pdfgenerator.qbroad
        self._pname_parameter_dict["scale"] = pdfgenerator.scale
        self._pname_parameter_dict["delta1"] = pdfgenerator.delta1
        self._pname_parameter_dict["delta2"] = pdfgenerator.delta2

    def _impose_symmertry_constraints(self, spacegroup: str):
        """
        Impose symmetry constraints on the structure based on the given
        crystal system and space group.
        """
        spacegroup = (
            self._inputs["spacegroup"] if spacegroup is None else spacegroup
        )
        stru_parset = self._residual_factory_components["pdfgenerator"].phase
        spacegroupparams = constrainAsSpaceGroup(stru_parset, spacegroup)
        for pname in self._possible_parameter_names_group[
            "structure_atom_xyz_parameters"
        ]:
            self._pname_parameter_dict.pop(pname)
        for pname in self._possible_parameter_names_group[
            "structure_lattice_parameters"
        ]:
            self._pname_parameter_dict.pop(pname)
        for pname in self._possible_parameter_names_group[
            "structure_atom_U_parameters"
        ]:
            self._pname_parameter_dict.pop(pname)
        for par in spacegroupparams.latpars:
            self._pname_parameter_dict[par.name] = par
        for par in spacegroupparams.adppars:
            self._pname_parameter_dict[par.name] = par

    def _bind_model_parameters(self):
        """
        Bind parameters to the variables used to compute residuals.
        """
        for pname, parameter in self._pname_parameter_dict.items():
            self._residual_factory_components["recipe"].addVar(
                parameter, name=pname, fixed=False
            )
        for pname, _ in self._pname_parameter_dict.items():
            self._pname_parameter_dict[pname] = (
                self._residual_factory_components["recipe"]._parameters.get(
                    pname
                )
            )
        self._residual_factory_components["recipe"].fix("all")
        self._parameters_ready = True

    def fix_parameters(self, parameter_names):
        """
        Fix parameters given their names. If it has been fixed, do nothing.
        If the parameter name is not found, raise KeyError.
        """
        for pname in parameter_names:
            if pname == "all":
                self._residual_factory_components["recipe"].fix("all")
                continue
            if pname not in self.parameters:
                raise KeyError(f"Parameter {pname} not found in the model.")
            self._residual_factory_components["recipe"].fix(pname)

    def free_parameters(self, parameter_names):
        """
        Free parameters given their names. If it has been freed, do nothing.
        If the parameter name is not found, raise KeyError.
        """
        for pname in parameter_names:
            if pname == "all":
                self._residual_factory_components["recipe"].free("all")
                continue
            if pname not in self.parameters:
                raise KeyError(f"Parameter {pname} not found in the model.")
            self._residual_factory_components["recipe"].free(pname)

    def show_parameters(self):
        """
        Show current parameter values and their fix/free status.
        """
        msg = "Current parameter values and their status:\n"
        for pname, parameter in self.parameters.items():
            status = (
                "free"
                if self._residual_factory_components["recipe"].isFree(
                    parameter
                )
                else "fixed"
            )
            msg += f"{pname}: {parameter.value} ({status})\n"
        print(msg)
        return msg
