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
from pathlib import Path
import numpy
from BaseAdapter import BaseAdapter

# xyz_par is constrained or not when symmetry condition is imposed?
# What happended whe "P1" is given? What is the difference of setting "P1"
#   and not setting spacegroup at all?
# setQmax and setQmin for PDFGenerator?


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

    Properties
    ----------
    residual: callable
        The current residual function.
    initial_values: dict
        The dictionary of initial parameter values for residual function.
    parameters: dict
        The dictionary of parameter objects (both fixed and free) used in
        the current instance.
    parameter_slots_mask: list
        The boolean mask list indicating which parameters in
        `parameters_names_in_slots` are present in the current model.
    parameter_names_in_slots: list
        The sorted list of all parameter names used in ML/RL model.
        The parameter name might not appear in the specific structure model.
    parameter_values_in_slots: list
        The sorted list of all parameter names used in ML/RL model.
        The parameter name might not appear in the specific structure model.

    Attributes
    ----------
    _inputs : dict
        A dictionary containing loaded 'structure' and 'profile' objects.
    _recipe : FitRecipe
        The FitRecipe object managing the fitting process.
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
        self._make_recipe(
            xmin=inputs.get("xmin", None),
            xmax=inputs.get("xmax", None),
            dx=inputs.get("dx", None),
            Qmin=inputs.get("qmin", None),
            Qmax=inputs.get("qmax", None),
        )

    @property
    def residual(self):
        return self._recipe.residual

    @property
    def initial_values(self):
        return self._recipe.values

    @property
    def parameters(self):
        return self._recipe._parameters

    @property
    def parameters_names_in_slots(self):
        names = [
            "qdamp",
            "qbroad",
            "scale",
            "delta1",
            "delta2",
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
        ]
        max_atoms = 64
        for i in range(max_atoms):
            names.extend(
                [
                    f"x_{i}",
                    f"y_{i}",
                    f"z_{i}",
                    f"Uiso_{i}",
                    f"U11_{i}",
                    f"U22_{i}",
                    f"U33_{i}",
                    f"U12_{i}",
                    f"U13_{i}",
                    f"U23_{i}",
                ]
            )
        return names

    @property
    def parameters_slots_mask(self):
        mask = numpy.zeros(len(self.parameters_names_in_slots), dtype=bool)
        for pname in self.parameters:
            if pname in self.parameters_names_in_slots:
                index = self.parameters_names_in_slots.index(pname)
                mask[index] = True
        return mask

    @property
    def parameters_values_in_slots(self):
        values = numpy.zeros(len(self.parameters_names_in_slots))
        for pname, parameter in self.parameters:
            if pname in self.parameters_names_in_slots:
                index = self.parameters_names_in_slots.index(pname)
                values[index] = parameter.value
        return values

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
        # self._inputs["spacegroup"] = sg.short_name if sg else "P1"
        self._inputs["spacegroup"] = sg.short_name if sg else None
        profile = Profile()
        parser = PDFParser()
        parser.parseFile(str(profile_path))
        profile.loadParsedData(parser)
        self._inputs["profile"] = profile

    def _make_recipe(
        self,
        xmin=None,
        xmax=None,
        dx=None,
        Qmin=None,
        Qmax=None,
    ):
        """
        Make the FitRecipe for PDF fitting.
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
        recipe.addContribution(contribution)
        RUN_PARALLEL = True
        if RUN_PARALLEL:
            try:
                import psutil
                import multiprocessing
                from multiprocessing import Pool
            except ImportError:
                print(
                    "\nYou don't appear to have the necessary packages for "
                    "parallelization"
                )
            syst_cores = multiprocessing.cpu_count()
            cpu_percent = psutil.cpu_percent()
            avail_cores = numpy.floor(
                (100 - cpu_percent) / (100.0 / syst_cores)
            )
            ncpu = int(numpy.max([1, avail_cores]))
            pool = Pool(processes=ncpu)
            pdfgenerator.parallel(ncpu=ncpu, mapfunc=pool.map)
        # find all parameters and add them to recipe variables
        parameter_names = {}
        parameter_names["pdfgenerator"] = [
            "qdamp",
            "qbroad",
            "scale",
            "delta1",
            "delta2",
        ]
        for pname in parameter_names["pdfgenerator"]:
            par = getattr(pdfgenerator, pname)
            recipe.addVar(par, name=pname, fixed=False)
        stru_parset = pdfgenerator.phase
        spacegroup = self._inputs.get("spacegroup", None)
        if spacegroup is not None:
            spacegroupparams = constrainAsSpaceGroup(stru_parset, spacegroup)
            if spacegroupparams.xyzpars is not None:
                for par in spacegroupparams.xyzpars:
                    recipe.addVar(par, name=par.name, fixed=False)
            if spacegroupparams.latpars is not None:
                for par in spacegroupparams.latpars:
                    recipe.addVar(par, name=par.name, fixed=False)
            if spacegroupparams.adppars is not None:
                for par in spacegroupparams.adppars:
                    recipe.addVar(par, name=par.name, fixed=False)
        else:
            for i, atom in enumerate(stru_parset.atoms):
                for coord in ["x", "y", "z"]:
                    par = atom._parameters[coord]
                    recipe.addVar(par, name=par.name + f"_{i}", fixed=False)
                for adp in [
                    "U11",
                    "U22",
                    "U33",
                    "U12",
                    "U13",
                    "U23",
                ]:
                    par = atom._parameters[adp]
                    recipe.addVar(par, name=par.name + f"_{i}", fixed=False)
            lattice = stru_parset.getLattice()
            for lattice_par_name in [
                "a",
                "b",
                "c",
                "alpha",
                "beta",
                "gamma",
            ]:
                par = lattice._parameters[lattice_par_name]
                recipe.addVar(par, name=lattice_par_name, fixed=False)
        recipe.fix("all")
        recipe.fithooks[0].verbose = 0
        self._recipe = recipe

    def fix_parameters(self, parameter_names):
        """
        Fix parameters given their names. If it has been fixed, do nothing.
        If the parameter name is not found, raise KeyError.
        """
        for pname in parameter_names:
            if pname == "all":
                self._recipe.fix("all")
                continue
            if pname not in self.parameters:
                raise KeyError(f"Parameter {pname} not found in the model.")
            self._recipe.fix(pname)

    def free_parameters(self, parameter_names):
        """
        Free parameters given their names. If it has been freed, do nothing.
        If the parameter name is not found, raise KeyError.
        """
        for pname in parameter_names:
            if pname == "all":
                self._recipe.free("all")
                continue
            if pname not in self.parameters:
                raise KeyError(f"Parameter {pname} not found in the model.")
            self._recipe.free(pname)

    def show_parameters(self):
        """
        Show current parameter values and their fix/free status for the
        current instance.
        """
        msg = "Current parameter values and their status:\n"
        pnames, pvalues, statuses = [], [], []
        for pname, parameter in self.parameters.items():
            status = (
                "free"
                if self.parameters["recipe"].isFree(parameter)
                else "fixed"
            )
            msg += f"{pname}: {parameter.value} ({status})\n"
            pnames.append(pname)
            pvalues.append(parameter.value)
            statuses.append(status)
        print(msg)
        return pnames, pvalues, statuses

    def apply_pv_dict(self, pv_dict: dict):
        """
        Apply all parameter values from the provided dictionary.
        Raise KeyError if any parameter is missing.

        Parameters
        ----------
        pv_dict : dict
            Dictionary mapping parameter names to their desired values.
        """
        variables = {
            pname: param
            for pname, param in self._recipe._parameters.items()
            if self._recipe.isFree(param)
        }
        for pname, pvalue in pv_dict.items():
            if pname not in variables:
                raise KeyError(f"Parameter {pname} not found or is fixed.")
            variables[pname].setValue(pvalue)
