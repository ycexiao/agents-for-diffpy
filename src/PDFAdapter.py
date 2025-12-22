from diffpy.srfit.pdf import PDFParser
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.structure.parsers import getParser
from diffpy.srfit.fitbase import (
    FitContribution,
    FitRecipe,
    Profile,
)
from diffpy.srfit.pdf import PDFGenerator
from pathlib import Path
import numpy
from BaseAdapter import BaseAdapter
from scipy.optimize import least_squares
import difflib

# xyz_par is constrained or not when symmetry condition is imposed?
# What happended whe "P1" is given? What is the difference of setting "P1"
#   and not setting spacegroup at all?
# setQmax and setQmin for PDFGenerator?


class PDFAdapter(BaseAdapter):
    """
    Adapter to expose PDF fitting interface for FitRunner.

    Attributes
    ----------
    _recipe : FitRecipe
        The FitRecipe object managing the fitting process.

    Methods
    -------
    load_inputs(inputs)
        Load inputs to initialize the adapter.
    apply_parameter_values(pv_dict)
        Update parameter values from the provided dictionary.
    get_parameter_values()
        Get current parameter values as a dictionary.
    show_parameters()
        Show current parameter values and their fix/free status.
    apply_action()
        Generate operations to be performed in the workflow.
    generate_observation()
    """

    def __init__(self):
        self.ready = False
        # hard-coded parameter names to standardize the action interface
        self._parameter_names = [
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
            self._parameter_names.extend(
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

    def if_ready(func):
        def wrapper(self, *args, **kwargs):
            if not self.ready:
                raise RuntimeError(
                    "PDFAdapter is not ready. Please load inputs first."
                )
            return func(self, *args, **kwargs)

        return wrapper

    def check_parameter_name(self, parameter_name):
        wrong_msg = ""
        if parameter_name not in self._parameter_names:
            wrong_msg = (
                f"Parameter {parameter_name} is not recognized. Did you mean: "
                f"{difflib.get_close_matches(parameter_name, self._parameter_names, cutoff=0.6)}\n"  # noqa: E501
            )
        return wrong_msg

    def load_inputs(self, inputs):
        recipe_input_keys = [
            "structure_path",
            "profile_path",
            "xmin",
            "xmax",
            "dx",
            "qmin",
            "qmax",
        ]
        recipe_inputs = {
            k: inputs[k] for k in recipe_input_keys if k in inputs
        }
        self._make_recipe(**recipe_inputs)

    def _make_recipe(
        self,
        structure_path,
        profile_path,
        xmin=None,
        xmax=None,
        dx=None,
        qmin=None,
        qmax=None,
    ):
        """
        Load inputs to create parameters and residuals.

        Attributes
        ----------
        inputs: dict
            The dictionary that should at least contain
            'structure_path' and 'profile_path'.
        """
        # load structure and profile
        stru_parser = getParser("cif")
        structure = stru_parser.parse(Path(structure_path).read_text())
        sg = getattr(stru_parser, "spacegroup", None)
        spacegroup = sg.short_name if sg else "P1"
        profile = Profile()
        parser = PDFParser()
        parser.parseFile(str(profile_path))
        profile.loadParsedData(parser)
        # set up PDF generator, contribution, and recipe
        xmin = xmin if xmin is not None else numpy.min(profile._xobs)
        xmax = xmax if xmax is not None else numpy.max(profile._xobs)
        dx = dx if dx is not None else numpy.mean(numpy.diff(profile._xobs))
        qmin = qmin if qmin is not None else 0.5
        qmax = qmax if qmax is not None else 25
        pdfgenerator = PDFGenerator()
        pdfgenerator.setStructure(structure)
        contribution = FitContribution("pdfcontribution")
        profile.setCalculationRange(xmin=xmin, xmax=xmax, dx=dx)
        contribution.setProfile(profile)
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
        for pname in [
            "qdamp",
            "qbroad",
            "scale",
            "delta1",
            "delta2",
        ]:
            par = getattr(pdfgenerator, pname)
            recipe.addVar(par, name=pname, fixed=False)
        stru_parset = pdfgenerator.phase
        spacegroupparams = constrainAsSpaceGroup(stru_parset, spacegroup)
        # FIXME: only support max 64 atoms. Refine the error messge later.
        for par in spacegroupparams.xyzpars:
            recipe.addVar(par, name=par.name, fixed=False)
            assert par.name in self._parameter_names
        for par in spacegroupparams.latpars:
            assert par.name in self._parameter_names
            recipe.addVar(par, name=par.name, fixed=False)
        for par in spacegroupparams.adppars:
            recipe.addVar(par, name=par.name, fixed=False)
            assert par.name in self._parameter_names
        recipe.fix("all")
        recipe.fithooks[0].verbose = 0
        self._recipe = recipe
        self.ready = True

    @if_ready
    def _apply_parameter_values(self, pv_dict):
        """
        Update parameter values from the provided dictionary.

        Parameters
        ----------
        pv_dict : dict
            Dictionary mapping parameter names to their desired values.
        """
        wrong_msg = ""
        for pname in pv_dict:
            wrong_msg += self.check_parameter_name(pname)
        if wrong_msg:
            raise KeyError(wrong_msg)
        for pname, pvalue in pv_dict.items():
            self._recipe._parameters[pname].setValue(pvalue)

    @if_ready
    def _get_parameter_values(self):
        """
        Get current parameter values as a dictionary.

        Returns
        -------
        pv_dict : dict
            Dictionary mapping parameter names to their current values.
        """
        return {
            pname: self._recipe._parameters[pname].value
            for pname in self._recipe._parameters
        }

    @if_ready
    def apply_payload(self, payload):
        if self.get_payload() == payload:
            return
        self._apply_parameter_values(payload)

    @if_ready
    def get_payload(self):
        return self._get_parameter_values()

    @if_ready
    def action_func_factory(self, action_names):
        """
        Generate operations to be performed in the workflow. Use factory to
        allow more flexible action functions in the future.

        Attributes
        ----------
        action_names: list of str
            The action_names appear in the DAG. This function maps the action
            names into the action function.
        """

        def action_func():
            # FIXME: currently only allow 'free' variables due to the
            # compatible issues encountered when initialize the self.conunc
            # variable in FitResults
            for name in action_names:
                if name == "all":
                    self._recipe.free("all")
                    break
                self._recipe.free(name)
            least_squares(
                self._recipe.residual, self._recipe.values, x_scale="jac"
            )

        return action_func

    @if_ready
    def _residual_scalar(self):
        """
        Copied from Fitresult._calculateMetrics
        """
        y = self._recipe.pdfcontribution.profile.y
        ycalc = self._recipe.pdfcontribution._eq()
        dy = self._recipe.pdfcontribution.profile.dy
        num = numpy.abs(y - ycalc)
        y = numpy.abs(y)
        chiv = num / dy
        cumchi2 = numpy.cumsum(chiv**2)
        # avoid index error for empty array
        yw = y / dy
        yw2tot = numpy.dot(yw, yw)
        if yw2tot == 0.0:
            yw2tot = 1.0
        cumrw = numpy.sqrt(cumchi2 / yw2tot)
        # avoid index error for empty array
        rw = cumrw[-1:].sum()
        return rw
