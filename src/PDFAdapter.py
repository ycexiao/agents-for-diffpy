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
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy
from queue import Queue

# xyz_par is constrained or not when symmetry condition is imposed?
# What happended whe "P1" is given? What is the difference of setting "P1"
#   and not setting spacegroup at all?
# setQmax and setQmin for PDFGenerator?


class PDFAdapter(BaseAdapter):
    """Adapter to expose PDF fitting interface for FitRunner.

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
        # for sanity check.
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
        self.inputs = None
        # Used to contain intermediate results during fitting
        self.snapshots = {}
        self.iteration_counts = 0

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
        self.inputs = inputs
        recipe_input_keys = [
            "structure_string",
            "profile_string",
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
        structure_string,
        profile_string,
        xmin=None,
        xmax=None,
        dx=None,
        qmin=None,
        qmax=None,
    ):
        """Load inputs to create parameters and residuals.

        Attributes
        ----------
        inputs: dict
            The dictionary that should at least contain
            'structure_path' and 'profile_path'.
        """
        # load structure and profile
        stru_parser = getParser("cif")
        structure = stru_parser.parse(structure_string)
        sg = getattr(stru_parser, "spacegroup", None)
        spacegroup = sg.short_name if sg else "P1"
        profile = Profile()
        parser = PDFParser()
        parser.parseString(profile_string)
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
        self._recipe._prepare()

    @if_ready
    def _apply_parameter_values(self, pv_dict):
        """Update parameter values from the provided dictionary.

        Parameters
        ----------
        pv_dict : dict
            Dictionary mapping parameter names to their desired values.
        """
        if pv_dict == {} or pv_dict is None:
            return
        wrong_msg = ""
        for pname in pv_dict:
            wrong_msg += self.check_parameter_name(pname)
        if wrong_msg:
            raise KeyError(wrong_msg)
        for pname, pvalue in pv_dict.items():
            self._recipe._parameters[pname].setValue(pvalue)
        self._recipe._prepare()
        for con in self._recipe._oconstraints:
            con.update()

    @if_ready
    def _get_parameter_values(self):
        """Get current parameter values as a dictionary.

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
        pv_dict = {
            pname: payload[pname]
            for pname in self._recipe._parameters
            if pname in payload
        }
        self._apply_parameter_values(pv_dict)

    @if_ready
    def get_payload(self):
        payload = self._get_parameter_values()
        return payload

    @if_ready
    def action_func_factory(self, action_names):
        """Generate operations to be performed in the workflow. Use factory to
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
            if action_names == []:
                return None
            for name in action_names:
                if name == "all":
                    self._recipe.free("all")
                    break
                self._recipe.free(name)
            least_squares(
                self._residual,
                self._recipe.values,
                x_scale="jac",
            )

        return action_func

    def _residual(self, p=[]):
        self._recipe._prepare()
        for con in self._recipe._oconstraints:
            con.update()
        self._recipe._applyValues(p)

        cons = list(self._recipe._contributions.values())
        ycalcs = [con._eq() for con in cons]
        ys = [con.profile.y for con in cons]
        residuals = [con._reseq() for con in cons]
        chiv = numpy.concatenate(
            [
                wi * residual.flatten()
                for wi, residual in zip(self._recipe._weights, residuals)
            ]
        )
        w = numpy.dot(chiv, chiv) / len(chiv)
        penalties = [
            numpy.sqrt(res.penalty(w)) for res in self._recipe._restraintlist
        ]
        chiv = numpy.concatenate([chiv, penalties])
        # Store the current snapshots.
        for i in range(len(cons)):
            self.snapshots[f"ycalc_{i}"] = ycalcs[i]
            self.snapshots[f"y_{i}"] = ys[i]
            self.snapshots[f"ydiff_{i}"] = ycalcs[i] - ys[i]
            self.snapshots[f"scalar_chi_{i}"] = w
        return chiv

    def clone(self):
        """Create a copy of the current PDFAdapter with the same inputs and parameter values."""
        adapter = PDFAdapter()
        adapter.load_inputs(self.inputs)
        adapter._apply_parameter_values(self._get_parameter_values())
        return adapter
