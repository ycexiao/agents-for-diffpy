from diffpy.srfit.pdf import PDFParser
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.structure.parsers import getParser
from diffpy.srfit.fitbase import (
    FitContribution,
    FitRecipe,
    Profile,
)
from diffpy.srfit.pdf import PDFGenerator
import numpy
from scipy.optimize import least_squares
import difflib


class PDFAdapter:
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
        # support max 64 atoms
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
        # Used to store intermediate results
        self.snapshots = {}

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
        if "remove_vars" in inputs:
            for var_name in inputs["remove_vars"]:
                self.delVar(var_name)

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
        contribution = FitContribution("pdfcontribution")
        profile.setCalculationRange(xmin=xmin, xmax=xmax, dx=dx)
        contribution.setProfile(profile)
        pdfgenerator = PDFGenerator("pdfgenerator")
        contribution.addProfileGenerator(pdfgenerator)
        pdfgenerator.setStructure(structure)
        if qmax is not None:
            pdfgenerator.setQmax(qmax)
        if qmin is not None:
            pdfgenerator.setQmin(qmin)
        recipe = FitRecipe()
        recipe.addContribution(contribution)
        RUN_PARALLEL = True
        if RUN_PARALLEL:
            try:
                import psutil
                import multiprocessing
                from multiprocessing import Pool

                syst_cores = multiprocessing.cpu_count()
                cpu_percent = psutil.cpu_percent()
                avail_cores = numpy.floor(
                    (100 - cpu_percent) / (100.0 / syst_cores)
                )
                ncpu = int(numpy.max([1, avail_cores]))
                pool = Pool(processes=ncpu)
                pdfgenerator.parallel(ncpu=ncpu, mapfunc=pool.map)
            except ImportError:
                print(
                    "\nYou don't appear to have the necessary packages for parallelization"
                )
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
    def delVar(self, var_name):
        """Delete a variable from the recipe.

        Variables are created automatically when the recipe is initialized. All
        variables will be refined by default. Use this method to remove a
        variable from the recipe if it should not be refined.

        Parameters
        ----------
        var_name : str
            The name of the variable to delete.
        """
        if var_name in self._recipe._parameters:
            self._recipe.delVar(self._recipe._parameters[var_name])
        else:
            raise KeyError(f"Variable '{var_name}' not found in the recipe.")

    @if_ready
    def _apply_parameter_values(self, pv_dict):
        """Update parameter values from the provided dictionary.

        Parameters
        ----------
        pv_dict : dict
            Dictionary mapping parameter names to their desired values.
            Unknown parameter names will raise a KeyError.
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
        """Apply a payload to the PDFAdapter.

        Payload is information cargo that will be updated at the end of each
        node and be passed to the child nodes (See FitDAG). A minimum example
        of it could be all the variables. Unknown keys will be ignored.

        Parameters
        ----------
        payload : dict
            Unknown parameter names will be ignored.
        """
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
        """Generate operations to be performed in the FitRunner.

        Attributes
        ----------
        action_names: list of str
            The instruction strings appeared at each node in FitDAG.
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
                method="trf",
            )

        return action_func

    def _residual(self, p=[]):
        """Residual function adapter from FitRecipe in order to capture the
        intermediate results, the snapshots, during the iterations."""
        # Prepare, if necessary
        self._recipe._prepare()

        for fithook in self._recipe.fithooks:
            fithook.precall(self._recipe)

        # Update the variable parameters.
        self._recipe._applyValues(p)

        # Update the constraints. These are ordered such that the list only
        # needs to be cycled once.
        for con in self._recipe._oconstraints:
            con.update()

        contributions = list(self._recipe._contributions.values())
        contribution_residuals = [
            ci.residual().flatten() for ci in contributions
        ]
        chiv = numpy.concatenate(
            [
                wi * residual
                for wi, residual in zip(
                    self._recipe._weights, contribution_residuals
                )
            ]
        )
        # Calculate the point-average chi^2
        w = numpy.dot(chiv, chiv) / len(chiv)
        # Now we must append the restraints
        penalties = [
            numpy.sqrt(res.penalty(w)) for res in self._recipe._restraintlist
        ]
        chiv = numpy.concatenate([chiv, penalties])

        for fithook in self._recipe.fithooks:
            fithook.postcall(self._recipe, chiv)
        ycalcs = [con._eq() for con in contributions]
        ys = [con.profile.y for con in contributions]
        # Store the current snapshots.
        for i in range(len(contributions)):
            self.snapshots[f"ycalc_{i}"] = ycalcs[i]
            self.snapshots[f"y_{i}"] = ys[i]
            self.snapshots[f"ydiff_{i}"] = ycalcs[i] - ys[i]
        return chiv

    def clone(self):
        """Create a copy of the current PDFAdapter with the same inputs and
        parameter values."""
        adapter = PDFAdapter()
        adapter.load_inputs(self.inputs)
        adapter._apply_parameter_values(self._get_parameter_values())
        return adapter
