from typing import Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiBinary

from diffpy.structure import Structure
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
from helper import (
    recipe_parameters_to_refinement_variales,
    get_structure_and_profile,
)


# Number of scatterers is not the same for different refinement tasks.
# Set to be the number of scatters unless space group constraint is set.

# calculated pdf and measured pdf are of different length.
# changing of rmin, rmax, and rstep also changes the observation space
# cast it into a (1, self._profile_length) array in observation

# How to refine lat and adp without
#   spacegroupparams = constrainAsSpaceGroup() ?


class SinglePhase(gym.Env):
    """
    A reinforcement learning environment for fitting a single structure to a PDF profile.
    """

    def __init__(
        self,
        structure: Union[Structure, str, Path],
        profile: Union[Profile, str, Path],
        initial_values: Optional[dict] = {},
        configurations: Optional[dict] = {},
        fixed_params: Optional[list] = [],
        step_limit=100,
        profile_length=100,
    ):
        """
        Initialize the environment.

        Parameters
        ----------
        structure: Union[Structure, str, Path]
            The crystal structure to be fitted. Can be a diffpy Structure object or a path to
            a structure file (e.g., CIF).

        profile: Union[Profile, str, Path]
            The target PDF profile. Can be a diffpy Profile object or a path to a data file.

        configurations: Optional[dict]
            A dictionary of configurations for the fitting process, including:
            - spacegroup: str, the space group symbol for constraints

        fixed_params: List,
            Name of params are fixed.

        initial_values: Optional[dict]
            the intial values for variable parms. It makes less sense since initial values will be
            overwritten by the action sent by the agent. Nevertheless, it is good to have it
            so we have more control over the internal process since it is required to initialize
            the env.

        Attributes
        ----------
        action_space
        observation_space

        _initial_values
        _fixed_params
        _variable_params
        _step_limit
        _nth_step
        _recipe
        _profile_length

        Methods
        ------
        self.reset
        self.step
        self._get_obs
        """
        self._nth_step = 0
        self._step_limit = step_limit
        self._profile_length = profile_length
        self._object_params_map = {
            "profile": ["xmin", "xmax", "dx"],
            "generator": ["qdamp", "qbroad", "scale", "delta1", "delta2"],
        }

        for name in fixed_params:
            if not name in initial_values.keys():
                raise ValueError(
                    "Fixed params should have given initial value."
                )
        self._fixed_params = {
            name: initial_values[name] for name in fixed_params
        }
        self._initial_values = initial_values

        profile_calculation_params = {
            key: self._initial_values[key]
            for key in self._object_params_map["profile"]
            if key in self._initial_values.keys()
        }

        structure, profile = get_structure_and_profile(
            structure, profile, profile_calculation_params
        )

        self._make_recipe_setup_params(structure, profile, configurations)
        self.action_space = MultiBinary(len(self._variable_params.keys()))
        self.observation_space = Dict(
            {
                "Rw": Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "residual": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._profile_length,),
                    dtype=np.float32,
                ),
            }
        )

    def _get_obs(self):
        results = FitResults(self._recipe)

        residual = self._recipe.residual()
        x_old = np.linspace(0, 1, len(residual))
        x_new = np.linspace(0, 1, self._profile_length)
        residual = np.interp(x_new, x_old, residual)
        residual = residual.astype(np.float32)

        Rw = results.rw.astype(np.float32)
        return {
            "Rw": Rw,
            "residual": residual,
        }

    def _get_info(self):
        out = {
            "step-limit": self._step_limit,
            "nth-step": self._nth_step,
            "profile-length": self._profile_length,
            "fixed-params": self._fixed_params,
            "variable-params": self._variable_params,
        }
        return out

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        self._setup_variable_params()
        self._apply_refinement_variables()
        self._nth_step = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """
        Parameters
        ----------
        action: dict
            A dict with variable names as keys and 0 or 1 as values.
            entries with value 1 are to be refined in this step.
        """
        free_variable_names = [
            name for name, value in action.items() if value == 1
        ]
        self._recipe.fix("all")
        for var_name in free_variable_names:
            self._recipe.free(var_name)
        least_squares(
            self._recipe.residual, self._recipe.values, x_scale="jac"
        )
        updated_parameters = recipe_parameters_to_refinement_variales(
            self._recipe._parameters.values()
        )

        self._variable_params.update(updated_parameters)

        observation = self._get_obs()
        reward = observation["Rw"]
        info = self._get_info()

        truncation = False
        termination = False
        self._nth_step += 1
        if self._nth_step >= self._step_limit:
            truncation = True
        return observation, reward, termination, truncation, info

    def _make_recipe_setup_params(
        self,
        structure: Structure,
        profile: Profile,
        configurations: dict = None,
    ):
        # load structure and target PDF into contribution
        pdfgenerator = PDFGenerator("G1")
        pdfgenerator.setStructure(structure, periodic=True)
        contribution = FitContribution("crystal")
        contribution.addProfileGenerator(pdfgenerator)
        contribution.setProfile(profile, xname="r")

        # load contribution into recipe
        recipe = FitRecipe()
        recipe.addContribution(contribution)

        # set instrumental parameter values
        generator_params = {
            key: self._initial_values[key]
            for key in self._object_params_map["generator"]
            if key in self._initial_values.keys()
        }
        for key, value in generator_params.items():
            attr = getattr(contribution, key, None)
            if attr is not None:
                attr.value = value

        # set equation
        contribution.setEquation("G1")

        # set verbose level
        for fithook in recipe.fithooks:
            fithook.verbose = 0

        # set variables
        self._recipe = recipe
        self._nscatterers = len(pdfgenerator.phase.getScatterers())

        # variables on contribution
        # space group constraints
        if configurations["spacegroup"]:
            spacegroupparams = constrainAsSpaceGroup(
                pdfgenerator.phase, configurations["spacegroup"]
            )
            lat_par_names = [par.name for par in spacegroupparams.latpars]
            adp_par_names = [par.name for par in spacegroupparams.adppars]
            self._valid_lat_adp_names = lat_par_names + adp_par_names
            self._setup_variable_params()
            for par in spacegroupparams.latpars:
                if par.name in self._fixed_params.keys():
                    continue
                self._recipe.addVar(
                    par,
                    value=self._variable_params[par.name],
                    fixed=True,
                    tag=par.name,
                )
            for par in spacegroupparams.adppars:
                if par.name in self._fixed_params.keys():
                    continue
                self._recipe.addVar(
                    par,
                    value=self._variable_params[par.name],
                    fixed=True,
                    tag=par.name,
                )
            left_variables_name = list(
                set(self._variable_params.keys())
                - set(self._valid_lat_adp_names)
            )
        else:
            self._setup_variable_params()
            left_variables_name = list(self._variable_params.keys())

        # variables on pdfgenerator
        for var_name in left_variables_name:
            var_value = self._variable_params[var_name]
            self._recipe.addVar(
                getattr(pdfgenerator, var_name),
                value=var_value,
                fixed=False,
                tag=var_name,
            )

        self._apply_refinement_variables()

    def _setup_variable_params(self):
        self._variable_params = OrderedDict(
            {
                # structure parameters
                "delta1": 0,  # r-independent peak broadening parameter
                "delta2": 0,  # r-dependent peak broadening parameter
                # dataset parameters
                "scale": 1,  # scale factor
                "qdamp": 0,  # Q-damp parameter
                "qbroad": 0,  # Q-broad parameter
            }
        )
        self.__latpars_names = ["a", "b", "c", "alpha", "beta", "gamma"]
        self.__iso_adppars_names = [
            f"Uiso_{i}" for i in range(self._nscatterers)
        ]
        self.__ani_adppars_names = [
            f"U{j+1}{k+1}_{i}"
            for i in range(self._nscatterers)
            for j in range(3)
            for k in range(3)
        ]
        if not self._valid_lat_adp_names:
            self._valid_lat_adp_names = list(
                self.__latpars_names
                + self.__iso_adppars_names
                + self.__ani_adppars_names
            )
        for name in self._valid_lat_adp_names:
            if name in ["a", "b", "c"]:
                self._variable_params[name] = np.random.random() * 20
            elif name in ["alpha", "beta", "gamma"]:
                self._variable_params[name] = np.random.random() * 120
            else:  # Usio and Uij
                self._variable_params[name] = 0

        self._variable_params.update(self._initial_values)
        for key in list(self._variable_params.keys()):
            if key in self._fixed_params.keys():
                self._variable_params.pop(key)

    def _apply_refinement_variables(self):
        for var in self._recipe._parameters.values():
            var.setValue(self._variable_params[var.name])
