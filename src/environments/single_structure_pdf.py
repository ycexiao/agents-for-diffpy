from typing import Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiBinary

from diffpy.structure import Structure, loadStructure
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.srfit.fitbase import FitContribution, FitRecipe, Profile
from diffpy.srfit.pdf import PDFParser, PDFGenerator
from collections import OrderedDict
from scipy.optimize import least_squares
from pathlib import Path
from diffpy.srfit.fitbase.parameter import ParameterProxy
from helper import recipe_parameters_to_refinement_variales


# Number of scatterers is not the same for different refinement tasks.
# Set to be the number of scatters unless space group constraint is set.

# calculated pdf and measured pdf are of different length.
# changing of rmin, rmax, and rstep also changes the observation space
# cast it into a (1,100) array


class SingleStructurePDFEnv(gym.Env):
    """
    A reinforcement learning environment for fitting a single structure to a PDF profile.
    """

    def __init__(
        self,
        structure: Union[Structure, str, Path],
        profile: Union[Profile, str, Path],
        configurations: Optional[dict] = None,
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
            - instrumental_params: dict, initial values for instrumental parameters, including
                - qdamp
                - qbroad
            - profile_calculation_params: dict, initial values for parameters for PDF calculation, including
                - rmin
                - rmax
                - rstep

        Attributes
        ----------
        action_space
        observation_space

        _step_limit
        _nth_step
        _recipe
        _refinement_variables_dict
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
        structure, profile = self._get_structure_and_profile(
            structure, profile, configurations["profile_calculation_params"]
        )
        self._make_recipe_setup_variables(
            structure,
            profile,
            configurations["spacegroup"],  # used as constraints
            configurations[
                "instrumental_params"
            ],  # used to set initial values
        )
        self.action_space = MultiBinary(
            len(self._refinement_variables_dict.keys())
        )
        self.observation_space = Dict(
            {
                "residual": Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "diff_pdf": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, self._profile_length),
                    dtype=np.float32,
                ),
            }
        )

    def _get_obs(self):
        residual = self._recipe.residual()
        pdf_calc = np.concatenate(
            [con.profile.ycalc for con in self._recipe._contributions.values()]
        )
        pdf_obs = np.concatenate(
            [con.profile.y for con in self._recipe._contributions.values()]
        )
        diff_pdf = pdf_calc - pdf_obs
        x_old = np.linspace(0, 1, len(diff_pdf))
        x_new = np.linspace(0, 1, self._profile_length)
        diff_pdf = np.interp(x_new, x_old, diff_pdf)
        parameter_names = list(self._refinement_variables_dict.keys())
        parameter_values = list(self._refinement_variables_dict.values())

        # print(self._refinement_variables_dict)
        # print(
        #     recipe_parameters_to_refinement_variales(
        #         self._recipe._parameters.values()
        #     )
        # )
        # print("\n")

        return {
            "parameter_names": parameter_names,
            "parameter_values": parameter_values,
            "residual": residual,
            "diff_pdf": diff_pdf,
        }

    def _get_info(self):
        return None

    def _setup_refinement_variables(self):
        self._refinement_variables_dict = OrderedDict(
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
                self._refinement_variables_dict[name] = np.random.random() * 20
            elif name in ["alpha", "beta", "gamma"]:
                self._refinement_variables_dict[name] = (
                    np.random.random() * 120
                )
            else:
                self._refinement_variables_dict[name] = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        self._setup_refinement_variables()
        self._nth_step = 0
        observation = self._get_obs()
        info = self._get_info()
        self._apply_refinement_variables()
        return observation, info

    def step(self, action):
        free_variables_index = np.where(action == 1)[0]
        free_variable_names = [
            name
            for i, name in enumerate(self._refinement_variables_dict.keys())
            if i in free_variables_index
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

        self._refinement_variables_dict.update(updated_parameters)

        observation = self._get_obs()
        reward = -observation["residual"]
        info = self._get_info()

        truncation = False
        termination = False
        self._nth_step += 1
        if self._nth_step >= self._step_limit:
            truncation = True

        return observation, reward, termination, truncation, info

    def _make_recipe_setup_variables(
        self,
        structure: Structure,
        profile: Profile,
        spacegroup_constraints: Optional[str] = None,
        instrumental_params: Optional[dict] = None,
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
        if instrumental_params:
            for key, value in instrumental_params.items():
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
        if spacegroup_constraints:
            spacegroupparams = constrainAsSpaceGroup(
                pdfgenerator.phase, spacegroup_constraints
            )
            lat_par_names = [par.name for par in spacegroupparams.latpars]
            adp_par_names = [par.name for par in spacegroupparams.adppars]
            self._valid_lat_adp_names = lat_par_names + adp_par_names
            self._setup_refinement_variables()
            for par in spacegroupparams.latpars:
                self._recipe.addVar(
                    par,
                    value=self._refinement_variables_dict[par.name],
                    fixed=True,
                    tag=par.name,
                )
            for par in spacegroupparams.adppars:
                self._recipe.addVar(
                    par,
                    value=self._refinement_variables_dict[par.name],
                    fixed=True,
                    tag=par.name,
                )
            left_variables_name = list(
                set(self._refinement_variables_dict.keys())
                - set(self._valid_lat_adp_names)
            )
        else:
            self._setup_refinement_variables()
            left_variables_name = list(self._refinement_variables_dict.keys())

        # variables on pdfgenerator
        for var_name in left_variables_name:
            var_value = self._refinement_variables_dict[var_name]
            self._recipe.addVar(
                getattr(pdfgenerator, var_name),
                value=var_value,
                fixed=False,
                tag=var_name,
            )

        self._apply_refinement_variables()

    def _apply_refinement_variables(self):
        for var in self._recipe._parameters.values():
            var.setValue(self._refinement_variables_dict[var.name])

    def _get_structure_and_profile(
        self, structure, profile, profile_calculation_params
    ):
        def ensure_obj_type(obj, expected_type, obj_to_type):
            if isinstance(obj, str) or isinstance(obj, Path):
                return obj_to_type(str(obj))
            elif isinstance(obj, expected_type):
                return obj
            else:
                raise ValueError(f"Unsupported type for {obj}: {type(obj)}")

        def data_to_profile(data_path):
            profile = Profile()
            parser = PDFParser()
            parser.parseFile(data_path)
            profile.loadParsedData(parser)
            profile.setCalculationRange(
                **profile_calculation_params
            )  # Set the calculation range for the PDF fit
            return profile

        structure = ensure_obj_type(structure, Structure, loadStructure)
        profile = ensure_obj_type(profile, Profile, data_to_profile)
        return structure, profile
