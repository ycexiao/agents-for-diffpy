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

# Number of scatterers is not the same for different refinement tasks and there are no upper limit.
# Set to be 8 for now. Agent can choose to activate/deactivate.


class SingleStructurePDFEnv(gym.Env):
    """
    A reinforcement learning environment for fitting a single structure to a PDF profile.
    """

    def __init__(
        self,
        structure: Union[Structure, str, Path],
        profile: Union[Profile, str, Path],
        configurations: Optional[dict] = None,
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
            - nscatterers: int, number of scatterers in the structure. Default 8
            - instrumental_params: dict, initial values for instrumental parameters, including
                - qdamp
                - qbroad
            - profile_calculation_params: dict, initial values for parameters for PDF calculation, including
                - rmin
                - rmax
                - rstep
            - step_limit: int, maximum number of steps per episode
        """
        self._nth_step = 0
        self._step_limit = configurations.get("step_limit", 100)
        self._nscatterers = configurations.get("nscatterers", 8)
        structure, profile = self._get_structure_and_profile(
            structure, profile, configurations["profile_calculation_params"]
        )
        self._reset_agent_location()
        self._make_recipe(
            structure,
            self._agent_location,
            profile,
            configurations["spacegroup"],  # used as constraints
            configurations[
                "instrumental_params"
            ],  # used to set initial values
        )
        self.action_space = MultiBinary(len(self._agent_location))
        self.observation_space = Dict(
            {
                "residual": Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "simulated_pdf": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, len(profile.xobs)),
                    dtype=np.float32,
                ),
                "diff_pdf": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, len(profile.xobs)),
                    dtype=np.float32,
                ),
            }
        )

    def _get_obs(self):
        residual = self._recipe.residual()
        simulated_pdf = self._recipe._contributions.values()
        diff_pdf = simulated_pdf - self._recipe._contributions.yobs
        return {
            "residual": residual,
            "simulated_pdf": simulated_pdf,
            "diff_pdf": diff_pdf,
        }

    def _get_info(self):
        pass

    def _reset_agent_location(self):
        self._latpars_names = ["a", "b", "c", "alpha", "beta", "gamma"]
        self._iso_adppars_names = [
            f"Uiso_{i}" for i in range(self._nscatterers)
        ]
        self._ani_adppars_names = [
            f"U{j}{k}_{i}"
            for i in range(self._nscatterers)
            for j in range(3)
            for k in range(3)
        ]
        self._agent_location = OrderedDict(
            {
                # structure parameters
                #   lattice parameters
                "a": np.random.random() * 20,
                "b": np.random.random() * 20,
                "c": np.random.random() * 20,
                "alpha": np.random.random() * 120,
                "beta": np.random.random() * 120,
                "gamma": np.random.random() * 120,
                "delta1": 0,  # r-independent peak broadening parameter
                "delta2": 0,  # r-dependent peak broadening parameter
                # dataset parameters
                "scale": 1,  # scale factor
                "qdamp": 0,  # Q-damp parameter
                "qbroad": 0,  # Q-broad parameter
            }
        )
        for i in range(self._nscatterers):
            # isotropic ADP
            self._agent_location[self._iso_adppars_names[i]] = (
                np.random.random()
            )
            # anisotropic ADP
            for j in range(9):
                ani_adp_name = self._ani_adppars_names[i * 9 + j]
                self._agent_location[ani_adp_name] = np.random.random()

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        self._reset_agent_location()
        self._nth_step = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        free_variables_index = np.where(action == 1)[0]
        free_variables_name = [
            self._agent_location.keys()[i] for i in free_variables_index
        ]

        self._recipe.fix("all")
        for var_name in free_variables_name:
            self._recipe.free(var_name)
        least_squares(
            self._recipe.residual, self._recipe.values, x_scale="jac"
        )
        self._agent_location.update(self._recipe._parameters.values())
        observation = self._get_obs()
        reward = -observation["residual"]
        info = self._get_info()

        truncation = False
        termination = False
        self._nth_step += 1
        if self._nth_step >= self._step_limit:
            truncation = True

        return observation, reward, termination, truncation, info

    def _make_recipe(
        self,
        structure: Structure,
        variables_dict: dict,
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
                attr = getattr(recipe.crystal.G1, key, None)
                if attr is not None:
                    attr.value = value

        # set equation
        contribution.setEquation("s1*G1")

        # set variables
        # scale
        recipe.addVar(contribution.s1, variables_dict["scale"], tag="scale")
        # space group constraints
        spacegroupparams = None
        if spacegroup_constraints:
            spacegroupparams = constrainAsSpaceGroup(
                pdfgenerator.phase, spacegroup_constraints
            )
            for par in spacegroupparams.latpars:
                recipe.addVar(
                    par,
                    value=variables_dict[par.name],
                    fixed=True,
                    tag=par.name,
                )
            for par in spacegroupparams.adppars:
                print(par.name)
                recipe.addVar(
                    par,
                    value=variables_dict[par.name],
                    fixed=True,
                    tag=par.name,
                )
        left_variables_name = list(
            set(variables_dict.keys())
            - set(
                self._iso_adppars_names
                + self._ani_adppars_names
                + self._latpars_names
                + ["scale"]
            )
        )

        for var_name in left_variables_name:
            var_value = variables_dict[var_name]
            recipe.addVar(
                getattr(pdfgenerator, var_name),
                value=var_value,
                fixed=True,
                tag=var_name,
            )
        self._recipe = recipe

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
