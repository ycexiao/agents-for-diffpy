from typing import Optional
import numpy as np
import gymnasium as gym
import diffpy.structure
from diffpy.srreal.pdfcalculator import PDFGenerator
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.srfit.fitbase import FitContribution, FitRecipe
from diffpy.srfit.fitbase import FitResults
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.pdf import PDFParser, PDFGenerator
from collections import OrderedDict
from scipy.optimize import least_squares


class SingleStructurePDFEnv(gym.Env):
    def __init__(
        self,
        structure: diffpy.structure.Structure,
        target_pdf: np.ndarray,
        configurations: Optional[dict] = None,
        mode="train",
    ):
        self._agent_location = OrderedDict(
            {
                # structure parameters
                "lat": [
                    *(np.random.random((1, 3)) * 20),
                    *(np.random.random(1, 3) * 120),
                ],  # lattice parameter
                "delta1": 0,  # r-independent peak broadening parameter
                "delta2": 0,  # r-dependent peak broadening parameter
                "biso": 1,  # isotropic atomic displacement parameter
                ## add it later
                # "uij": np.random.random(1)
                # * 0.005,  # anisotropic atomic displacement parameter
                # dataset parameters
                "scale": 1,  # scale factor
                "qdamp": 0,  # Q-damp parameter
                "qbroad": 0,  # Q-broad parameter
            }
        )
        self._set_recipe(
            structure,
            self._agent_location,
            target_pdf,
            configurations["spacegroup"],  # used as constraints
            configurations[
                "instrumental_params"
            ],  # used to set initial values
        )
        self.action_space = gym.spaces.MultiBinary(len(self._agent_location))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, len(target_pdf)),
            dtype=np.float32,
        )
        self._nth_step = 0
        self._step_limit = configurations.get("step_limit", 100)
        self._mode = mode  # "train" or "test"
        self._target_pdf = target_pdf

    def _get_obs(self):
        residual = self._recipe.residual()
        simulated_pdf = self._recipe._contributions.values()
        diff_pdf = simulated_pdf - self._target_pdf
        return {
            "residual": residual,
            "simulated_pdf": simulated_pdf,
            "diff_pdf": diff_pdf,
        }

    def _get_info(self):
        return None

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        self._agent_location = OrderedDict(
            {
                # structure parameters
                "lat": [
                    *(np.random.random((1, 3)) * 20),
                    *(np.random.random(1, 3) * 120),
                ],  # lattice parameter
                "delta1": 0,  # r-independent peak broadening parameter
                "delta2": 0,  # r-dependent peak broadening parameter
                "biso": 1,  # isotropic atomic displacement parameter
                ## add it later
                # "uij": np.random.random(1)
                # * 0.005,  # anisotropic atomic displacement parameter
                # dataset parameters
                "scale": 1,  # scale factor
                "qdamp": 0,  # Q-damp parameter
                "qbroad": 0,  # Q-broad parameter
            }
        )
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

    def _set_recipe(
        self,
        structure: diffpy.structure.Structure,
        variables: dict,
        target_pdf: np.ndarray,
        spacegroup_constraints: Optional[str] = None,
        instrumental_params: Optional[dict] = None,
    ):
        # load structure and target PDF into contribution
        pdfgenerator = PDFGenerator("G1")
        pdfgenerator.setStructure(structure, periodic=True)
        contribution = FitContribution("crystal")
        contribution.addProfileGenerator(pdfgenerator)
        contribution.setProfile(target_pdf, xname="r")

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
        recipe.addVar(contribution.s1, variables["scale"], tag="scale")
        # space group constraints
        spacegroupparams = None
        if spacegroup_constraints:
            spacegroupparams = constrainAsSpaceGroup(
                pdfgenerator.phase, spacegroup_constraints
            )
            for par in spacegroupparams.latpars:
                recipe.addVar(
                    par,
                    value=variables[par.name],
                    fixed=True,
                    tag=par.name,
                )
            for par in spacegroupparams.adppars:
                recipe.addVar(
                    par,
                    value=variables[par.name],
                    fixed=True,
                    tag=par.name,
                )
            left_variables_name = list(
                set(variables.keys()) - set(["scale", "lat", "biso", "uij"])
            )
        else:
            left_variables_name = list(
                set(variables.keys()) - set(["scale", "lat"])
            )
        for var_name in left_variables_name:
            var_value = variables[var_name]
            recipe.addVar(
                getattr(pdfgenerator, var_name),
                value=var_value,
                fixed=True,
                tag=var_name,
            )
        self._recipe = recipe
