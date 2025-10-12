from pathlib import Path
import pytest
import sys

from diffpy.structure import Structure, loadStructure
from diffpy.srfit.fitbase import FitContribution, FitRecipe, Profile
from diffpy.srfit.pdf import PDFParser, PDFGenerator
from diffpy.srfit.structure import constrainAsSpaceGroup

from matplotlib import pyplot as plt
import numpy as np
from helper import recipe_parameters_to_refinement_variales


@pytest.mark.skipif(True, reason="Coding up other tests")
def test_iinfo(structure_and_profile_path):
    profile = structure_and_profile_path["Ni"]["profile"]

    def data_to_profile(data_path):
        profile = Profile()
        parser = PDFParser()
        parser.parseFile(data_path)
        profile.loadParsedData(parser)
        return profile

    profile = data_to_profile(profile)
    structure = loadStructure(structure_and_profile_path["Ni"]["structure"])
    pdfgenerator = PDFGenerator("G1")
    pdfgenerator.setStructure(structure, periodic=True)
    spacegroupparams = constrainAsSpaceGroup(pdfgenerator.phase, "Fm-3m")
    print(spacegroupparams.__dict__.keys())
    print(spacegroupparams.adpsymbols)
    print(spacegroupparams.isosymbol)
    print(spacegroupparams.scatterers)
    print(len(spacegroupparams.scatterers))
    print(pdfgenerator.phase.getScatterers())


# @pytest.mark.skipif(True, reason="Coding up other tests")
def test_environment_initialization(structure_and_profile_path):
    from single_structure_pdf import SingleStructurePDFEnv

    # Initialization
    env = SingleStructurePDFEnv(
        structure=structure_and_profile_path["Ni"]["structure"],
        profile=structure_and_profile_path["Ni"]["profile"],
        configurations={
            "spacegroup": "Fm-3m",
            "instrumental_params": {"qdamp": 0.03, "qbroad": 0.01},
            "step_limit": 50,
            "profile_calculation_params": {
                "xmin": 0.5,
                "xmax": 20.0,
                "dx": 0.01,
            },
        },
    )

    history = np.zeros((env._step_limit, len(env._refinement_variables_dict)))

    # reset
    observation, info = env.reset()
    history[0] = observation["parameter_values"]

    # step
    variables = env._refinement_variables_dict
    action_vector = np.zeros(len(variables))
    scale_index = list(variables.keys()).index("scale")
    action_vector[scale_index] = 1
    observation, reward, termination, truncation, info = env.step(
        action_vector
    )
    history[1] = observation["parameter_values"]
    print(observation["parameter_names"])
    print(history[1] - history[0])
