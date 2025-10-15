import pytest

from diffpy.structure import loadStructure
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.pdf import PDFParser, PDFGenerator

import gymnasium as gym


@pytest.mark.skipif(
    True,
    reason=(
        "Just a playground to test code functionality. "
        "Comment this to run it."
    ),
)
def test_iinfo(structure_and_profile_path):
    env = gym.make(  # noqa: F841
        "pdf-single-phase",
        structure=structure_and_profile_path["Ni"]["structure"],
        profile=structure_and_profile_path["Ni"]["profile"],
        configurations={
            "spacegroup": "Fm-3m",
            "fixed_params": {
                "profile": {
                    "xmin": 0.5,
                    "xmax": 20.0,
                    "dx": 0.01,
                },
                "generator": {
                    "qdamp": 0.03,
                    "qbroad": 0.01,
                },
            },
        },
    )
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
    # spacegroupparams = constrainAsSpaceGroup(pdfgenerator.phase, "Fm-3m")
    # print(spacegroupparams.__dict__.keys())
    # print(spacegroupparams.adpsymbols)
    # print(spacegroupparams.isosymbol)
    # print(spacegroupparams.scatterers)
    # print(len(spacegroupparams.scatterers))
    # print(pdfgenerator.phase.getScatterers())


# @pytest.mark.skipif(True, reason="Coding up other tests")
def test_environment_initialization(structure_and_profile_path):
    env = gym.make(
        "pdf-single-phase",
        structure=structure_and_profile_path["Ni"]["structure"],
        profile=structure_and_profile_path["Ni"]["profile"],
        configurations={
            "spacegroup": "Fm-3m",
        },
        fixed_params=["xmin", "xmax", "dx"],
        initial_values={
            "xmin": 0.5,
            "xmax": 20.0,
            "dx": 0.01,
            "qdamp": 0.03,
            "qbroad": 0.01,
        },
    )

    # reset
    observation, info = env.reset()
    print(info["variable-params"])
    # step
    action = {"scale": 1, "qdamp": 0}  # refine the scale
    observation, reward, termination, truncation, info = env.step(action)
    print(info["variable-params"])
