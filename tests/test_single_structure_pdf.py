import pytest

from diffpy.structure import loadStructure
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.pdf import PDFParser, PDFGenerator

import gymnasium as gym
import environments  # noqa: F401
import time


@pytest.mark.skipif(
    True,
    reason=(
        "Just a playground to test code functionality. "
        "Comment this to run it."
    ),
)
def test_info(structure_and_profile_path):
    env = gym.make(  # noqa: F841
        "pdf-single-phase",
        structure=structure_and_profile_path["Ni"]["structure"],
        profile=structure_and_profile_path["Ni"]["profile"],
        configurations={
            "spacegroup": "Fm-3m",
        },
        fixed_params=["xmin", "xmax", "dx", "Qmin", "Qmax"],
        initial_values={
            "a": 3.52,
            "xmin": 1.5,
            "xmax": 50.0,
            "dx": 0.01,
            "qdamp": 0.04,
            "qbroad": 0.02,
            "scale": 0.4,
            "Uiso_0": 0.005,
            "delta2": 2,
            "Qmin": 0.1,
            "Qmax": 25,
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
    print(pdfgenerator.qmin)
    # spacegroupparams = constrainAsSpaceGroup(pdfgenerator.phase, "Fm-3m")
    # print(spacegroupparams.__dict__.keys())
    # print(spacegroupparams.adpsymbols)
    # print(spacegroupparams.isosymbol)
    # print(spacegroupparams.scatterers)
    # print(len(spacegroupparams.scatterers))
    # print(pdfgenerator.phase.getScatterers())


# @pytest.mark.skipif(True, reason="Coding up other tests")
def test_Ni_bulk(structure_and_profile_path):
    last_time = time.time()
    env = gym.make(
        "pdf-single-phase",
        structure=structure_and_profile_path["Ni"]["structure"],
        profile=structure_and_profile_path["Ni"]["profile"],
        configurations={
            "spacegroup": "Fm-3m",
        },
        fixed_params=["xmin", "xmax", "dx", "Qmin", "Qmax"],
        initial_values={
            "xmin": 1.5,
            "xmax": 50.0,
            "dx": 0.01,
            "qdamp": 0.04,
            "qbroad": 0.02,
            "scale": 0.4,
            "Uiso_0": 0.005,
            "delta2": 2,
            "Qmin": 0.1,
            "Qmax": 25,
        },
    )
    current_time = time.time()
    print("Initialization cost %s seconds" % (current_time - last_time))
    last_time = time.time()
    Rws = []
    # reset
    observation, info = env.reset()
    print(info["variable-params"])
    Rws.append(observation["Rw"][0])

    # Agent made the following decisions
    actions = [
        {"a": 1},
        {"scale": 1},
        {"Uiso_0": 1},
        {"delta2": 1},
        {key: 1 for key, value in info["variable-params"].items()},
    ]
    # step
    for i, action in enumerate(actions):
        observation, reward, termination, truncation, info = env.step(action)
        Rws.append(observation["Rw"][0])
        current_time = time.time()
        print(f"{i}-th iteraction cost {current_time-last_time} seconds.")
        print(info["variable-params"])
        last_time = time.time()
        if termination or truncation:
            break
    # print(Rws)
    # fig, ax = plt.subplots()
    # ax.plot(range(len(Rws)), Rws)
    # ax.set_ylabel("Rw")
    # ax.set_xlabel("Step")
    # plt.show()


"""
Totally around 10 seconds
Initialization cost 0.18345427513122559 seconds
0-th iteraction cost 2.2763824462890625 seconds.
1-th iteraction cost 0.007516384124755859 seconds.
2-th iteraction cost 1.0915942192077637 seconds.
3-th iteraction cost 1.290797472000122 seconds.
4-th iteraction cost 5.222010374069214 seconds.
"""
