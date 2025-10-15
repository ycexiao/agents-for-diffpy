"""
Helper funcitons to handle the interface compatibility
"""

from collections import OrderedDict
from diffpy.srfit.pdf import PDFParser
from diffpy.srfit.fitbase import Profile
from diffpy.structure import Structure, loadStructure


def recipe_parameters_to_refinement_variales(parameters: OrderedDict):
    out = {}
    for par in parameters:
        out[par.name] = par.value
    return out


def get_structure_and_profile(structure, profile, profile_calculation_params):
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
