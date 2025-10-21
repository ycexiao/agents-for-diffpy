"""Helper funcitons to handle the interface compatibility."""

from collections import OrderedDict
from diffpy.srfit.pdf import PDFParser
from diffpy.srfit.fitbase import Profile
from diffpy.structure import Structure, loadStructure
from pathlib import Path


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


def adpmatrix_from_ij(U_dict):
    """Create ADP matrix from Uij components."""
    U11 = U_dict.get("U11", 0)
    U22 = U_dict.get("U22", 0)
    U33 = U_dict.get("U33", 0)
    U12 = U_dict.get("U12", 0)
    U13 = U_dict.get("U13", 0)
    U23 = U_dict.get("U23", 0)
    return [[U11, U12, U13], [U12, U22, U23], [U13, U23, U33]]
