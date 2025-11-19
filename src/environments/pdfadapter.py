from pathlib import Path
from diffpy.structure import Structure, loadStructure
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.srfit.fitbase import (
    FitContribution,
    FitRecipe,
    Profile,
    FitResults,
)
from diffpy.srfit.pdf import PDFParser, PDFGenerator
import re


class PDFAdapter:
    def __init__(self, profile_path: Path, structure_path: Path):
        self.structure = loadStructure(str(structure_path))
        profile = Profile()
        parser = PDFParser()
        parser.parseFile(str(profile_path))
        profile.loadParsedData(parser)
        self.profile = profile
        self.generator = PDFGenerator()
        self.generator.setStructure(self.structure)
        self.contribution = FitContribution()
        self.contribution.setGenerator(self.generator)
        self.contribution.setProfile(self.profile)
        self.recipe = FitRecipe()
        self.recipe.addContribution(self.contribution)

    def init_action_space(self):
        special_params_names = [
            "spdiameter",
            "stepcut",
            "delta1",
            "delta2",
            "sratio",
            "rcut",
        ]  # not implement yet
        instrument_params_names = [
            "qdamp",
            "qbroad",
            "qmin",
            "qmax",
            "rmin",
            "rmax",
            "dx",
        ]
        atom_params_names = [
            f"x_{i}, y_{i}, z_{i}, "
            f"u11_{i}, u22_{i}, u33_{i}, u12_{i}, u13_{i}, u23_{i}"
            for i in range(len(self.structure))
        ]
        atom_params_names = [*(item.split(", ") for item in atom_params_names)]
        instrument_params = dict.fromkeys(instrument_params_names)
        atom_params = dict.fromkeys(atom_params_names)

    def get_var(self, var_name):
        if var_name is "qmin":
            return self.generator.getQmin()
        elif var_name is "qmax":
            return self.generator.getQmax()
        elif var_name is ["qdamp", "qbroad", "scale", "delta1", "delta2"]:
            return getattr(self.generator, var_name)
        elif bool(re.search(var_name, "\d$")):
            atom_index = int(var_name.split("_")[1])
            atom = self.structure[atom_index]
            name = re.sub(r"_\d$", "", var_name)
            if name == "x":
                return atom.xyz[0]
            elif name == "y":
                return atom.xyz[1]
            elif name == "z":
                return atom.xyz[2]
            elif name in ["u11", "u22", "u33", "u12", "u13", "u23"]:
                i = name[-2]
                j = name[-1]
                return atom.U[i, j]
            else:
                raise ValueError(f"Unsupported variable name: {var_name}")
        else:
            raise ValueError(f"Unsupported variable name: {var_name}")

    def set_var(self, var_name, value):
        if var_name is "qmin":
            self.generator.setQmin(value)
        elif var_name is "qmax":
            self.generator.setQmax(value)
        elif var_name is ["qdamp", "qbroad", "scale", "delta1", "delta2"]:
            setattr(self.generator, var_name, value)
        elif bool(re.search(var_name, "\d$")):
            atom_index = int(var_name.split("_")[1])
            atom = self.structure[atom_index]
            name = re.sub(r"_\d$", "", var_name)
            if name == "x":
                atom.xyz = (value, atom.xyz[1], atom.xyz[2])
            elif name == "y":
                atom.xyz = (atom.xyz[0], value, atom.xyz[2])
            elif name == "z":
                atom.xyz = (atom.xyz[0], atom.xyz[1], value)
            elif name in ["u11", "u22", "u33", "u12", "u13", "u23"]:
                i = name[-2]
                j = name[-1]
                U_matrix = atom.U
                U_matrix[i, j] = value
                if i != j:
                    U_matrix[j, i] = value
                atom.U = U_matrix
            else:
                raise ValueError(f"Unsupported variable name: {var_name}")
        else:
            raise ValueError(f"Unsupported variable name: {var_name}")

    def generate_loss_fun(self, action):
        pass

    def action_space(self):
        pass
