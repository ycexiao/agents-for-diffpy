import random
from diffpy.structure.parsers import getParser
from diffpy.srfit.structure import struToParameterSet
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srfit.fitbase.parameter import ParameterAdapter


class PDFexperiment:
    def __init__(self, cfg=None):
        if not cfg:
            cfg = {
                "qmax": 25,
                "rmin": 0,
                "rmax": 10.001,
                "rstep": 0.05,
            }
        self.pdfcalculator = PDFCalculator(**cfg)
        self.default_parameter_values = {}
        self.structure = None
        self.spacegroup_params = None
        self.constraints = None
        self.parameters = []

    def load_inputs(self, structure_txt):
        stru_parser = getParser("cif")
        structure = stru_parser.parse(structure_txt)
        stru_parset = struToParameterSet("phase", structure)
        sg = getattr(stru_parser, "spacegroup", None)
        spacegroup = sg if sg is not None else "P1"
        spacegroup_params = constrainAsSpaceGroup(stru_parset, spacegroup)
        self.constraints = list(stru_parset._getConstraints().values())
        self.structure = structure
        parameters = []
        parsets = ["latpars", "xyzpars", "adppars"]
        for ps in parsets:
            if hasattr(spacegroup_params, ps):
                parameters.extend(list(getattr(spacegroup_params, ps)))
        calc_pnames = ["scale", "delta1", "delta2", "qdamp", "qbroad"]
        parameters.extend(
            [
                ParameterAdapter(pname, self.pdfcalculator, attr=pname)
                for pname in calc_pnames
            ]
        )
        self.parameters = {p.name: p for p in parameters}

    def update_consrtaints(self):
        iter_n = len(self.constraints)
        # No need to sort constraints if we just iterate enough times
        for _ in range(iter_n):
            for cons in self.constraints:
                cons.update()

    def perturb_parameter(self, pname, probability, magnitude):
        if random.random() > probability:
            return
        current_value = self.parameters[pname].value
        current_value = 1 if current_value == 0 else current_value
        self.parameters[pname].setValue(
            random.random() * magnitude * current_value
        )
        self.update_consrtaints()

    def generate(self):
        r, g = self.pdfcalculator(self.structure)
        return r, g
