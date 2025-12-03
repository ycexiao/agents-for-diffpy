from PDFAdapter import PDFAdapter
from pathlib import Path
from diffpy.structure import loadStructure
from matplotlib import pyplot as plt

profile_path = Path().cwd() / "data" / "Ni.gr"
structure_path = Path().cwd() / "data" / "Ni.cif"

adapter = PDFAdapter(str(profile_path), str(structure_path))
stru = adapter.residual_factory_components["pdfgenerator"].phase
# print(type(stru))
lattice_par = stru.lattice
# print(stru.__dict__.keys())
# print(lattice_par.__dict__.keys())
print(stru.atoms[0].__dict__.keys())
pars = stru.atoms[0]._parameters
print(type(stru.atoms[0]))
print(pars.keys())
# lattice = adapter.inputs["structure"].getLattice()
# print(adapter.inputs["structure"])
# print(adapter.inputs["structure"].__dict__.keys())
# print(adapter.inputs["structure"]._lattice)
# print(adapter.inputs["structure"]._lattice.__dict__.keys())
# print(adapter.inputs["profile"].__dict__.keys())
# print(adapter.inputs["profile"].meta)
# print(adapter.inputs["profile"]._xobs)
# print(adapter.inputs["profile"]._observers)
