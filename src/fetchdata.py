from mp_api.client import MPRester  # noqa: F401
import pickle
from diffpy.structure.parsers import getParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import random
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal import DebyeWallerFactor
from matplotlib import pyplot as plt


# with MPRester() as mpr:
#     docs = mpr.materials.summary.search(material_ids=["mp-149"])
# docs_data = [doc.model_dump() for doc in docs]
# with open("tmp.pkl", "wb") as f:
#     pickle.dump(docs_data, f)
with open("tmp.pkl", "rb") as f:
    docs_data = pickle.load(f)
# print(docs_data[0].keys())
doc = docs_data[0]
stru_matgen = doc["structure"]
sga = SpacegroupAnalyzer(stru_matgen)
spacegroupnumber = sga.get_space_group_number()

string = stru_matgen.to(fmt="cif")
stru = getParser("cif").parse(string)
perturbed_structure = stru

# -------------------------
# 2. Randomly perturb lattice parameters
# -------------------------
# (small random variations around ±2%)
lat = perturbed_structure.lattice
perturb_frac = 0.02  # ±2%
lat.setLatPar(
    lat.a * (1 + random.uniform(-perturb_frac, perturb_frac)),
    lat.b * (1 + random.uniform(-perturb_frac, perturb_frac)),
    lat.c * (1 + random.uniform(-perturb_frac, perturb_frac)),
    lat.alpha * (1 + random.uniform(-perturb_frac, perturb_frac)),
    lat.beta * (1 + random.uniform(-perturb_frac, perturb_frac)),
    lat.gamma * (1 + random.uniform(-perturb_frac, perturb_frac)),
)

# -------------------------
# 3. Randomly perturb atomic Uiso or Uij
# -------------------------
for atom in perturbed_structure:
    if atom.U is not None:
        atom.U = [
            [u + random.uniform(-0.001, 0.001) for u in row] for row in atom.U
        ]
    else:
        atom.Uiso = abs(random.gauss(0.005, 0.002))  # assign new Uiso

# -------------------------
# 4. Set up the PDF calculator
# -------------------------
pdfcalc = PDFCalculator()
dwf = DebyeWallerFactor()
pdfcalc.addContribution(dwf)  # << this makes Uiso matter
pdfcalc = PDFCalculator()
pdfcalc.rmin = 0.0
pdfcalc.rmax = 20.0
pdfcalc.rstep = 0.01

# -------------------------
# 5. Randomly perturb PDF parameters
# -------------------------
pdfcalc.delta1 = abs(random.gauss(0.0, 0.05))  # correlated motion
pdfcalc.delta2 = abs(random.gauss(0.0, 0.05))  # uncorrelated motion
pdfcalc.qdamp = abs(random.gauss(0.03, 0.005))
pdfcalc.qbroad = abs(random.gauss(0.05, 0.01))
pdfcalc.scale = random.uniform(0.8, 1.2)

# -------------------------
# 6. Compute the perturbed PDF
# -------------------------
r, G = pdfcalc(perturbed_structure)

# -------------------------
# 7. Plot the result
# -------------------------
plt.plot(r, G, label="PDF with Debye–Waller Broadening")
plt.xlabel("r (Å)")
plt.ylabel("G(r)")
plt.title("PDF with Uiso Effects (Debye–Waller)")
plt.legend()
plt.show()

# Optional: print parameters for record
print("Applied perturbations:")
print(f"  delta1 = {pdfcalc.delta1:.4f}")
print(f"  delta2 = {pdfcalc.delta2:.4f}")
print(f"  qdamp  = {pdfcalc.qdamp:.4f}")
print(f"  qbroad = {pdfcalc.qbroad:.4f}")
print(f"  scale  = {pdfcalc.scale:.4f}")
print(
    "  a,b,c, alpha, beta, gamma= "
    f" {lat.a:.3f}, {lat.b:.3f}, {lat.c:.3f}, "
    f"{lat.alpha:.2f}, {lat.beta:.2f}, {lat.gamma:.2f}"
)


# string = doc["structure"].to(fmt="cif")
# cifparser = getParser("cif")
# stru = cifparser.parse(string)
# print(stru.lattice.spacegroup)
