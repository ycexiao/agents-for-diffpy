from mp_api.client import MPRester
import pickle


def fetch_mp_structure():
    with MPRester() as mpr:
        docs = mpr.materials.summary.search(
            elements=[
                "Li",
            ],
            energy_above_hull=(0, 0.05),  # stable or nearly stable
            num_elements=(1, 5),  # optional: 1-3 elements
            fields=[
                "nsites",
                "material_id",
                "structure",
                "formula_pretty",
            ],
            chunk_size=10,
            num_chunks=1,
        )
    return docs


docs = fetch_mp_structure()

# with open("mp_docs.pkl", "wb") as f:
#     pickle.dump(docs, f)
