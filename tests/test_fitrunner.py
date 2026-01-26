from FitRunner import FitRunner
from PDFAdapter import PDFAdapter
from FitDAG import FitDAG
from pathlib import Path
import unittest
import sys
from scipy.optimize import least_squares

sys.path.append(str(Path(__file__).parent / "diffpycmi_scripts.py"))
from diffpycmi_scripts import make_recipe


class TestFitRunner(unittest.TestCase):
    def setUp(self):
        profile_path = Path("tests/data/Ni.gr")
        structure_path = Path("tests/data/Ni.cif")
        inputs = {
            "profile_string": profile_path.read_text(),
            "structure_string": structure_path.read_text(),
            "xmin": 1.5,
            "xmax": 50,
            "dx": 0.01,
            "qmax": 25.0,
            "qmin": 0.1,
            "remove_vars": ["delta1"],
        }
        payload = {
            "scale": 0.4,
            "a": 3.52,
            "Uiso_0": 0.005,
            "delta2": 2.0,
            "qdamp": 0.04,
            "qbroad": 0.02,
        }
        dag = FitDAG()
        dag.from_str("a->scale->qdamp->Uiso_0->delta2->all")
        runner = FitRunner()
        self.runner = runner
        self.dag = dag
        self.payload = payload
        self.inputs = inputs
        self.profile_path = profile_path
        self.structure_path = structure_path

    def test_run(self):
        # C0: Run the same Fit using diffpy.cmi and FitRunner.
        #  Expect to get similar results.
        diffpycmi_recipe = make_recipe(
            str(self.structure_path), str(self.profile_path)
        )
        diffpycmi_recipe.fithooks[0].verbose = 0
        diffpycmi_recipe.fix("all")
        tags = ["lat", "scale", "adp", "d2", "all"]
        for tag in tags:
            diffpycmi_recipe.free(tag)
            least_squares(
                diffpycmi_recipe.residual,
                diffpycmi_recipe.values,
                x_scale="jac",
            )
        diffpynames = [
            "fcc_Lat",
            "s1",
            "fcc_ADP",
            "Ni_Delta2",
            "Calib_Qdamp",
            "Calib_Qbroad",
            "all",
        ]
        pname = ["a", "scale", "Uiso_0", "delta2", "qdamp", "qbroad", "all"]
        diffpy2pname = {
            diffpyname: pname[i] for i, diffpyname in enumerate(diffpynames)
        }
        diffpy_pv_dict = {}
        for tag, parameter in diffpycmi_recipe._parameters.items():
            pname = diffpy2pname[tag]
            diffpy_pv_dict[pname] = parameter.value

        dag = FitDAG()
        dag.from_str("a->scale->qdamp->Uiso_0->delta2->all")
        runner = FitRunner()
        # Inputs and payload are the same as those used in the diffpy.cmi
        runner._run_dag(dag, PDFAdapter, self.inputs, self.payload)
        last_node_id = dag.leaf_nodes[0]
        last_node = dag.nodes[last_node_id]
        runner_pv_dict = {
            pname: last_node["payload"][pname]
            for pname in diffpy_pv_dict.keys()
        }

        for pname in diffpy_pv_dict.keys():
            self.assertAlmostEqual(
                diffpy_pv_dict[pname], runner_pv_dict[pname], places=4
            )

    def test_watch(self):
        # C1: Collect data for parameter "a" from the payload of each node
        #  Expect 6 data points collected for parameter "a"
        self.runner.watch(
            lambda dag, node_id: True,
            pname="a",
            update_mode="append",
            source="payload",
        )
        self.runner._run_dag(self.dag, PDFAdapter, self.inputs, self.payload)
        self.assertEqual(
            list(self.runner.data_for_plot.values())[0]["ydata"].qsize(), 6
        )

    def test_mark(self):
        # C1: Mark a node "hasAdapter" and "hasPayload"
        #  Expect runner.is_marked(node_id, "initialized") to be True
        node_id = list(self.dag.nodes.keys())[0]
        self.runner.mark(node_id, "hasAdapter")
        self.runner.mark(node_id, "hasPayload")
        self.assertTrue(self.runner.is_marked(node_id, "initialized"))
        # C2: Mark a node "hasAdapter" only
        #  Expect runner.is_marked(node_id, "initialized") to be False
        node_id = list(self.dag.nodes.keys())[1]
        self.runner.mark(node_id, "hasAdapter")
        self.assertFalse(self.runner.is_marked(node_id, "initialized"))

        # C3: Mark a node "hasPayload" only
        #  Expect runner.is_marked(node_id, "initialized") to be False
        node_id = list(self.dag.nodes.keys())[2]
        self.runner.mark(node_id, "hasPayload")
        self.assertFalse(self.runner.is_marked(node_id, "initialized"))

        # C4: Mark a node "hasAdapter", "hasPayload" and "completed"
        #  Expect runner.is_marked(node_id, "initialized") to be False
        #  runner.is_marked(node_id, "initialized") to be True
        node_id = list(self.dag.nodes.keys())[3]
        self.runner.mark(node_id, "hasAdapter")
        self.runner.mark(node_id, "hasPayload")
        self.runner.mark(node_id, "completed")
        self.assertFalse(self.runner.is_marked(node_id, "initialized"))
        self.assertTrue(self.runner.is_marked(node_id, "completed"))
