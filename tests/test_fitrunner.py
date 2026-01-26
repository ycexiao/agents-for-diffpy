from FitRunner import FitRunner
from PDFAdapter import PDFAdapter
from FitDAG import FitDAG
from pathlib import Path
import re
import unittest


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
