from PDFAdapter import PDFAdapter
from pathlib import Path
from unittest import TestCase


class TestPDFAdapter(TestCase):
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
        adapter = PDFAdapter()
        adapter.load_inputs(inputs)
        self.inputs = inputs
        self.adapter = adapter

    def test_payload(self):
        # C1: Use adapter.apply_payload.
        #   Expect the instance to have a new payload.
        expected_payload = {
            "scale": 0.4,
            "a": 3.52,
            "Uiso_0": 0.005,
            "delta2": 2.0,
            "qdamp": 0.04,
            "qbroad": 0.02,
        }
        self.adapter.apply_payload(expected_payload)
        current_payload = self.adapter.get_payload()
        current_payload = {
            key: current_payload[key] for key in expected_payload
        }
        self.assertEqual(current_payload, expected_payload)
        # C2: Use adapter.apply_payload.
        #  Expect the residual to change after applying a new payload
        residual_before = self.adapter._residual()
        self.adapter.apply_payload({"scale": 0.5})
        residual_after = self.adapter._residual()
        self.assertNotEqual(sum(residual_before), sum(residual_after))

    def test_action(self):
        # C4: Use different initial payload values to test the optimization.
        #  Expect the final optimized value to be the same.
        self.adapter.load_inputs(self.inputs)
        self.adapter.apply_payload({"scale": 0.4})
        self.adapter.action_func_factory(["scale"])()
        pv_dict_1 = self.adapter._get_parameter_values()
        self.adapter.apply_payload({"scale": 0.4})
        self.adapter.action_func_factory(["scale"])()
        pv_dict_2 = self.adapter._get_parameter_values()
        self.assertEqual(pv_dict_1, pv_dict_2)

    def test_clone(self):
        # C5: clone the adapter.
        #  Expect the cloned adapter to have the same payload as the original
        new_adapter = self.adapter.clone()
        self.assertEqual(new_adapter.get_payload(), self.adapter.get_payload())
