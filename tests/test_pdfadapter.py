from PDFAdapter import PDFAdapter
from pathlib import Path
import numpy

profile_path = Path("data/Ni.gr")
structure_path = Path("data/Ni.cif")

inputs = {
    "profile_string": profile_path.read_text(),
    "structure_string": structure_path.read_text(),
    "xmin": 1.5,
    "xmax": 50,
    "dx": 0.01,
    "qmax": 25.0,
    "qmin": 0.1,
}

def test_load_inputs():
    adapter = PDFAdapter()
    adapter.load_inputs(inputs)


def test_payloads():
    adapter = PDFAdapter()
    adapter.load_inputs(inputs)
    expected_payload = {
        "scale": 0.4,
        "a": 3.52,
        "Uiso_0": 0.005,
        "delta2": 2.0,
        "qdamp": 0.04,
        "qbroad": 0.02,
    }
    current_payload = adapter.get_payload()
    assert current_payload == {key:current_payload[key] for key in current_payload}
    adapter.apply_payload(expected_payload)
    current_payload = adapter.get_payload()
    assert current_payload == {key:current_payload[key] for key in current_payload}


def test_action():
    adapter = PDFAdapter()
    adapter.load_inputs(inputs)
    expected_a = 3.522729608872
    adapter.apply_payload({"a":3.52})  # even using 3.53 as the initial value won't stable convegence to the expected_a. This is with diffpy.srfit
    assert not numpy.isclose(adapter.get_payload()['a'], expected_a)
    adapter.action_func_factory('a')()
    print(adapter.get_payload()['a'])
    assert numpy.isclose(adapter.get_payload()['a'], expected_a)


def test_clone():
    adapter = PDFAdapter()
    adapter.load_inputs(inputs)
    new_adapter = adapter.clone()
    assert new_adapter.get_payload() == adapter.get_payload()
    
