import unittest
from FitDAG import FitDAG
import tempfile
from pathlib import Path
import networkx as nx


class TestFitDAG(unittest.TestCase):
    def test_from(self):
        # C1: Test from_str
        #  Expect the DAG to have nodes in the order of actions
        def dag_is_correct(dag):
            actions = ["a", "scale", "delta2", "Uiso", "all"]
            node_ids = nx.topological_sort(dag)
            assert len(dag.nodes) == len(actions)
            for i, node_id in enumerate(node_ids):
                assert dag.nodes[node_id]["action"][0] == actions[i]

        dag_str = "a->scale->delta2->Uiso->all"
        dag_from_str = FitDAG()
        dag_from_str.from_str(dag_str)
        dag_is_correct(dag_from_str)
        # C2: Test from_dict
        #  Expect the DAG to have nodes in the order of actions
        dag_dict = {
            "nodes": [
                {"id": "1", "action": "a"},
                {"id": "2", "action": "scale"},
                {"id": "3", "action": "delta2"},
                {"id": "4", "action": "Uiso"},
                {"id": "5", "action": "all"},
            ],
            "edges": [
                {"source": "1", "target": "2"},
                {"source": "2", "target": "3"},
                {"source": "3", "target": "4"},
                {"source": "4", "target": "5"},
            ],
        }
        dag_from_dict = FitDAG()
        dag_from_dict.from_dict(dag_dict)
        dag_is_correct(dag_from_dict)
        # C3: Test from_json
        #  Expect the DAG to have nodes in the order of actions
        dag_from_json = FitDAG()
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "graph.json"
            dag_from_dict.to_json(str(file_path))
            dag_from_json.from_json(str(file_path))
        dag_is_correct(dag_from_json)

    def test_copy(self):
        dag_str = "a->scale->delta2->Uiso->all"
        dag_from_str = FitDAG()
        dag_from_str.from_str(dag_str)
        for node_id in dag_from_str.nodes():
            dag_from_str.nodes[node_id]["payload"] = {"pname": 0}
        # C1: with_payload=True, with_same_id=True
        #  Expect copied_dag = original_dag
        copied_dag = dag_from_str.copy(with_payload=True, with_same_id=True)
        self.assertTrue(nx.utils.graphs_equal(dag_from_str, copied_dag))
        # C2: with_payload=False, with_same_id=True
        #  Expect copied_dag to have the same node_ids, same structure,
        #  but payloads should be None
        copied_dag = dag_from_str.copy(with_payload=False, with_same_id=True)
        for node_id in dag_from_str.nodes():
            self.assertTrue(node_id in copied_dag.nodes)
            self.assertTrue(copied_dag.nodes[node_id]["payload"] is None)
        self.assertTrue(nx.is_isomorphic(dag_from_str, copied_dag))
        # C3: with_payload=False, with_same_id=False
        #  Expect copied_dag to have different node_ids, same structure,
        #  and payloads should be None
        copied_dag = dag_from_str.copy(with_payload=False, with_same_id=False)
        for node_id in dag_from_str.nodes():
            self.assertTrue(node_id not in copied_dag.nodes)
        for node_id in copied_dag.nodes():
            self.assertTrue(copied_dag.nodes[node_id]["payload"] is None)
        self.assertTrue(nx.is_isomorphic(dag_from_str, copied_dag))
