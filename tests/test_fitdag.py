from FitDAG import FitDAG
import tempfile
from pathlib import Path
import json


def test_from():
    def check_dag(dag):
        actions = ["a", "scale", "delta2", "Uiso", "all"]
        assert len(dag.nodes) == len(actions)
        for node_id, node_content in dag.nodes(data=True):
            assert len(node_content["action"]) == 1
            assert node_content["action"][0] in actions

    dag_str = "a->scale->delta2->Uiso->all"
    dag_from_str = FitDAG()
    dag_from_str.from_str(dag_str)

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

    dag_from_json = FitDAG()
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "graph.json"
        dag_from_dict.to_json(str(file_path))
        dag_from_json.from_json(str(file_path))

    check_dag(dag_from_str)
    check_dag(dag_from_dict)
    check_dag(dag_from_json)
