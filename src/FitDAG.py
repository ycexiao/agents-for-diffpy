import networkx as nx
import uuid
from BaseAdapter import BaseAdapter
import numpy
import json


class FitDAG:
    "A directed acyclic graph (DAG) representing fitting instructions."

    def __init__(self, adapter=BaseAdapter):
        self.init_from_adapter(adapter)

    def init_from_adapter(self, adapter):
        self.allowed_actions = [
            adapter.parameter_names_in_slots[i]
            for i in range(len(adapter.parameter_names_in_slots))
            if adapter.parameter_slots_mask[i] == 1
        ]
        self.allowed_actions.append("all")

    def from_dict(self, data):
        """
        Initialize the fitting diagram from a dictionary representation.

        Parameters
        ----------
        data : dict
            A dictionary representation of the fitting diagram.
            e.g.:
            {
                "nodes": [
                    {"id": "1", "value": "scale"},
                    {"id": "2", "value": "alpha"},
                    {"id": "3", "value": "all"},
                ],
                "edges": [("1", "2"), ("2", "3")],
            }
        """
        self.graph = nx.DiGraph()
        all_node_values = []
        for node in data["nodes"]:
            value = [v.strip() for v in node["value"].split(",")]
            self.graph.add_node(
                node["id"], value=value, runtime_status="pending"
            )
            all_node_values.extend(value)
        for edge in data["edges"]:
            self.graph.add_edge(edge[0], edge[1])
        all_node_values = list(set(all_node_values))
        self.__check_node_values(all_node_values)

    def from_str(self, dag_str):
        """
        Parse a linear DIAG from a string representation.

        Parameters
        ----------
        diag_str : str
            String representation of the fitting diagram.
            E.g.: "scale->alpha->a->qdamp->all"
        """
        self.graph = nx.DiGraph()
        nodes_values = [
            [v.strip() for v in value.split(",")]
            for value in dag_str.split("->")
        ]
        all_node_values = list(
            set([v for sublist in nodes_values for v in sublist])
        )
        parent_node_id = str(uuid.uuid4())
        self.graph.add_node(
            parent_node_id, value=nodes_values[0], runtime_status="pending"
        )
        for i in range(1, len(nodes_values)):
            child_node_id = str(uuid.uuid4())
            self.graph.add_node(
                child_node_id, value=nodes_values[i], runtime_status="pending"
            )
            self.graph.add_edge(parent_node_id, child_node_id)
            parent_node_id = child_node_id
        self.__check_node_values(all_node_values)

    def tag_node(self, node_id, tag_value):
        """
        Tag a node in the fitting diagram with a specific value.

        Parameters
        ----------
        node_id : str
            The ID of the node to be tagged.
        tag_value : any
            The value to tag the node with.
        """
        assert tag_value in [
            "pending",
            "initialized",
            "completed",
            "failed",
        ]
        self.__check_node_ids([node_id])
        self.graph.nodes[node_id]["tag"] = tag_value

    def tag_is(self, node_id, tag_value):
        assert tag_value in [
            "pending",
            "initialized",
            "completed",
            "failed",
        ]
        self.__check_node_ids([node_id])
        return self.graph.nodes[node_id]["tag"] == tag_value

    def to_json(self):
        """
        Serialize the fitting diagram to a JSON-compatible dictionary.

        Returns
        -------
        dict
            A dictionary representation of the fitting diagram.
        """
        data = {
            "nodes": [
                {"id": node, "value": self.graph.nodes[node]["value"]}
                for node in self.graph.nodes
            ],
            "edges": [[u, v] for u, v in self.graph.edges],
        }
        json_str = json.dumps(data, indent=4)
        return json_str

    def __check_node_values(self, nodes_values):
        undifined_operations = set(nodes_values) - set(self.allowed_actions)
        if undifined_operations:
            raise ValueError(
                f"Fitting diagram contains undefined operations: "
                f"{sorted(undifined_operations)}."
            )

    def __check_node_ids(self, nodes_ids):
        undifined_node_ids = [
            node_id for node_id in nodes_ids if node_id not in self.graph.nodes
        ]
        if undifined_node_ids:
            raise ValueError(
                f"Fitting diagram contains undefined node IDs: "
                f"{sorted(undifined_node_ids)}."
            )
