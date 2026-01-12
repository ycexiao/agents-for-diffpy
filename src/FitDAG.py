import networkx as nx
import uuid
import copy
from collections import defaultdict
import pickle
import re
from networkx.readwrite.json_graph import node_link_data
import json


class FitDAG(nx.DiGraph):
    """
    A directed acyclic graph (DAG) representing fitting instructions.

    Each node defines an action executed under the context of the input from
    root node and the payload from the parent node. Each action updates
    the payload in the current node.

    # FIXME: This is the current policy. We could allow more flexible policies
    # e.g. combine the payloads from multiple parent nodes or root nodes.
    # modify get_input_source and get_payload_source methods accordingly.

    If multiple parent nodes exist, use the one has the nearest distance to
    the root node.

    If multiple root nodes exist, use the one has the furthest distance to the
    current node.

    payload is any data structure that is updated along the DAG.


    Node Attributes
    ----------------
    description: str
        string description of the current node
    inputs: dict
    action: list
        list of operations being executed in the current node

    Edge Attributes
    ---------------
    description: str
        string description of the current edge
    """

    def __init__(self):
        super().__init__()
        self.names = []
        self.default_node = {
            "description": "",
            "id": "",
            "name": "",
            "buffer": {},
            "payload": {},
            "action": [],
        }
        self.default_edge = {
            "description": "",
            "source": None,
            "target": None,
        }

    @property
    def root_nodes(self):
        return [
            node_id for node_id in self.nodes() if self.in_degree(node_id) == 0
        ]

    @property
    def leaf_nodes(self):
        return [
            node_id
            for node_id in self.nodes()
            if self.out_degree(node_id) == 0
        ]

    def furnish_node_dict(self, node_dict):
        template_node = self.default_node
        node_dict = {
            key: (
                node_dict[key]
                if key in node_dict.keys()
                else template_node[key]
            )
            for key in template_node.keys()
        }
        # FIXME: refine the error later
        assert "action" in node_dict
        name = (
            node_dict["name"]
            if node_dict["name"] != ""
            else ", ".join(node_dict["action"])
        )
        node_dict["name"] = name
        self.names.append(name)
        return node_dict

    def furnish_edge_dict(self, edge_dict):
        template_edge = self.default_edge
        edge_dict = {
            key: (
                edge_dict[key]
                if key in edge_dict.keys()
                else template_edge[key]
            )
            for key in template_edge.keys()
        }
        return edge_dict

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
                    {"id": "1", "name": "node1", "action": "scale"},
                    {"id": "2", "name": "node2", "action": "alpha"},
                    {"id": "3", "name": "node3", "action": "all"},
                ],
                "edges": [
                {"source": "1", "target": "2"},
                {"source": "2", "target": "3"}
                ],
            }
        """
        self.clear()
        for node_content in data["nodes"]:
            action = node_content["action"]
            if isinstance(action, str):
                node_content["action"] = [v.strip() for v in action.split(",")]
            elif isinstance(action, list):
                node_content["action"] = action
            else:
                raise TypeError(
                    "Only str and list of str supported for desginating actions."
                )
            node_content = self.furnish_node_dict(node_content)
            node_id = node_content.get("id", str(uuid.uuid4()))
            self.add_node(node_id, **node_content)
        for edge in data["edges"]:
            edge = self.furnish_edge_dict(edge)
            self.add_edge(edge.pop("source"), edge.pop("target"), **edge)

    def from_str(self, dag_str):
        """
        Parse a linear DAG from a string representation.

        Parameters
        ----------
        diag_str : str
            String representation of the fitting diagram.
            E.g.: "scale->alpha->a->qdamp->all"
        """
        self.clear()
        actions = [
            [v.strip() for v in value.split(",")]
            for value in dag_str.split("->")
        ]
        node_content = {"action": actions[0]}
        node_content = self.furnish_node_dict(node_content)
        parent_node_id = str(uuid.uuid4())
        self.add_node(parent_node_id, **node_content)
        for i in range(1, len(actions)):
            node_content = {"action": actions[i]}
            node_content = self.furnish_node_dict(node_content)
            child_node_id = str(uuid.uuid4())
            self.add_node(child_node_id, **node_content)
            edge = self.furnish_edge_dict({})
            self.add_edge(parent_node_id, child_node_id, **edge)
            parent_node_id = child_node_id

    def get_node_by_name(self, name):
        assert self.names.count(name) == 1
        nodes = []
        for node_id, node_content in self.nodes(data=True):
            if node_content["name"] == name:
                return node_id, node_content

    def get_edge_by_name(self, u_name, v_name):
        u_node_id, _ = self.get_node_by_name(u_name)[0]
        v_node_id, _ = self.get_node_by_name(v_name)[0]
        return self[u_node_id][v_node_id]

    def clean_copy(
        self,
        with_payload=False,
        with_same_id=True,
    ):
        graph = (
            nx.DiGraph()
        )  # use nx.DiGraph so copies can be serialized with minimal compatible issues
        id_maps = {}
        for node_id, node_content in self.nodes(data=True):
            node_content["buffer"] = None
            if with_payload:
                pass
            else:
                node_content["payload"] = None
            if with_same_id:
                id_maps[node_id] = node_id
            else:
                new_node_id = str(uuid.uuid4())
                id_maps[node_id] = new_node_id
                node_id = new_node_id
            graph.add_node(node_id, **node_content)

        for u, v, edge_content in self.edges(data=True):
            graph.add_edge(id_maps[u], id_maps[v], **edge_content)
        return graph

    def to_json(self, filename="graph.json"):
        graph = self.clean_copy(with_payload=True, with_same_id=True)

        data = node_link_data(
            graph,
            source="source",  # edge source key
            target="target",  # edge target key
            name="id",  # node id key
            edges="edges",  # what to call the edge list
            nodes="nodes",  # what to call the node list
        )
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def from_json(self, filename):
        with open(filename, "r") as f:
            graph_dict = json.load(f)
        self.from_dict(graph_dict)

    def render(self, filename="graph.html"):
        from pyvis.network import Network

        net = Network(
            directed=True,
            notebook=False,
            cdn_resources="remote",
        )
        graph = self.clean_copy(with_payload=False)
        node_ids, node_lables = [], []
        for node_id, node_content in graph.nodes(data=True):
            node_ids.append(node_id)
            node_lables.append(node_content["name"])
        net.add_nodes(node_ids, label=node_lables)
        net.add_edges(graph.edges())
        # Save and render
        # net.write_html(filename)
        net.show(filename, notebook=False)

    def clear(self):
        super().clear()
        self.names = []

    def mark(self, node_id, tag):
        if "node_status" not in self.running_metadata:
            self.running_metadata["node_status"] = defaultdict(list)
        allowed_tags = ["hasPayload", "hasAdapter", "completed"]
        # FIXME: implement the error later
        assert tag in allowed_tags
        self.running_metadata["node_status"][node_id].append(tag)

    def is_marked(self, node_id, status):
        if "node_status" not in self.running_metadata:
            return False
        if status == "initialized":
            return set(["hasPayload", "hasAdapter"]) == set(
                self.running_metadata["node_status"][node_id]
            )
        elif status == "completed":
            return (
                len(self.running_metadata["node_status"][node_id]) == 3
                and "completed"
                == self.running_metadata["node_status"][node_id][-1]
            )
